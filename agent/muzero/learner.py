"""MuZero learner: self-play -> replay -> unrolled updates -> checkpoint.

Simple synchronization (§28): the learner process owns the network. Self-play
workers (processes) each receive a model *snapshot* for their episode batch,
run real-simulator episodes with the latent MCTS, and return numeric episode
arrays to the parent, which appends them to the replay buffer and then trains.

    learner
      |  periodic model snapshot
      v
    self-play workers (ProcessPoolExecutor)
      |  numeric episode arrays
      v
    replay buffer
      |
      v
    unrolled updates (train_step)  ->  checkpoint / artifact
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import numpy as np
import torch

from ..simulator import GameConfig
from .config import MuZeroConfig
from .metrics import Metrics
from .networks import MuZeroNetwork, save_checkpoint
from .replay import Episode, ReplayBuffer, episode_from_dict, episode_to_dict
from .self_play import MuZeroSelfPlay
from .train import train_step


def _sp_worker(
    payload: tuple[
        dict[str, object],
        int,
        int,
        int,
        int,
        dict[str, object],
        GameConfig,
        list[int],
    ],
) -> list[dict[str, object]]:
    """Self-play worker: build a network from the snapshot and run episodes."""
    (state_dict, obs_dim, action_dim, latent_dim, hidden_dim, config_dict, game_config, seeds) = payload
    config = MuZeroConfig.from_dict(config_dict)
    net = MuZeroNetwork(obs_dim, action_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, seed=0)
    net.load_state_dict(state_dict)
    net.eval()
    sp = MuZeroSelfPlay(net, config, game_config)
    out: list[dict[str, object]] = []
    for seed in seeds:
        out.append(episode_to_dict(sp.run_episode(seed=int(seed))))
    return out


class MuZeroLearner:
    """Runs the closed learning loop (self-play -> replay -> train -> improve)."""

    def __init__(
        self,
        config: MuZeroConfig,
        game_config: GameConfig | None = None,
        *,
        metrics: Metrics | None = None,
    ) -> None:
        self._config = config
        self._game_config = game_config if game_config is not None else GameConfig(
            episode_steps=config.episode_steps, seed=config.seed
        )
        self._metrics = metrics if metrics is not None else Metrics(
            os.path.join(config.checkpoint_dir, "metrics.json")
        )
        self._net = MuZeroNetwork(
            139, 60, latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, seed=config.seed
        )
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=config.learning_rate)
        self._replay = ReplayBuffer(config.replay_capacity)
        self._rng = np.random.RandomState(config.seed)
        self._step = 0
        self._target: MuZeroNetwork | None = None
        if config.target_update_interval > 0:
            self._target = self._net.copy_target()

    # -- public API ---------------------------------------------------------

    def train(self, num_updates: int | None = None) -> dict[str, object]:
        """Run the loop for ``num_updates`` training updates (or config default)."""
        config = self._config
        total_updates = num_updates if num_updates is not None else config.num_updates
        workers = config.workers if config.workers > 0 else self._default_workers()
        start = time.perf_counter()
        episode_counter = 0
        last_ep = 0
        last_gen = 0

        while self._step < total_updates:
            # 1) self-play generation (real simulator; latent MCTS)
            episodes = self._gather_episodes(
                config.episodes_per_generation, workers, episode_counter
            )
            self._replay.extend(episodes)
            episode_counter += len(episodes)
            for ep in episodes:
                self._record_episode(ep)

            # 2) training updates from replay
            n_updates = min(config.updates_per_generation, total_updates - self._step)
            for _ in range(n_updates):
                positions = [
                    self._replay.sample_position(self._rng, config.unroll_steps)
                    for _ in range(config.batch_size)
                ]
                m = train_step(self._net, self._optimizer, positions, config)
                self._step += 1
                self._metrics.record(
                    self._step,
                    {
                        "policy_loss": m["policy"],
                        "value_loss": m["value"],
                        "reward_loss": m["reward"],
                        "total_loss": m["total"],
                        "policy_entropy": m["entropy"],
                        "replay_size": float(self._replay.size),
                        "replay_transitions": float(self._replay.total_transitions),
                    },
                )
                if self._target is not None and self._step % config.target_update_interval == 0:
                    self._net.sync_target(self._target)
                if config.verbose and self._step % config.log_interval == 0:
                    print(
                        f"[step {self._step}/{total_updates}] "
                        f"total={m['total']:.4f} p={m['policy']:.4f} v={m['value']:.4f} "
                        f"r={m['reward']:.4f} ent={m['entropy']:.3f} "
                        f"replay={self._replay.size} "
                        f"episodes={episode_counter} ({time.perf_counter()-start:.0f}s)"
                    )
            self.checkpoint(tag="latest")

        total = time.perf_counter() - start
        self.checkpoint(tag="final")
        summary: dict[str, object] = {
            "updates": self._step,
            "episodes": episode_counter,
            "wall_seconds": total,
            "params": self._net.num_parameters(),
        }
        print(f"[learner] done: {summary}")
        return summary

    # -- self-play gathering ------------------------------------------------

    def _gather_episodes(self, num_episodes: int, workers: int, offset: int) -> list[Episode]:
        if workers <= 1:
            sp = MuZeroSelfPlay(self._net, self._config, self._game_config)
            return [sp.run_episode(seed=self._config.seed + offset + i) for i in range(num_episodes)]
        seeds = [self._config.seed + offset + i for i in range(num_episodes)]
        chunks = [seeds[w::workers] for w in range(workers)]
        chunks = [c for c in chunks if c]
        snapshot = (
            self._net.state_dict(),
            self._net.obs_dim,
            self._net.action_dim,
            self._net.latent_dim,
            self._net.hidden_dim,
            self._config.to_dict(),
            self._game_config,
        )
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [
                pool.submit(
                    _sp_worker,
                    (snapshot[0], snapshot[1], snapshot[2], snapshot[3], snapshot[4],
                     snapshot[5], snapshot[6], chunk),
                )
                for chunk in chunks
            ]
            results = [f.result() for f in futures]
        episodes: list[Episode] = []
        for r in results:
            episodes.extend(episode_from_dict(d) for d in r)
        return episodes

    # -- checkpoint / resume ------------------------------------------------

    def checkpoint(self, tag: str = "latest") -> str:
        os.makedirs(self._config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self._config.checkpoint_dir, f"muzero_{tag}.pt")
        save_checkpoint(
            self._net,
            path,
            optimizer=self._optimizer,
            step=self._step,
            extra={
                "config": self._config.to_dict(),
                "replay_transitions": self._replay.total_transitions,
            },
        )
        replay_path = os.path.join(self._config.checkpoint_dir, f"replay_{tag}.pkl")
        self._replay.save(replay_path)
        return path

    def resume(self, checkpoint_path: str | None = None) -> None:
        from .networks import build_network_from_checkpoint, load_checkpoint

        ckpt_dir = self._config.checkpoint_dir
        path = checkpoint_path or os.path.join(ckpt_dir, "muzero_latest.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no checkpoint to resume: {path}")
        ckpt = load_checkpoint(path)
        net = build_network_from_checkpoint(ckpt)
        self._net.load_state_dict(net.state_dict())
        if "optimizer" in ckpt:
            self._optimizer.load_state_dict(cast(dict[str, Any], ckpt["optimizer"]))
        self._step = int(cast(int, ckpt.get("step", 0)))
        replay_path = os.path.join(ckpt_dir, "replay_latest.pkl")
        if os.path.exists(replay_path):
            self._replay = ReplayBuffer.load(replay_path)
        print(f"[resume] step {self._step}, replay {self._replay.size}")

    # -- artifact export (deployment-only: model + encoders + config) -------

    def export_artifact(self, path: str | None = None) -> str:
        artifact = path or os.path.join(self._config.artifact_dir, "muzero_model.pt")
        os.makedirs(os.path.dirname(artifact), exist_ok=True)
        save_checkpoint(
            self._net, artifact, step=self._step, extra={"config": self._config.to_dict()}
        )
        return artifact

    # -- helpers ------------------------------------------------------------

    def _record_episode(self, ep: Episode) -> None:
        # Terminal cash = starting money + sum(scaled rewards) / reward_scale,
        # per player (reward_scale is a fixed invertible multiplier).
        mask0 = ep.players == 0
        mask1 = ep.players == 1
        starting = float(self._game_config.starting_money)
        cash0 = starting + float(ep.rewards[mask0].sum()) / self._config.reward_scale
        cash1 = starting + float(ep.rewards[mask1].sum()) / self._config.reward_scale
        win0 = 1.0 if cash0 > cash1 else 0.0
        self._metrics.record(
            self._step,
            {
                "episode_len": float(ep.length),
                "terminal_cash0": cash0,
                "terminal_cash1": cash1,
                "win0": win0,
            },
        )

    def _default_workers(self) -> int:
        return max(1, min(4, (os.cpu_count() or 2) // 2))
