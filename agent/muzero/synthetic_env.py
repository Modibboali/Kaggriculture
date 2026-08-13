"""Tiny synthetic environment for the end-to-end MuZero smoke test.

State 0 with two candidate actions:

    state 0
      |-- action A -> reward 10, next state 0
      `-- action B -> reward  0, next state 0

The episode runs ``episode_length`` steps; the optimal policy always picks A
(total reward = 10 * episode_length). This environment deliberately does NOT
depend on Kaggriculture: observation is a single float, action embeddings are
one-hot 2-vectors, and the dynamics/reward are exact. It exercises the full
closed loop — representation, PUCT, dynamics, reward, policy, value, replay
and the unrolled training update — before any real Kaggriculture training.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .config import MuZeroConfig
from .metrics import Metrics
from .networks import MuZeroNetwork
from .puct import MuZeroPUCT
from .replay import Episode, ReplayBuffer, compute_returns
from .train import train_step

OBS_DIM = 1
ACTION_DIM = 2
REWARD_A = 10.0
REWARD_B = 0.0


def action_embeddings() -> np.ndarray:
    """A = [1, 0], B = [0, 1]."""
    return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def run_episode(
    net: MuZeroNetwork,
    config: MuZeroConfig,
    *,
    episode_length: int,
    rng: np.random.RandomState,
) -> Episode:
    """One synthetic self-play episode (MCTS-driven actions)."""
    puct = MuZeroPUCT(net, config)
    obs = np.zeros((episode_length, OBS_DIM), dtype=np.float32)
    cand = [action_embeddings() for _ in range(episode_length)]
    acts: list[int] = []
    policies: list[np.ndarray] = []
    rewards: list[float] = []
    for t in range(episode_length):
        probs, best, _ = puct.search(
            np.asarray([0.0], dtype=np.float32),
            action_embeddings(),
            num_simulations=config.simulations,
            dirichlet=True,
            rng=rng,
        )
        if config.temperature > 1e-6:
            logits = np.log(probs + 1e-8) / config.temperature
            logits = logits - logits.max()
            p = np.exp(logits) / np.exp(logits).sum()
            chosen = int(rng.choice(len(p), p=p))
        else:
            chosen = best
        r = REWARD_A if chosen == 0 else REWARD_B
        acts.append(chosen)
        policies.append(probs)
        rewards.append(r * config.reward_scale)
        obs[t] = 0.0
    rets = compute_returns(np.asarray(rewards, dtype=np.float32), config.gamma)
    return Episode(
        obs=obs,
        candidate_embs=cand,
        action_indices=np.asarray(acts, dtype=np.int32),
        policy_targets=policies,
        rewards=np.asarray(rewards, dtype=np.float32),
        returns=rets,
        players=np.zeros(episode_length, dtype=np.int8),
        terminal=True,
    )


def train_synthetic(
    *,
    episode_length: int = 4,
    episodes_per_generation: int = 8,
    updates_per_generation: int = 30,
    num_generations: int = 10,
    latent_dim: int = 16,
    hidden_dim: int = 32,
    simulations: int = 20,
    c_puct: float = 1.5,
    learning_rate: float = 1e-2,
    seed: int = 0,
    out_dir: str | None = None,
) -> dict[str, object]:
    """Train MuZero on the synthetic env and return final metrics."""
    config = MuZeroConfig(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        simulations=simulations,
        c_puct=c_puct,
        learning_rate=learning_rate,
        episodes_per_generation=episodes_per_generation,
        updates_per_generation=updates_per_generation,
        temperature=1.0,
        seed=seed,
        reward_scale=0.1,
        checkpoint_dir=os.path.join(out_dir, "ckpt") if out_dir else "output/muzero/synth/ckpt",
    )
    net = MuZeroNetwork(OBS_DIM, ACTION_DIM, latent_dim=latent_dim, hidden_dim=hidden_dim, seed=seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(10000)
    rng = np.random.RandomState(seed)
    metrics = Metrics(os.path.join(out_dir, "synthetic_metrics.json") if out_dir else None)

    for gen in range(num_generations):
        for _ in range(episodes_per_generation):
            replay.append(run_episode(net, config, episode_length=episode_length, rng=rng))
        for _ in range(updates_per_generation):
            positions = [
                replay.sample_position(rng, config.unroll_steps) for _ in range(config.batch_size)
            ]
            m = train_step(net, optimizer, positions, config)
            metrics.record(gen * updates_per_generation + _, m)
    return {
        "config": config.to_dict(),
        "metrics": metrics.history,
        "final": {k: v for k, v in metrics.history[-1].items()} if metrics.history else {},
        "net": net,
    }


def root_policy_probs(
    net: MuZeroNetwork, config: MuZeroConfig, *, num_simulations: int = 40, dirichlet: bool = False
) -> np.ndarray:
    """Eval-mode root visit distribution over [A, B] (0 = A, 1 = B)."""
    puct = MuZeroPUCT(net, config)
    probs, _, _ = puct.search(
        np.asarray([0.0], dtype=np.float32),
        action_embeddings(),
        num_simulations=num_simulations,
        dirichlet=dirichlet,
        rng=np.random.RandomState(0),
    )
    return probs
