"""Replay buffer for MuZero training.

Stores complete episodes in an efficient numeric format: numpy arrays for the
fixed-width observation / action-index / reward / return sequences, and
per-step lists for the *ragged* candidate embeddings and policy targets
(each step has a variable number of candidates). No ``GameState`` /
``TurnAction`` objects are stored — the buffer never touches simulator object
graphs, so it is cheap to save / load / shuffle.

Support (uniform sampling for now):

* ``append(episode)``
* ``sample_position(rng, unroll_steps)`` — a training position with the
  unrolled action / reward / value / policy targets
* ``save(path)`` / ``load(path)``

Value targets are the **exact Monte-Carlo returns** from the stored episode
(undiscounted by default, ``gamma`` configurable), so no bootstrap / target
network is needed for the value head unless ``bootstrap`` is enabled.
"""

from __future__ import annotations

import os
import pickle
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """One complete self-play episode (numeric arrays only)."""

    obs: np.ndarray  # (T, D_obs) float32
    candidate_embs: list[np.ndarray]  # per-step (n_t, D_a) float32
    action_indices: np.ndarray  # (T,) int32 chosen action index per step
    policy_targets: list[np.ndarray]  # per-step (n_t,) visit distribution
    rewards: np.ndarray  # (T,) float32 scaled rewards (cash delta * reward_scale)
    returns: np.ndarray  # (T,) float32 exact future returns (suffix sums)
    players: np.ndarray  # (T,) int8 acting player id
    terminal: bool
    length: int = 0

    def __post_init__(self) -> None:
        self.length = int(self.obs.shape[0])


def compute_returns(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Exact future returns ``returns[t] = sum_{i>=t} gamma^(i-t) r[i]``."""
    t = len(rewards)
    rets = np.zeros(t, dtype=np.float32)
    acc = 0.0
    for i in range(t - 1, -1, -1):
        acc = float(rewards[i]) + gamma * acc
        rets[i] = acc
    return rets


@dataclass(frozen=True, slots=True)
class TrainingPosition:
    """A sampled position with everything the unrolled loss needs."""

    obs: np.ndarray  # (D_obs,)
    cand_embs: list[np.ndarray]  # (n_{t+k}, D_a) for k = 0..num_unroll
    action_indices: np.ndarray  # (num_unroll,) chosen actions for the unroll
    policy_targets: list[np.ndarray]  # (n_{t+k},) for k = 0..num_unroll
    reward_targets: np.ndarray  # (num_unroll,) r[t+1 .. t+num_unroll]
    value_targets: np.ndarray  # (num_unroll + 1,) v_target(t .. t+num_unroll)
    num_unroll: int
    player: int


class ReplayBuffer:
    """Uniform-sampling replay buffer of complete episodes."""

    def __init__(self, capacity: int = 10000) -> None:
        self._capacity = max(1, int(capacity))
        self._episodes: deque[Episode] = deque(maxlen=self._capacity)
        self._total_transitions = 0

    @property
    def size(self) -> int:
        return len(self._episodes)

    @property
    def total_transitions(self) -> int:
        return self._total_transitions

    def append(self, episode: Episode) -> None:
        self._total_transitions += episode.length
        self._episodes.append(episode)

    def extend(self, episodes: list[Episode]) -> None:
        for ep in episodes:
            self.append(ep)

    def sample_position(self, rng: np.random.RandomState, unroll_steps: int) -> TrainingPosition:
        """Sample a random transition and build its unrolled targets."""
        if not self._episodes:
            raise RuntimeError("replay buffer is empty")
        ep = self._episodes[rng.randint(len(self._episodes))]
        t = int(rng.randint(ep.length))
        return make_position(ep, t, unroll_steps)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "episodes": [
                {
                    "obs": ep.obs,
                    "candidate_embs": ep.candidate_embs,
                    "action_indices": ep.action_indices,
                    "policy_targets": ep.policy_targets,
                    "rewards": ep.rewards,
                    "returns": ep.returns,
                    "players": ep.players,
                    "terminal": ep.terminal,
                }
                for ep in self._episodes
            ],
            "capacity": self._capacity,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301 (trusted local artifact)
        buf = cls(capacity=int(payload["capacity"]))
        for d in payload["episodes"]:
            buf.append(
                Episode(
                    obs=np.asarray(d["obs"], dtype=np.float32),
                    candidate_embs=[np.asarray(e, dtype=np.float32) for e in d["candidate_embs"]],
                    action_indices=np.asarray(d["action_indices"], dtype=np.int32),
                    policy_targets=[np.asarray(p, dtype=np.float32) for p in d["policy_targets"]],
                    rewards=np.asarray(d["rewards"], dtype=np.float32),
                    returns=np.asarray(d["returns"], dtype=np.float32),
                    players=np.asarray(d["players"], dtype=np.int8),
                    terminal=bool(d["terminal"]),
                )
            )
        return buf


def episode_to_dict(ep: Episode) -> dict[str, object]:
    """Picklable dict form for shipping episodes across processes."""
    return {
        "obs": ep.obs,
        "candidate_embs": ep.candidate_embs,
        "action_indices": ep.action_indices,
        "policy_targets": ep.policy_targets,
        "rewards": ep.rewards,
        "returns": ep.returns,
        "players": ep.players,
        "terminal": ep.terminal,
    }


def episode_from_dict(d: dict[str, object]) -> Episode:
    from typing import cast

    return Episode(
        obs=np.asarray(d["obs"], dtype=np.float32),
        candidate_embs=[np.asarray(e, dtype=np.float32) for e in cast(list[object], d["candidate_embs"])],
        action_indices=np.asarray(d["action_indices"], dtype=np.int32),
        policy_targets=[np.asarray(p, dtype=np.float32) for p in cast(list[object], d["policy_targets"])],
        rewards=np.asarray(d["rewards"], dtype=np.float32),
        returns=np.asarray(d["returns"], dtype=np.float32),
        players=np.asarray(d["players"], dtype=np.int8),
        terminal=bool(d["terminal"]),
    )


def make_position(ep: Episode, t: int, unroll_steps: int) -> TrainingPosition:
    """Build a training position starting at step ``t`` of ``ep``.

    Unrolls as far as the episode allows (``num_unroll <= unroll_steps``);
    reward targets are the stored rewards at t+1.., value targets the stored
    exact returns at t.. (the future return is known because the whole episode
    is stored). The policy target at the *end* of the window (t+num_unroll) is
    included so every latent state in the unroll has a policy supervision.
    """
    t = min(t, ep.length - 1)
    obs = np.asarray(ep.obs[t], dtype=np.float32)
    cand_embs = [np.asarray(ep.candidate_embs[t], dtype=np.float32)]
    policy_targets = [np.asarray(ep.policy_targets[t], dtype=np.float32)]
    actions: list[int] = []
    rewards: list[float] = []
    value_targets = [float(ep.returns[t])]

    k = 0
    while k < unroll_steps and (t + k + 1) < ep.length:
        actions.append(int(ep.action_indices[t + k]))
        rewards.append(float(ep.rewards[t + k]))
        value_targets.append(float(ep.returns[t + k + 1]))
        cand_embs.append(np.asarray(ep.candidate_embs[t + k + 1], dtype=np.float32))
        policy_targets.append(np.asarray(ep.policy_targets[t + k + 1], dtype=np.float32))
        k += 1

    return TrainingPosition(
        obs=obs,
        cand_embs=cand_embs,
        action_indices=np.asarray(actions, dtype=np.int32),
        policy_targets=policy_targets,
        reward_targets=np.asarray(rewards, dtype=np.float32),
        value_targets=np.asarray(value_targets, dtype=np.float32),
        num_unroll=len(actions),
        player=int(ep.players[t]),
    )
