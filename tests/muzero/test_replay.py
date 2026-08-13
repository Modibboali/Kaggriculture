"""Replay buffer tests: append, sample, save/load, sequence validity."""

from __future__ import annotations

import numpy as np
import pytest

from agent.muzero.replay import (
    Episode,
    ReplayBuffer,
    compute_returns,
    make_position,
)


def _episode(seed: int = 0, length: int = 6, n_cands: int = 4) -> Episode:
    rng = np.random.RandomState(seed)
    obs = rng.randn(length, 3).astype(np.float32)
    embs = [rng.randn(n_cands, 2).astype(np.float32) for _ in range(length)]
    acts = rng.randint(0, n_cands, size=length).astype(np.int32)
    policies = [np.full(n_cands, 1.0 / n_cands, dtype=np.float32) for _ in range(length)]
    rewards = rng.randn(length).astype(np.float32) * 0.5
    returns = compute_returns(rewards)
    return Episode(
        obs=obs,
        candidate_embs=embs,
        action_indices=acts,
        policy_targets=policies,
        rewards=rewards,
        returns=returns,
        players=np.zeros(length, dtype=np.int8),
        terminal=True,
    )


def test_compute_returns_equivalence() -> None:
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    rets = compute_returns(rewards, gamma=1.0)
    assert rets[0] == pytest.approx(6.0)
    assert rets[1] == pytest.approx(5.0)
    assert rets[2] == pytest.approx(3.0)
    rets_d = compute_returns(rewards, gamma=0.5)
    assert rets_d[0] == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 3.0)


def test_append_and_size() -> None:
    buf = ReplayBuffer(capacity=10)
    assert buf.size == 0
    buf.append(_episode(0))
    buf.append(_episode(1))
    assert buf.size == 2
    assert buf.total_transitions == 12


def test_capacity_eviction() -> None:
    buf = ReplayBuffer(capacity=2)
    for i in range(5):
        buf.append(_episode(i))
    assert buf.size == 2


def test_sample_position_validity() -> None:
    buf = ReplayBuffer()
    ep = _episode(0, length=6, n_cands=4)
    buf.append(ep)
    rng = np.random.RandomState(0)
    pos = buf.sample_position(rng, unroll_steps=3)
    assert pos.obs.shape == (3,)
    assert len(pos.cand_embs) == pos.num_unroll + 1
    assert len(pos.policy_targets) == pos.num_unroll + 1
    assert pos.action_indices.shape == (pos.num_unroll,)
    assert pos.reward_targets.shape == (pos.num_unroll,)
    assert pos.value_targets.shape == (pos.num_unroll + 1,)
    # action indices within bounds
    for k in range(pos.num_unroll):
        assert 0 <= pos.action_indices[k] < pos.cand_embs[k].shape[0]


def test_make_position_matches_returns() -> None:
    ep = _episode(1, length=8)
    pos = make_position(ep, 2, unroll_steps=4)
    # value target at step 0 == stored exact return at t=2
    assert pos.value_targets[0] == pytest.approx(float(ep.returns[2]))
    # reward target k == stored reward at t+k
    assert pos.reward_targets[0] == pytest.approx(float(ep.rewards[2]))
    # value target after unroll == stored return at t+num_unroll
    assert pos.value_targets[-1] == pytest.approx(float(ep.returns[2 + pos.num_unroll]))


def test_make_position_truncated_at_episode_end() -> None:
    ep = _episode(2, length=3)
    pos = make_position(ep, 2, unroll_steps=5)  # only 1 transition left
    assert pos.num_unroll == 0
    assert pos.value_targets.shape == (1,)


def test_save_load_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    buf = ReplayBuffer(capacity=50)
    for i in range(4):
        buf.append(_episode(i))
    path = str(tmp_path / "replay.pkl")
    buf.save(path)
    loaded = ReplayBuffer.load(path)
    assert loaded.size == 4
    assert loaded.total_transitions == buf.total_transitions
    # loaded positions reproduce the original values
    rng = np.random.RandomState(0)
    p1 = buf.sample_position(rng, 2)
    rng2 = np.random.RandomState(0)
    p2 = loaded.sample_position(rng2, 2)
    assert np.array_equal(p1.obs, p2.obs)
    assert np.array_equal(p1.action_indices, p2.action_indices)
    assert np.allclose(p1.reward_targets, p2.reward_targets)
