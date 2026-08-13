"""Training tests: one step, no NaNs, gradient flow, loss composition."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.muzero.config import MuZeroConfig
from agent.muzero.networks import MuZeroNetwork
from agent.muzero.replay import ReplayBuffer
from agent.muzero.train import compute_batch_loss, train_step

from tests.muzero.test_replay import _episode


def _config(**kw) -> MuZeroConfig:  # type: ignore[no-untyped-def]
    base = dict(latent_dim=8, hidden_dim=16, unroll_steps=3, batch_size=4, seed=0)
    base.update(kw)
    return MuZeroConfig(**base)


def _positions(buf: ReplayBuffer, n: int, unroll: int):  # type: ignore[no-untyped-def]
    rng = np.random.RandomState(0)
    return [buf.sample_position(rng, unroll) for _ in range(n)]


def test_loss_finite_and_composition() -> None:
    net = MuZeroNetwork(3, 2, latent_dim=8, hidden_dim=16, seed=0)
    buf = ReplayBuffer()
    for i in range(6):
        buf.append(_episode(i, length=10, n_cands=4))
    cfg = _config()
    losses = compute_batch_loss(net, _positions(buf, 4, 3), cfg)
    for k in ("total", "policy", "value", "reward", "entropy"):
        assert torch.isfinite(losses[k])
    expected = (
        cfg.policy_loss_weight * float(losses["policy"].detach())
        + cfg.value_loss_weight * float(losses["value"].detach())
        + cfg.reward_loss_weight * float(losses["reward"].detach())
    )
    assert float(losses["total"].detach()) == pytest.approx(expected, rel=1e-5)


def test_train_step_no_nan_and_gradient_flow() -> None:
    net = MuZeroNetwork(3, 2, latent_dim=8, hidden_dim=16, seed=0)
    buf = ReplayBuffer()
    for i in range(6):
        buf.append(_episode(i, length=10, n_cands=4))
    cfg = _config(learning_rate=1e-2)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    m = train_step(net, optimizer, _positions(buf, 4, 3), cfg)
    assert np.isfinite(list(m.values())).all()
    after = net.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed  # gradient flowed and updated weights


def test_train_step_reduces_loss_on_synthetic_task() -> None:
    """On a simple task the unrolled loss should decrease after a few steps."""
    from agent.muzero.synthetic_env import train_synthetic

    result = train_synthetic(
        episode_length=3,
        episodes_per_generation=4,
        updates_per_generation=20,
        num_generations=3,
        latent_dim=16,
        hidden_dim=32,
        simulations=10,
        seed=0,
    )
    hist = result["metrics"]
    assert len(hist) > 0
    first = hist[0]["total"]
    last = hist[-1]["total"]
    assert last < first
    assert np.isfinite(hist[-1]["total"])
