"""Synthetic end-to-end test (Task 22): MuZero learns to prefer the good action."""

from __future__ import annotations

import numpy as np

from agent.muzero.config import MuZeroConfig
from agent.muzero.synthetic_env import REWARD_A, root_policy_probs, train_synthetic


def test_synthetic_env_learns_preferred_action() -> None:
    """After training, the root policy must prefer action A (reward 10)."""
    result = train_synthetic(
        episode_length=4,
        episodes_per_generation=5,
        updates_per_generation=30,
        num_generations=8,
        latent_dim=16,
        hidden_dim=32,
        simulations=20,
        seed=0,
    )
    net = result["net"]
    cfg = MuZeroConfig.from_dict(result["config"])
    probs = root_policy_probs(net, cfg, num_simulations=40)
    # action 0 = A, action 1 = B
    assert float(probs[0]) > 0.8
    assert float(probs[1]) < float(probs[0])


def test_synthetic_loss_decreases_no_nan() -> None:
    result = train_synthetic(
        episode_length=3,
        episodes_per_generation=4,
        updates_per_generation=20,
        num_generations=3,
        seed=1,
    )
    hist = result["metrics"]
    assert hist[-1]["total"] < hist[0]["total"]
    for entry in hist:
        assert np.isfinite(entry["total"])
        assert np.isfinite(entry["policy"])
        assert np.isfinite(entry["value"])
        assert np.isfinite(entry["reward"])


def test_reward_equivalence() -> None:
    """Sum of dense rewards = final cash - initial cash (synthetic proof)."""
    import numpy as np

    from agent.muzero.replay import compute_returns

    rewards = np.asarray([REWARD_A, REWARD_A, 0.0, REWARD_A], dtype=np.float32)
    rets = compute_returns(rewards)
    assert rets[0] == rewards.sum()  # undiscounted return from the start
    assert rets[1] == rewards[1:].sum()
