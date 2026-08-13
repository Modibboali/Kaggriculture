"""Network shape / determinism / action-sensitivity tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.muzero.networks import (
    DynamicsNetwork,
    MuZeroNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)


def test_representation_shape() -> None:
    rep = RepresentationNetwork(obs_dim=10, hidden_dim=16, latent_dim=8, seed=0)
    out = rep(torch.randn(4, 10))
    assert out.shape == (4, 8)


def test_dynamics_shape_and_reward() -> None:
    dyn = DynamicsNetwork(latent_dim=8, action_dim=5, hidden_dim=16, seed=0)
    lat = torch.randn(4, 8)
    a = torch.randn(4, 5)
    next_lat, reward = dyn(lat, a)
    assert next_lat.shape == (4, 8)
    assert reward.shape == (4,)


def test_prediction_policy_and_value_shapes() -> None:
    pred = PredictionNetwork(latent_dim=8, hidden_dim=16, action_dim=5, seed=0)
    lat = torch.randn(3, 8)
    aembs = torch.randn(3, 7, 5)  # 7 candidates
    logits, value = pred(lat, aembs)
    assert logits.shape == (3, 7)
    assert value.shape == (3,)


def test_muzero_network_initial_and_recurrent() -> None:
    net = MuZeroNetwork(obs_dim=10, action_dim=5, latent_dim=8, hidden_dim=16, seed=0)
    obs = torch.randn(2, 10)
    aembs = torch.randn(2, 6, 5)
    latent, logits, value = net.initial_inference(obs, aembs)
    assert latent.shape == (2, 8)
    assert logits.shape == (2, 6)
    assert value.shape == (2,)
    next_lat, reward = net.recurrent_inference(latent, torch.randn(2, 5))
    assert next_lat.shape == (2, 8)
    assert reward.shape == (2,)
    assert net.num_parameters() > 0


def test_dynamics_deterministic_and_finite() -> None:
    net = MuZeroNetwork(obs_dim=10, action_dim=5, latent_dim=8, hidden_dim=16, seed=0)
    lat = torch.randn(1, 8)
    a = torch.randn(1, 5)
    l1, r1 = net.recurrent_inference(lat, a)
    l2, r2 = net.recurrent_inference(lat, a)
    assert torch.equal(l1, l2)
    assert torch.equal(r1, r2)
    assert torch.isfinite(l1).all()
    assert torch.isfinite(r1).all()


def test_dynamics_action_sensitive() -> None:
    """Different action embeddings must (almost surely) give different latents."""
    net = MuZeroNetwork(obs_dim=10, action_dim=5, latent_dim=8, hidden_dim=16, seed=1)
    lat = torch.randn(1, 8)
    a1 = torch.randn(1, 5)
    a2 = torch.randn(1, 5)
    l1, _ = net.recurrent_inference(lat, a1)
    l2, _ = net.recurrent_inference(lat, a2)
    assert not torch.allclose(l1, l2, atol=1e-4)


def test_parameter_count_scales() -> None:
    small = MuZeroNetwork(139, 60, latent_dim=16, hidden_dim=32, seed=0)
    large = MuZeroNetwork(139, 60, latent_dim=128, hidden_dim=256, seed=0)
    assert small.num_parameters() < large.num_parameters()


def test_seeded_init_reproducible() -> None:
    a = MuZeroNetwork(10, 5, latent_dim=8, hidden_dim=16, seed=42)
    b = MuZeroNetwork(10, 5, latent_dim=8, hidden_dim=16, seed=42)
    c = MuZeroNetwork(10, 5, latent_dim=8, hidden_dim=16, seed=43)
    for ka, kb, kc in zip(a.state_dict(), b.state_dict(), c.state_dict()):
        assert torch.equal(a.state_dict()[ka], b.state_dict()[kb])
        assert not torch.equal(a.state_dict()[ka], c.state_dict()[kc])


def test_target_copy_and_sync() -> None:
    net = MuZeroNetwork(10, 5, latent_dim=8, hidden_dim=16, seed=0)
    target = net.copy_target()
    for tp, op in zip(target.parameters(), net.parameters()):
        assert torch.equal(tp, op)
        assert not tp.requires_grad
    # mutate online, sync, then target matches
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    net.sync_target(target)
    for tp, op in zip(target.parameters(), net.parameters()):
        assert torch.equal(tp, op)


def test_save_load_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from agent.muzero.networks import build_network_from_checkpoint, load_checkpoint, save_checkpoint

    net = MuZeroNetwork(10, 5, latent_dim=8, hidden_dim=16, seed=0)
    path = str(tmp_path / "net.pt")
    save_checkpoint(net, path, step=7)
    ckpt = load_checkpoint(path)
    restored = build_network_from_checkpoint(ckpt)
    assert restored.latent_dim == 8
    assert restored.hidden_dim == 16
    obs = torch.randn(2, 10)
    aembs = torch.randn(2, 4, 5)
    a1 = net.initial_inference(obs, aembs)[1]
    a2 = restored.initial_inference(obs, aembs)[1]
    assert torch.allclose(a1, a2, atol=1e-6)
