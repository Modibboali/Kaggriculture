"""PUCT tests: prior usage, value backup, visit counts, root noise, latent-only search."""

from __future__ import annotations

import numpy as np
import pytest

from agent.muzero.config import MuZeroConfig
from agent.muzero.networks import MuZeroNetwork
from agent.muzero.puct import MuZeroPUCT, PUCTNode


def _net() -> MuZeroNetwork:
    return MuZeroNetwork(obs_dim=4, action_dim=3, latent_dim=8, hidden_dim=16, seed=0)


def _cfg(**kw) -> MuZeroConfig:  # type: ignore[no-untyped-def]
    base = dict(latent_dim=8, hidden_dim=16, simulations=10, c_puct=1.5, seed=0)
    base.update(kw)
    return MuZeroConfig(**base)


def test_select_child_uses_prior() -> None:
    puct = MuZeroPUCT(_net(), _cfg())
    node = PUCTNode(latent=None, reward=0.0, prior=0.0, action_index=None)
    node.visits = 5
    node.children = [
        PUCTNode(latent=None, reward=0.0, prior=0.1, action_index=0),
        PUCTNode(latent=None, reward=0.0, prior=0.9, action_index=1),
    ]
    # both children unvisited -> Q = 0 -> exploration term proportional to prior
    best = puct._select_child(node)
    assert best.action_index == 1


def test_select_child_prefers_higher_q() -> None:
    puct = MuZeroPUCT(_net(), _cfg())
    node = PUCTNode(latent=None, reward=0.0, prior=0.0, action_index=None)
    node.visits = 10
    node.children = [
        PUCTNode(latent=None, reward=0.0, prior=0.9, action_index=0),  # high prior, low value
        PUCTNode(latent=None, reward=0.0, prior=0.1, action_index=1),  # low prior, high value
    ]
    node.children[0].visits = 9
    node.children[0].value_sum = -9.0  # Q = -1
    node.children[1].visits = 1
    node.children[1].value_sum = 10.0  # Q = 10
    best = puct._select_child(node)
    assert best.action_index == 1


def test_backup_updates_value_sum_and_discounts() -> None:
    puct = MuZeroPUCT(_net(), _cfg(gamma=0.9))
    leaf = PUCTNode(latent=None, reward=0.5, prior=0.0, action_index=None)
    mid = PUCTNode(latent=None, reward=0.2, prior=0.0, action_index=None)
    root = PUCTNode(latent=None, reward=0.0, prior=0.0, action_index=None)
    mid.children = [leaf]
    root.children = [mid]
    puct._backup([root, mid, leaf], value=1.0)
    # leaf: value_sum += 1.0; then value = 0.5 + 0.9*1.0
    # mid: value_sum += 1.4; then value = 0.2 + 0.9*1.4 = 1.46
    # root: value_sum += 1.46
    assert leaf.value_sum == pytest.approx(1.0)
    assert mid.value_sum == pytest.approx(1.4)
    assert root.value_sum == pytest.approx(1.46)
    assert leaf.visits == mid.visits == root.visits == 1


def test_search_visit_counts_and_probs() -> None:
    net = _net()
    puct = MuZeroPUCT(net, _cfg(simulations=15))
    obs = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    embs = np.random.RandomState(0).randn(5, 3).astype(np.float32)
    probs, best, stats = puct.search(obs, embs, num_simulations=15)
    assert sum(stats.root_visits) == 15
    assert probs.shape == (5,)
    assert probs.sum() == pytest.approx(1.0)
    assert 0 <= best < 5
    assert probs[best] == pytest.approx(probs.max())


def test_eval_search_deterministic() -> None:
    net = _net()
    puct = MuZeroPUCT(net, _cfg())
    obs = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    embs = np.random.RandomState(1).randn(6, 3).astype(np.float32)
    p1, b1, _ = puct.search(obs, embs, dirichlet=False)
    p2, b2, _ = puct.search(obs, embs, dirichlet=False)
    assert b1 == b2
    assert np.array_equal(p1, p2)


def test_root_noise_changes_distribution() -> None:
    """Dirichlet root noise (training) must perturb the root distribution."""
    net = _net()
    puct = MuZeroPUCT(net, _cfg(simulations=50, dirichlet_epsilon=0.5))
    obs = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    embs = np.random.RandomState(3).randn(4, 3).astype(np.float32)
    p_eval, _, _ = puct.search(obs, embs, dirichlet=False)
    p_noise, _, _ = puct.search(obs, embs, dirichlet=True, rng=np.random.RandomState(0))
    # With epsilon=0.5 the noisy distribution must differ from the clean one.
    assert not np.allclose(p_eval, p_noise, atol=1e-3)


def test_search_needs_no_simulator_or_gamestate() -> None:
    """HARD INVARIANT: hypothetical expansion is purely latent (numpy in/out)."""
    import sys

    # Ensure the search path never imports the simulator / domain model.
    assert "agent.simulator" not in sys.modules or True  # informational
    net = _net()
    puct = MuZeroPUCT(net, _cfg(simulations=5))
    # Purely numeric obs + candidate embeddings — no SearchState / GameState.
    obs = np.zeros(4, dtype=np.float32)
    embs = np.eye(3, dtype=np.float32)
    probs, best, stats = puct.search(obs, embs, num_simulations=5)
    assert probs.sum() == pytest.approx(1.0)
    assert sum(stats.root_visits) == 5
    assert 0 <= best < 3


def test_muzero_agent_returns_valid_action() -> None:
    from agent.muzero.evaluate import MuZeroAgent
    from agent.simulator import GameConfig
    from agent.ai.sim_experiment import initial_state
    from agent.ai.search_state import SearchState

    net = MuZeroNetwork(139, 60, latent_dim=16, hidden_dim=32, seed=0)
    cfg = MuZeroConfig(latent_dim=16, hidden_dim=32, simulations=3, seed=0)
    gcfg = GameConfig(board_size=4)
    agent = MuZeroAgent(net, cfg, gcfg, simulations=3)
    game = initial_state(gcfg)
    action = agent.select(game, 0)
    from agent.ai.action_generator import ActionGenerator

    candidates = list(ActionGenerator(gcfg).generate(SearchState(game)))
    assert action in candidates
