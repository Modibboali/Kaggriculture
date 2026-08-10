"""MCTS integration with the real simulator (no Kaggle environment)."""

from __future__ import annotations

import random

from agent.ai import (
    ActionGenerator,
    Evaluator,
    HeuristicRolloutPolicy,
    MCTS,
    MCTSConfig,
    SearchState,
    SimulatorAdapter,
    Terminal,
)
from agent.actions import TurnAction

from ._build import make_components, make_search_state


def _mcts(iterations: int = 40, *, seed: int = 0) -> tuple[MCTS[SearchState], SimulatorAdapter]:
    config, adapter, generator, evaluator, terminal = make_components()
    rollout = HeuristicRolloutPolicy(generator)
    mcts = MCTS(
        MCTSConfig(iterations=iterations, max_simulation_steps=15, seed=seed),
        transition=adapter.transition,
        generate=generator.generate,
        is_terminal=terminal.is_terminal,
        terminal_value=terminal.value,
        evaluate=evaluator.evaluate,
        rollout=rollout.choose,
        rng=random.Random(seed),
    )
    return mcts, adapter


def test_mcts_searches_real_simulator() -> None:
    mcts, _ = _mcts()
    action = mcts.search(make_search_state(), 0)
    assert isinstance(action, TurnAction)


def test_mcts_is_deterministic_on_simulator() -> None:
    a, _ = _mcts(seed=3)
    b, _ = _mcts(seed=3)
    state = make_search_state()
    assert a.search(state, 0) == b.search(state, 0)


def test_mcts_respects_iteration_budget() -> None:
    mcts, _ = _mcts(iterations=25)
    root = mcts.search_root(make_search_state(), 0)
    assert root.visits == 25


def test_mcts_plays_multiple_turns() -> None:
    mcts, adapter = _mcts(iterations=20)
    state = make_search_state()
    for _ in range(8):
        action = mcts.search(state, 0)
        state = adapter.transition(state, action)
    assert state.game.step == 8


def test_mcts_counts_transitions() -> None:
    mcts, adapter = _mcts(iterations=30)
    before = adapter.transitions
    mcts.search(make_search_state(), 0)
    assert adapter.transitions > before


def test_mcts_rollout_policies_both_work() -> None:
    from agent.ai import RandomRolloutPolicy

    config, adapter, generator, evaluator, terminal = make_components()
    for rollout in (RandomRolloutPolicy(generator), HeuristicRolloutPolicy(generator)):
        mcts = MCTS(
            MCTSConfig(iterations=15, seed=1),
            transition=adapter.transition,
            generate=generator.generate,
            is_terminal=terminal.is_terminal,
            terminal_value=terminal.value,
            evaluate=evaluator.evaluate,
            rollout=rollout.choose,
            rng=random.Random(1),
        )
        action = mcts.search(make_search_state(), 0)
        assert isinstance(action, TurnAction)
