"""Synthetic MCTS correctness tests.

These verify the UCT algorithm itself against a tiny hand-built environment
with no Kaggriculture rules, so the algorithm is tested independently of the
simulator.
"""

from __future__ import annotations

from agent.ai import MCTS, MCTSConfig
from agent.actions import PlantAction, TurnAction, WaterAction
from agent.state import CropType

A = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
B = TurnAction(farmer_action=WaterAction())


def _mcts(iterations: int = 200, *, seed: int = 0) -> MCTS[int]:
    """Tiny env: state 0 -> A -> 1 (terminal value 10), B -> 2 (value 0)."""

    def transition(state: int, action: TurnAction) -> int:
        return 1 if action == A else 2

    def generate(state: int) -> tuple[TurnAction, ...]:
        del state
        return (A, B)

    def is_terminal(state: int) -> bool:
        return state in (1, 2)

    def terminal_value(state: int, player: int) -> float:
        del player
        return 10.0 if state == 1 else 0.0

    def evaluate(state: int, player: int) -> float:
        del state, player
        return 0.0

    def rollout(state: int, rng: object) -> TurnAction:
        del state, rng
        return A

    return MCTS(
        MCTSConfig(iterations=iterations, seed=seed),
        transition=transition,
        generate=generate,
        is_terminal=is_terminal,
        terminal_value=terminal_value,
        evaluate=evaluate,
        rollout=rollout,
    )


def test_mcts_selects_obvious_best_action() -> None:
    chosen = _mcts(iterations=300).search(0, 0)
    assert chosen == A


def test_mcts_is_deterministic_with_fixed_seed() -> None:
    assert _mcts(iterations=100, seed=7).search(0, 0) == _mcts(iterations=100, seed=7).search(0, 0)


def test_children_are_expanded_and_visits_increase() -> None:
    mcts = _mcts(iterations=50)
    root = mcts.search_root(0, 0)
    assert root.visits == 50  # one backpropagation per iteration
    assert len(root.children) == 2  # both actions eventually expanded
    child_a = next(c for c in root.children if c.action == A)
    child_b = next(c for c in root.children if c.action == B)
    assert child_a.visits > child_b.visits  # A is favored
    assert child_a.total_value == 10.0 * child_a.visits
    assert child_b.total_value == 0.0


def test_unvisited_actions_are_explored() -> None:
    mcts = _mcts(iterations=2)
    root = mcts.search_root(0, 0)
    assert {c.action for c in root.children} == {A, B}


def test_budget_is_respected() -> None:
    iterations = 25
    mcts = _mcts(iterations=iterations)
    root = mcts.search_root(0, 0)
    assert root.visits == iterations


def test_terminal_state_stops_simulation() -> None:
    # A root that is already terminal: no expansion, no rollout, value = money.
    mcts = _mcts(iterations=10)
    root = mcts.search_root(1, 0)
    assert root.terminal
    assert root.children == []
