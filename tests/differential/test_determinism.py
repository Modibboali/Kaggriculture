"""Determinism and end-to-end differential tests against the real environment.

These tests launch the official Kaggriculture environment (no mocks). If the
environment is not installed, the module is skipped via ``pytest``.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

pytest.importorskip("kaggle_environments")

from agent.actions import TurnAction  # noqa: E402
from agent.state import GameState  # noqa: E402
from agent.testing.differential import (  # noqa: E402
    DifferentialRunner,
    SCENARIO_BUY_PLANT_WATER,
    SCENARIO_BUY_SEED,
    SCENARIO_MOVE_FARMER,
    render_differential_report,
    render_summary,
)

pytestmark = pytest.mark.integration


class NoOpSimulator:
    """Test double returning the input state unchanged (no rules implemented)."""

    def apply(
        self, state: GameState, actions: tuple[TurnAction, TurnAction]
    ) -> GameState:
        del actions
        return state


class ReplaySimulator:
    """Test double replaying pre-recorded GameStates in order."""

    def __init__(self, states: Sequence[GameState]) -> None:
        self._states = list(states)
        self._index = 0

    def apply(
        self, state: GameState, actions: tuple[TurnAction, TurnAction]
    ) -> GameState:
        del state, actions
        if self._index >= len(self._states):
            raise AssertionError("replay exhausted")
        result = self._states[self._index]
        self._index += 1
        return result


def test_run_kaggle_produces_canonical_states() -> None:
    runner = DifferentialRunner()
    run = runner.run_kaggle(SCENARIO_BUY_PLANT_WATER)

    assert len(run.turns) == len(SCENARIO_BUY_PLANT_WATER)
    assert run.initial_state.step == 0
    # Turn 1 bought a WHEAT seed.
    assert run.turns[0].canonical["private"]["seeds"]["WHEAT"] == 1
    # Turn 2 planted it at the farmer's tile (4,4).
    assert run.turns[1].canonical["farms"][0]["tiles"][4][4]["kind"] == "PLANT"
    assert run.turns[1].canonical["farms"][0]["tiles"][4][4]["plant"]["crop"] == "WHEAT"
    # Turn 3 watered it.
    assert run.turns[2].canonical["farms"][0]["tiles"][4][4]["plant"]["watered_today"] is True


def test_kaggle_runs_are_deterministic() -> None:
    runner = DifferentialRunner()
    first = runner.run_kaggle(SCENARIO_BUY_PLANT_WATER)
    second = runner.run_kaggle(SCENARIO_BUY_PLANT_WATER)
    assert first.initial_state == second.initial_state
    assert [turn.canonical for turn in first.turns] == [
        turn.canonical for turn in second.turns
    ]


def test_move_farmer_scenario_moves_position() -> None:
    runner = DifferentialRunner()
    run = runner.run_kaggle(SCENARIO_MOVE_FARMER)
    assert run.turns[0].canonical["farms"][0]["farmer"] == [4, 3]
    assert run.turns[1].canonical["farms"][0]["farmer"] == [4, 2]


def test_compare_matches_with_replay_simulator() -> None:
    """The framework reports a match when the simulator reproduces Kaggle."""
    runner = DifferentialRunner()
    kaggle = runner.run_kaggle(SCENARIO_BUY_PLANT_WATER)
    recorded = [turn.state for turn in kaggle.turns]
    result = runner.compare(SCENARIO_BUY_PLANT_WATER, ReplaySimulator(recorded))
    assert result.has_mismatch is False
    assert result.mismatched_turns == ()


def test_compare_reports_mismatch_with_noop_simulator() -> None:
    """A simulator that applies no rules produces a useful mismatch report."""
    runner = DifferentialRunner()
    result = runner.compare(SCENARIO_BUY_SEED, NoOpSimulator())
    assert result.has_mismatch is True

    report = render_differential_report(result)
    assert "Scenario: buy_seed" in report
    assert "Turn: 1" in report
    assert "private.seeds.WHEAT" in report
    assert "expected:" in report
    assert "actual:" in report

    summary = render_summary(result)
    assert summary.startswith("Scenario buy_seed: MISMATCH")


def test_runner_reports_helpful_paths() -> None:
    """Verify the report exposes exact paths for a planted-then-noop mismatch."""
    runner = DifferentialRunner()
    result = runner.compare(SCENARIO_BUY_PLANT_WATER, NoOpSimulator())
    report = render_differential_report(result)
    assert "farms[0].tiles[4][4]" in report
