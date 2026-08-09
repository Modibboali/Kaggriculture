"""Differential tests: first verified simulator layer vs the real Kaggle engine.

Follows the mandated verification order: PASS, Movement, BUY_SEED, PLANT,
WATER. Each transition must reproduce the official environment's resulting
state turn-for-turn before the next is considered verified. If any diverges,
the rendered report shows the exact differing paths.
"""

from __future__ import annotations

import pytest

pytest.importorskip("kaggle_environments")

from agent.actions import MovementAction, TurnAction  # noqa: E402
from agent.simulator import Simulator  # noqa: E402
from agent.state import Direction  # noqa: E402
from agent.testing.differential import (  # noqa: E402
    DifferentialResult,
    DifferentialRunner,
    SCENARIO_BUY_PLANT_WATER,
    SCENARIO_BUY_SEED,
    SCENARIO_MOVE_FARMER,
    SCENARIO_MULTIPLE_TURNS,
    SCENARIO_PASS,
    SCENARIO_PLANT_CROP,
    SCENARIO_WATER_CROP,
    Scenario,
    render_differential_report,
)

pytestmark = pytest.mark.integration


def _assert_match(result: DifferentialResult) -> None:
    if result.has_mismatch:
        raise AssertionError(
            f"Simulator diverged from Kaggle on turns {result.mismatched_turns}:\n"
            + render_differential_report(result)
        )


def test_pass_matches_kaggle() -> None:
    _assert_match(DifferentialRunner().compare(SCENARIO_PASS, Simulator()))


def test_move_farmer_matches_kaggle() -> None:
    _assert_match(DifferentialRunner().compare(SCENARIO_MOVE_FARMER, Simulator()))


def test_move_farmer_all_directions_match_kaggle() -> None:
    for direction in Direction:
        move = TurnAction(farmer_action=MovementAction(direction=direction))
        scenario = Scenario.single_player(
            f"move_{direction.label.lower()}",
            (move, move),
        )
        _assert_match(DifferentialRunner().compare(scenario, Simulator()))


def test_move_farmer_into_wall_matches_kaggle() -> None:
    """Six NORTH moves from (4,4) push against the board edge (y=0)."""
    north = TurnAction(farmer_action=MovementAction(direction=Direction.NORTH))
    scenario = Scenario.single_player("move_north_into_wall", tuple(north for _ in range(6)))
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))


def test_buy_seed_matches_kaggle() -> None:
    _assert_match(DifferentialRunner().compare(SCENARIO_BUY_SEED, Simulator()))


def test_plant_crop_matches_kaggle() -> None:
    _assert_match(DifferentialRunner().compare(SCENARIO_PLANT_CROP, Simulator()))


def test_water_crop_matches_kaggle() -> None:
    _assert_match(DifferentialRunner().compare(SCENARIO_WATER_CROP, Simulator()))


def test_long_sequence_matches_kaggle() -> None:
    """End-to-end: PASS, movement, BUY_SEED, PLANT and WATER in one scenario."""
    _assert_match(DifferentialRunner().compare(SCENARIO_MULTIPLE_TURNS, Simulator()))


def test_buy_plant_water_matches_kaggle() -> None:
    _assert_match(DifferentialRunner().compare(SCENARIO_BUY_PLANT_WATER, Simulator()))
