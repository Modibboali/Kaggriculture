"""Differential tests for FERTILIZE, DIG, BUY_LAND, and combined sequences.

Every scenario runs on the real Kaggle environment AND on the simulator; the
canonical states must match turn-for-turn. A mismatch renders the exact
differing paths for diagnosis.
"""

from __future__ import annotations

import pytest

pytest.importorskip("kaggle_environments")

from agent.simulator import Simulator  # noqa: E402
from agent.testing.differential import (  # noqa: E402
    DIG_SCENARIOS,
    DifferentialResult,
    DifferentialRunner,
    FERTILIZE_SCENARIOS,
    LAND_SCENARIOS,
    SEQUENCE_SCENARIOS,
    Scenario,
    render_differential_report,
)

pytestmark = pytest.mark.integration


def _assert_match(result: DifferentialResult) -> None:
    if result.has_mismatch:
        raise AssertionError(
            f"Simulator diverged from Kaggle on turns {result.mismatched_turns} "
            f"(scenario {result.scenario}):\n" + render_differential_report(result)
        )


@pytest.mark.parametrize("scenario", FERTILIZE_SCENARIOS, ids=lambda s: s.name)
def test_fertilize_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))


@pytest.mark.parametrize("scenario", DIG_SCENARIOS, ids=lambda s: s.name)
def test_dig_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))


@pytest.mark.parametrize("scenario", LAND_SCENARIOS, ids=lambda s: s.name)
def test_land_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))


@pytest.mark.parametrize("scenario", SEQUENCE_SCENARIOS, ids=lambda s: s.name)
def test_sequence_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))
