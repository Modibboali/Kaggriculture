"""Differential tests for structures, animals, workers, FEED/CARE/COLLECT, and
combined mechanic sequences."""

from __future__ import annotations

import pytest

pytest.importorskip("kaggle_environments")

from agent.simulator import Simulator  # noqa: E402
from agent.testing.differential import (  # noqa: E402
    ANIMAL_SCENARIOS,
    CARE_SCENARIOS,
    COLLECT_SCENARIOS,
    DifferentialResult,
    DifferentialRunner,
    FEED_SCENARIOS,
    MECHANIC_SEQUENCE_SCENARIOS,
    STRUCTURE_SCENARIOS,
    Scenario,
    WORKER_SCENARIOS,
    render_differential_report,
)

pytestmark = pytest.mark.integration


def _assert_match(result: DifferentialResult) -> None:
    if result.has_mismatch:
        raise AssertionError(
            f"Simulator diverged from Kaggle on turns {result.mismatched_turns} "
            f"(scenario {result.scenario}):\n" + render_differential_report(result)
        )


@pytest.mark.parametrize("scenario", STRUCTURE_SCENARIOS, ids=lambda s: s.name)
def test_structure_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))


@pytest.mark.parametrize("scenario", ANIMAL_SCENARIOS, ids=lambda s: s.name)
def test_animal_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))


@pytest.mark.parametrize("scenario", FEED_SCENARIOS, ids=lambda s: s.name)
def test_feed_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))


@pytest.mark.parametrize("scenario", CARE_SCENARIOS, ids=lambda s: s.name)
def test_care_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))


@pytest.mark.parametrize("scenario", COLLECT_SCENARIOS, ids=lambda s: s.name)
def test_collect_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))


@pytest.mark.parametrize("scenario", WORKER_SCENARIOS, ids=lambda s: s.name)
def test_worker_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))


@pytest.mark.parametrize("scenario", MECHANIC_SEQUENCE_SCENARIOS, ids=lambda s: s.name)
def test_mechanic_sequence_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))
