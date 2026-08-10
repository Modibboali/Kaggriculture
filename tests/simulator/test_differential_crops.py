"""Differential tests for HARVEST, crop lifecycle, and day-boundary dynamics.

Runs each crop/harvest scenario on the real Kaggle environment AND on the
simulator, then asserts the canonical states match turn-for-turn. A mismatch
renders the exact differing paths for diagnosis.
"""

from __future__ import annotations

import pytest

pytest.importorskip("kaggle_environments")

from agent.simulator import Simulator  # noqa: E402
from agent.testing.differential import (  # noqa: E402
    CROP_SCENARIOS,
    DifferentialResult,
    DifferentialRunner,
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


@pytest.mark.parametrize("scenario", CROP_SCENARIOS, ids=lambda s: s.name)
def test_crop_scenario_matches_kaggle(scenario: Scenario) -> None:
    _assert_match(DifferentialRunner().compare(scenario, Simulator()))
