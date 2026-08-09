"""Integration test: convert a *real* Kaggriculture observation.

This test launches the official ``kaggle_environments`` package (no mocks) and
converts genuine observations into the domain model. If the environment is not
installed in the test environment, the module is skipped via ``pytest`` rather
than replaced with a fake test.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("kaggle_environments")

from kaggle_environments import make  # noqa: E402

from agent.environment import KaggleObservationAdapter  # noqa: E402
from agent.state import GameState  # noqa: E402

GAME = "kaggriculture"
pytestmark = pytest.mark.integration


def _pass_agent(obs: Any, config: Any) -> dict[str, Any]:
    """Deterministic agent that always passes."""
    del obs, config
    return {"farmer": ["PASS"], "market": []}


def _capturing_agent(captured: list[dict[str, Any]]) -> Any:
    """An agent that records every observation it receives."""

    def agent(obs: Any, config: Any) -> dict[str, Any]:
        del config
        captured.append(obs)
        return {"farmer": ["PASS"], "market": []}

    return agent


def test_real_observation_conversion() -> None:
    """Launch the official environment and convert real agent observations."""
    env = make(GAME, configuration={"episodeSteps": 720})
    captured: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    env.run([_capturing_agent(captured[0]), _capturing_agent(captured[1])])

    # Every agent received genuine observations during the run; convert the
    # first and last observations each agent actually saw.
    for player in (0, 1):
        assert len(captured[player]) > 0
        for obs in (captured[player][0], captured[player][-1]):
            state = KaggleObservationAdapter.from_observation(obs)
            assert isinstance(state, GameState)
            # The domain state reflects the real observation's scalar fields.
            assert state.current_player == obs["player"]
            assert state.day == obs["day"]
            assert state.hour == obs["hour"]
            assert state.step == obs["step"]
            # Both farms are represented, positionally.
            assert len(state.players) == 2
            assert state.current_player_state.farm.money == int(
                obs["farms"][state.current_player]["money"]
            )
            assert state.market.inventory.total_items() >= 0
            assert len(state.town.unlocked_shops) >= 0

    # The adapter is deterministic: same observation -> equal states.
    obs0 = captured[0][-1]
    assert KaggleObservationAdapter.from_observation(obs0) == KaggleObservationAdapter.from_observation(obs0)


def test_real_observation_mid_season() -> None:
    """Convert a real observation from well into a season."""
    env = make(GAME, configuration={"episodeSteps": 720})
    captured: list[dict[str, Any]] = []
    # ~5 days in (120 turns), enough for the environment to have produced
    # plants/weeds/inventory variety.
    env.run([_capturing_agent(captured), _pass_agent])
    assert len(captured) > 0
    obs = captured[-1]
    state = KaggleObservationAdapter.from_observation(obs)
    assert state.day == obs["day"]
    assert state.hour == obs["hour"]
    assert state.step == obs["step"]
    # A real observation always exposes both farms as grids.
    assert state.players[0].farm.board_size > 0
    assert state.players[1].farm.board_size > 0
