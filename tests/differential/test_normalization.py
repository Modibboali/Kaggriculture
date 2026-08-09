"""Unit tests for the canonical state normalizer (no environment needed)."""

from __future__ import annotations

from typing import Any

from agent.environment import KaggleObservationAdapter
from agent.testing.differential import normalize

EMPTY = None


def _observation() -> dict[str, Any]:
    """A synthetic observation covering every tile kind and both players."""
    grid: list[list[Any]] = [
        [EMPTY, "LOCKED", "LOCKED", "LOCKED"],
        [EMPTY, EMPTY,
         {"kind": "PLANT", "crop": "WHEAT", "planted_day": 1, "watered_today": False,
          "consecutive_unwatered": 1, "yield_units": 2, "max_lifespan_step": -1,
          "fertilized_until_day": -1}, "LOCKED"],
        [EMPTY,
         {"kind": "COOP", "animal": "GOOSE", "placed_day": 0, "yield_units": 3,
          "fed_today": False, "consecutive_unfed": 0, "cared_today": False,
          "fertilizer_available": True, "pending_care_bonus": 0},
         {"kind": "PASTURE", "animal": None, "placed_day": 0, "yield_units": 0,
          "fed_today": False, "consecutive_unfed": 0, "cared_today": False,
          "fertilizer_available": False, "pending_care_bonus": 0}, "LOCKED"],
        [EMPTY, EMPTY, {"kind": "WEED"}, EMPTY],
    ]

    def farm(money: float) -> dict[str, Any]:
        return {
            "money": money,
            "tiles": grid,
            "farmer": [1, 1],
            "hands": [[2, 1], [1, 2]],
            "unlocked_quadrants": ["NW"],
            "hires_today": 1,
        }

    return {
        "remainingOverageTime": 1.234,
        "step": 51,
        "player": 0,
        "day": 2,
        "hour": 3,
        "farms": [farm(1234.0), farm(500.0)],
        "market": {
            "inventory": {"WHEAT": 10000, "CARROT": 9000, "EGG": 10000},
            "prices": {
                "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
                "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
            },
        },
        "town": {"unlocked_shops": ["BAKERY", "PIZZA SHOP"]},
        "private": {
            "shed": {"WHEAT": 3, "EGG": 2},
            "seeds": {"WHEAT": 4, "CARROT": 1},
            "inventories": [
                {"WHEAT": 1},
                {"FERTILIZER": 1},
                {"WHEAT": 1},
            ],
        },
    }


def _canonical() -> dict[str, Any]:
    state = KaggleObservationAdapter.from_observation(_observation())
    return normalize(state)


def test_top_level_fields() -> None:
    canonical = _canonical()
    assert canonical["day"] == 2
    assert canonical["hour"] == 3
    assert canonical["step"] == 51
    assert canonical["current_player"] == 0
    assert set(canonical) == {"day", "hour", "step", "current_player", "farms", "private", "market", "town"}


def test_farm_normalization() -> None:
    canonical = _canonical()
    player0 = canonical["farms"][0]
    assert player0["money"] == 1234
    assert player0["farmer"] == [1, 1]
    assert player0["hands"] == [[2, 1], [1, 2]]
    assert player0["unlocked_quadrants"] == ["NW"]
    assert player0["hires_today"] == 1
    assert len(player0["tiles"]) == 4
    assert canonical["farms"][1]["money"] == 500


def test_tile_normalization() -> None:
    tiles = _canonical()["farms"][0]["tiles"]
    # tiles[y][x]: empty at (0,0), locked at (1,0), plant at (2,1),
    # coop at (1,2), pasture at (2,2), weed at (2,3).
    assert tiles[0][0] == {"kind": "EMPTY"}
    assert tiles[0][1] == {"kind": "LOCKED"}
    assert tiles[3][2] == {"kind": "WEED"}
    assert tiles[1][2]["kind"] == "PLANT"
    assert tiles[1][2]["plant"]["crop"] == "WHEAT"
    assert tiles[1][2]["plant"]["watered_today"] is False
    assert tiles[1][2]["plant"]["yield_units"] == 2
    assert tiles[2][1]["kind"] == "COOP"
    assert tiles[2][1]["animal"]["animal"] == "GOOSE"
    assert tiles[2][1]["animal"]["fertilizer_available"] is True
    assert tiles[2][2]["kind"] == "PASTURE"
    assert tiles[2][2]["animal"] is None


def test_private_normalization() -> None:
    private = _canonical()["private"]
    assert private["shed"]["WHEAT"] == 3
    assert private["shed"]["EGG"] == 2
    assert private["seeds"]["WHEAT"] == 4
    assert private["seeds"]["CARROT"] == 1
    assert private["inventories"][0]["WHEAT"] == 1
    assert private["inventories"][1]["FERTILIZER"] == 1


def test_market_and_town_normalization() -> None:
    canonical = _canonical()
    assert canonical["market"]["inventory"]["WHEAT"] == 10000
    assert canonical["market"]["prices"]["MELON"] == 250
    assert canonical["town"]["unlocked_shops"] == ["BAKERY", "PIZZA_SHOP"]


def test_normalization_is_deterministic() -> None:
    assert _canonical() == _canonical()


def test_normalization_is_value_based() -> None:
    # Two separately-built GameStates with equal content normalize identically.
    state_a = KaggleObservationAdapter.from_observation(_observation())
    state_b = KaggleObservationAdapter.from_observation(_observation())
    assert normalize(state_a) == normalize(state_b)


def test_normalized_output_is_json_safe() -> None:
    import json

    json.dumps(_canonical())
