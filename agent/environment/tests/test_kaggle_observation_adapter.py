"""Unit tests for :mod:`agent.environment` (adapter + validation).

The observation fixtures here mirror the schema verified against the live
Kaggriculture environment by ``agent.diagnostics``.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import pytest

from agent.environment import (
    InvalidObservationError,
    KaggleObservationAdapter,
)
from agent.state import (
    AnimalType,
    CoopTile,
    CropType,
    EmptyTile,
    GameState,
    ItemType,
    LockedTile,
    PastureTile,
    PlantTile,
    Position,
    Quadrant,
    ShopType,
    StructureType,
    WeedTile,
)

EMPTY = None


def _farm(money: float, grid: list[list[Any]], unlocked: list[str]) -> dict[str, Any]:
    return {
        "money": money,
        "tiles": grid,
        "farmer": [1, 1],
        "hands": [[2, 1], [1, 2]],
        "unlocked_quadrants": unlocked,
        "hires_today": 1,
    }


def _grid() -> list[list[Any]]:
    return [
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


def _make_observation() -> dict[str, Any]:
    """A valid, representative observation (player 0 is the observer)."""
    p0_grid = _grid()
    p1_grid = copy.deepcopy(_grid())
    return {
        "remainingOverageTime": 1.234,
        "step": 51,
        "player": 0,
        "day": 2,
        "hour": 3,
        "farms": [_farm(1234.0, p0_grid, ["NW"]), _farm(500.0, p1_grid, ["NW"])],
        "market": {
            "inventory": {"WHEAT": 10000, "CARROT": 9000, "EGG": 10000, "FERTILIZER": 10000},
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
                {"WHEAT": 1},        # main farmer
                {"FERTILIZER": 1},   # hand 1
                {"WHEAT": 1},        # hand 2
            ],
        },
    }


def convert(obs: dict[str, Any]) -> GameState:
    return KaggleObservationAdapter.from_observation(obs)


# --- 1. minimal valid observation ------------------------------------------
def test_minimal_valid_observation() -> None:
    state = convert(_make_observation())
    assert isinstance(state, GameState)
    assert state.day == 2
    assert state.hour == 3
    assert state.step == 51
    assert state.current_player == 0


# --- 3. multiple players ---------------------------------------------------
def test_multiple_players() -> None:
    state = convert(_make_observation())
    assert len(state.players) == 2
    # Farm identity is positional: players[p] corresponds to farms[p].
    assert state.players[0].farm.money == 1234
    assert state.players[1].farm.money == 500
    # The observing player has private data; the opponent does not.
    assert state.players[0].seeds.total() == 5
    assert state.players[1].inventory.total_items() == 0
    assert state.players[1].seeds.total() == 0


# --- 4. farm parsing -------------------------------------------------------
def test_farm_parsing() -> None:
    state = convert(_make_observation())
    farm = state.current_player_state.farm
    assert farm.money == 1234
    assert farm.board_size == 4
    assert farm.farmer.position == Position(1, 1)
    assert farm.farmer.is_main_farmer is True
    assert len(farm.workers) == 2
    assert Quadrant.NW in farm.unlocked_quadrants
    assert farm.hires_today == 1


# --- 5. tile parsing -------------------------------------------------------
def test_tile_parsing() -> None:
    farm = convert(_make_observation()).current_player_state.farm
    assert isinstance(farm.tile_at(Position(0, 0)), EmptyTile)
    assert isinstance(farm.tile_at(Position(1, 0)), LockedTile)
    assert isinstance(farm.tile_at(Position(2, 3)), WeedTile)
    assert farm.tile_at(Position(-1, 0)) is None


# --- 6. plant parsing ------------------------------------------------------
def test_plant_parsing() -> None:
    tile = convert(_make_observation()).current_player_state.farm.tile_at(Position(2, 1))
    assert isinstance(tile, PlantTile)
    assert tile.plant.crop is CropType.WHEAT
    assert tile.plant.planted_day == 1
    assert tile.plant.watered_today is False
    assert tile.plant.consecutive_unwatered == 1
    assert tile.plant.yield_units == 2
    assert tile.plant.fertilized_until_day == -1
    assert tile.plant.max_lifespan_step == -1


# --- 7. animal parsing -----------------------------------------------------
def test_animal_parsing() -> None:
    farm = convert(_make_observation()).current_player_state.farm
    coop = farm.tile_at(Position(1, 2))
    assert isinstance(coop, CoopTile)
    assert coop.animal is not None
    assert coop.animal.animal is AnimalType.GOOSE
    assert coop.animal.placed_day == 0
    assert coop.animal.yield_units == 3
    assert coop.animal.fed_today is False
    assert coop.animal.cared_today is False
    assert coop.animal.fertilizer_available is True
    assert coop.animal.pending_care_bonus == 0

    pasture = farm.tile_at(Position(2, 2))
    assert isinstance(pasture, PastureTile)
    assert pasture.animal is None


# --- 8. worker parsing -----------------------------------------------------
def test_worker_parsing() -> None:
    state = convert(_make_observation())
    player = state.current_player_state
    assert len(player.workers) == 3  # farmer + 2 hands
    assert player.workers[0].id == 0 and player.workers[0].is_main_farmer is True
    assert player.workers[1].id == 1 and player.workers[1].position == Position(2, 1)
    assert player.workers[2].id == 2 and player.workers[2].position == Position(1, 2)
    # farm.farmer / farm.workers agree with the full roster
    assert player.farm.farmer is player.workers[0]
    assert player.farm.workers == player.workers[1:]
    # opponent workers exist (public positions) but carry no inventory
    opponent = state.opponent_state
    assert opponent.workers[0].is_main_farmer is True
    assert opponent.workers[0].inventory.total_items() == 0


# --- 9. inventory parsing --------------------------------------------------
def test_inventory_parsing() -> None:
    state = convert(_make_observation())
    shed = state.current_player_state.inventory
    assert shed.get(ItemType.WHEAT) == 3
    assert shed.get(ItemType.EGG) == 2
    assert shed.get(ItemType.GOOSE) == 0
    # worker inventories
    assert state.current_player_state.workers[1].inventory.get(ItemType.FERTILIZER) == 1
    assert state.current_player_state.workers[2].inventory.get(ItemType.WHEAT) == 1
    # opponent shed is empty
    assert state.opponent_state.inventory.total_items() == 0


# --- 10. market parsing ----------------------------------------------------
def test_market_parsing() -> None:
    state = convert(_make_observation())
    assert state.market.inventory.get(ItemType.WHEAT) == 10000
    assert state.market.inventory.get(ItemType.CARROT) == 9000
    assert state.market.price(ItemType.MELON) == 250
    assert state.market.price(ItemType.FERTILIZER) == 100


# --- 11. town parsing ------------------------------------------------------
def test_town_parsing() -> None:
    state = convert(_make_observation())
    assert state.town.has_shop(ShopType.BAKERY) is True
    assert state.town.has_shop(ShopType.PIZZA_SHOP) is True
    assert state.town.has_shop(ShopType.YARN_STORE) is False


# --- 12. day/hour/step parsing ---------------------------------------------
def test_day_hour_step_parsing() -> None:
    state = convert(_make_observation())
    assert state.day == 2
    assert state.hour == 3
    assert state.step == 51


# --- 13. current player parsing --------------------------------------------
def test_current_player_parsing() -> None:
    state = convert(_make_observation())
    assert state.current_player == 0
    assert state.current_player_state is state.players[0]
    assert state.opponent_state is state.players[1]


# --- 14. missing required field --------------------------------------------
@pytest.mark.parametrize(
    ("description", "mutate", "expected_path"),
    [
        ("missing town", lambda obs: obs.pop("town"), "town"),
        ("missing day", lambda obs: obs.pop("day"), "day"),
        ("missing step", lambda obs: obs.pop("step"), "step"),
        ("missing player", lambda obs: obs.pop("player"), "player"),
        ("missing farms", lambda obs: obs.pop("farms"), "farms"),
        ("missing private", lambda obs: obs.pop("private"), "private"),
        ("missing market", lambda obs: obs.pop("market"), "market"),
        ("missing shed", lambda obs: obs["private"].pop("shed"), r"private\.shed"),
        ("missing market inventory", lambda obs: obs["market"].pop("inventory"), r"market\.inventory"),
        ("missing farm tiles", lambda obs: obs["farms"][0].pop("tiles"), r"farms\[0\]\.tiles"),
        ("missing farm farmer", lambda obs: obs["farms"][0].pop("farmer"), r"farms\[0\]\.farmer"),
        ("missing farm hires_today", lambda obs: obs["farms"][0].pop("hires_today"), r"farms\[0\]\.hires_today"),
    ],
)
def test_missing_required_field(
    description: str,
    mutate: Callable[[dict[str, Any]], None],
    expected_path: str,
) -> None:
    del description
    obs = _make_observation()
    mutate(obs)
    with pytest.raises(InvalidObservationError, match=expected_path):
        convert(obs)


# --- 15. malformed field ---------------------------------------------------
@pytest.mark.parametrize(
    ("description", "mutate", "expected_path"),
    [
        ("player out of range", lambda obs: obs.update(player=5), "player"),
        ("player not an int", lambda obs: obs.update(player="0"), "player"),
        ("day not an int", lambda obs: obs.update(day="x"), "day"),
        ("farmer not a position", lambda obs: obs["farms"][0].update(farmer=[4]), r"farms\[0\]\.farmer"),
        ("farmer position non-int", lambda obs: obs["farms"][0].update(farmer=["a", 4]), r"farms\[0\]\.farmer"),
        ("tiles empty", lambda obs: obs["farms"][0].update(tiles=[]), r"farms\[0\]\.tiles"),
        ("tile row not a list", lambda obs: obs["farms"][0]["tiles"].__setitem__(0, "LOCKED"), r"farms\[0\]\.tiles\[0\]"),
        ("unknown tile kind", lambda obs: obs["farms"][0]["tiles"].__setitem__(1, [None, {"kind": "BARN"}, None, None]), r"farms\[0\]\.tiles\[1\]\[1\]"),
        ("plant missing crop", lambda obs: obs["farms"][0]["tiles"][1].__setitem__(2, {"kind": "PLANT"}), r"farms\[0\]\.tiles\[1\]\[2\]\.crop"),
        ("money not a number", lambda obs: obs["farms"][0].update(money="rich"), r"farms\[0\]\.money"),
        ("shed count non-int", lambda obs: obs["private"]["shed"].__setitem__("WHEAT", "many"), r"private\.shed\.WHEAT"),
        ("market price non-int", lambda obs: obs["market"]["prices"].__setitem__("WHEAT", "low"), r"market\.prices\.WHEAT"),
    ],
)
def test_malformed_field(
    description: str,
    mutate: Callable[[dict[str, Any]], None],
    expected_path: str,
) -> None:
    del description
    obs = _make_observation()
    mutate(obs)
    with pytest.raises(InvalidObservationError, match=expected_path):
        convert(obs)


# --- 16. input mutation after conversion -----------------------------------
def test_mutating_input_does_not_change_state() -> None:
    obs = _make_observation()
    state = convert(obs)

    tile00 = state.current_player_state.farm.tile_at(Position(0, 0))
    assert tile00 is not None

    # Snapshot domain values derived from the observation.
    snapshot = {
        "day": state.day,
        "hour": state.hour,
        "step": state.step,
        "current_player": state.current_player,
        "money": state.current_player_state.farm.money,
        "shed_wheat": state.current_player_state.inventory.get(ItemType.WHEAT),
        "market_wheat_price": state.market.price(ItemType.WHEAT),
        "town_shops": state.town.unlocked_shops,
        "tile00": tile00.tile_type,
        "farmer_pos": state.current_player_state.farm.farmer.position,
        "hand1_pos": state.current_player_state.farm.workers[0].position,
        "hand1_fert": state.current_player_state.workers[1].inventory.get(ItemType.FERTILIZER),
        "seeds_wheat": state.current_player_state.seeds.get(CropType.WHEAT),
    }

    # Mutate the input observation deeply and broadly.
    obs["day"] = 99
    obs["hour"] = 99
    obs["step"] = 999
    obs["player"] = 1
    obs["farms"][0]["money"] = 1.0
    obs["farms"][0]["tiles"][0][0] = {"kind": "WEED"}
    obs["farms"][0]["farmer"][0] = 9
    obs["farms"][0]["hands"][0][0] = 9
    obs["private"]["shed"]["WHEAT"] = 1000
    obs["private"]["seeds"]["WHEAT"] = 1000
    obs["private"]["inventories"][1]["FERTILIZER"] = 1000
    obs["market"]["prices"]["WHEAT"] = 1
    obs["market"]["inventory"]["WHEAT"] = 1
    obs["town"]["unlocked_shops"].append("YARN_STORE")

    assert state.day == snapshot["day"]
    assert state.hour == snapshot["hour"]
    assert state.step == snapshot["step"]
    assert state.current_player == snapshot["current_player"]
    assert state.current_player_state.farm.money == snapshot["money"]
    assert state.current_player_state.inventory.get(ItemType.WHEAT) == snapshot["shed_wheat"]
    assert state.market.price(ItemType.WHEAT) == snapshot["market_wheat_price"]
    assert state.town.unlocked_shops == snapshot["town_shops"]
    tile_after = state.current_player_state.farm.tile_at(Position(0, 0))
    assert tile_after is not None
    assert tile_after.tile_type == snapshot["tile00"]
    assert state.current_player_state.farm.farmer.position == snapshot["farmer_pos"]
    assert state.current_player_state.farm.workers[0].position == snapshot["hand1_pos"]
    assert state.current_player_state.workers[1].inventory.get(ItemType.FERTILIZER) == snapshot["hand1_fert"]
    assert state.current_player_state.seeds.get(CropType.WHEAT) == snapshot["seeds_wheat"]


# --- 17. repeated conversion is equivalent ---------------------------------
def test_repeated_conversion_equivalent() -> None:
    obs = _make_observation()
    first = convert(obs)
    second = convert(obs)
    assert first == second
    assert hash(first) == hash(second)


# --- backward-compatibility shim -------------------------------------------
def test_game_state_from_observation_shim_delegates() -> None:
    obs = _make_observation()
    assert GameState.from_observation(obs) == convert(obs)


# --- validation is non-mutating --------------------------------------------
def test_validation_does_not_mutate_input() -> None:
    obs = _make_observation()
    before = copy.deepcopy(obs)
    KaggleObservationAdapter.validate(obs)
    assert obs == before
