"""Unit tests for the simulator's first verified transition layer.

These tests run the simulator on a small synthetic 4x4 board (no Kaggle
environment required) so each transition's local behavior can be checked in
isolation, without the differential harness.
"""

from __future__ import annotations

from agent.actions import BuySeedAction, MovementAction, PlantAction, TurnAction, WaterAction
from agent.simulator import GameConfig, Simulator
from agent.simulator.game_config import DEFAULT_MARKET_PARAMS, MARKET_I0, PRODUCT_ITEMS
from agent.state import (
    CropType,
    Direction,
    EmptyTile,
    Farm,
    GameState,
    Inventory,
    ItemType,
    Market,
    PlantTile,
    PlayerState,
    Position,
    Seeds,
    Tile,
    Town,
    Worker,
)

CONFIG = GameConfig(board_size=4)


def _grid(size: int) -> tuple[tuple[Tile, ...], ...]:
    return tuple(tuple(EmptyTile() for _ in range(size)) for _ in range(size))


def _market() -> Market:
    inventory = Inventory({item: MARKET_I0 for item in PRODUCT_ITEMS})
    prices = {item: params.base for item, params in DEFAULT_MARKET_PARAMS.items()}
    return Market(inventory, prices)


def make_state(
    *,
    money0: int = 3000,
    money1: int = 3000,
    step: int = 0,
    day: int = 0,
    hour: int = 0,
) -> GameState:
    """A minimal 4x4 state with both farmers standing on the NW shed tile."""
    tiles = _grid(4)
    farmer0 = Worker(0, Position(1, 1), Inventory.empty(), True)
    farmer1 = Worker(0, Position(1, 1), Inventory.empty(), True)
    farm0 = Farm(money=money0, tiles=tiles, farmer=farmer0, workers=(farmer0,), unlocked_quadrants=frozenset(), hires_today=0)
    farm1 = Farm(money=money1, tiles=tiles, farmer=farmer1, workers=(farmer1,), unlocked_quadrants=frozenset(), hires_today=0)
    player0 = PlayerState(farm0, Inventory.empty(), Seeds.empty(), (farmer0,))
    player1 = PlayerState(farm1, Inventory.empty(), Seeds.empty(), (farmer1,))
    return GameState(
        day=day,
        hour=hour,
        step=step,
        market=_market(),
        town=Town(frozenset()),
        players=(player0, player1),
        current_player=0,
    )


# --- turn lifecycle --------------------------------------------------------


def test_pass_advances_turn() -> None:
    simulator = Simulator(CONFIG)
    after = simulator.apply(make_state(), TurnAction())
    assert after.step == 1
    assert after.day == 0
    assert after.hour == 1


def test_turn_advances_day_at_turns_per_day_boundary() -> None:
    simulator = Simulator(CONFIG)
    after = simulator.apply(make_state(step=23, hour=23, day=0), TurnAction())
    assert after.step == 24
    assert after.day == 1
    assert after.hour == 0


# --- movement --------------------------------------------------------------


def test_movement_moves_farmer_north() -> None:
    simulator = Simulator(CONFIG)
    action = TurnAction(farmer_action=MovementAction(direction=Direction.NORTH))
    after = simulator.apply(make_state(), action)
    assert after.players[0].farm.farmer.position == Position(1, 0)


def test_movement_noops_at_board_edge() -> None:
    simulator = Simulator(CONFIG)
    west = TurnAction(farmer_action=MovementAction(direction=Direction.WEST))
    once = simulator.apply(make_state(), west)
    assert once.players[0].farm.farmer.position == Position(0, 1)
    twice = simulator.apply(once, west)
    assert twice.players[0].farm.farmer.position == Position(0, 1)


# --- market ----------------------------------------------------------------


def test_buy_seed_spends_money_and_adds_seed() -> None:
    simulator = Simulator(CONFIG)
    action = TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),))
    after = simulator.apply(make_state(), action)
    assert after.players[0].farm.money == 3000 - 10
    assert after.players[0].seeds.get(CropType.WHEAT) == 1


def test_step_zero_town_consumes_center_products() -> None:
    simulator = Simulator(CONFIG)
    after = simulator.apply(make_state(), TurnAction())
    # The town center consumes 1 of every non-fertilizer product at step 0.
    assert after.market.inventory.get(ItemType.WHEAT) == MARKET_I0 - 1
    assert after.market.inventory.get(ItemType.FERTILIZER) == MARKET_I0
    # ... and prices are refreshed to reflect the new supply.
    assert after.market.price(ItemType.WHEAT) == 26


def test_market_orders_do_not_touch_other_player() -> None:
    simulator = Simulator(CONFIG)
    action = TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),))
    after = simulator.apply(make_state(), action)
    assert after.players[1].farm.money == 3000
    assert after.players[1].seeds.total() == 0


# --- farming ---------------------------------------------------------------


def test_plant_creates_plant_and_consumes_seed() -> None:
    simulator = Simulator(CONFIG)
    buy = simulator.apply(
        make_state(),
        TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),)),
    )
    after = simulator.apply(buy, TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT)))
    tile = after.players[0].farm.tile_at(Position(1, 1))
    assert tile is not None
    assert isinstance(tile, PlantTile)
    assert tile.plant.crop == CropType.WHEAT
    assert tile.plant.watered_today is False
    assert tile.plant.yield_units == 1  # one-time crops start with 1 unit
    assert after.players[0].seeds.get(CropType.WHEAT) == 0


def test_plant_is_noop_without_seed() -> None:
    simulator = Simulator(CONFIG)
    after = simulator.apply(make_state(), TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT)))
    tile = after.players[0].farm.tile_at(Position(1, 1))
    assert isinstance(tile, EmptyTile)


def test_water_marks_plant_watered() -> None:
    simulator = Simulator(CONFIG)
    buy = simulator.apply(
        make_state(),
        TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),)),
    )
    planted = simulator.apply(buy, TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT)))
    watered = simulator.apply(planted, TurnAction(farmer_action=WaterAction()))
    tile = watered.players[0].farm.tile_at(Position(1, 1))
    assert isinstance(tile, PlantTile)
    assert tile.plant.watered_today is True
    # Planted on day 0 -> age 0 is before the bonus window, so no yield bump.
    assert tile.plant.yield_units == 1


def test_water_is_noop_without_plant() -> None:
    simulator = Simulator(CONFIG)
    after = simulator.apply(make_state(), TurnAction(farmer_action=WaterAction()))
    assert after.step == 1
    assert after.players[0].farm.farmer.position == Position(1, 1)
    assert isinstance(after.players[0].farm.tile_at(Position(1, 1)), EmptyTile)
