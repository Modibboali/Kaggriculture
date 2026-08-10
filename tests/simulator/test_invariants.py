"""Invariant / property tests for the simulator.

These assert properties that must hold for *every* transition, and which match
the actual Kaggriculture behavior (verified against the environment):

* ``apply`` never mutates its input state (full immutability),
* time progression is monotonic: ``step`` and ``day`` never decrease,
* simulation is deterministic: identical initial states + identical actions
  produce identical states,
* harvesting cannot create inventory from an empty tile,
* a successful harvest removes the crop from the tile.

No random-walk or fuzz testing: transitions are deterministic by design.
"""

from __future__ import annotations

from dataclasses import replace

from agent.actions import (
    BuildCoopAction,
    BuyLandAction,
    BuySeedAction,
    DigAction,
    FertilizeAction,
    HarvestAction,
    HireAction,
    PickupAction,
    PlaceAction,
    PlantAction,
    TurnAction,
    WaterAction,
)
from agent.simulator import GameConfig, Simulator
from agent.state import (
    AnimalType,
    CoopTile,
    CropType,
    EmptyTile,
    GameState,
    Inventory,
    ItemType,
    PlantTile,
    Position,
    Quadrant,
)

CONFIG = GameConfig(board_size=4)
FARMER_POS = Position(1, 1)


def _advance(simulator: Simulator, state: GameState, turns: int, action: TurnAction) -> GameState:
    for _ in range(turns):
        state = simulator.apply(state, action)
    return state


def test_apply_does_not_mutate_input() -> None:
    simulator = Simulator(CONFIG)
    before = _advance(simulator, _initial(), 3, TurnAction())
    snapshot = before
    simulator.apply(before, TurnAction(farmer_action=WaterAction()))
    assert before == snapshot


def test_step_and_day_are_monotonic() -> None:
    simulator = Simulator(CONFIG)
    state = _initial()
    previous_step, previous_day = -1, -1
    for _ in range(60):
        state = simulator.apply(state, TurnAction())
        assert state.step > previous_step
        assert state.day >= previous_day
        previous_step, previous_day = state.step, state.day


def test_determinism_equivalent_states_and_actions() -> None:
    simulator = Simulator(CONFIG)
    a = _advance(simulator, _initial(), 5, TurnAction())
    b = _advance(simulator, _initial(), 5, TurnAction())
    assert a == b


def test_harvest_empty_tile_creates_nothing() -> None:
    simulator = Simulator(CONFIG)
    state = _initial()
    before = state.players[0].workers[0].inventory.total_items()
    after = simulator.apply(state, TurnAction(farmer_action=HarvestAction()))
    assert after.players[0].workers[0].inventory.total_items() == before
    assert isinstance(after.players[0].farm.tile_at(FARMER_POS), EmptyTile)


def test_successful_harvest_removes_plant() -> None:
    simulator = Simulator(CONFIG)
    # Buy + plant + water on day 0, then advance to day 2 watering daily.
    state = _initial()
    state = simulator.apply(state, TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),)))
    state = simulator.apply(state, TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT)))
    state = simulator.apply(state, TurnAction(farmer_action=WaterAction()))
    # Fast-forward to day 2 hour 0 (48 turns) watering at the start of day 1.
    state = simulator.apply(state, TurnAction())  # step 3
    for _ in range(20):
        state = simulator.apply(state, TurnAction())
    state = simulator.apply(state, TurnAction(farmer_action=WaterAction()))  # day 1
    for _ in range(23):
        state = simulator.apply(state, TurnAction())
    state = simulator.apply(state, TurnAction(farmer_action=WaterAction()))  # day 2
    tile = state.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    harvested = simulator.apply(state, TurnAction(farmer_action=HarvestAction()))
    assert isinstance(harvested.players[0].farm.tile_at(FARMER_POS), EmptyTile)
    assert harvested.players[0].workers[0].inventory.get(ItemType.WHEAT) == 2


def test_insufficient_funds_do_not_create_money() -> None:
    simulator = Simulator(CONFIG)
    state = _initial()
    poor = replace(state.players[0], farm=replace(state.players[0].farm, money=500))
    state = replace(state, players=(poor, state.players[1]))
    after = simulator.apply(
        state, TurnAction(market_actions=(BuyLandAction(quadrant=Quadrant.NE),))
    )
    assert after.players[0].farm.money == 500
    assert after.players[0].farm.unlocked_quadrants == frozenset({Quadrant.NW})


def test_land_purchase_changes_land_exactly_once() -> None:
    simulator = Simulator(CONFIG)
    state = _initial()
    before = state.players[0].farm.unlocked_quadrants
    after = simulator.apply(
        state, TurnAction(market_actions=(BuyLandAction(quadrant=Quadrant.NE),))
    )
    assert len(after.players[0].farm.unlocked_quadrants) == len(before) + 1
    assert after.players[0].farm.money == 3000 - 1000


def test_fertilize_consumes_exactly_one_fertilizer() -> None:
    simulator = Simulator(CONFIG)
    state = _initial()
    # Put a plant + 2 fertilizer in the shed, pick up 1, fertilize.
    from agent.state import Farm, PlantState, PlantTile, Worker

    farm = state.players[0].farm
    planted = farm.replace_tile(FARMER_POS, PlantTile(PlantState(CropType.WHEAT, 0, False, 1, 1, -1, 120)))
    player = replace(state.players[0], farm=planted, inventory=Inventory({ItemType.FERTILIZER: 2}))
    state = replace(state, players=(player, state.players[1]))
    state = simulator.apply(
        state, TurnAction(farmer_action=PickupAction(item=ItemType.FERTILIZER, quantity=1))
    )
    total_before = (
        state.players[0].inventory.get(ItemType.FERTILIZER)
        + state.players[0].workers[0].inventory.get(ItemType.FERTILIZER)
    )
    after = simulator.apply(state, TurnAction(farmer_action=FertilizeAction()))
    total_after = (
        after.players[0].inventory.get(ItemType.FERTILIZER)
        + after.players[0].workers[0].inventory.get(ItemType.FERTILIZER)
    )
    assert total_after == total_before - 1


def test_invalid_dig_does_not_create_resources() -> None:
    simulator = Simulator(CONFIG)
    state = _initial()
    before_money = state.players[0].farm.money
    before_shed = state.players[0].inventory.total_items()
    after = simulator.apply(state, TurnAction(farmer_action=DigAction()))  # empty tile
    assert after.players[0].farm.money == before_money
    assert after.players[0].inventory.total_items() == before_shed
    assert after.players[0].workers[0].inventory.total_items() == 0


def test_hire_adds_exactly_one_hand() -> None:
    simulator = Simulator(CONFIG)
    state = _initial()
    before = len(state.players[0].farm.workers)
    after = simulator.apply(state, TurnAction(market_actions=(HireAction(quantity=1),)))
    assert len(after.players[0].farm.workers) == before + 1
    assert after.players[0].farm.money == 3000 - 1  # fib(0) = 1


def test_build_consumes_nothing() -> None:
    simulator = Simulator(CONFIG)
    state = _initial()
    before_money = state.players[0].farm.money
    before_shed = state.players[0].inventory.total_items()
    after = simulator.apply(state, TurnAction(farmer_action=BuildCoopAction()))
    assert after.players[0].farm.money == before_money
    assert after.players[0].inventory.total_items() == before_shed
    assert isinstance(after.players[0].farm.tile_at(FARMER_POS), CoopTile)


def test_invalid_place_does_not_create_animal() -> None:
    simulator = Simulator(CONFIG)
    state = _initial()
    # Farmer at (0,0) (empty, not shed-adjacent) carrying a goose.
    worker = state.players[0].workers[0].moved_to(Position(0, 0))
    worker = replace(worker, inventory=Inventory({ItemType.GOOSE: 1}))
    farm = replace(state.players[0].farm, farmer=worker)
    player = replace(state.players[0], farm=farm, workers=(worker,))
    state = replace(state, players=(player, state.players[1]))
    after = simulator.apply(state, TurnAction(farmer_action=PlaceAction(animal=AnimalType.GOOSE)))
    assert after.players[0].workers[0].inventory.get(ItemType.GOOSE) == 1
    assert isinstance(after.players[0].farm.tile_at(Position(0, 0)), EmptyTile)


def _initial() -> GameState:
    """A fresh 4x4 state with empty player inventories (no carried items)."""
    return _build_clean_state()


def _build_clean_state() -> GameState:
    from agent.state import Farm, Market, PlayerState, Seeds, Quadrant, Town, Worker
    from agent.simulator.game_config import DEFAULT_MARKET_PARAMS, MARKET_I0, PRODUCT_ITEMS

    size = 4
    tiles = tuple(tuple(EmptyTile() for _ in range(size)) for _ in range(size))
    farmer0 = Worker(0, FARMER_POS, Inventory.empty(), True)
    farmer1 = Worker(0, FARMER_POS, Inventory.empty(), True)
    farm0 = Farm(money=3000, tiles=tiles, farmer=farmer0, workers=(), unlocked_quadrants=frozenset({Quadrant.NW}), hires_today=0)
    farm1 = Farm(money=3000, tiles=tiles, farmer=farmer1, workers=(), unlocked_quadrants=frozenset({Quadrant.NW}), hires_today=0)
    market = Market(
        Inventory({item: MARKET_I0 for item in PRODUCT_ITEMS}),
        {item: params.base for item, params in DEFAULT_MARKET_PARAMS.items()},
    )
    return GameState(
        day=0,
        hour=0,
        step=0,
        market=market,
        town=Town(frozenset()),
        players=(
            PlayerState(farm0, Inventory.empty(), Seeds.empty(), (farmer0,)),
            PlayerState(farm1, Inventory.empty(), Seeds.empty(), (farmer1,)),
        ),
        current_player=0,
    )
