"""Unit tests for the simulator's verified transition layers.

These tests run the simulator on a small synthetic 4x4 board (no Kaggle
environment required) so each transition's local behavior can be checked in
isolation, without the differential harness.
"""

from __future__ import annotations

from dataclasses import replace

from agent.actions import (
    BuildCoopAction,
    BuildPastureAction,
    BuyLandAction,
    BuySeedAction,
    CareAction,
    CollectFertilizerAction,
    DigAction,
    DropAction,
    FeedAction,
    FertilizeAction,
    HarvestAction,
    HireAction,
    MovementAction,
    PickupAction,
    PlaceAction,
    PlantAction,
    TurnAction,
    WaterAction,
)
from agent.simulator import CropLifecycleTransition, EndOfDayProcessor, GameConfig, Simulator
from agent.simulator.game_config import DEFAULT_MARKET_PARAMS, MARKET_I0, PRODUCT_ITEMS
from agent.state import (
    AnimalState,
    AnimalType,
    CoopTile,
    CropType,
    Direction,
    EmptyTile,
    Farm,
    GameState,
    Inventory,
    ItemType,
    LOCKED_TILE,
    Market,
    PastureTile,
    PlantState,
    PlantTile,
    PlayerState,
    Position,
    Quadrant,
    Seeds,
    Tile,
    Town,
    WEED_TILE,
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
    farm0 = Farm(money=money0, tiles=tiles, farmer=farmer0, workers=(), unlocked_quadrants=frozenset({Quadrant.NW}), hires_today=0)
    farm1 = Farm(money=money1, tiles=tiles, farmer=farmer1, workers=(), unlocked_quadrants=frozenset({Quadrant.NW}), hires_today=0)
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


# --- test helpers ----------------------------------------------------------
FARMER_POS = Position(1, 1)


def _plant(
    *,
    crop: CropType = CropType.WHEAT,
    yield_units: int = 1,
    planted_day: int = 0,
    watered_today: bool = False,
    consecutive: int = 1,
    max_lifespan: int = 120,
    fertilized_until: int = -1,
) -> PlantState:
    return PlantState(
        crop=crop,
        planted_day=planted_day,
        watered_today=watered_today,
        consecutive_unwatered=consecutive,
        yield_units=yield_units,
        fertilized_until_day=fertilized_until,
        max_lifespan_step=max_lifespan,
    )


def _with_plant(state: GameState, position: Position, plant: PlantState) -> GameState:
    return _with_tile_at(state, position, PlantTile(plant))


def _with_tile_at(state: GameState, position: Position, tile: Tile) -> GameState:
    farm = state.players[0].farm
    player = replace(state.players[0], farm=farm.replace_tile(position, tile))
    return replace(state, players=(player, state.players[1]))


def _with_carried(state: GameState, items: dict[ItemType, int]) -> GameState:
    player = state.players[0]
    worker = player.workers[0]
    new_worker = replace(worker, inventory=Inventory(items))
    new_workers = (new_worker, *player.workers[1:])
    farm = replace(player.farm, farmer=new_worker, workers=new_workers[1:])
    new_player = replace(player, farm=farm, workers=new_workers)
    return replace(state, players=(new_player, state.players[1]))


# --- HARVEST ---------------------------------------------------------------


def test_harvest_mature_adds_to_farmer_inventory() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(day=2, hour=0, step=48)
    state = _with_plant(state, FARMER_POS, _plant(yield_units=2, consecutive=0))
    after = simulator.apply(state, TurnAction(farmer_action=HarvestAction()))
    assert after.players[0].workers[0].inventory.get(ItemType.WHEAT) == 2
    assert isinstance(after.players[0].farm.tile_at(FARMER_POS), EmptyTile)


def test_harvest_immature_is_noop() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(day=0, hour=0, step=0)
    state = _with_plant(state, FARMER_POS, _plant(yield_units=1))
    after = simulator.apply(state, TurnAction(farmer_action=HarvestAction()))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    assert tile.plant.yield_units == 1
    assert after.players[0].workers[0].inventory.total_items() == 0


def test_harvest_zero_yield_is_noop() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(day=2, hour=0, step=48)
    state = _with_plant(state, FARMER_POS, _plant(yield_units=0, consecutive=0))
    after = simulator.apply(state, TurnAction(farmer_action=HarvestAction()))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    assert after.players[0].workers[0].inventory.total_items() == 0


def test_harvest_empty_tile_is_noop() -> None:
    simulator = Simulator(CONFIG)
    after = simulator.apply(make_state(day=2, hour=0, step=48), TurnAction(farmer_action=HarvestAction()))
    assert isinstance(after.players[0].farm.tile_at(FARMER_POS), EmptyTile)
    assert after.players[0].workers[0].inventory.total_items() == 0


def test_harvest_ongoing_keeps_plant_and_resets_yield() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(day=10, hour=0, step=240)
    state = _with_plant(
        state,
        FARMER_POS,
        _plant(crop=CropType.TOMATO, yield_units=2, consecutive=0, max_lifespan=288),
    )
    after = simulator.apply(state, TurnAction(farmer_action=HarvestAction()))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    assert tile.plant.yield_units == 0
    assert after.players[0].workers[0].inventory.get(ItemType.TOMATO) == 2


# --- crop lifecycle --------------------------------------------------------


def test_daily_refresh_unwatered_plant_dies() -> None:
    lifecycle = CropLifecycleTransition(CONFIG)
    state = make_state(day=0, hour=23, step=23)
    state = _with_plant(state, FARMER_POS, _plant(consecutive=1))
    after = lifecycle.daily_refresh(state)
    assert after.players[0].farm.tile_at(FARMER_POS) is WEED_TILE


def test_daily_refresh_watered_resets_consecutive() -> None:
    lifecycle = CropLifecycleTransition(CONFIG)
    state = make_state(day=0, hour=23, step=23)
    state = _with_plant(state, FARMER_POS, _plant(watered_today=True, consecutive=1))
    after = lifecycle.daily_refresh(state)
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    assert tile.plant.watered_today is False
    assert tile.plant.consecutive_unwatered == 0
    assert tile.plant.yield_units == 1


def test_daily_refresh_ongoing_regrows() -> None:
    lifecycle = CropLifecycleTransition(CONFIG)
    # Tomato planted day 0: at end of day 7 (next_day 8) it produces its first unit.
    state = make_state(day=7, hour=23, step=191)
    state = _with_plant(
        state,
        FARMER_POS,
        _plant(crop=CropType.TOMATO, yield_units=0, watered_today=True, consecutive=0, max_lifespan=-1),
    )
    after = lifecycle.daily_refresh(state)
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    assert tile.plant.yield_units == 1


def test_decay_reduces_yield_then_weeds() -> None:
    lifecycle = CropLifecycleTransition(CONFIG)
    # Plant with max_lifespan_step=10 decays every two steps from step 10.
    state = make_state(day=0, hour=0, step=10)
    state = _with_plant(state, FARMER_POS, _plant(yield_units=2, max_lifespan=10))
    first = lifecycle.decay(state)
    tile = first.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    assert tile.plant.yield_units == 1
    # Next step (parity mismatch) does not decay...
    state_11 = replace(first, step=11)
    assert lifecycle.decay(state_11).players[0].farm.tile_at(FARMER_POS) is not WEED_TILE
    # ...but the step after that does, dropping yield to 0 -> weed.
    state_12 = replace(first, step=12)
    assert lifecycle.decay(state_12).players[0].farm.tile_at(FARMER_POS) is WEED_TILE


# --- end-of-day ------------------------------------------------------------


def test_end_of_day_drops_carried_inventory_to_shed() -> None:
    processor = EndOfDayProcessor(CONFIG)
    state = make_state(day=0, hour=23, step=23)
    state = _with_carried(state, {ItemType.WHEAT: 2})
    after = processor.process(state)
    assert after.players[0].inventory.get(ItemType.WHEAT) == 2
    assert after.players[0].workers[0].inventory.total_items() == 0


def test_end_of_day_resets_farmer_and_clears_hands() -> None:
    processor = EndOfDayProcessor(CONFIG)
    state = make_state(day=0, hour=23, step=23)
    farmer = state.players[0].workers[0]
    moved = replace(state.players[0], workers=(farmer.moved_to(Position(0, 0)),))
    state = replace(state, players=(moved, state.players[1]))
    after = processor.process(state)
    assert after.players[0].farm.farmer.position == Position(1, 1)
    assert after.players[0].farm.hires_today == 0


def test_simulator_runs_end_of_day_at_boundary() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(day=0, hour=23, step=23)
    state = _with_plant(state, FARMER_POS, _plant(watered_today=True, consecutive=0))
    state = _with_carried(state, {ItemType.WHEAT: 1})
    after = simulator.apply(state, TurnAction())
    assert after.step == 24
    assert after.day == 1
    assert after.hour == 0
    # Carried wheat dropped to the shed, plant survived and reset.
    assert after.players[0].inventory.get(ItemType.WHEAT) == 1
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    assert tile.plant.watered_today is False
    assert tile.plant.consecutive_unwatered == 0


# --- FERTILIZE -------------------------------------------------------------


def test_fertilize_consumes_carried_fertilizer_and_sets_window() -> None:
    simulator = Simulator(CONFIG)
    state = _with_plant(make_state(day=0, hour=0, step=0), FARMER_POS, _plant())
    state = _with_carried(state, {ItemType.FERTILIZER: 1})
    after = simulator.apply(state, TurnAction(farmer_action=FertilizeAction()))
    assert after.players[0].workers[0].inventory.get(ItemType.FERTILIZER) == 0
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    assert tile.plant.fertilized_until_day == 2  # day 0 + 2


def test_fertilize_extends_existing_window_with_max() -> None:
    simulator = Simulator(CONFIG)
    state = _with_plant(
        make_state(day=0, hour=0, step=0), FARMER_POS, _plant(fertilized_until=5)
    )
    state = _with_carried(state, {ItemType.FERTILIZER: 1})
    after = simulator.apply(state, TurnAction(farmer_action=FertilizeAction()))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    assert tile.plant.fertilized_until_day == 5  # max(5, day+2)


def test_fertilize_noop_without_fertilizer() -> None:
    simulator = Simulator(CONFIG)
    state = _with_plant(make_state(day=0, hour=0, step=0), FARMER_POS, _plant())
    after = simulator.apply(state, TurnAction(farmer_action=FertilizeAction()))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PlantTile)
    assert tile.plant.fertilized_until_day == -1


def test_fertilize_noop_on_empty_tile() -> None:
    simulator = Simulator(CONFIG)
    state = _with_carried(make_state(day=0, hour=0, step=0), {ItemType.FERTILIZER: 1})
    after = simulator.apply(state, TurnAction(farmer_action=FertilizeAction()))
    assert after.players[0].workers[0].inventory.get(ItemType.FERTILIZER) == 1
    assert isinstance(after.players[0].farm.tile_at(FARMER_POS), EmptyTile)


# --- DIG -------------------------------------------------------------------


def test_dig_removes_plant_without_yield() -> None:
    simulator = Simulator(CONFIG)
    state = _with_plant(make_state(day=0, hour=0, step=0), FARMER_POS, _plant(yield_units=3))
    after = simulator.apply(state, TurnAction(farmer_action=DigAction()))
    assert isinstance(after.players[0].farm.tile_at(FARMER_POS), EmptyTile)
    assert after.players[0].workers[0].inventory.total_items() == 0


def test_dig_removes_weed() -> None:
    simulator = Simulator(CONFIG)
    state = _with_tile_at(make_state(day=0, hour=0, step=0), FARMER_POS, WEED_TILE)
    after = simulator.apply(state, TurnAction(farmer_action=DigAction()))
    assert isinstance(after.players[0].farm.tile_at(FARMER_POS), EmptyTile)


def test_dig_noop_on_empty_tile() -> None:
    simulator = Simulator(CONFIG)
    after = simulator.apply(make_state(day=0, hour=0, step=0), TurnAction(farmer_action=DigAction()))
    assert isinstance(after.players[0].farm.tile_at(FARMER_POS), EmptyTile)


def test_dig_noop_on_locked_tile() -> None:
    simulator = Simulator(CONFIG)
    state = _with_tile_at(make_state(day=0, hour=0, step=0), FARMER_POS, LOCKED_TILE)
    after = simulator.apply(state, TurnAction(farmer_action=DigAction()))
    assert after.players[0].farm.tile_at(FARMER_POS) is LOCKED_TILE


# --- PICKUP / DROP ---------------------------------------------------------


def test_pickup_moves_shed_item_to_carried() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(day=0, hour=0, step=0)
    player = replace(state.players[0], inventory=Inventory({ItemType.FERTILIZER: 2}))
    state = replace(state, players=(player, state.players[1]))
    after = simulator.apply(
        state, TurnAction(farmer_action=PickupAction(item=ItemType.FERTILIZER, quantity=1))
    )
    assert after.players[0].inventory.get(ItemType.FERTILIZER) == 1
    assert after.players[0].workers[0].inventory.get(ItemType.FERTILIZER) == 1


def test_drop_moves_carried_to_shed() -> None:
    simulator = Simulator(CONFIG)
    state = _with_carried(make_state(day=0, hour=0, step=0), {ItemType.WHEAT: 2})
    after = simulator.apply(state, TurnAction(farmer_action=DropAction()))
    assert after.players[0].inventory.get(ItemType.WHEAT) == 2
    assert after.players[0].workers[0].inventory.total_items() == 0


# --- BUY_LAND --------------------------------------------------------------


def test_buy_land_purchases_next_quadrant_and_unlocks_tiles() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(money0=3000)
    after = simulator.apply(
        state, TurnAction(market_actions=(BuyLandAction(quadrant=Quadrant.NE),))
    )
    assert after.players[0].farm.money == 2000
    assert Quadrant.NE in after.players[0].farm.unlocked_quadrants
    assert isinstance(after.players[0].farm.tile_at(Position(2, 0)), EmptyTile)  # NE tile


def test_buy_land_insufficient_money_is_noop() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(money0=500)
    after = simulator.apply(
        state, TurnAction(market_actions=(BuyLandAction(quadrant=Quadrant.NE),))
    )
    assert after.players[0].farm.money == 500
    assert after.players[0].farm.unlocked_quadrants == frozenset({Quadrant.NW})


def test_buy_land_purchases_in_fixed_order() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(money0=3000)
    buy = TurnAction(market_actions=(BuyLandAction(quadrant=Quadrant.NE),))
    first = simulator.apply(state, buy)
    assert first.players[0].farm.unlocked_quadrants == frozenset({Quadrant.NW, Quadrant.NE})
    second = simulator.apply(first, buy)
    assert second.players[0].farm.unlocked_quadrants == frozenset(
        {Quadrant.NW, Quadrant.NE, Quadrant.SW}
    )
    assert second.players[0].farm.money == 0


# --- structures / animals / workers -----------------------------------------


def _animal(
    *,
    animal: AnimalType = AnimalType.GOOSE,
    placed_day: int = 0,
    yield_units: int = 0,
    fed_today: bool = False,
    consec_unfed: int = 0,
    cared_today: bool = False,
    fert_available: bool = False,
    pending: int = 0,
) -> AnimalState:
    return AnimalState(
        animal=animal,
        placed_day=placed_day,
        yield_units=yield_units,
        consecutive_unfed=consec_unfed,
        fed_today=fed_today,
        cared_today=cared_today,
        fertilizer_available=fert_available,
        pending_care_bonus=pending,
    )


def _with_animal_state(
    state: GameState, position: Position, animal: AnimalState
) -> GameState:
    return _with_tile_at(state, position, CoopTile(animal))


def test_build_coop_on_empty_tile_is_free() -> None:
    simulator = Simulator(CONFIG)
    after = simulator.apply(make_state(), TurnAction(farmer_action=BuildCoopAction()))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, CoopTile)
    assert tile.animal is None
    assert after.players[0].farm.money == 3000


def test_build_noop_on_occupied_tile() -> None:
    simulator = Simulator(CONFIG)
    state = _with_plant(make_state(), FARMER_POS, _plant())
    after = simulator.apply(state, TurnAction(farmer_action=BuildCoopAction()))
    assert isinstance(after.players[0].farm.tile_at(FARMER_POS), PlantTile)


def test_place_animal_on_matching_structure() -> None:
    simulator = Simulator(CONFIG)
    state = _with_tile_at(make_state(), FARMER_POS, CoopTile(None))
    state = _with_carried(state, {ItemType.GOOSE: 1})
    after = simulator.apply(state, TurnAction(farmer_action=PlaceAction(animal=AnimalType.GOOSE)))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, CoopTile)
    assert tile.animal is not None and tile.animal.animal == AnimalType.GOOSE
    assert after.players[0].workers[0].inventory.get(ItemType.GOOSE) == 0


def test_place_animal_wrong_structure_is_noop() -> None:
    simulator = Simulator(CONFIG)
    state = _with_tile_at(make_state(), FARMER_POS, PastureTile(None))
    state = _with_carried(state, {ItemType.GOOSE: 1})
    after = simulator.apply(state, TurnAction(farmer_action=PlaceAction(animal=AnimalType.GOOSE)))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, PastureTile)
    assert tile.animal is None
    assert after.players[0].workers[0].inventory.get(ItemType.GOOSE) == 1


def test_feed_consumes_carried_wheat() -> None:
    simulator = Simulator(CONFIG)
    state = _with_animal_state(make_state(), FARMER_POS, _animal())
    state = _with_carried(state, {ItemType.WHEAT: 1})
    after = simulator.apply(state, TurnAction(farmer_action=FeedAction()))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, CoopTile) and tile.animal is not None
    assert tile.animal.fed_today is True
    assert after.players[0].workers[0].inventory.get(ItemType.WHEAT) == 0


def test_feed_noop_without_wheat() -> None:
    simulator = Simulator(CONFIG)
    state = _with_animal_state(make_state(), FARMER_POS, _animal())
    after = simulator.apply(state, TurnAction(farmer_action=FeedAction()))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, CoopTile) and tile.animal is not None
    assert tile.animal.fed_today is False


def test_care_sets_cared_today() -> None:
    simulator = Simulator(CONFIG)
    state = _with_animal_state(make_state(), FARMER_POS, _animal())
    after = simulator.apply(state, TurnAction(farmer_action=CareAction()))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, CoopTile) and tile.animal is not None
    assert tile.animal.cared_today is True


def test_collect_fertilizer_requires_available_flag() -> None:
    simulator = Simulator(CONFIG)
    state = _with_animal_state(make_state(), FARMER_POS, _animal(fert_available=True))
    after = simulator.apply(state, TurnAction(farmer_action=CollectFertilizerAction()))
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, CoopTile) and tile.animal is not None
    assert tile.animal.fertilizer_available is False
    assert after.players[0].workers[0].inventory.get(ItemType.FERTILIZER) == 1


def test_hire_spawns_hand_and_costs() -> None:
    simulator = Simulator(CONFIG)
    after = simulator.apply(make_state(), TurnAction(market_actions=(HireAction(quantity=1),)))
    assert after.players[0].farm.money == 2999
    assert len(after.players[0].farm.workers) == 1
    assert after.players[0].farm.workers[0].position == Position(2, 1)  # 4x4 spawn
    assert after.players[0].farm.hires_today == 1


def test_hire_insufficient_money_is_noop() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(money0=0)
    after = simulator.apply(state, TurnAction(market_actions=(HireAction(quantity=1),)))
    assert len(after.players[0].farm.workers) == 0
    assert after.players[0].farm.money == 0


def test_animal_daily_refresh_produces_and_sets_fertilizer() -> None:
    lifecycle = CropLifecycleTransition(CONFIG)
    state = make_state(day=3, hour=23, step=95)
    state = _with_animal_state(state, FARMER_POS, _animal(fed_today=True, placed_day=0))
    after = lifecycle.refresh_animals(state)
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, CoopTile) and tile.animal is not None
    assert tile.animal.yield_units == 1  # first goose production (day 4)
    assert tile.animal.fertilizer_available is True
    assert tile.animal.fed_today is False


def test_animal_escapes_after_two_unfed_days() -> None:
    lifecycle = CropLifecycleTransition(CONFIG)
    state = make_state(day=1, hour=23, step=47)
    state = _with_animal_state(state, FARMER_POS, _animal(consec_unfed=1))
    after = lifecycle.refresh_animals(state)
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, CoopTile)
    assert tile.animal is None  # escaped; structure remains


def test_harvest_animal_product() -> None:
    simulator = Simulator(CONFIG)
    state = make_state(day=4, hour=0, step=96)
    state = _with_animal_state(state, FARMER_POS, _animal(yield_units=4))
    after = simulator.apply(state, TurnAction(farmer_action=HarvestAction()))
    assert after.players[0].workers[0].inventory.get(ItemType.EGG) == 4
    tile = after.players[0].farm.tile_at(FARMER_POS)
    assert isinstance(tile, CoopTile) and tile.animal is not None
    assert tile.animal.yield_units == 0
