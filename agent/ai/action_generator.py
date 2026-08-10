"""Action generation for search.

The generator produces a focused set of *legal, meaningful* candidate
``TurnAction``\\ s for the acting player — deliberately not every
syntactically possible action (the naive space is enormous). It reads the
current tile, inventory, seeds, money, land and market to avoid obviously
dominated no-ops. Illegal-but-submitted actions are silent no-ops in the
simulator, so the generator needs only light legality checks; it never
reimplements transition rules.

Candidates are farmer-action turns and market-order turns (the first baseline
does not combine both in one turn). Generation is deterministic (no RNG).
"""

from __future__ import annotations

from ..actions import (
    BuyAnimalAction,
    BuyLandAction,
    BuyProductAction,
    BuySeedAction,
    BuildCoopAction,
    BuildPastureAction,
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
    SellAction,
    TurnAction,
    WaterAction,
)
from ..simulator.game_config import LAND_ORDER, LAND_PRICES, PRODUCT_ITEMS, GameConfig
from ..state import (
    AnimalType,
    CoopTile,
    CropType,
    Direction,
    EmptyTile,
    ItemType,
    PastureTile,
    PlantTile,
    Position,
    StructureType,
    WeedTile,
)
from .search_state import SearchState

# Products that may be sold back to the market (SELL draws from the shed).
_SELLABLE = frozenset(PRODUCT_ITEMS)


def _fib(n: int) -> int:
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _is_shed_adjacent(position: Position, board_size: int) -> bool:
    half = board_size // 2
    return (position.x, position.y) in {
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    }


def _dedupe(actions: list[TurnAction]) -> list[TurnAction]:
    seen: set[TurnAction] = set()
    result: list[TurnAction] = []
    for action in actions:
        if action not in seen:
            seen.add(action)
            result.append(action)
    return result


class ActionGenerator:
    """Generates legal, meaningful candidate actions for the acting player."""

    def __init__(self, config: GameConfig, *, include_movement: bool = True) -> None:
        self._config = config
        self._include_movement = include_movement

    def generate(self, state: SearchState) -> tuple[TurnAction, ...]:
        game = state.game
        player = game.current_player
        player_state = game.players[player]
        farm = player_state.farm
        farmer = farm.farmer
        tile = farm.tile_at(farmer.position)
        board = farm.board_size
        money = farm.money
        carried = farmer.inventory
        shed = player_state.inventory
        seeds = player_state.seeds

        actions: list[TurnAction] = [TurnAction()]

        # --- movement (kept small: 4 directions) ---------------------------
        if self._include_movement:
            for direction in Direction:
                dx, dy = direction.delta
                nx, ny = farmer.position.x + dx, farmer.position.y + dy
                if 0 <= nx < board and 0 <= ny < board:
                    actions.append(TurnAction(farmer_action=MovementAction(direction=direction)))

        # --- tile-based farmer actions -------------------------------------
        if isinstance(tile, EmptyTile):
            for crop in CropType:
                if seeds.get(crop) > 0:
                    actions.append(TurnAction(farmer_action=PlantAction(crop=crop)))
            actions.append(TurnAction(farmer_action=BuildCoopAction()))
            actions.append(TurnAction(farmer_action=BuildPastureAction()))
        elif isinstance(tile, PlantTile):
            plant = tile.plant
            if not plant.watered_today:
                actions.append(TurnAction(farmer_action=WaterAction()))
            crop_spec = self._config.crops[plant.crop]
            if plant.yield_units > 0 and game.day - plant.planted_day >= crop_spec.first_yield_day:
                actions.append(TurnAction(farmer_action=HarvestAction()))
            if carried.get(ItemType.FERTILIZER) > 0:
                actions.append(TurnAction(farmer_action=FertilizeAction()))
            actions.append(TurnAction(farmer_action=DigAction()))
        elif isinstance(tile, WeedTile):
            actions.append(TurnAction(farmer_action=DigAction()))
        elif isinstance(tile, (CoopTile, PastureTile)):
            if tile.animal is None:
                for animal_type in AnimalType:
                    animal_spec = self._config.animals[animal_type]
                    matches = (
                        (animal_spec.structure == StructureType.COOP and isinstance(tile, CoopTile))
                        or (animal_spec.structure == StructureType.PASTURE and isinstance(tile, PastureTile))
                    )
                    if matches and carried.get(animal_type.as_item) > 0:
                        actions.append(TurnAction(farmer_action=PlaceAction(animal=animal_type)))
            else:
                animal_state = tile.animal
                if not animal_state.fed_today and carried.get(ItemType.WHEAT) > 0:
                    actions.append(TurnAction(farmer_action=FeedAction()))
                if not animal_state.cared_today:
                    actions.append(TurnAction(farmer_action=CareAction()))
                if animal_state.fertilizer_available:
                    actions.append(TurnAction(farmer_action=CollectFertilizerAction()))
                if animal_state.yield_units > 0:
                    actions.append(TurnAction(farmer_action=HarvestAction()))

        # --- shed-adjacent pickup / drop -----------------------------------
        if _is_shed_adjacent(farmer.position, board):
            if carried.total_items() > 0:
                actions.append(TurnAction(farmer_action=DropAction()))
            for item, count in shed.items.items():
                if count > 0:
                    actions.append(TurnAction(farmer_action=PickupAction(item=item, quantity=1)))

        # --- market orders -------------------------------------------------
        for crop in CropType:
            if money >= self._config.crops[crop].seed_cost:
                actions.append(TurnAction(market_actions=(BuySeedAction(crop=crop, quantity=1),)))
        wheat_price = game.market.price(ItemType.WHEAT)
        if money >= wheat_price:
            actions.append(TurnAction(market_actions=(BuyProductAction(item=ItemType.WHEAT, quantity=1),)))
        fert_price = game.market.price(ItemType.FERTILIZER)
        if money >= fert_price:
            actions.append(TurnAction(market_actions=(BuyProductAction(item=ItemType.FERTILIZER, quantity=1),)))
        for item, count in shed.items.items():
            if item in _SELLABLE and count > 0:
                actions.append(TurnAction(market_actions=(SellAction(item=item, quantity=1),)))
        n_extra = len(farm.unlocked_quadrants) - 1  # NW is always present
        if n_extra < len(LAND_ORDER) and money >= LAND_PRICES[n_extra]:
            actions.append(TurnAction(market_actions=(BuyLandAction(quadrant=LAND_ORDER[n_extra]),)))
        hire_cost = self._config.farm_hand_cost_mult * _fib(farm.hires_today)
        if money >= hire_cost:
            actions.append(TurnAction(market_actions=(HireAction(quantity=1),)))
        for animal in AnimalType:
            if money >= self._config.animals[animal].cost:
                actions.append(TurnAction(market_actions=(BuyAnimalAction(animal=animal, quantity=1),)))

        return tuple(_dedupe(actions))
