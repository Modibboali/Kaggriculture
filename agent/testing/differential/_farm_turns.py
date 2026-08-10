"""Shared turn-action building blocks and day-sequence helpers for scenarios.

These are typed ``TurnAction`` objects and small sequence builders used by the
farming / land scenario groups, so each group stays self-contained while the
actual turn lists are explicit and reproducible.
"""

from __future__ import annotations

from ...actions import (
    BuildCoopAction,
    BuildPastureAction,
    BuyAnimalAction,
    BuyLandAction,
    BuyProductAction,
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
from ...state import AnimalType, CropType, Direction, ItemType, Quadrant

TURNS_PER_DAY = 24

PASS_TURN = TurnAction()

BUY_WHEAT = TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),))
BUY_FERT = TurnAction(market_actions=(BuyProductAction(item=ItemType.FERTILIZER, quantity=1),))
BUY_FERT2 = TurnAction(market_actions=(BuyProductAction(item=ItemType.FERTILIZER, quantity=2),))

PICKUP_FERT = TurnAction(farmer_action=PickupAction(item=ItemType.FERTILIZER, quantity=1))
PICKUP_FERT2 = TurnAction(farmer_action=PickupAction(item=ItemType.FERTILIZER, quantity=2))

PLANT_WHEAT = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
WATER = TurnAction(farmer_action=WaterAction())
HARVEST = TurnAction(farmer_action=HarvestAction())
FERTILIZE = TurnAction(farmer_action=FertilizeAction())
DIG = TurnAction(farmer_action=DigAction())
DROP = TurnAction(farmer_action=DropAction())
MOVE_WEST = TurnAction(farmer_action=MovementAction(direction=Direction.WEST))
MOVE_EAST = TurnAction(farmer_action=MovementAction(direction=Direction.EAST))
MOVE_NORTH = TurnAction(farmer_action=MovementAction(direction=Direction.NORTH))

BUY_LAND_NE = TurnAction(market_actions=(BuyLandAction(quadrant=Quadrant.NE),))

# --- structures / animals / workers ----------------------------------------
BUILD_COOP = TurnAction(farmer_action=BuildCoopAction())
BUILD_PASTURE = TurnAction(farmer_action=BuildPastureAction())

BUY_GOOSE = TurnAction(market_actions=(BuyAnimalAction(animal=AnimalType.GOOSE, quantity=1),))
BUY_COW = TurnAction(market_actions=(BuyAnimalAction(animal=AnimalType.COW, quantity=1),))
BUY_SHEEP = TurnAction(market_actions=(BuyAnimalAction(animal=AnimalType.SHEEP, quantity=1),))
PICKUP_GOOSE = TurnAction(farmer_action=PickupAction(item=ItemType.GOOSE, quantity=1))
PICKUP_COW = TurnAction(farmer_action=PickupAction(item=ItemType.COW, quantity=1))
PICKUP_SHEEP = TurnAction(farmer_action=PickupAction(item=ItemType.SHEEP, quantity=1))
PLACE_GOOSE = TurnAction(farmer_action=PlaceAction(animal=AnimalType.GOOSE))
PLACE_COW = TurnAction(farmer_action=PlaceAction(animal=AnimalType.COW))
PLACE_SHEEP = TurnAction(farmer_action=PlaceAction(animal=AnimalType.SHEEP))

FEED = TurnAction(farmer_action=FeedAction())
CARE = TurnAction(farmer_action=CareAction())
COLLECT_FERT = TurnAction(farmer_action=CollectFertilizerAction())
HIRE = TurnAction(market_actions=(HireAction(quantity=1),))

PICKUP_WHEAT1 = TurnAction(farmer_action=PickupAction(item=ItemType.WHEAT, quantity=1))

_Turns = tuple[TurnAction, ...]


def buy_wheat_product(quantity: int) -> TurnAction:
    """Buy ``quantity`` WHEAT from the market into the shed."""
    return TurnAction(market_actions=(BuyProductAction(item=ItemType.WHEAT, quantity=quantity),))


def pickup_wheat(quantity: int) -> TurnAction:
    """Pick ``quantity`` WHEAT up from the shed into the carried inventory."""
    return TurnAction(farmer_action=PickupAction(item=ItemType.WHEAT, quantity=quantity))


def hand_plant(crop: CropType) -> TurnAction:
    """Farmer passes; the first hired hand plants ``crop``."""
    return TurnAction(worker_actions=(PlantAction(crop=crop),))


def hand_move(direction: Direction) -> TurnAction:
    """Farmer passes; the first hired hand moves ``direction``."""
    return TurnAction(worker_actions=(MovementAction(direction=direction),))


def hand_build_coop() -> TurnAction:
    """Farmer passes; the first hired hand builds a coop."""
    return TurnAction(worker_actions=(BuildCoopAction(),))


def pad_day0(prelude: _Turns) -> _Turns:
    """``prelude`` on day 0 padded with PASS to fill the whole day."""
    assert len(prelude) <= TURNS_PER_DAY
    return (*prelude, *((PASS_TURN,) * (TURNS_PER_DAY - len(prelude))))


def append_days(turns: _Turns, days: int, *, water: bool = True) -> _Turns:
    """Append ``days`` full days; WATER at the first hour when ``water``."""
    result: list[TurnAction] = list(turns)
    for _ in range(days):
        result.append(WATER if water else PASS_TURN)
        result.extend([PASS_TURN] * (TURNS_PER_DAY - 1))
    return tuple(result)


def feed_care_days(prelude: _Turns, days: int) -> _Turns:
    """``prelude`` ends on day 0 with the farmer carrying wheat at the shed;
    FEED + CARE close day 0, then each of ``days`` further days picks up 1
    wheat (dropped to the shed each night), feeds, and cares.

    The prelude must buy/pick up at least ``days + 1`` WHEAT so the shed stays
    stocked for the daily pickups.
    """
    turns: list[TurnAction] = list(prelude)
    used = len(prelude) + 2  # + FEED + CARE
    assert used <= TURNS_PER_DAY
    turns.extend([FEED, CARE])
    turns.extend([PASS_TURN] * (TURNS_PER_DAY - used))
    for _ in range(days):
        turns.append(PICKUP_WHEAT1)
        turns.append(FEED)
        turns.append(CARE)
        turns.extend([PASS_TURN] * (TURNS_PER_DAY - 3))
    return tuple(turns)
