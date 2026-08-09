"""Legal market orders for the acting player.

Legality here is derived from the player's farm (money, unlocked land, hires
made today) and the shared market (product prices). The fixed costs below are
game constants documented in the Kaggriculture overview's "Object Types" and
"Town Buildings" tables; they are static configuration, not heuristics, and
are embedded here (rather than in the state model) to keep the domain model
free of business logic.
"""

from __future__ import annotations

from ..actions import (
    Action,
    BuyAnimalAction,
    BuyLandAction,
    BuyProductAction,
    BuySeedAction,
    HireAction,
    SellAction,
)
from ..state import AnimalType, CropType, ItemType, Market, PlayerState, Quadrant

# Fixed, non-configurable seed prices (overview "Object Types" table).
SEED_COST: dict[CropType, int] = {
    CropType.WHEAT: 10,
    CropType.CARROT: 20,
    CropType.TOMATO: 50,
    CropType.STRAWBERRY: 100,
    CropType.MELON: 80,
}

# Fixed, non-configurable animal prices (overview "Object Types" table).
ANIMAL_COST: dict[AnimalType, int] = {
    AnimalType.GOOSE: 300,
    AnimalType.COW: 400,
    AnimalType.SHEEP: 500,
}

# Only WHEAT and FERTILIZER can be bought back from the market.
BUYABLE_PRODUCTS: tuple[ItemType, ...] = (ItemType.WHEAT, ItemType.FERTILIZER)

# Cost of the 1st / 2nd / 3rd land unlock (overview "Costs are: $1k, $2k, $4k").
LAND_COST: tuple[int, int, int] = (1000, 2000, 4000)


def _fib(n: int) -> int:
    """Fibonacci with ``fib(1) == fib(2) == 1`` (used for hire pricing)."""
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


class MarketGenerator:
    """Generates every legal market order for the acting player."""

    def generate(self, player: PlayerState, market: Market) -> tuple[Action, ...]:
        """The legal market orders for ``player`` against ``market``.

        Only quantity-1 (atomic) orders are generated; larger orders are
        compositions the planner can build from these.
        """
        farm = player.farm
        money = farm.money
        actions: list[Action] = []

        # Buy seeds at fixed prices.
        for crop in CropType:
            if money >= SEED_COST[crop]:
                actions.append(BuySeedAction(crop=crop, quantity=1))

        # Buy animals at fixed prices.
        for animal in AnimalType:
            if money >= ANIMAL_COST[animal]:
                actions.append(BuyAnimalAction(animal=animal, quantity=1))

        # Buy WHEAT / FERTILIZER. The buy price is quoted just after a
        # one-unit purchase, so the market's sell price is an accurate
        # in-state proxy for the cost.
        for item in BUYABLE_PRODUCTS:
            if money >= market.price(item):
                actions.append(BuyProductAction(item=item, quantity=1))

        # Sell anything the shed holds (selling is unrestricted).
        for item in ItemType:
            if player.inventory.contains(item):
                actions.append(SellAction(item=item, quantity=1))

        # Next hire costs fib(hires_today + 1) with the default multiplier 1.
        if money >= _fib(farm.hires_today + 1):
            actions.append(HireAction(quantity=1))

        # Buying land: one purchase per not-yet-unlocked quadrant, at a cost
        # that increases with the number of quadrants already unlocked.
        unlocked = len(farm.unlocked_quadrants)
        if unlocked < 4:
            land_cost = LAND_COST[unlocked - 1]
            for quadrant in Quadrant:
                if quadrant not in farm.unlocked_quadrants and money >= land_cost:
                    actions.append(BuyLandAction(quadrant=quadrant))

        return tuple(actions)
