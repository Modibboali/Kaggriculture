"""Static game data and configuration for the simulator.

These tables are transcribed from the official Kaggriculture environment
(``kaggle_environments/envs/kaggriculture/kaggriculture.py``), which is the
ground-truth source of the rules. They are game *rules/data* and deliberately
do not live in the environment-independent domain model; the simulator is the
rules layer that consumes them. The differential framework feeds the same
configuration to both sides, so these defaults must match the official engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..state import AnimalType, CropType, ItemType, ShopType


@dataclass(frozen=True, slots=True)
class CropSpec:
    """Per-crop growth parameters (from the environment's CROPS table)."""

    seed_cost: int
    first_yield_day: int
    max_yield_day: int
    interval: int
    max_yield: int
    ongoing: bool


@dataclass(frozen=True, slots=True)
class MarketParam:
    """Per-product price-curve parameters (from MARKET_PARAMS)."""

    base: int
    i0: int
    t: int
    below_func: str
    below_target: float
    above_func: str
    above_target: float


MARKET_I0 = 10000
PRICE_FLOOR = 1

DEFAULT_CROPS: Mapping[CropType, CropSpec] = {
    CropType.WHEAT: CropSpec(10, 2, 4, 0, 6, False),
    CropType.CARROT: CropSpec(20, 2, 3, 0, 4, False),
    CropType.TOMATO: CropSpec(50, 8, 8, 1, 4, True),
    CropType.STRAWBERRY: CropSpec(100, 10, 10, 2, 4, True),
    CropType.MELON: CropSpec(80, 10, 12, 0, 6, False),
}

DEFAULT_MARKET_PARAMS: Mapping[ItemType, MarketParam] = {
    ItemType.WHEAT: MarketParam(25, MARKET_I0, 400, "sqrt", 0.80, "log", 0.20),
    ItemType.CARROT: MarketParam(35, MARKET_I0, 450, "log", 0.20, "sqrt", 0.70),
    ItemType.TOMATO: MarketParam(60, MARKET_I0, 200, "linear", 0.40, "sqrt", 0.60),
    ItemType.STRAWBERRY: MarketParam(120, MARKET_I0, 100, "sqrt", 0.70, "linear", 1.60),
    ItemType.MELON: MarketParam(250, MARKET_I0, 300, "log", 0.20, "sq", 3.60),
    ItemType.EGG: MarketParam(50, MARKET_I0, 332, "linear", 0.40, "log", 0.20),
    ItemType.MILK: MarketParam(160, MARKET_I0, 122, "sqrt", 0.60, "linear", 1.60),
    ItemType.WOOL: MarketParam(200, MARKET_I0, 105, "log", 0.20, "sq", 3.20),
    ItemType.FERTILIZER: MarketParam(100, MARKET_I0, 200, "linear", 0.40, "linear", 0.40),
}

ANIMAL_COST: Mapping[AnimalType, int] = {
    AnimalType.GOOSE: 300,
    AnimalType.COW: 400,
    AnimalType.SHEEP: 500,
}

# Tradable products on the market (excludes animals and seeds).
PRODUCT_ITEMS: tuple[ItemType, ...] = (
    ItemType.WHEAT,
    ItemType.CARROT,
    ItemType.TOMATO,
    ItemType.STRAWBERRY,
    ItemType.MELON,
    ItemType.EGG,
    ItemType.MILK,
    ItemType.WOOL,
    ItemType.FERTILIZER,
)

# Only WHEAT and FERTILIZER can be bought back from the market.
BUYABLE_PRODUCTS: tuple[ItemType, ...] = (ItemType.WHEAT, ItemType.FERTILIZER)

SHOPS: Mapping[ShopType, tuple[ItemType, ...]] = {
    ShopType.BAKERY: (ItemType.EGG, ItemType.WHEAT),
    ShopType.PIZZA_SHOP: (ItemType.MILK, ItemType.TOMATO, ItemType.WHEAT),
    ShopType.BRUNCH_SPOT: (ItemType.EGG, ItemType.WHEAT, ItemType.STRAWBERRY),
    ShopType.YARN_STORE: (ItemType.WOOL,),
    ShopType.ICE_CREAM_SHOP: (ItemType.STRAWBERRY, ItemType.MILK, ItemType.WHEAT),
    ShopType.PET_CAFE: (ItemType.CARROT,),
    ShopType.SMOOTHIE_SHOP: (ItemType.STRAWBERRY, ItemType.MILK),
    ShopType.FARMERS_MARKET: (
        ItemType.WHEAT,
        ItemType.CARROT,
        ItemType.TOMATO,
        ItemType.STRAWBERRY,
    ),
}

# Town-center demand schedule: (day_threshold, multiplier), highest first.
TOWN_CENTER_SCHEDULE: tuple[tuple[int, int], ...] = ((20, 4), (10, 2), (0, 1))


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Tunable game configuration; defaults match the official environment."""

    turns_per_day: int = 24
    board_size: int = 10
    shed_capacity: int = 100
    starting_money: int = 3000
    max_market_orders_per_turn: int = 10
    farm_hand_cost_mult: int = 1
    town_shop_sell_interval: int = 4
    town_center_sell_interval: int = 12
    town_shop_unlock_interval: int = 3
    weed_spawn_chance: float = 0.005
    price_floor: int = PRICE_FLOOR
    crops: Mapping[CropType, CropSpec] = field(default_factory=lambda: DEFAULT_CROPS)
    market_params: Mapping[ItemType, MarketParam] = field(
        default_factory=lambda: DEFAULT_MARKET_PARAMS
    )
