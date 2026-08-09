"""Type aliases and key constants for the Kaggriculture observation schema.

These describe the *raw* observation as produced by the official Kaggle
environment (verified by the diagnostics suite in ``agent.diagnostics``). They
are typing and documentation aids only: runtime validation lives in
:mod:`agent.environment.observation_validation` and translation into immutable
domain objects lives in :mod:`agent.environment.kaggle_observation_adapter`.
"""

from __future__ import annotations

from typing import Any

# A raw Kaggriculture observation is an untyped dict from the Kaggle env.
Observation = dict[str, Any]

# Top-level observation keys (observed from the live environment).
TOP_LEVEL_FIELDS = (
    "remainingOverageTime",
    "step",
    "player",
    "farms",
    "private",
    "market",
    "town",
    "day",
    "hour",
)

# Per-player private keys.
PRIVATE_FIELDS = ("shed", "seeds", "inventories")

# Per-farm keys.
FARM_FIELDS = ("money", "tiles", "farmer", "hands", "unlocked_quadrants", "hires_today")

# Tile structure kinds observed from the environment.
TILE_KINDS = ("WEED", "PLANT", "COOP", "PASTURE")

# Plant tile fields.
PLANT_FIELDS = (
    "kind",
    "crop",
    "planted_day",
    "watered_today",
    "consecutive_unwatered",
    "yield_units",
    "max_lifespan_step",
    "fertilized_until_day",
)

# Animal structure tile fields (coop / pasture).
ANIMAL_TILE_FIELDS = (
    "kind",
    "animal",
    "placed_day",
    "yield_units",
    "fed_today",
    "consecutive_unfed",
    "cared_today",
    "fertilizer_available",
    "pending_care_bonus",
)

# Market and town fields.
MARKET_FIELDS = ("inventory", "prices")
TOWN_FIELDS = ("unlocked_shops",)

# Nested structural aliases (documentation only).
PositionData = list[int]  # [x, y]
FarmData = dict[str, Any]
MarketData = dict[str, Any]
TownData = dict[str, Any]
PrivateData = dict[str, Any]
