"""Differential scenarios for BUY_LAND."""

from __future__ import annotations

from ._farm_turns import (
    BUY_LAND_NE,
    BUY_WHEAT,
    DIG,
    MOVE_EAST,
    PASS_TURN,
    PLANT_WHEAT,
    WATER,
    pad_day0,
)
from .scenarios import Scenario

# Valid purchase: buys NE for 1000 (the submitted quadrant is ignored).
SCENARIO_LAND_VALID = Scenario.single_player(
    "land_valid",
    (BUY_LAND_NE,),
    "Buy the next land (NE) for 1000 coins.",
)

# Insufficient money: three purchases drain 3000; the third (SE, 4000) fails.
SCENARIO_LAND_INSUFFICIENT = Scenario.single_player(
    "land_insufficient",
    (BUY_LAND_NE, BUY_LAND_NE, BUY_LAND_NE),
    "Buy NE, SW, then attempt SE with no money left: no-op.",
)

# All land already bought: a fourth purchase is a no-op.
SCENARIO_LAND_NONE_LEFT = Scenario.single_player(
    "land_none_left",
    (BUY_LAND_NE, BUY_LAND_NE, BUY_LAND_NE, BUY_LAND_NE),
    "Attempt to buy land when all three quadrants are already owned.",
)

# Purchase followed by movement onto the newly unlocked land.
SCENARIO_LAND_THEN_MOVE = Scenario.single_player(
    "land_then_move",
    (BUY_LAND_NE, MOVE_EAST, MOVE_EAST),
    "Buy NE, then walk two tiles east onto the newly unlocked land.",
)

# Purchase followed by DIG on the new land (empty tile: no-op).
SCENARIO_LAND_THEN_DIG = Scenario.single_player(
    "land_then_dig",
    (BUY_LAND_NE, MOVE_EAST, MOVE_EAST, DIG),
    "Buy NE, move onto it, and dig an empty tile there.",
)

# Purchase followed by PLANT on the new land.
SCENARIO_LAND_THEN_PLANT = Scenario.single_player(
    "land_then_plant",
    (BUY_LAND_NE, MOVE_EAST, MOVE_EAST, BUY_WHEAT, PLANT_WHEAT, WATER),
    "Buy NE, move onto it, plant, and water a crop on the new land.",
)

# Purchase across a day boundary: unlocked land persists, weeds may spawn there.
SCENARIO_LAND_DAY_BOUNDARY = Scenario.single_player(
    "land_day_boundary",
    pad_day0((BUY_LAND_NE,)),
    "Buy NE and cross into day 1.",
)

LAND_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_LAND_VALID,
    SCENARIO_LAND_INSUFFICIENT,
    SCENARIO_LAND_NONE_LEFT,
    SCENARIO_LAND_THEN_MOVE,
    SCENARIO_LAND_THEN_DIG,
    SCENARIO_LAND_THEN_PLANT,
    SCENARIO_LAND_DAY_BOUNDARY,
)
