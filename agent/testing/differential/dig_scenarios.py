"""Differential scenarios for DIG."""

from __future__ import annotations

from ._farm_turns import (
    BUY_WHEAT,
    DIG,
    MOVE_EAST,
    PASS_TURN,
    PLANT_WHEAT,
    pad_day0,
)
from .scenarios import Scenario

# Dig a plant (valid target): the plant is removed with no yield.
SCENARIO_DIG_VALID = Scenario.single_player(
    "dig_valid",
    (BUY_WHEAT, PLANT_WHEAT, DIG),
    "Dig a planted tile: the plant is removed with no yield.",
)

# Dig an empty tile: no-op.
SCENARIO_DIG_EMPTY = Scenario.single_player(
    "dig_empty",
    (DIG,),
    "Dig an empty tile: silent no-op.",
)

# Dig a LOCKED tile (move into NE then dig): no-op.
SCENARIO_DIG_LOCKED = Scenario.single_player(
    "dig_locked",
    (MOVE_EAST, DIG),
    "Dig a locked tile: silent no-op (land stays locked).",
)

# Repeated DIG: first removes the plant, the second finds an empty tile.
SCENARIO_DIG_REPEATED = Scenario.single_player(
    "dig_repeated",
    (BUY_WHEAT, PLANT_WHEAT, DIG, DIG),
    "Dig twice: the first clears the plant, the second is a no-op.",
)

# DIG across a day boundary: the cleared tile stays empty.
SCENARIO_DIG_DAY_BOUNDARY = Scenario.single_player(
    "dig_day_boundary",
    pad_day0((BUY_WHEAT, PLANT_WHEAT, DIG)),
    "Dig a plant then cross into day 1.",
)

# DIG followed by PLANT on the cleared tile.
SCENARIO_DIG_THEN_PLANT = Scenario.single_player(
    "dig_then_plant",
    (BUY_WHEAT, PLANT_WHEAT, DIG, BUY_WHEAT, PLANT_WHEAT),
    "Dig a plant, then replant on the now-empty tile.",
)

# DIG a weed: plant that dies unwatered becomes a weed at the end of day 0.
SCENARIO_DIG_WEED = Scenario.single_player(
    "dig_weed",
    pad_day0((BUY_WHEAT, PLANT_WHEAT)) + (DIG,),
    "Let a plant die into a weed, then dig it on day 1.",
)

DIG_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_DIG_VALID,
    SCENARIO_DIG_EMPTY,
    SCENARIO_DIG_LOCKED,
    SCENARIO_DIG_REPEATED,
    SCENARIO_DIG_DAY_BOUNDARY,
    SCENARIO_DIG_THEN_PLANT,
    SCENARIO_DIG_WEED,
)
