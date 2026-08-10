"""Differential scenarios for BUILD_COOP / BUILD_PASTURE."""

from __future__ import annotations

from ._farm_turns import (
    BUILD_COOP,
    BUILD_PASTURE,
    BUY_LAND_NE,
    MOVE_EAST,
    MOVE_WEST,
    PASS_TURN,
    pad_day0,
)
from .scenarios import Scenario

# Build a coop on an empty tile: free, tile becomes a COOP.
SCENARIO_STRUCTURE_COOP = Scenario.single_player(
    "structure_coop",
    (BUILD_COOP,),
    "Build a coop on the farmer's empty tile.",
)

# Build a pasture on an empty tile.
SCENARIO_STRUCTURE_PASTURE = Scenario.single_player(
    "structure_pasture",
    (BUILD_PASTURE,),
    "Build a pasture on the farmer's empty tile.",
)

# Repeated BUILD on an occupied tile: no-op.
SCENARIO_STRUCTURE_REPEATED = Scenario.single_player(
    "structure_repeated",
    (BUILD_COOP, BUILD_PASTURE),
    "Try to build a pasture over an existing coop: no-op.",
)

# BUILD on a locked tile (move into NE without buying it): no-op.
SCENARIO_STRUCTURE_LOCKED = Scenario.single_player(
    "structure_locked",
    (MOVE_EAST, BUILD_COOP),
    "Try to build on a locked tile: no-op.",
)

# BUILD on newly bought land: works after BUY_LAND + movement.
SCENARIO_STRUCTURE_ON_NEW_LAND = Scenario.single_player(
    "structure_on_new_land",
    (BUY_LAND_NE, MOVE_EAST, MOVE_EAST, BUILD_PASTURE),
    "Buy NE, walk onto it, and build a pasture there.",
)

# Structure survives a day rollover.
SCENARIO_STRUCTURE_DAY_ROLLOVER = Scenario.single_player(
    "structure_day_rollover",
    pad_day0((BUILD_COOP,)),
    "Build a coop and cross into day 1.",
)

# BUILD has no resource cost: money is unchanged.
SCENARIO_STRUCTURE_NO_COST = Scenario.single_player(
    "structure_no_cost",
    (BUILD_COOP, BUILD_PASTURE, MOVE_WEST, PASS_TURN),
    "Two builds leave money untouched.",
)

STRUCTURE_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_STRUCTURE_COOP,
    SCENARIO_STRUCTURE_PASTURE,
    SCENARIO_STRUCTURE_REPEATED,
    SCENARIO_STRUCTURE_LOCKED,
    SCENARIO_STRUCTURE_ON_NEW_LAND,
    SCENARIO_STRUCTURE_DAY_ROLLOVER,
    SCENARIO_STRUCTURE_NO_COST,
)
