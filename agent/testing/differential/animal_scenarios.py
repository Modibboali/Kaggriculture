"""Differential scenarios for animal placement and lifecycle."""

from __future__ import annotations

from ._farm_turns import (
    BUILD_COOP,
    BUILD_PASTURE,
    BUY_GOOSE,
    HARVEST,
    MOVE_WEST,
    PICKUP_GOOSE,
    PLACE_GOOSE,
    append_days,
    buy_wheat_product,
    feed_care_days,
    pad_day0,
    pickup_wheat,
)
from .scenarios import Scenario

# Setup: build a coop at the farmer's tile and place a goose there on day 0.
GOOSE_SETUP = (BUILD_COOP, BUY_GOOSE, PICKUP_GOOSE, PLACE_GOOSE)

# Introduce + place an animal (valid placement + resulting animal state).
SCENARIO_ANIMAL_PLACE = Scenario.single_player(
    "animal_place",
    GOOSE_SETUP,
    "Build a coop, buy and pick up a goose, and place it.",
)

# PLACE on an empty, non-shed tile with no structure: silent no-op.
SCENARIO_ANIMAL_INVALID_PLACE = Scenario.single_player(
    "animal_invalid_place",
    (BUY_GOOSE, PICKUP_GOOSE, MOVE_WEST, PLACE_GOOSE),
    "Place a goose on an empty tile away from the shed: no-op.",
)

# PLACE on the wrong structure kind (goose on a pasture): no-op.
SCENARIO_ANIMAL_WRONG_STRUCTURE = Scenario.single_player(
    "animal_wrong_structure",
    (BUY_GOOSE, PICKUP_GOOSE, MOVE_WEST, BUILD_PASTURE, PLACE_GOOSE),
    "Place a goose on a pasture: no-op.",
)

# Capacity: a second animal cannot be placed on an occupied structure.
SCENARIO_ANIMAL_CAPACITY = Scenario.single_player(
    "animal_capacity",
    (*GOOSE_SETUP, BUY_GOOSE, PICKUP_GOOSE, PLACE_GOOSE),
    "Try to place a second goose on the occupied coop: no-op.",
)

# Multiple animals on separate structures.
SCENARIO_ANIMAL_MULTIPLE = Scenario.single_player(
    "animal_multiple",
    (*GOOSE_SETUP, BUY_GOOSE, PICKUP_GOOSE, MOVE_WEST, BUILD_COOP, PLACE_GOOSE),
    "Two geese on two coops.",
)

# Animal across a day: fed/cared daily -> fertilizer available, flags reset.
SCENARIO_ANIMAL_DAY_PROGRESSION = Scenario.single_player(
    "animal_day_progression",
    feed_care_days((*GOOSE_SETUP, buy_wheat_product(3), pickup_wheat(3)), 1),
    "Feed and care the goose on day 0 and day 1, then observe the day state.",
)

# Animal production: goose produces eggs from day 4 (care bonus applied).
SCENARIO_ANIMAL_PRODUCTION = Scenario.single_player(
    "animal_production",
    feed_care_days((*GOOSE_SETUP, buy_wheat_product(4), pickup_wheat(4)), 3) + (HARVEST,),
    "Feed/care a goose to day 4, then harvest its eggs.",
)

# Animal lifecycle: an unfed animal escapes and leaves an empty structure.
SCENARIO_ANIMAL_ESCAPE = Scenario.single_player(
    "animal_escape",
    pad_day0(GOOSE_SETUP) + append_days((), 1, water=False),
    "Leave the goose unfed for two days: it escapes, leaving an empty coop.",
)

ANIMAL_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_ANIMAL_PLACE,
    SCENARIO_ANIMAL_INVALID_PLACE,
    SCENARIO_ANIMAL_WRONG_STRUCTURE,
    SCENARIO_ANIMAL_CAPACITY,
    SCENARIO_ANIMAL_MULTIPLE,
    SCENARIO_ANIMAL_DAY_PROGRESSION,
    SCENARIO_ANIMAL_PRODUCTION,
    SCENARIO_ANIMAL_ESCAPE,
)
