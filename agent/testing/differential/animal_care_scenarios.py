"""Differential scenarios for FEED / CARE / COLLECT_FERTILIZER."""

from __future__ import annotations

from ._farm_turns import (
    BUILD_COOP,
    BUY_GOOSE,
    CARE,
    COLLECT_FERT,
    FEED,
    PICKUP_GOOSE,
    PLACE_GOOSE,
    buy_wheat_product,
    feed_care_days,
    pickup_wheat,
)
from .scenarios import Scenario

# Setup: build a coop at the farmer's tile and place a goose there on day 0.
GOOSE_SETUP = (BUILD_COOP, BUY_GOOSE, PICKUP_GOOSE, PLACE_GOOSE)

# --- FEED ----------------------------------------------------------------


SCENARIO_FEED_VALID = Scenario.single_player(
    "feed_valid",
    (*GOOSE_SETUP, buy_wheat_product(1), pickup_wheat(1), FEED),
    "Feed the goose with a carried wheat: fed_today becomes True.",
)

SCENARIO_FEED_MISSING_FOOD = Scenario.single_player(
    "feed_missing_food",
    (*GOOSE_SETUP, FEED),
    "Feed with no carried wheat: silent no-op.",
)

SCENARIO_FEED_INVALID_TARGET = Scenario.single_player(
    "feed_invalid_target",
    (FEED,),
    "Feed on an empty tile: silent no-op.",
)

SCENARIO_FEED_REPEATED = Scenario.single_player(
    "feed_repeated",
    (*GOOSE_SETUP, buy_wheat_product(2), pickup_wheat(2), FEED, FEED),
    "Feed twice: the second feed is a no-op because the animal is already fed.",
)

SCENARIO_FEED_DAY_ROLLOVER = Scenario.single_player(
    "feed_day_rollover",
    feed_care_days((*GOOSE_SETUP, buy_wheat_product(2), pickup_wheat(2)), 0),
    "Feed on day 0 and cross into day 1: fed_today resets at the boundary.",
)


# --- CARE ----------------------------------------------------------------


SCENARIO_CARE_VALID = Scenario.single_player(
    "care_valid",
    (*GOOSE_SETUP, CARE),
    "Care for the goose: cared_today becomes True.",
)

SCENARIO_CARE_INVALID_TARGET = Scenario.single_player(
    "care_invalid_target",
    (CARE,),
    "Care on an empty tile: silent no-op.",
)

SCENARIO_CARE_REPEATED = Scenario.single_player(
    "care_repeated",
    (*GOOSE_SETUP, CARE, CARE),
    "Care twice: the second care is a no-op.",
)

SCENARIO_CARE_DAY_ROLLOVER = Scenario.single_player(
    "care_day_rollover",
    feed_care_days((*GOOSE_SETUP, buy_wheat_product(1), pickup_wheat(1)), 0),
    "Care on day 0 and cross into day 1: cared_today resets.",
)


# --- COLLECT_FERTILIZER --------------------------------------------------


SCENARIO_COLLECT_BEFORE_PRODUCTION = Scenario.single_player(
    "collect_before_production",
    (*GOOSE_SETUP, COLLECT_FERT),
    "Collect fertilizer on day 0 before it is available: silent no-op.",
)

SCENARIO_COLLECT_AFTER_DAY = Scenario.single_player(
    "collect_after_day",
    feed_care_days((*GOOSE_SETUP, buy_wheat_product(2), pickup_wheat(2)), 0)
    + (COLLECT_FERT,),
    "Collect the fertilizer made available at the end of day 0.",
)

SCENARIO_COLLECT_REPEATED = Scenario.single_player(
    "collect_repeated",
    feed_care_days((*GOOSE_SETUP, buy_wheat_product(2), pickup_wheat(2)), 0)
    + (COLLECT_FERT, COLLECT_FERT),
    "Collecting twice: the second collection finds the flag cleared.",
)

FEED_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_FEED_VALID,
    SCENARIO_FEED_MISSING_FOOD,
    SCENARIO_FEED_INVALID_TARGET,
    SCENARIO_FEED_REPEATED,
    SCENARIO_FEED_DAY_ROLLOVER,
)

CARE_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_CARE_VALID,
    SCENARIO_CARE_INVALID_TARGET,
    SCENARIO_CARE_REPEATED,
    SCENARIO_CARE_DAY_ROLLOVER,
)

COLLECT_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_COLLECT_BEFORE_PRODUCTION,
    SCENARIO_COLLECT_AFTER_DAY,
    SCENARIO_COLLECT_REPEATED,
)
