"""Differential scenarios for FERTILIZE.

Each scenario drives the same typed action list through the official Kaggle
environment and the simulator; the canonical states must match turn-for-turn.
"""

from __future__ import annotations

from ._farm_turns import (
    BUY_FERT,
    BUY_FERT2,
    BUY_WHEAT,
    FERTILIZE,
    HARVEST,
    PICKUP_FERT,
    PICKUP_FERT2,
    PLANT_WHEAT,
    WATER,
    append_days,
    pad_day0,
)
from .scenarios import Scenario

# Fertilize on the planting day (immature wheat): valid, no age requirement.
SCENARIO_FERTILIZE_VALID = Scenario.single_player(
    "fertilize_valid",
    (BUY_FERT, PICKUP_FERT, BUY_WHEAT, PLANT_WHEAT, FERTILIZE),
    "Buy/pickup fertilizer, plant, then fertilize on the planting day.",
)

SCENARIO_FERTILIZE_BEFORE_WATER = Scenario.single_player(
    "fertilize_before_water",
    (BUY_FERT, PICKUP_FERT, BUY_WHEAT, PLANT_WHEAT, FERTILIZE, WATER),
    "Fertilize then water on day 0.",
)

SCENARIO_FERTILIZE_AFTER_WATER = Scenario.single_player(
    "fertilize_after_water",
    (BUY_FERT, PICKUP_FERT, BUY_WHEAT, PLANT_WHEAT, WATER, FERTILIZE),
    "Water then fertilize on day 0.",
)

SCENARIO_FERTILIZE_IMMATURE = Scenario.single_player(
    "fertilize_immature",
    (BUY_FERT, PICKUP_FERT, BUY_WHEAT, PLANT_WHEAT, FERTILIZE, WATER),
    "Fertilize an immature crop on the planting day.",
)

SCENARIO_FERTILIZE_MATURE = Scenario.single_player(
    "fertilize_mature",
    pad_day0((BUY_FERT, PICKUP_FERT, BUY_WHEAT, PLANT_WHEAT, WATER))
    + append_days((), 1)
    + (PICKUP_FERT, FERTILIZE),
    "Reach day 2 (re-pickup fertilizer after the shed drop), then fertilize a mature crop.",
)

SCENARIO_FERTILIZE_REPEATED = Scenario.single_player(
    "fertilize_repeated",
    (BUY_FERT2, PICKUP_FERT2, BUY_WHEAT, PLANT_WHEAT, FERTILIZE, FERTILIZE),
    "Fertilize twice in a row with two units of fertilizer.",
)

SCENARIO_FERTILIZE_DAY_ROLLOVER = Scenario.single_player(
    "fertilize_day_rollover",
    pad_day0((BUY_FERT, PICKUP_FERT, BUY_WHEAT, PLANT_WHEAT, FERTILIZE, WATER)),
    "Fertilize on day 0 and cross into day 1: the window must persist.",
)

SCENARIO_FERTILIZE_HARVEST = Scenario.single_player(
    "fertilize_harvest",
    append_days(
        pad_day0((BUY_FERT, PICKUP_FERT, BUY_WHEAT, PLANT_WHEAT, WATER, FERTILIZE)),
        1,
    )
    + (WATER, HARVEST),
    "Fertilized wheat watered on day 2: window bonus is +2, harvest reflects it.",
)

SCENARIO_FERTILIZE_INVALID_TARGET = Scenario.single_player(
    "fertilize_invalid_target",
    (FERTILIZE,),
    "Fertilize an empty tile: silent no-op.",
)

SCENARIO_FERTILIZE_MISSING_RESOURCE = Scenario.single_player(
    "fertilize_missing_resource",
    (BUY_WHEAT, PLANT_WHEAT, FERTILIZE),
    "Fertilize with no fertilizer in the carried inventory: silent no-op.",
)

FERTILIZE_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_FERTILIZE_VALID,
    SCENARIO_FERTILIZE_BEFORE_WATER,
    SCENARIO_FERTILIZE_AFTER_WATER,
    SCENARIO_FERTILIZE_IMMATURE,
    SCENARIO_FERTILIZE_MATURE,
    SCENARIO_FERTILIZE_REPEATED,
    SCENARIO_FERTILIZE_DAY_ROLLOVER,
    SCENARIO_FERTILIZE_HARVEST,
    SCENARIO_FERTILIZE_INVALID_TARGET,
    SCENARIO_FERTILIZE_MISSING_RESOURCE,
)
