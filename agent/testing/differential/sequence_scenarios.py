"""Combined differential scenarios: realistic sequences combining FERTILIZE,
DIG, and BUY_LAND with the already-verified farming loop."""

from __future__ import annotations

from ._farm_turns import (
    BUY_FERT,
    BUY_LAND_NE,
    BUY_WHEAT,
    DIG,
    DROP,
    FERTILIZE,
    HARVEST,
    MOVE_EAST,
    PASS_TURN,
    PICKUP_FERT,
    PLANT_WHEAT,
    WATER,
    append_days,
    pad_day0,
)
from .scenarios import Scenario

# BUY_SEED -> PLANT -> WATER -> FERTILIZE -> PASS -> HARVEST -> DROP
SCENARIO_SEQ_FERTILIZE_HARVEST = Scenario.single_player(
    "seq_fertilize_harvest",
    append_days(
        pad_day0((BUY_FERT, PICKUP_FERT, BUY_WHEAT, PLANT_WHEAT, WATER, FERTILIZE)), 1
    )
    + (WATER, HARVEST, DROP),
    "Fertilized wheat loop: water daily, harvest the boosted yield on day 2, drop it to the shed.",
)

# BUY_LAND -> move -> PLANT -> WATER -> day rollover -> re-walk -> WATER -> HARVEST
SCENARIO_SEQ_LAND_PLANT_HARVEST = Scenario.single_player(
    "seq_land_plant_harvest",
    pad_day0((BUY_LAND_NE, MOVE_EAST, MOVE_EAST, BUY_WHEAT, PLANT_WHEAT, WATER))
    + (MOVE_EAST, MOVE_EAST, WATER)  # day 1: re-walk to the crop on the new land
    + ((PASS_TURN,) * 21)
    + (MOVE_EAST, MOVE_EAST, WATER, HARVEST),  # day 2
    "Plant on newly bought land, keep watering it across day rollovers, then harvest.",
)

# DIG (weed) -> BUY_SEED -> PLANT -> WATER -> day rollover -> WATER -> HARVEST
SCENARIO_SEQ_DIG_PLANT_HARVEST = Scenario.single_player(
    "seq_dig_plant_harvest",
    pad_day0((BUY_WHEAT, PLANT_WHEAT))  # plant dies into a weed at end of day 0
    + (DIG, BUY_WHEAT, PLANT_WHEAT, WATER)  # day 1: dig the weed, replant, water
    + ((PASS_TURN,) * 20)
    + (WATER, HARVEST),  # day 2
    "Dig a weed, replant, water across a day rollover, then harvest.",
)

SEQUENCE_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_SEQ_FERTILIZE_HARVEST,
    SCENARIO_SEQ_LAND_PLANT_HARVEST,
    SCENARIO_SEQ_DIG_PLANT_HARVEST,
)
