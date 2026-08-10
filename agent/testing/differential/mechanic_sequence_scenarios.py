"""Combined differential scenarios: end-to-end sequences integrating
structures, animals, workers, FEED/CARE/COLLECT with the farming loop."""

from __future__ import annotations

from ...state import CropType, Direction
from ._farm_turns import (
    BUILD_COOP,
    BUILD_PASTURE,
    BUY_GOOSE,
    BUY_LAND_NE,
    BUY_SHEEP,
    BUY_WHEAT,
    CARE,
    COLLECT_FERT,
    FEED,
    HARVEST,
    HIRE,
    MOVE_EAST,
    MOVE_WEST,
    PASS_TURN,
    PICKUP_GOOSE,
    PICKUP_SHEEP,
    PICKUP_WHEAT1,
    PLACE_GOOSE,
    PLACE_SHEEP,
    WATER,
    append_days,
    buy_wheat_product,
    feed_care_days,
    hand_move,
    hand_plant,
    pad_day0,
    pickup_wheat,
)
from .scenarios import Scenario

GOOSE_SETUP = (BUILD_COOP, BUY_GOOSE, PICKUP_GOOSE, PLACE_GOOSE)

# HIRE -> hand plants -> WATER -> day rollover -> HARVEST.
SCENARIO_SEQ_WORKER_FARMING = Scenario.single_player(
    "seq_worker_farming",
    pad_day0((HIRE, BUY_WHEAT, hand_move(Direction.WEST), hand_plant(CropType.WHEAT), WATER))
    + append_days((), 1)
    + (WATER, HARVEST),
    "A hired hand plants wheat; the farmer waters it across day rollovers and harvests.",
)

# BUILD COOP -> PLACE goose -> FEED -> CARE -> production -> COLLECT -> HARVEST.
SCENARIO_SEQ_ANIMAL_LOOP = Scenario.single_player(
    "seq_animal_loop",
    feed_care_days((*GOOSE_SETUP, buy_wheat_product(4), pickup_wheat(4)), 3)
    + (COLLECT_FERT, HARVEST),
    "Full goose loop: feed/care to day 4, collect fertilizer, harvest the eggs.",
)

# BUY_LAND -> BUILD PASTURE -> PLACE sheep -> FEED/CARE -> day rollover.
SCENARIO_SEQ_LAND_PASTURE_SHEEP = Scenario.single_player(
    "seq_land_pasture_sheep",
    (
        BUY_LAND_NE,
        MOVE_EAST,
        MOVE_EAST,
        BUILD_PASTURE,
        MOVE_WEST,
        MOVE_WEST,
        BUY_SHEEP,
        PICKUP_SHEEP,
        MOVE_EAST,
        MOVE_EAST,
        PLACE_SHEEP,
        MOVE_WEST,
        MOVE_WEST,
        buy_wheat_product(2),
        pickup_wheat(2),
        MOVE_EAST,
        MOVE_EAST,
        FEED,
        CARE,
    )
    + ((PASS_TURN,) * 5)  # rest of day 0 (farmer resets to the shed)
    + (PICKUP_WHEAT1, MOVE_EAST, MOVE_EAST, FEED, CARE)  # day 1 feeding
    + ((PASS_TURN,) * 19),
    "Sheep on a pasture built on bought land, fed/cared across a day boundary.",
)

MECHANIC_SEQUENCE_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_SEQ_WORKER_FARMING,
    SCENARIO_SEQ_ANIMAL_LOOP,
    SCENARIO_SEQ_LAND_PASTURE_SHEEP,
)
