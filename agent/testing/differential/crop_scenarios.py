"""Focused differential scenarios for HARVEST + crop lifecycle + day rollover.

Each scenario is an explicit per-turn action list built from typed actions so
the same sequence drives both the official environment and the simulator.
Scenarios that cross day boundaries water at the first hour of each day to keep
crops alive and exercise the deterministic end-of-day transition (weed spawns,
shed drop, farmer reset, shop unlocks).
"""

from __future__ import annotations

from ...actions import (
    BuySeedAction,
    HarvestAction,
    MovementAction,
    PlantAction,
    TurnAction,
    WaterAction,
)
from ...state import CropType, Direction
from .scenarios import Scenario

TURNS_PER_DAY = 24

# --- Reusable per-player turns ---------------------------------------------
PASS_TURN = TurnAction()
BUY_WHEAT = TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),))
BUY_CARROT = TurnAction(market_actions=(BuySeedAction(crop=CropType.CARROT, quantity=1),))
BUY_TOMATO = TurnAction(market_actions=(BuySeedAction(crop=CropType.TOMATO, quantity=1),))
PLANT_WHEAT = TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
PLANT_CARROT = TurnAction(farmer_action=PlantAction(crop=CropType.CARROT))
PLANT_TOMATO = TurnAction(farmer_action=PlantAction(crop=CropType.TOMATO))
WATER = TurnAction(farmer_action=WaterAction())
HARVEST = TurnAction(farmer_action=HarvestAction())
MOVE_WEST = TurnAction(farmer_action=MovementAction(direction=Direction.WEST))
MOVE_EAST = TurnAction(farmer_action=MovementAction(direction=Direction.EAST))

_Prelude = tuple[TurnAction, ...]


def _water_every_day(prelude: _Prelude, days: int) -> tuple[TurnAction, ...]:
    """``prelude`` on day 0, then WATER at the first hour of each of the next
    ``days`` days and PASS for the rest of the day.

    The prelude waters the crop on day 0 (planting day counts as unwatered, so
    this keeps it alive across the first day boundary).
    """
    turns: list[TurnAction] = list(prelude)
    turns.extend([PASS_TURN] * (TURNS_PER_DAY - len(prelude)))
    for _ in range(days):
        turns.append(WATER)
        turns.extend([PASS_TURN] * (TURNS_PER_DAY - 1))
    return tuple(turns)


def _append_days(turns: tuple[TurnAction, ...], days: int) -> tuple[TurnAction, ...]:
    """Append ``days`` full days of WATER-at-first-hour / PASS after ``turns``."""
    result: list[TurnAction] = list(turns)
    for _ in range(days):
        result.append(WATER)
        result.extend([PASS_TURN] * (TURNS_PER_DAY - 1))
    return tuple(result)


# --- Phase 1: focused HARVEST scenarios ------------------------------------
SCENARIO_HARVEST_IMMATURE = Scenario.single_player(
    "harvest_immature",
    (BUY_WHEAT, PLANT_WHEAT, HARVEST),
    "Harvest on the planting day: immature crop is a silent no-op.",
)

SCENARIO_HARVEST_EMPTY = Scenario.single_player(
    "harvest_empty",
    (HARVEST,),
    "Harvest an empty tile: silent no-op.",
)

SCENARIO_HARVEST_WRONG_POSITION = Scenario.single_player(
    "harvest_wrong_position",
    (BUY_WHEAT, PLANT_WHEAT, MOVE_WEST, HARVEST),
    "Harvest while standing off the plant: silent no-op.",
)

# Water daily through day 1, then harvest on day 2 WITHOUT watering day 2.
SCENARIO_HARVEST_MATURE = Scenario.single_player(
    "harvest_mature",
    (*_water_every_day((BUY_WHEAT, PLANT_WHEAT, WATER), 1), HARVEST),
    "Harvest a mature crop on day 2 (no water that day): base yield 1.",
)

# Water daily through day 1, water + harvest on day 2 (window bonus applies).
SCENARIO_HARVEST_AFTER_WATERING = Scenario.single_player(
    "harvest_after_watering",
    (*_water_every_day((BUY_WHEAT, PLANT_WHEAT, WATER), 1), WATER, HARVEST),
    "Water then harvest on day 2: watering inside the bonus window adds yield.",
)

SCENARIO_REPEATED_HARVEST = Scenario.single_player(
    "harvest_repeated",
    (*_water_every_day((BUY_WHEAT, PLANT_WHEAT, WATER), 1), WATER, HARVEST, HARVEST),
    "Harvest a one-time crop twice: the second harvest finds an empty tile.",
)

SCENARIO_HARVEST_AFTER_MULTIPLE_DAYS = Scenario.single_player(
    "harvest_after_multiple_days",
    (*_water_every_day((BUY_WHEAT, PLANT_WHEAT, WATER), 3), HARVEST),
    "Water daily for 4 days, harvest on day 4: yield grows across the window.",
)

# --- Phase 6: lifecycle / day-boundary scenarios ---------------------------
SCENARIO_PLANT_PASS = Scenario.single_player(
    "plant_pass",
    (BUY_WHEAT, PLANT_WHEAT, PASS_TURN, PASS_TURN),
    "Plant then pass: the plant persists.",
)

SCENARIO_PLANT_WATER_PASS = Scenario.single_player(
    "plant_water_pass",
    (BUY_WHEAT, PLANT_WHEAT, WATER, PASS_TURN),
    "Plant, water, then pass within day 0.",
)

SCENARIO_PLANT_SEVERAL_PASSES = Scenario.single_player(
    "plant_several_passes",
    (BUY_WHEAT, PLANT_WHEAT, WATER, *((PASS_TURN,) * 5)),
    "Plant, water, then several PASS turns within day 0.",
)

SCENARIO_CROSS_DAY_BOUNDARY = Scenario.single_player(
    "cross_day_boundary",
    (*_water_every_day((BUY_WHEAT, PLANT_WHEAT, WATER), 1), WATER),
    "Cross day 0 -> 1 with a watered wheat; exercises the end-of-day transition.",
)

SCENARIO_REPEATED_WATER_PASS = Scenario.single_player(
    "repeated_water_pass",
    _water_every_day((BUY_WHEAT, PLANT_WHEAT, WATER), 2),
    "Water at the start of each of three days, pass otherwise.",
)

SCENARIO_MATURITY = Scenario.single_player(
    "maturity",
    _water_every_day((BUY_WHEAT, PLANT_WHEAT, WATER), 3),
    "Carry a watered wheat across three day boundaries to maturity.",
)

SCENARIO_CROSS_MULTIPLE_DAYS = Scenario.single_player(
    "cross_multiple_days",
    _water_every_day((BUY_WHEAT, PLANT_WHEAT, WATER), 3),
    "Four days of daily watering: weeds, shed drop, farmer reset, shop unlocks.",
)

SCENARIO_MULTIPLE_CROPS = Scenario.single_player(
    "multiple_crops",
    (
        BUY_WHEAT,
        PLANT_WHEAT,
        WATER,
        *((PASS_TURN,) * 21),
        BUY_TOMATO,
        MOVE_WEST,
        PLANT_TOMATO,
        WATER,
        MOVE_EAST,
        WATER,
        PASS_TURN,
    ),
    "Wheat planted day 0 and tomato planted day 1, both watered, in parallel.",
)

SCENARIO_ONGOING_CROP = Scenario.single_player(
    "ongoing_crop_tomato",
    _append_days(
        (*_water_every_day((BUY_TOMATO, PLANT_TOMATO, WATER), 7), HARVEST), 3
    )
    + (HARVEST,),
    "Tomato: water to day 8, harvest, regrow three days, harvest again.",
)

CROP_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_HARVEST_IMMATURE,
    SCENARIO_HARVEST_EMPTY,
    SCENARIO_HARVEST_WRONG_POSITION,
    SCENARIO_HARVEST_MATURE,
    SCENARIO_HARVEST_AFTER_WATERING,
    SCENARIO_REPEATED_HARVEST,
    SCENARIO_HARVEST_AFTER_MULTIPLE_DAYS,
    SCENARIO_PLANT_PASS,
    SCENARIO_PLANT_WATER_PASS,
    SCENARIO_PLANT_SEVERAL_PASSES,
    SCENARIO_CROSS_DAY_BOUNDARY,
    SCENARIO_REPEATED_WATER_PASS,
    SCENARIO_MATURITY,
    SCENARIO_CROSS_MULTIPLE_DAYS,
    SCENARIO_MULTIPLE_CROPS,
    SCENARIO_ONGOING_CROP,
)
