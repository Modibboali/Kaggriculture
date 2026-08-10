"""Differential scenarios for HIRE and worker behavior."""

from __future__ import annotations

from ...actions import PlantAction, TurnAction
from ...state import CropType, Direction
from ._farm_turns import (
    BUY_LAND_NE,
    BUY_WHEAT,
    HIRE,
    PASS_TURN,
    hand_build_coop,
    hand_move,
    hand_plant,
    pad_day0,
)
from .scenarios import Scenario

# HIRE one hand: cost 1, spawns on the least-occupied shed-access tile.
SCENARIO_WORKER_HIRE = Scenario.single_player(
    "worker_hire",
    (HIRE,),
    "Hire a hand on day 0 (cost 1).",
)

# Insufficient funds: drain money with land, then the hire is a no-op.
SCENARIO_WORKER_INSUFFICIENT = Scenario.single_player(
    "worker_insufficient",
    (BUY_LAND_NE, BUY_LAND_NE, HIRE),
    "Spend all money on land; the following hire fails.",
)

# Worker state + movement + a worker action (plant).
SCENARIO_WORKER_ACTION = Scenario.single_player(
    "worker_action",
    (HIRE, BUY_WHEAT, hand_move(Direction.WEST), hand_plant(CropType.WHEAT)),
    "Hire, then the hand walks west and plants wheat.",
)

# Worker + structure interaction: the hand builds a coop at its spawn tile.
SCENARIO_WORKER_STRUCTURE = Scenario.single_player(
    "worker_structure",
    (HIRE, hand_build_coop()),
    "A hired hand builds a coop on its spawn tile.",
)

# Worker across a day: hands are cleared, hires reset, inventories reset.
SCENARIO_WORKER_DAY = Scenario.single_player(
    "worker_day",
    pad_day0((HIRE,)),
    "Hire a hand on day 0 and cross into day 1.",
)

# Multiple workers: two hires cost 1 + 1.
SCENARIO_WORKER_MULTIPLE = Scenario.single_player(
    "worker_multiple",
    (HIRE, HIRE),
    "Hire two hands on day 0.",
)

# Atomic PLANT validation: farmer + hands over-request a scarce seed; all drop.
PLANT_ALL = TurnAction(
    worker_actions=(
        PlantAction(crop=CropType.WHEAT),
        PlantAction(crop=CropType.WHEAT),
    ),
)
SCENARIO_WORKER_ATOMIC_PLANT = Scenario.single_player(
    "worker_atomic_plant",
    (HIRE, HIRE, BUY_WHEAT, PLANT_ALL),
    "Farmer and two hands all try to plant the single wheat seed: all are blocked.",
)

WORKER_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIO_WORKER_HIRE,
    SCENARIO_WORKER_INSUFFICIENT,
    SCENARIO_WORKER_ACTION,
    SCENARIO_WORKER_STRUCTURE,
    SCENARIO_WORKER_DAY,
    SCENARIO_WORKER_MULTIPLE,
    SCENARIO_WORKER_ATOMIC_PLANT,
)
