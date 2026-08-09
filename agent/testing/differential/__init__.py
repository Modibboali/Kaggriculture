"""Differential testing framework.

Compares the official Kaggriculture environment against a (future) lightweight
simulator using the same initial state and action sequence. The pipeline is:

    Kaggle observation -> adapter -> GameState -> canonical state
    simulator output   -> GameState -> canonical state
    canonical states compared path-by-path (StateDiff)

The framework may depend on the Kaggle environment; the simulator itself must
remain environment-independent (only the ``Simulator`` protocol is defined
here).
"""

from .observation_normalizer import CanonicalState, normalize
from .report import render_differential_report, render_summary
from .runner import (
    DifferentialResult,
    DifferentialRunner,
    KaggleRun,
    Simulator,
    SimulatorRun,
    TurnComparison,
    TurnTrace,
    compare,
)
from .scenarios import (
    SCENARIO_BUY_PLANT_WATER,
    SCENARIO_BUY_SEED,
    SCENARIO_MARKET_TRANSACTION,
    SCENARIO_MOVE_FARMER,
    SCENARIO_MULTIPLE_TURNS,
    SCENARIO_PASS,
    SCENARIO_PLANT_CROP,
    SCENARIO_TWO_PLAYER,
    SCENARIO_WATER_CROP,
    SCENARIOS,
    Scenario,
    TurnActions,
    all_scenarios,
)
from .state_diff import MISSING, DiffEntry, StateDiff, diff

__all__ = [
    "CanonicalState",
    "DiffEntry",
    "DifferentialResult",
    "DifferentialRunner",
    "KaggleRun",
    "MISSING",
    "SCENARIO_BUY_PLANT_WATER",
    "SCENARIO_BUY_SEED",
    "SCENARIO_MARKET_TRANSACTION",
    "SCENARIO_MOVE_FARMER",
    "SCENARIO_MULTIPLE_TURNS",
    "SCENARIO_PASS",
    "SCENARIO_PLANT_CROP",
    "SCENARIO_TWO_PLAYER",
    "SCENARIO_WATER_CROP",
    "SCENARIOS",
    "Scenario",
    "Simulator",
    "SimulatorRun",
    "StateDiff",
    "TurnActions",
    "TurnComparison",
    "TurnTrace",
    "all_scenarios",
    "compare",
    "diff",
    "normalize",
    "render_differential_report",
    "render_summary",
]
