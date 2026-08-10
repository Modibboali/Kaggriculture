"""Kaggriculture simulator: the verified transition layers.

This package forward-simulates a subset of the official environment's rules on
the immutable domain model: PASS, movement, BUY_SEED, PLANT, WATER, HARVEST,
FERTILIZE, DIG, BUY_LAND, PICKUP/DROP, crop time progression (per-step decay +
daily refresh), and the deterministic end-of-day transition. The rules are
transcribed from the official Kaggriculture source and verified by
differential testing against the real ``kaggle-environments`` engine.
"""

from __future__ import annotations

from .game_config import AnimalSpec, CropSpec, GameConfig, MarketParam
from .simulator import Simulator
from .transition_engine import TransitionEngine
from .transitions.animals import AnimalTransition
from .transitions.crop_lifecycle import CropLifecycleTransition
from .transitions.dig import DigTransition
from .transitions.end_of_day import EndOfDayProcessor
from .transitions.fertilize import FertilizeTransition
from .transitions.harvest import HarvestTransition
from .transitions.items import PickupDropTransition
from .transitions.land import LandTransition
from .transitions.structure import StructureTransition
from .transitions.workers import WorkerTransition

__all__ = [
    "AnimalSpec",
    "AnimalTransition",
    "CropLifecycleTransition",
    "CropSpec",
    "DigTransition",
    "EndOfDayProcessor",
    "FertilizeTransition",
    "GameConfig",
    "HarvestTransition",
    "LandTransition",
    "MarketParam",
    "PickupDropTransition",
    "Simulator",
    "StructureTransition",
    "TransitionEngine",
    "WorkerTransition",
]
