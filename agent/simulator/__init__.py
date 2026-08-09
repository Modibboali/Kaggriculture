"""Kaggriculture simulator: the verified first transition layer.

This package forward-simulates a subset of the official environment's rules
(PASS, movement, BUY_SEED, PLANT, WATER) on the immutable domain model. The
rules are transcribed from the official Kaggriculture source and verified by
differential testing against the real ``kaggle-environments`` engine.
"""

from __future__ import annotations

from .game_config import CropSpec, GameConfig, MarketParam
from .simulator import Simulator
from .transition_engine import TransitionEngine

__all__ = ["CropSpec", "GameConfig", "MarketParam", "Simulator", "TransitionEngine"]
