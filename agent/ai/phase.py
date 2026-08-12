"""Deterministic game-phase model for phase-aware action prioritisation.

Phases are derived from the *remaining actionable time* and the actual crop /
animal cycle lengths in the game configuration — not from the raw day number
and not from any learned model.

    TERMINAL          h_steps <= 0                     (no actions)
    CASH_CONVERSION   h_days <  shortest crop->cash    (only convert assets)
    PRODUCTION        shortest <= h_days < longest     (short crop cycles fit)
    DEVELOPMENT       h_days >= longest crop->cash     (long investments fit)

``shortest/longest_crop_cash_days`` use each crop's ``first_yield_day`` plus
~2 days of harvest/sell overhead, so the boundaries follow the simulator's
own growth tables.
"""

from __future__ import annotations

from enum import Enum

from ..simulator import GameConfig
from .search_state import SearchState


class GamePhase(str, Enum):
    """The strategic phase of the episode for the acting player."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    CASH_CONVERSION = "cash_conversion"
    TERMINAL = "terminal"

    @property
    def label(self) -> str:
        return str(self.value)


def shortest_crop_cash_days(config: GameConfig) -> int:
    """Fewest days for a seed to become cash: grow to first yield + ~2 overhead."""
    return min(spec.first_yield_day for spec in config.crops.values()) + 2


def longest_crop_cash_days(config: GameConfig) -> int:
    """Most days for the longest crop to become cash (first_yield + ~2)."""
    return max(spec.first_yield_day for spec in config.crops.values()) + 2


def phase_for(h_steps: int, h_days: int, config: GameConfig) -> GamePhase:
    """The deterministic phase for a remaining ``h_steps`` / ``h_days``."""
    if h_steps <= 0:
        return GamePhase.TERMINAL
    if h_days < shortest_crop_cash_days(config):
        return GamePhase.CASH_CONVERSION
    if h_days < longest_crop_cash_days(config):
        return GamePhase.PRODUCTION
    return GamePhase.DEVELOPMENT


def phase_of(state: SearchState, config: GameConfig) -> GamePhase:
    """The phase of a search state (its remaining horizon)."""
    from .evaluation import horizon_days, horizon_remaining

    h_steps = horizon_remaining(state, config.episode_steps)
    h_days = horizon_days(state, config)
    return phase_for(h_steps, h_days, config)
