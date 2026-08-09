"""Kaggle environment boundary: observation validation and translation.

The official ``kaggle_environments`` package is the authoritative game engine.
This package defines the clean boundary between it and our immutable domain
model: ``KaggleObservationAdapter`` validates a raw observation and translates
it into an ``agent.state.GameState``. Nothing in ``agent.state`` (except the
backward-compatible ``GameState.from_observation`` shim) knows the raw
observation format.
"""

from .kaggle_observation_adapter import KaggleObservationAdapter
from .observation_validation import InvalidObservationError, validate_observation

__all__ = [
    "InvalidObservationError",
    "KaggleObservationAdapter",
    "validate_observation",
]
