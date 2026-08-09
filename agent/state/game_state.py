"""Top-level immutable game snapshot.

This module owns the single object that represents the whole game. All rules,
search, and evaluation live elsewhere; ``GameState`` only stores state and
offers the *stable* API surface (``clone``, ``apply``, ``legal_actions``,
``is_terminal``, ``winner``) that the future planner will implement on top of
it.

Observation translation is deliberately NOT part of this module: the Kaggle
observation -> domain boundary lives in :mod:`agent.environment`. The
``from_observation`` classmethod below is a thin, backward-compatible shim
that delegates to
:class:`agent.environment.kaggle_observation_adapter.KaggleObservationAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .market import Market
from .player import PlayerState
from .town import Town


@dataclass(frozen=True, slots=True)
class GameState:
    """An immutable snapshot of the whole game.

    ``players[current_player]`` is the acting player. ``step`` is the absolute
    turn index reported by the environment (0-based; ``day * turns_per_day +
    hour`` when the episode is configured with 24 turns/day). Every component
    is frozen and hashable, so a ``GameState`` can be used directly as a key in
    a transposition table or cache owned by a search algorithm.
    """

    day: int
    hour: int
    step: int
    market: Market
    town: Town
    players: tuple[PlayerState, PlayerState]
    current_player: int

    @property
    def current_player_state(self) -> PlayerState:
        """The state of the player whose turn it is."""
        return self.players[self.current_player]

    @property
    def opponent_state(self) -> PlayerState:
        """The state of the other player."""
        return self.players[1 - self.current_player]

    def clone(self) -> "GameState":
        """Return an independent copy of this state.

        The whole state graph is immutable, so the copy is just ``self``; the
        method exists so call sites written against a mutable API keep working
        if the model ever gains mutable internals.
        """
        return self

    def apply(self, action: Any) -> "GameState":
        """Advance the state by one ``action``.

        Intentionally unimplemented: game rules belong to a future environment
        / logic layer, not to the domain model.
        """
        raise NotImplementedError(
            "GameState.apply() is intentionally unimplemented: game rules are "
            "out of scope for the domain model."
        )

    def legal_actions(self) -> list[Any]:
        """The actions legal for the current player this turn."""
        raise NotImplementedError(
            "GameState.legal_actions() is intentionally unimplemented: action "
            "legality is game logic, out of scope for the domain model."
        )

    def is_terminal(self) -> bool:
        """Whether the season is over."""
        raise NotImplementedError(
            "GameState.is_terminal() is intentionally unimplemented."
        )

    def winner(self) -> int | None:
        """The winning player id (0 or 1), or ``None`` on a tie."""
        raise NotImplementedError(
            "GameState.winner() is intentionally unimplemented."
        )

    # -- Observation translation ---------------------------------------------
    # The Kaggle observation -> domain translation is owned by the adapter in
    # agent.environment. This method is a thin, backward-compatible alias.

    @classmethod
    def from_observation(cls, observation: dict[str, Any]) -> "GameState":
        """Build a ``GameState`` from a Kaggle observation dict.

        Delegates to :class:`agent.environment.kaggle_observation_adapter.KaggleObservationAdapter`,
        the canonical validation + translation boundary between the official
        Kaggriculture environment and the domain model.
        """
        from ..environment.kaggle_observation_adapter import (
            KaggleObservationAdapter,
        )

        return KaggleObservationAdapter.from_observation(observation)
