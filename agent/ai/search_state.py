"""Compact immutable planning state.

A :class:`SearchState` is a thin, hashable wrapper around the simulator's
:class:`~agent.state.game_state.GameState`. It exists so the search layer:

* has a single, stable value type to carry through MCTS (independent of the
  raw Kaggle observation format),
* can compute a fast, cached, deterministic state key for transposition use,
* never mutates — every transition returns a new ``SearchState``.

It deliberately does **not** duplicate the game state: ``game`` holds the full
immutable domain model, which already contains every field the simulator
needs to transition and evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..state import GameState, Market, PlayerState, Town


@dataclass(frozen=True, slots=True)
class SearchState:
    """An immutable planning snapshot wrapping a ``GameState``."""

    game: GameState
    _key: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_key", None)

    # -- convenience delegation ---------------------------------------------
    @property
    def day(self) -> int:
        return self.game.day

    @property
    def hour(self) -> int:
        return self.game.hour

    @property
    def step(self) -> int:
        return self.game.step

    @property
    def current_player(self) -> int:
        return self.game.current_player

    @property
    def players(self) -> tuple[PlayerState, PlayerState]:
        return self.game.players

    @property
    def market(self) -> Market:
        return self.game.market

    @property
    def town(self) -> Town:
        return self.game.town

    # -- identity -----------------------------------------------------------
    def state_key(self) -> int:
        """A deterministic, value-based key (cached after the first call).

        Equal ``SearchState``s always produce the same key; materially
        different states produce (with overwhelming probability) different
        keys. The key is independent of Python object identity.
        """
        key = self._key
        if key is None:
            key = hash(self.game)
            object.__setattr__(self, "_key", key)
        return key

    def __hash__(self) -> int:
        return self.state_key()


def state_key(state: SearchState) -> int:
    """The deterministic search-state key (module-level convenience)."""
    return state.state_key()
