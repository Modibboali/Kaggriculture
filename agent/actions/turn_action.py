"""Everything a player submits for a single turn."""

from __future__ import annotations

from dataclasses import dataclass

from .action import Action, PASS


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnAction:
    """The full set of actions submitted for one turn.

    ``farmer_action`` is the main farmer's action (defaults to ``PASS``),
    ``worker_actions`` the hired hands' actions (one entry per acting hand, in
    roster order), and ``market_actions`` the ordered market orders. This maps
    onto the environment's submission shape: the farmer list is
    ``[farmer_action, *worker_actions]`` and the market list is
    ``market_actions``.

    Immutable and hashable, so a ``TurnAction`` can be used directly as a key
    in the action cache of an MCTS / AlphaZero search.
    """

    farmer_action: Action = PASS
    worker_actions: tuple[Action, ...] = ()
    market_actions: tuple[Action, ...] = ()

    def all_actions(self) -> tuple[Action, ...]:
        """Every action in this turn: farmer, then hands, then market."""
        return (self.farmer_action, *self.worker_actions, *self.market_actions)

    def num_actions(self) -> int:
        """Total number of actions submitted this turn."""
        return 1 + len(self.worker_actions) + len(self.market_actions)

    def workers_used(self) -> int:
        """Number of hired hands given an action this turn.

        The main farmer is tracked separately in ``farmer_action``, so it is
        not counted here.
        """
        return len(self.worker_actions)
