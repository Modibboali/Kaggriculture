"""Worker transition: HIRE.

Verified against the official environment (``_do_hire`` inside
``_process_market``):

* HIRE is an atomic market order: the n-th hire of the current day costs
  ``farmHandCostMult * fib(hires_today)`` (fib: 1, 1, 2, 3, 5, ...), paid from
  the player's farm money; an unaffordable hire is a silent no-op;
* on success the new hand spawns on the least-occupied shed-access tile (NWSE
  tie-break) and gains an empty carried inventory; ``hires_today`` increments.

``hire`` is called by the market processor inside the per-order-index atomic
step, so its money effect interleaves correctly with SELL/BUY orders in the
same turn.
"""

from __future__ import annotations

from dataclasses import replace

from ...state import Farm, GameState, Inventory, Position, Worker
from ..game_config import GameConfig
from . import shed_access_tiles, with_player


def _fib(n: int) -> int:
    """Indexed so _fib(0)=1, _fib(1)=1, _fib(2)=2, _fib(3)=3, _fib(4)=5..."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


class WorkerTransition:
    """Hires farm hands for a player."""

    def __init__(self, config: GameConfig) -> None:
        self._config = config

    def hire(self, state: GameState, player: int, money: int) -> tuple[GameState, int]:
        """Atomically hire one hand; returns ``(state, money)``.

        ``money`` is the player's current (possibly working-copy) farm money; it
        is decremented only on a successful hire.
        """
        player_state = state.players[player]
        farm = player_state.farm
        cost = self._config.farm_hand_cost_mult * _fib(farm.hires_today)
        if money < cost:
            return state, money

        spawn = self._spawn_hand(farm)
        new_hand = Worker(len(player_state.workers), spawn, Inventory.empty(), False)
        new_workers = (*player_state.workers, new_hand)
        new_farm = Farm(
            money=farm.money,
            tiles=farm.tiles,
            farmer=farm.farmer,
            workers=(*farm.workers, new_hand),
            unlocked_quadrants=farm.unlocked_quadrants,
            hires_today=farm.hires_today + 1,
        )
        new_player = replace(player_state, farm=new_farm, workers=new_workers)
        return with_player(state, player, new_player), money - cost

    def _spawn_hand(self, farm: Farm) -> Position:
        """The first free shed-access tile (min occupancy, NWSE tie-break)."""
        access = shed_access_tiles(farm.board_size)
        occupants: dict[Position, int] = {pos: 0 for pos in access}
        for pos in (farm.farmer.position, *(worker.position for worker in farm.workers)):
            if pos in occupants:
                occupants[pos] += 1
        return min(occupants, key=lambda pos: (occupants[pos], access.index(pos)))
