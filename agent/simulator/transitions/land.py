"""Land transition: BUY_LAND.

Verified against the official environment (``_do_buy_land`` inside
``_process_market``):

* BUY_LAND is an atomic market order: the quadrant named in the action is
  IGNORED and the next quadrant in ``LAND_ORDER`` (NE, SW, SE) is purchased;
* the cost (1000 / 2000 / 4000) is paid from the player's farm money, and a
  purchase with insufficient money is a silent no-op;
* on success the quadrant is added to ``unlocked_quadrants`` and every LOCKED
  tile inside it becomes empty (unlocked), which makes it traversable and
  buildable.

``buy_land`` is called by the market processor inside the per-order-index
atomic step, so its money effect interleaves correctly with SELL/BUY orders in
the same turn.
"""

from __future__ import annotations

from dataclasses import replace

from ...state import (
    Farm,
    GameState,
    LockedTile,
    Quadrant,
    EMPTY_TILE,
)
from ..game_config import LAND_ORDER, LAND_PRICES, GameConfig
from . import with_player


def quadrant_of(x: int, y: int, half: int) -> Quadrant:
    """The quadrant a board coordinate lies in (the env's ``_quadrant_of``)."""
    ns = "N" if y < half else "S"
    ew = "W" if x < half else "E"
    return Quadrant[ns + ew]


class LandTransition:
    """Buys the next quadrant of land for a player."""

    def __init__(self, config: GameConfig) -> None:
        self._config = config

    def buy_land(self, state: GameState, player: int, money: int) -> tuple[GameState, int]:
        """Atomically purchase the next land in order; returns ``(state, money)``.

        ``money`` is the player's current (possibly working-copy) farm money;
        it is decremented only on a successful purchase.
        """
        player_state = state.players[player]
        farm = player_state.farm
        n_extra = len(farm.unlocked_quadrants) - 1  # NW is always present
        if n_extra >= len(LAND_ORDER):
            return state, money
        cost = LAND_PRICES[n_extra]
        if money < cost:
            return state, money
        quadrant = LAND_ORDER[n_extra]
        new_farm = self._unlock_quadrant(farm, quadrant)
        new_player = replace(player_state, farm=new_farm)
        return with_player(state, player, new_player), money - cost

    @staticmethod
    def _unlock_quadrant(farm: Farm, quadrant: Quadrant) -> Farm:
        """Return a farm with ``quadrant`` unlocked: LOCKED tiles become empty
        and the quadrant joins ``unlocked_quadrants`` (unconditionally, exactly
        as the environment does)."""
        half = farm.board_size // 2
        new_rows = []
        for y, row in enumerate(farm.tiles):
            new_row = list(row)
            for x, tile in enumerate(row):
                if isinstance(tile, LockedTile) and quadrant_of(x, y, half) == quadrant:
                    new_row[x] = EMPTY_TILE
            new_rows.append(tuple(new_row))
        new_unlocked = frozenset((*farm.unlocked_quadrants, quadrant))
        return Farm(
            money=farm.money,
            tiles=tuple(new_rows),
            farmer=farm.farmer,
            workers=farm.workers,
            unlocked_quadrants=new_unlocked,
            hires_today=farm.hires_today,
        )
