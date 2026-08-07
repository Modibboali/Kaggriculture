"""A single player's farm board (public per-player state)."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Quadrant
from .position import Position
from .tile import Tile
from .worker import Worker


@dataclass(frozen=True, slots=True)
class Farm:
    """The immutable board for one player.

    ``tiles`` is stored row-major and indexed ``tiles[y][x]`` to match the
    observation layout. ``farmer`` is the main farmer; ``workers`` are the
    hired hands for the current day. All mutating operations return a new
    ``Farm``, so a snapshot of a game state is just a reference.
    """

    money: int
    tiles: tuple[tuple[Tile, ...], ...]
    farmer: Worker
    workers: tuple[Worker, ...]
    unlocked_quadrants: frozenset[Quadrant]
    hires_today: int

    @property
    def board_size(self) -> int:
        """Width/height of the (square) board."""
        return len(self.tiles[0]) if self.tiles else 0

    def is_in_bounds(self, position: Position) -> bool:
        """Whether ``position`` lies within the board."""
        if not self.tiles:
            return False
        size = self.board_size
        return 0 <= position.x < size and 0 <= position.y < len(self.tiles)

    def tile_at(self, position: Position) -> Tile | None:
        """The tile at ``position``, or ``None`` when out of bounds.

        ``None`` unambiguously means "out of bounds" because tiles are never
        ``None`` in this model (empty cells are ``EmptyTile``). Returning
        ``None`` keeps neighbour / move legality checks allocation-free for
        search.
        """
        if not self.is_in_bounds(position):
            return None
        return self.tiles[position.y][position.x]

    def replace_tile(self, position: Position, tile: Tile) -> "Farm":
        """Return a copy of this farm with the tile at ``position`` replaced."""
        if not self.is_in_bounds(position):
            raise ValueError(f"position out of bounds: {position}")
        row = self.tiles[position.y]
        new_row = row[: position.x] + (tile,) + row[position.x + 1 :]
        new_tiles = (
            self.tiles[: position.y] + (new_row,) + self.tiles[position.y + 1 :]
        )
        return Farm(
            money=self.money,
            tiles=new_tiles,
            farmer=self.farmer,
            workers=self.workers,
            unlocked_quadrants=self.unlocked_quadrants,
            hires_today=self.hires_today,
        )

    def move_worker(self, worker_id: int, position: Position) -> "Farm":
        """Return a copy with the named worker standing at ``position``."""
        if not self.is_in_bounds(position):
            raise ValueError(f"target position out of bounds: {position}")

        moved = False

        def relocate(worker: Worker) -> Worker:
            nonlocal moved
            if worker.id != worker_id:
                return worker
            moved = True
            return worker.moved_to(position)

        farmer = relocate(self.farmer)
        workers = tuple(relocate(worker) for worker in self.workers)
        if not moved:
            raise ValueError(f"no worker with id {worker_id}")
        return Farm(
            money=self.money,
            tiles=self.tiles,
            farmer=farmer,
            workers=workers,
            unlocked_quadrants=self.unlocked_quadrants,
            hires_today=self.hires_today,
        )
