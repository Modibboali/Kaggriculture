"""Immutable 2-D grid positions.

Positions use screen-style coordinates: ``x`` grows east, ``y`` grows south.
This matches the observation's ``tiles[y][x]`` layout, so the tile at
``Position(x, y)`` is ``farm.tiles[y][x]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Direction


@dataclass(frozen=True, slots=True)
class Position:
    """A point on the farm grid."""

    x: int
    y: int

    def neighbors(self) -> tuple["Position", ...]:
        """The four orthogonal neighbours in N/S/E/W order.

        May include out-of-bounds positions; callers should check bounds
        (e.g. via ``Farm.is_in_bounds``) before using them.
        """
        return (
            Position(self.x, self.y - 1),
            Position(self.x, self.y + 1),
            Position(self.x - 1, self.y),
            Position(self.x + 1, self.y),
        )

    def distance(self, other: "Position") -> int:
        """Manhattan distance, matching four-directional movement."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def move(self, direction: Direction) -> "Position":
        """Return the position one tile away in ``direction``.

        The result may be out of bounds; callers should check before use.
        """
        dx, dy = direction.delta
        return Position(self.x + dx, self.y + dy)
