"""Semantic state diffing.

A :class:`StateDiff` is the path-level difference between two canonical states.
It identifies added, removed, and changed fields, and classifies each change by
domain (money, inventory, seeds, tile, crop, animal, worker, market, town,
day/hour, ...) so reports can be grouped and debugged quickly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .observation_normalizer import CanonicalState

# Sentinel used for values present on only one side of the comparison.
MISSING: Any = object()


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One path-level difference between two canonical states.

    ``kind`` is ``"added"``, ``"removed"``, or ``"changed"``. ``expected`` is
    the value in the reference (Kaggle) state; ``actual`` is the value in the
    candidate (simulator) state. Missing values are the ``MISSING`` sentinel.
    """

    path: str
    kind: str
    domain: str
    expected: Any
    actual: Any


@dataclass(frozen=True, slots=True)
class StateDiff:
    """The full set of differences between two canonical states."""

    reference: CanonicalState
    candidate: CanonicalState
    entries: tuple[DiffEntry, ...]

    @property
    def matches(self) -> bool:
        """Whether the two states are semantically identical."""
        return not self.entries

    def __bool__(self) -> bool:
        return not self.entries

    @property
    def by_domain(self) -> dict[str, tuple[DiffEntry, ...]]:
        """Differences grouped by domain (money, inventory, seeds, ...)."""
        grouped: dict[str, list[DiffEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.domain, []).append(entry)
        return {domain: tuple(entries) for domain, entries in grouped.items()}


def diff(reference: CanonicalState, candidate: CanonicalState) -> StateDiff:
    """Compute the path-level differences between two canonical states."""
    return StateDiff(
        reference=reference,
        candidate=candidate,
        entries=_diff(reference, candidate, ""),
    )


def _diff(expected: Any, actual: Any, path: str) -> tuple[DiffEntry, ...]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _diff_mappings(expected, actual, path)
    if isinstance(expected, list) and isinstance(actual, list):
        return _diff_lists(expected, actual, path)
    if expected != actual:
        return (DiffEntry(path or "<root>", "changed", classify(path), expected, actual),)
    return ()


def _diff_mappings(
    expected: Mapping[str, Any], actual: Mapping[str, Any], path: str
) -> tuple[DiffEntry, ...]:
    entries: list[DiffEntry] = []
    for key in sorted(set(expected) | set(actual)):
        child = f"{path}.{key}" if path else key
        if key not in expected:
            entries.append(DiffEntry(child, "added", classify(child), MISSING, actual[key]))
        elif key not in actual:
            entries.append(DiffEntry(child, "removed", classify(child), expected[key], MISSING))
        else:
            entries.extend(_diff(expected[key], actual[key], child))
    return tuple(entries)


def _diff_lists(expected: list[Any], actual: list[Any], path: str) -> tuple[DiffEntry, ...]:
    entries: list[DiffEntry] = []
    for index in range(max(len(expected), len(actual))):
        child = f"{path}[{index}]"
        if index >= len(expected):
            entries.append(DiffEntry(child, "added", classify(child), MISSING, actual[index]))
        elif index >= len(actual):
            entries.append(DiffEntry(child, "removed", classify(child), expected[index], MISSING))
        else:
            entries.extend(_diff(expected[index], actual[index], child))
    return tuple(entries)


def classify(path: str) -> str:
    """Classify a diff path into a domain bucket for reporting/grouping."""
    if path.startswith(("day", "hour", "step")):
        return "day/hour/step"
    if "market" in path:
        return "market"
    if "town" in path:
        return "town"
    if path.startswith("private.seeds"):
        return "seeds"
    if path.startswith("private.shed") or ".inventories" in path:
        return "inventory"
    if ".money" in path:
        return "money"
    if ".farmer" in path or ".hands" in path:
        return "worker/position"
    if "unlocked_quadrants" in path:
        return "land"
    if "hires_today" in path:
        return "hiring"
    if ".tiles" in path:
        if ".plant" in path:
            return "crop"
        if ".animal" in path:
            return "animal"
        return "tile"
    return "other"
