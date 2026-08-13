"""Lightweight metrics accumulator for MuZero training / evaluation."""

from __future__ import annotations

import json
import os
import statistics


class Metrics:
    """Tracks scalar metrics and writes a JSON log."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._history: list[dict[str, float]] = []

    def record(self, step: int, values: dict[str, float]) -> None:
        entry: dict[str, float] = {"step": float(step)}
        entry.update(values)
        self._history.append(entry)
        if self._path is not None:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._history, f, indent=2)

    def summary(self, key: str) -> dict[str, float]:
        vals = [float(e[key]) for e in self._history if key in e]
        if not vals:
            return {"count": 0.0, "mean": 0.0, "last": 0.0}
        return {
            "count": float(len(vals)),
            "mean": statistics.fmean(vals),
            "last": vals[-1],
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    @property
    def history(self) -> list[dict[str, float]]:
        return self._history
