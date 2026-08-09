"""Human-readable differential reports."""

from __future__ import annotations

import json
from typing import Any

from ...actions import TurnAction
from .runner import DifferentialResult
from .state_diff import MISSING


def format_value(value: Any) -> str:
    """Render a canonical value compactly for a report."""
    if value is MISSING:
        return "<missing>"
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


def _turn_summary(action: TurnAction) -> str:
    parts = [str(action.farmer_action)]
    parts.extend(str(part) for part in action.worker_actions)
    parts.extend(str(part) for part in action.market_actions)
    return " | ".join(parts)


def render_differential_report(result: DifferentialResult) -> str:
    """Render a full report of every mismatch in ``result``."""
    lines = [f"Scenario: {result.scenario}", ""]
    for comparison in result.comparisons:
        if comparison.diff.matches:
            continue
        lines.append(f"Turn: {comparison.step}")
        lines.append("Action:")
        lines.append(f"    P0: {_turn_summary(comparison.actions[0])}")
        lines.append(f"    P1: {_turn_summary(comparison.actions[1])}")
        lines.append("Differences:")
        for entry in comparison.diff.entries:
            lines.append("")
            lines.append(f"    {entry.path}:")
            lines.append(f"        expected: {format_value(entry.expected)}")
            lines.append(f"        actual: {format_value(entry.actual)}")
        lines.append("")
    if not result.has_mismatch:
        lines.append("No differences detected.")
    return "\n".join(lines)


def render_summary(result: DifferentialResult) -> str:
    """A one-line summary of a differential result."""
    if result.has_mismatch:
        turns = ", ".join(str(turn) for turn in result.mismatched_turns)
        return f"Scenario {result.scenario}: MISMATCH at turn(s) [{turns}]"
    return f"Scenario {result.scenario}: match ({len(result.comparisons)} turns)"
