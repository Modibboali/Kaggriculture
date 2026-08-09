"""Unit tests for StateDiff and report rendering (no environment needed)."""

from __future__ import annotations

from typing import Any

from agent.actions import TurnAction
from agent.testing.differential import (
    MISSING,
    DiffEntry,
    DifferentialResult,
    StateDiff,
    TurnComparison,
    diff,
    render_differential_report,
    render_summary,
)


def _sample_states() -> tuple[dict[str, Any], dict[str, Any]]:
    reference = {
        "day": 1,
        "hour": 2,
        "farms": [
            {
                "money": 90,
                "tiles": [[{"kind": "EMPTY"}, {"kind": "WEED"}]],
                "hands": [[1, 1]],
            }
        ],
        "private": {"seeds": {"WHEAT": 1}, "shed": {"EGG": 0}},
    }
    candidate = {
        "day": 1,
        "hour": 3,
        "farms": [
            {
                "money": 80,
                "tiles": [[{"kind": "EMPTY"}, {"kind": "PLANT", "plant": {"crop": "WHEAT"}}]],
                "hands": [[1, 1], [2, 2]],
            }
        ],
        "private": {"seeds": {"WHEAT": 0}, "shed": {"EGG": 1}},
    }
    return reference, candidate


def test_equal_states_match() -> None:
    reference, _ = _sample_states()
    result = diff(reference, dict(reference))
    assert result.matches is True
    assert bool(result) is True
    assert result.entries == ()


def test_identifies_changed_scalars_with_paths() -> None:
    reference, candidate = _sample_states()
    result = diff(reference, candidate)
    paths = {entry.path: entry for entry in result.entries}
    # day is equal -> no entry; hour differs -> changed.
    assert "day" not in paths
    assert paths["hour"].kind == "changed"
    assert paths["farms[0].money"].kind == "changed"
    assert paths["farms[0].money"].expected == 90
    assert paths["farms[0].money"].actual == 80
    assert paths["farms[0].money"].domain == "money"
    assert paths["private.seeds.WHEAT"].expected == 1
    assert paths["private.seeds.WHEAT"].actual == 0


def test_nested_differences() -> None:
    reference, candidate = _sample_states()
    result = diff(reference, candidate)
    paths = {entry.path: entry for entry in result.entries}
    # The weed tile became a plant tile: the kind field changed.
    kind_diff = paths["farms[0].tiles[0][1].kind"]
    assert kind_diff.expected == "WEED"
    assert kind_diff.actual == "PLANT"
    assert kind_diff.domain == "tile"
    # The plant sub-object is new in the candidate -> added.
    plant_diff = paths["farms[0].tiles[0][1].plant"]
    assert plant_diff.kind == "added"
    assert plant_diff.expected is MISSING
    assert plant_diff.actual == {"crop": "WHEAT"}
    assert plant_diff.domain == "crop"


def test_added_and_removed_fields() -> None:
    reference, candidate = _sample_states()
    result = diff(reference, candidate)
    paths = {entry.path: entry for entry in result.entries}
    # hands gained a second entry -> added.
    assert paths["farms[0].hands[1]"].kind == "added"
    assert paths["farms[0].hands[1]"].expected is MISSING
    # shed.EGG went from 0 -> 1 (changed, not added).
    assert paths["private.shed.EGG"].kind == "changed"

    removed = diff(candidate, reference)
    removed_paths = {entry.path: entry for entry in removed.entries}
    assert removed_paths["farms[0].hands[1]"].kind == "removed"
    assert removed_paths["farms[0].hands[1]"].actual is MISSING


def test_by_domain_groups() -> None:
    reference, candidate = _sample_states()
    result = diff(reference, candidate)
    grouped = result.by_domain
    assert "money" in grouped
    assert "seeds" in grouped
    assert any(entry.domain == "worker/position" for entry in result.entries)


def test_list_length_mismatch() -> None:
    reference = {"farms": [{"hands": [[0, 0]]}]}
    candidate = {"farms": [{"hands": [[0, 0], [1, 1], [2, 2]]}]}
    result = diff(reference, candidate)
    assert len(result.entries) == 2
    assert all(entry.kind == "added" for entry in result.entries)


def test_classify_domains() -> None:
    from agent.testing.differential.state_diff import classify

    assert classify("farms[0].money") == "money"
    assert classify("private.seeds.WHEAT") == "seeds"
    assert classify("private.shed.EGG") == "inventory"
    assert classify("private.inventories[0].WHEAT") == "inventory"
    assert classify("farms[0].tiles[2][3].plant.watered_today") == "crop"
    assert classify("farms[0].tiles[2][3].animal.fed_today") == "animal"
    assert classify("farms[0].tiles[0][0]") == "tile"
    assert classify("farms[0].farmer") == "worker/position"
    assert classify("market.prices.WHEAT") == "market"
    assert classify("town.unlocked_shops") == "town"
    assert classify("day") == "day/hour/step"


def test_report_renders_no_differences() -> None:
    reference, _ = _sample_states()
    result = DifferentialResult(
        scenario="pass",
        comparisons=(
            TurnComparison(
                step=1,
                actions=(TurnAction(), TurnAction()),
                diff=diff(reference, dict(reference)),
            ),
        ),
    )
    report = render_differential_report(result)
    assert "Scenario: pass" in report
    assert "No differences detected." in report
    assert render_summary(result) == "Scenario pass: match (1 turns)"


def test_report_renders_mismatch() -> None:
    reference, candidate = _sample_states()
    result = DifferentialResult(
        scenario="demo",
        comparisons=(
            TurnComparison(
                step=3,
                actions=(TurnAction(), TurnAction()),
                diff=diff(reference, candidate),
            ),
        ),
    )
    report = render_differential_report(result)
    assert "Scenario: demo" in report
    assert "Turn: 3" in report
    assert "farms[0].money:" in report
    assert "expected: 90" in report
    assert "actual: 80" in report
    assert render_summary(result).startswith("Scenario demo: MISMATCH")
