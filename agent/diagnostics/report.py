"""Report generation for the environment diagnostics suite.

Turns a :class:`~agent.diagnostics.utils.DiagnosticRun` into a console summary,
a Markdown report, a JSON report, and benchmark statistics, and computes the
final A/B verdict and recommendations for the project architecture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import DiagnosticRun, ProbeResult

# Rough minimum search throughput (steps/sec) below which using the official
# environment as the in-tree MCTS rollout engine is impractical. This is a
# documented heuristic, not a hard science number.
MIN_STEPS_PER_SECOND = 20.0


def _find(run: DiagnosticRun, name: str) -> ProbeResult | None:
    """Return the probe result with the given name, if present."""
    for result in run.results:
        if result.name == name:
            return result
    return None


def _flag(run: DiagnosticRun, probe: str, key: str) -> bool:
    """Whether ``run``'s ``probe`` result has a truthy ``key`` in details."""
    result = _find(run, probe)
    return bool(result and result.details.get(key))


def performance_numbers(run: DiagnosticRun) -> dict[str, Any]:
    """Flatten all numeric details across probes into one metric table."""
    numbers: dict[str, Any] = {}
    for result in run.results:
        for key, value in result.details.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numbers[f"{result.name}.{key}"] = value
    return numbers


def supported_operations(run: DiagnosticRun) -> list[str]:
    """What the environment demonstrably supports."""
    ops: list[str] = []
    if _flag(run, "actions", "both_players_controllable"):
        ops.append("Inject independent actions for both players, simultaneously")
    if _flag(run, "cloning", "any_clone_works"):
        ops.append("Clone the environment (copy / pickle)")
    if _flag(run, "serialization", "any_roundtrip_works"):
        ops.append("Serialize and restore environment state")
    if _flag(run, "replay", "tojson_during"):
        ops.append("Access replay while the game is running")
    observations = _find(run, "observations")
    if observations is not None and observations.success:
        ops.append("Read observations / internal state / history without rerunning")
    if _flag(run, "determinism", "same_seed_identical"):
        ops.append("Deterministic replay under a fixed seed (with a deterministic agent)")
    return ops or ["None detected"]


def unsupported_operations(run: DiagnosticRun) -> list[str]:
    """What the environment demonstrably does not support."""
    ops: list[str] = []
    if _find(run, "cloning") is not None and not _flag(run, "cloning", "any_clone_works"):
        ops.append("Clone the environment (copy / deepcopy / pickle / cloudpickle all failed)")
    if _find(run, "serialization") is not None and not _flag(run, "serialization", "any_roundtrip_works"):
        ops.append("Serialize / restore environment state to bytes")
    if _find(run, "determinism") is not None and _flag(run, "determinism", "same_seed_identical") is not True:
        ops.append("Fully deterministic replay under a fixed seed")
    return ops or ["None detected"]


def recommendations(run: DiagnosticRun) -> list[str]:
    """Architecture recommendations driven by the probe findings."""
    recs: list[str] = []
    if not _flag(run, "cloning", "any_clone_works"):
        recs.append(
            "The environment cannot be cloned cheaply: do not branch search "
            "state by copying the environment. Keep the domain GameState as the "
            "single source of truth and re-derive it from observations."
        )
    if not _flag(run, "serialization", "any_roundtrip_works"):
        recs.append(
            "Environment state cannot be serialized: persist agent data through "
            "the domain model (hashable GameState) instead of environment objects."
        )
    if _find(run, "determinism") is not None and not _flag(run, "determinism", "same_seed_identical"):
        recs.append(
            "The environment is not fully deterministic under `seed`: drive "
            "self-play reproducibility from your own RNG, not the environment's."
        )
    steps = _find(run, "benchmark")
    steps_per_sec = float(steps.details.get("steps_per_sec", 0.0)) if steps else 0.0
    if steps_per_sec < MIN_STEPS_PER_SECOND:
        recs.append(
            f"Search throughput is low ({steps_per_sec:.1f} steps/sec < "
            f"{MIN_STEPS_PER_SECOND:g}): the official environment is too slow to "
            "be the MCTS rollout engine."
        )
    recs.append(
        "Drive all search with agent.state.GameState.from_observation + "
        "agent.generators.LegalActionGenerator; these are environment-independent "
        "and hashable."
    )
    recs.append(
        "Plan for a lightweight in-process simulator on the domain model "
        "(agent.state + agent.actions) to serve as the MCTS rollout engine, with "
        "the official environment used only for training/validation games."
    )
    return recs


def verdict(run: DiagnosticRun) -> tuple[str, list[str]]:
    """Compute the final A/B suitability verdict with reasons.

    Criteria (documented heuristics):
      * both players are independently controllable via injected actions,
      * the environment can be cloned OR state serialized/restored,
      * measured throughput meets MIN_STEPS_PER_SECOND.
    """
    step_injectable = _flag(run, "actions", "both_players_controllable")
    any_clone = _flag(run, "cloning", "any_clone_works")
    any_serial = _flag(run, "serialization", "any_roundtrip_works")
    bench = _find(run, "benchmark")
    steps_per_sec = float(bench.details.get("steps_per_sec", 0.0)) if bench else 0.0

    reasons = [
        f"Controllable stepping for both players: {'yes' if step_injectable else 'no'}",
        f"Environment cloning: {'yes' if any_clone else 'no'}",
        f"State save/restore: {'yes' if any_serial else 'no'}",
        f"Search throughput: {steps_per_sec:.1f} steps/sec (MCTS minimum assumed: {MIN_STEPS_PER_SECOND:g})",
    ]

    suitable = step_injectable and (any_clone or any_serial) and steps_per_sec >= MIN_STEPS_PER_SECOND
    if suitable:
        reasons.append(
            "The official environment exposes controllable stepping, state "
            "save/restore, and adequate throughput for in-tree use."
        )
        return "A", reasons
    reasons.append(
        "The official environment does not meet all requirements for in-tree "
        "use; a lightweight domain-model simulator is recommended for MCTS."
    )
    return "B", reasons


def generate_markdown(run: DiagnosticRun) -> str:
    """Render the full human-readable Markdown report."""
    lines: list[str] = []
    lines.append("# Kaggriculture Environment Compatibility Report")
    lines.append("")
    lines.append(f"- **Environment:** `{run.environment}` (official `kaggle-environments` package)")
    lines.append(f"- **Generated:** {run.timestamp}")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    letter, reasons = verdict(run)
    lines.append(
        f"**{letter}** — "
        f"{'Suitable for in-tree MCTS use' if letter == 'A' else 'NOT suitable for in-tree MCTS use'}"
    )
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Probe | Result | Summary |")
    lines.append("|-------|--------|---------|")
    for result in run.results:
        status = "PASS" if result.success else "FAIL"
        safe_summary = result.summary.replace("|", "\\|")
        lines.append(f"| {result.name} | {status} | {safe_summary} |")
    lines.append("")
    lines.append("## Supported Operations")
    lines.append("")
    for op in supported_operations(run):
        lines.append(f"- {op}")
    lines.append("")
    lines.append("## Unsupported Operations")
    lines.append("")
    for op in unsupported_operations(run):
        lines.append(f"- {op}")
    lines.append("")
    lines.append("## Performance Numbers")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, value in performance_numbers(run).items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    for rec in recommendations(run):
        lines.append(f"- {rec}")
    lines.append("")
    lines.append("## Detailed Probe Results")
    lines.append("")
    for result in run.results:
        lines.append(f"### {result.name}")
        lines.append("")
        lines.append(f"- **Success:** {result.success}")
        lines.append(f"- **Summary:** {result.summary}")
        lines.append(f"- **Duration (s):** {result.duration_s:.4f}")
        lines.append("- **Details:**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result.details, indent=2, default=str))
        lines.append("```")
        for error in result.errors:
            lines.append(f"- **Error ({error.exception_type}):** {error.message}")
            lines.append("")
            lines.append("```")
            lines.append(error.traceback)
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def generate_json(run: DiagnosticRun) -> str:
    """Render the full JSON report."""
    return json.dumps(run.to_dict(), indent=2, default=str)


def generate_benchmark_json(run: DiagnosticRun) -> str:
    """Render the benchmark statistics JSON."""
    bench = _find(run, "benchmark")
    data = bench.to_dict() if bench else {"probe": "benchmark", "success": False}
    return json.dumps(data, indent=2, default=str)


def write_report(run: DiagnosticRun, output_dir: Path) -> tuple[Path, Path, Path]:
    """Write report.md, report.json, and benchmark.json, returning their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "report.md"
    json_path = output_dir / "report.json"
    bench_path = output_dir / "benchmark.json"
    md_path.write_text(generate_markdown(run), encoding="utf-8")
    json_path.write_text(generate_json(run), encoding="utf-8")
    bench_path.write_text(generate_benchmark_json(run), encoding="utf-8")
    return md_path, json_path, bench_path


def print_console_summary(run: DiagnosticRun) -> None:
    """Print the required console summary (the suite's stdout deliverable)."""
    sep = "=" * 72
    print(sep)
    print("Kaggriculture Environment Compatibility Report")
    print(f"Environment: {run.environment}   Generated: {run.timestamp}")
    print(sep)
    for result in run.results:
        status = "PASS" if result.success else "FAIL"
        print(f"[{status:4}] {result.name:<24} {result.summary}")
    print("-" * 72)
    letter, reasons = verdict(run)
    print(f"VERDICT: {letter}")
    for reason in reasons:
        print(f"  - {reason}")
    print(sep)
