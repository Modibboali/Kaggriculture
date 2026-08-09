"""Environment Compatibility & Benchmark Suite entry point.

Usage:
    python -m agent.diagnostics.environment_probe [probe ...]

With no arguments every probe and the benchmark run. Otherwise only the named
probes run (e.g. ``python -m agent.diagnostics.environment_probe cloning
stepping``). Reports are written to ``agent/diagnostics/output`` and a console
summary is printed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import benchmark
from .probes import (
    actions,
    cloning,
    determinism,
    environment_creation,
    observations,
    replay,
    serialization,
    stepping,
)
from .report import print_console_summary, write_report
from .utils import DiagnosticRun, ProbeResult, configure_logging, now_iso, run_probe

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Ordered mapping of probe name -> module with a ``run() -> ProbeResult``.
_MODULES: dict[str, Any] = {
    "environment_creation": environment_creation,
    "stepping": stepping,
    "cloning": cloning,
    "serialization": serialization,
    "observations": observations,
    "actions": actions,
    "replay": replay,
    "determinism": determinism,
    "benchmark": benchmark,
}


def run_all(selected: list[str] | None = None) -> DiagnosticRun:
    """Run the chosen probes (or all of them) and aggregate the results."""
    names = list(_MODULES) if selected is None else [n for n in _MODULES if n in selected]
    results = [run_probe(name, _MODULES[name].run) for name in names]
    return DiagnosticRun(
        environment="kaggriculture",
        timestamp=now_iso(),
        results=tuple(results),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point: run probes, write reports, print the console summary."""
    configure_logging()
    selected = list(argv) if argv is not None else None
    run = run_all(selected=selected)
    paths = write_report(run, OUTPUT_DIR)
    for path in paths:
        logger.info("wrote %s", path)
    print_console_summary(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
