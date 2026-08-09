"""Shared helpers for the environment diagnostics suite.

Contains result containers, timing/memory measurement, safe probing helpers,
and JSON-safe conversion. No Kaggle-specific code lives here; the probes live
in :mod:`agent.diagnostics.probes`.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypeVar, cast

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProbeError:
    """A structured record of one exception raised by a probe."""

    probe: str
    message: str
    exception_type: str
    traceback: str


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Structured output of a single probe or benchmark.

    ``details`` is a JSON-safe mapping of everything the probe learned.
    ``mcts_verdict`` is an optional per-probe opinion on whether the finding
    supports using the official environment inside MCTS.
    """

    name: str
    success: bool
    summary: str
    details: dict[str, Any]
    duration_s: float
    errors: tuple[ProbeError, ...] = ()
    mcts_verdict: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable dictionary view of this result."""
        return {
            "probe": self.name,
            "success": self.success,
            "summary": self.summary,
            "details": self.details,
            "duration_s": round(self.duration_s, 6),
            "errors": [asdict(error) for error in self.errors],
            "mcts_verdict": self.mcts_verdict,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticRun:
    """The aggregated output of the whole diagnostics suite."""

    environment: str
    timestamp: str
    results: tuple[ProbeResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable dictionary view of the whole run."""
        return {
            "environment": self.environment,
            "timestamp": self.timestamp,
            "results": [result.to_dict() for result in self.results],
        }


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple stdout logging handler."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def now_iso() -> str:
    """Current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def measure(fn: Callable[[], T]) -> tuple[T, float, int]:
    """Run ``fn`` once, returning ``(result, duration_s, peak_bytes)``.

    Peak memory uses :mod:`tracemalloc`, which measures Python heap
    allocations and works on every platform without extra dependencies.
    """
    import tracemalloc

    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = fn()
    finally:
        duration = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, duration, peak


def attempt(label: str, fn: Callable[[], T]) -> tuple[bool, T | None, str]:
    """Try ``fn``; return ``(ok, result, description)``.

    On failure the description is ``"ExceptionType: message"``. Callers that
    need the full traceback should catch the exception themselves.
    """
    del label  # kept for readable log lines if a probe chooses to log it
    try:
        value = fn()
        return True, value, f"ok ({type(value).__name__})"
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash
        return False, None, f"{type(exc).__name__}: {exc}"


def run_probe(name: str, fn: Callable[[], ProbeResult]) -> ProbeResult:
    """Run a probe with wall-clock timing and a safety net.

    The probe's own ``duration_s`` is overwritten with the measured wall time
    so callers see the total cost including any setup the probe performs.
    """
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        result = ProbeResult(
            name=name,
            success=False,
            summary=f"probe raised {type(exc).__name__}: {exc}",
            details={},
            duration_s=0.0,
            errors=(
                ProbeError(
                    probe=name,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    traceback=tb,
                ),
            ),
            mcts_verdict=False,
        )
    return replace(result, duration_s=time.perf_counter() - start)


def public_attrs(obj: Any) -> list[str]:
    """Public (non-underscore) attribute names of ``obj``, sorted."""
    return sorted(name for name in dir(obj) if not name.startswith("_"))


def shape_of(value: Any) -> str:
    """A compact description of ``value``: type plus length where relevant."""
    try:
        if isinstance(value, (list, tuple, dict, set, str, bytes)):
            return f"{type(value).__name__}(len={len(value)})"
        return type(value).__name__
    except Exception:  # noqa: BLE001
        return "<unknown>"


def safe_repr(value: Any) -> str:
    """``repr`` that never raises."""
    try:
        return repr(value)
    except Exception:  # noqa: BLE001
        return f"<{type(value).__name__}: repr failed>"


def json_safe(value: Any) -> Any:
    """Convert a value into a JSON-serializable form, recursively."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return repr(value)
    if is_dataclass(value):
        # is_dataclass() narrows to a DataclassInstance union that asdict()
        # rejects; the value is known to be a dataclass instance here.
        return json_safe(asdict(cast(Any, value)))
    return safe_repr(value)


def extract_unsupported(tb: str) -> str:
    """Pull the most useful line out of a clone/serialize traceback."""
    for line in tb.splitlines():
        lowered = line.lower()
        if "cannot pickle" in lowered or "not picklable" in lowered:
            return line.strip()
    lines = tb.strip().splitlines()
    return lines[-1] if lines else ""
