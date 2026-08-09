"""Probe 9: performance benchmark.

Measures the numbers the rest of the architecture depends on: environment
creation throughput, steps/sec, peak memory, clone speed, and serialization
speed.
"""

from __future__ import annotations

import copy
import json as jsonlib
import logging
import pickle
import time
from typing import Any

from .utils import ProbeResult, measure

logger = logging.getLogger(__name__)

GAME = "kaggriculture"


def run() -> ProbeResult:
    """Benchmark creation, stepping, memory, cloning, and serialization."""
    try:
        import kaggle_environments
    except ImportError as exc:
        return ProbeResult(
            name="benchmark",
            success=False,
            summary=f"kaggle_environments is not installed: {exc}",
            details={"installed": False},
            duration_s=0.0,
            errors=(),
            mcts_verdict=None,
        )

    from .probes.stepping import pass_agent

    details: dict[str, Any] = {}

    # --- Creation throughput -------------------------------------------------
    n_creations = 10
    start = time.perf_counter()
    for _ in range(n_creations):
        kaggle_environments.make(GAME)
    creation_total = time.perf_counter() - start
    details["creation_per_sec"] = round(n_creations / creation_total, 1) if creation_total else 0.0
    details["creation_avg_s"] = round(creation_total / n_creations, 6)

    # --- Full-run throughput -------------------------------------------------
    env = kaggle_environments.make(GAME)
    start = time.perf_counter()
    try:
        env.run([pass_agent, pass_agent])
        details["full_run_ok"] = True
        details["full_run_detail"] = "completed"
    except Exception as exc:  # noqa: BLE001
        details["full_run_ok"] = False
        details["full_run_detail"] = f"{type(exc).__name__}: {exc}"
    run_total = time.perf_counter() - start
    steps = len(getattr(env, "steps", []))
    details["full_run_steps"] = steps
    details["full_run_s"] = round(run_total, 6)
    details["steps_per_sec"] = round(steps / run_total, 1) if run_total > 0 and steps else 0.0
    details["avg_step_s"] = round(run_total / steps, 6) if steps else 0.0

    # --- Memory during create + short run ------------------------------------
    def create_and_run() -> Any:
        e = kaggle_environments.make(GAME, configuration={"episodeSteps": 60})
        e.run([pass_agent, pass_agent])
        return e

    _, _, peak = measure(create_and_run)
    details["peak_memory_run_bytes"] = peak

    # --- Clone speed (deepcopy) ----------------------------------------------
    try:
        _, clone_s, _ = measure(lambda: copy.deepcopy(env))
        details["clone_ok"] = True
        details["clone_s"] = round(clone_s, 6)
        details["clone_per_sec"] = round(1.0 / clone_s, 1) if clone_s > 0 else 0.0
    except Exception as exc:  # noqa: BLE001
        details["clone_ok"] = False
        details["clone_detail"] = f"{type(exc).__name__}: {exc}"

    # --- Serialization speed (pickle + toJSON) -------------------------------
    try:
        data, dump_s, _ = measure(lambda: pickle.dumps(env))
        details["pickle_ok"] = True
        details["pickle_dump_s"] = round(dump_s, 6)
        details["pickle_size_bytes"] = len(data)
    except Exception as exc:  # noqa: BLE001
        details["pickle_ok"] = False
        details["pickle_detail"] = f"{type(exc).__name__}: {exc}"

    # Native clone speed (the environment ships its own clone()).
    try:
        _, native_clone_s, _ = measure(lambda: env.clone())
        details["native_clone_ok"] = True
        details["native_clone_s"] = round(native_clone_s, 6)
        details["native_clone_per_sec"] = (
            round(1.0 / native_clone_s, 1) if native_clone_s > 0 else 0.0
        )
    except Exception as exc:  # noqa: BLE001
        details["native_clone_ok"] = False
        details["native_clone_detail"] = f"{type(exc).__name__}: {exc}"

    # Replay serialization speed (full step history to JSON).
    try:
        js, js_s, _ = measure(lambda: jsonlib.dumps(env.steps, default=str))
        details["replay_json_ok"] = True
        details["replay_json_len"] = len(js)
        details["replay_json_s"] = round(js_s, 6)
    except Exception as exc:  # noqa: BLE001
        details["replay_json_ok"] = False
        details["replay_json_detail"] = f"{type(exc).__name__}: {exc}"

    summary = (
        f"creation={details.get('creation_per_sec', 0.0)}/s, "
        f"steps={details.get('steps_per_sec', 0.0)}/s, "
        f"peak mem={peak / 1024 / 1024:.1f} MiB"
    )
    logger.info(summary)
    return ProbeResult(
        name="benchmark",
        success=bool(details.get("full_run_ok")),
        summary=summary,
        details=details,
        duration_s=0.0,
        mcts_verdict=None,
    )


if __name__ == "__main__":
    from .utils import configure_logging, run_probe

    configure_logging()
    result = run_probe("benchmark", run)
    logger.info("probe result: success=%s summary=%s", result.success, result.summary)
