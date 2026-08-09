"""Probe 1: environment creation.

Determines whether the Kaggriculture environment can be created
programmatically, how long creation takes, whether several environments can
coexist, and what memory creation consumes.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from ..utils import ProbeError, ProbeResult, attempt, json_safe, measure, shape_of

logger = logging.getLogger(__name__)

GAME = "kaggriculture"


def run() -> ProbeResult:
    """Create the environment and record timing, memory, and coexistence."""
    try:
        from kaggle_environments import make
    except ImportError as exc:
        return ProbeResult(
            name="environment_creation",
            success=False,
            summary=f"kaggle_environments is not installed: {exc}",
            details={"installed": False},
            duration_s=0.0,
            errors=(
                ProbeError(
                    probe="environment_creation",
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    traceback=traceback.format_exc(),
                ),
            ),
            mcts_verdict=False,
        )

    details: dict[str, Any] = {}
    envs: list[Any] = []

    def create_one() -> Any:
        env = make(GAME)
        envs.append(env)
        return env

    env, creation_s, peak = measure(create_one)
    details["installed"] = True
    details["creation_time_s"] = round(creation_s, 6)
    details["peak_memory_bytes"] = peak
    details["env_type"] = type(env).__name__

    ok_multi, extra, desc_multi = attempt(
        "create 3 environments", lambda: [make(GAME) for _ in range(3)]
    )
    details["multiple_simultaneous"] = ok_multi
    details["multiple_detail"] = desc_multi
    if ok_multi and extra is not None:
        details["multiple_created"] = len(extra)

    # Record compact metadata about the created environment.
    for attr in ("name", "agents", "configuration", "specification", "episodeSteps"):
        value = getattr(env, attr, None)
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            details[f"env.{attr}"] = shape_of(value)
        else:
            details[f"env.{attr}"] = json_safe(value)

    summary = (
        f"created '{GAME}' in {creation_s:.4f}s, peak {peak / 1024 / 1024:.1f} MiB; "
        f"simultaneous envs: {ok_multi}"
    )
    logger.info(summary)
    return ProbeResult(
        name="environment_creation",
        success=True,
        summary=summary,
        details=details,
        duration_s=0.0,
        mcts_verdict=True,
    )


if __name__ == "__main__":
    from ..utils import configure_logging, run_probe

    configure_logging()
    result = run_probe("environment_creation", run)
    logger.info("probe result: success=%s summary=%s", result.success, result.summary)
