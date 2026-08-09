"""Probe 7: replay access.

Determines whether replay information can be accessed while the game is still
running, or only after it completes.
"""

from __future__ import annotations

import logging
from typing import Any

from ..utils import ProbeResult, attempt

logger = logging.getLogger(__name__)

GAME = "kaggriculture"


def run() -> ProbeResult:
    """Check replay access mid-game and after a full run."""
    try:
        import kaggle_environments
    except ImportError as exc:
        return ProbeResult(
            name="replay",
            success=False,
            summary=f"kaggle_environments is not installed: {exc}",
            details={"installed": False},
            duration_s=0.0,
            errors=(),
            mcts_verdict=False,
        )

    from .stepping import pass_agent

    details: dict[str, Any] = {}

    # Mid-game access: is the step history readable while running?
    env = kaggle_environments.make(GAME)
    for _ in range(5):
        try:
            env.step([pass_agent, pass_agent])
        except Exception:  # noqa: BLE001
            break
    steps_during = len(getattr(env, "steps", []))
    details["steps_during"] = steps_during
    details["replay_during"] = steps_during > 0

    # toJSON() in this version returns environment *metadata*, not the replay.
    ok_meta, meta, desc_meta = attempt("toJSON (metadata)", lambda: env.toJSON())
    details["tojson_ok"] = ok_meta
    details["tojson_detail"] = desc_meta
    details["tojson_is_dict"] = isinstance(meta, dict)
    if isinstance(meta, dict):
        details["tojson_keys"] = list(meta.keys())

    # Post-game access.
    env2 = kaggle_environments.make(GAME)
    try:
        env2.run([pass_agent, pass_agent])
        details["full_run_ok"] = True
        details["full_run_detail"] = "ran to completion"
    except Exception as exc:  # noqa: BLE001
        details["full_run_ok"] = False
        details["full_run_detail"] = f"{type(exc).__name__}: {exc}"
    steps_after = len(getattr(env2, "steps", []))
    details["steps_after"] = steps_after
    details["replay_after"] = steps_after > 0
    try:
        details["step0_keys"] = list(env2.steps[0][0].keys())
    except Exception:  # noqa: BLE001
        details["step0_keys"] = None

    ok_render, _, desc_render = attempt("render json", lambda: env2.render(mode="json"))
    details["render_json"] = ok_render
    details["render_json_detail"] = desc_render

    replay_during = bool(details["replay_during"])
    replay_after = bool(details["replay_after"])
    summary = (
        f"replay during={replay_during}, after={replay_after}; "
        "replay is env.steps (toJSON() is metadata)"
    )
    logger.info(summary)
    return ProbeResult(
        name="replay",
        success=replay_after,
        summary=summary,
        details=details,
        duration_s=0.0,
        mcts_verdict=replay_during,
    )


if __name__ == "__main__":
    from ..utils import configure_logging, run_probe

    configure_logging()
    result = run_probe("replay", run)
    logger.info("probe result: success=%s summary=%s", result.success, result.summary)
