"""Probe 2: environment stepping API and step performance.

Determines which stepping entry points exist (``step`` / ``run`` / ``act``),
whether actions for both players can be injected manually, and how fast the
environment advances (100 and 1000 steps).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..utils import ProbeResult, attempt, public_attrs

logger = logging.getLogger(__name__)

GAME = "kaggriculture"

# A valid Kaggriculture action for one turn (see the competition overview).
PASS_ACTION: dict[str, Any] = {"farmer": ["PASS"], "market": []}


def pass_agent(obs: Any, config: Any) -> dict[str, Any]:
    """A trivial deterministic agent that always passes."""
    del obs, config
    return PASS_ACTION


def run() -> ProbeResult:
    """Discover the stepping API and measure step throughput."""
    try:
        import kaggle_environments
    except ImportError as exc:
        return ProbeResult(
            name="stepping",
            success=False,
            summary=f"kaggle_environments is not installed: {exc}",
            details={"installed": False},
            duration_s=0.0,
            errors=(),
            mcts_verdict=False,
        )

    details: dict[str, Any] = {}
    env = kaggle_environments.make(GAME)
    attrs = public_attrs(env)
    for name in ("step", "run", "act", "reset", "toJSON", "render", "clone", "play", "train"):
        details[f"has_{name}"] = name in attrs

    # 1) Raw dict actions for both players.
    ok_dict, _, desc_dict = attempt(
        "step with raw dict actions",
        lambda: env.step([PASS_ACTION, PASS_ACTION]),
    )
    details["step_dict_actions"] = ok_dict
    details["step_dict_detail"] = desc_dict
    if ok_dict:
        details["steps_after_step"] = len(getattr(env, "steps", []))

    # 2) Callable actions for both players.
    ok_fn, _, desc_fn = attempt(
        "step with callable actions",
        lambda: env.step([pass_agent, pass_agent]),
    )
    details["step_callable_actions"] = ok_fn
    details["step_callable_detail"] = desc_fn

    # 3) Throughput via ``run`` over a controlled number of steps.
    def timed_run(episode_steps: int) -> tuple[float, int]:
        env_run = kaggle_environments.make(
            GAME, configuration={"episodeSteps": episode_steps}
        )
        start = time.perf_counter()
        env_run.run([pass_agent, pass_agent])
        duration = time.perf_counter() - start
        return duration, len(getattr(env_run, "steps", []))

    try:
        dur100, steps100 = timed_run(100)
        details["run_100_steps_s"] = round(dur100, 6)
        details["run_100_steps_per_s"] = round(steps100 / dur100, 1) if dur100 > 0 else 0.0
    except Exception as exc:  # noqa: BLE001
        details["run_100"] = f"{type(exc).__name__}: {exc}"

    try:
        dur1000, steps1000 = timed_run(1000)
        details["run_1000_steps_s"] = round(dur1000, 6)
        details["run_1000_steps_per_s"] = round(steps1000 / dur1000, 1) if dur1000 > 0 else 0.0
        details["avg_step_s"] = round(dur1000 / steps1000, 6) if steps1000 > 0 else 0.0
    except Exception as exc:  # noqa: BLE001
        details["run_1000"] = f"{type(exc).__name__}: {exc}"

    success = bool(ok_dict or ok_fn)
    summary = (
        f"step(): dict={ok_dict}, callable={ok_fn}; "
        f"{details.get('run_1000_steps_per_s', 0.0)} steps/sec @1000"
    )
    logger.info(summary)
    return ProbeResult(
        name="stepping",
        success=success,
        summary=summary,
        details=details,
        duration_s=0.0,
        mcts_verdict=success,
    )


if __name__ == "__main__":
    from ..utils import configure_logging, run_probe

    configure_logging()
    result = run_probe("stepping", run)
    logger.info("probe result: success=%s summary=%s", result.success, result.summary)
