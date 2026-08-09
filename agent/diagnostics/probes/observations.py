"""Probe 5: observation and internal-state access.

Determines whether the current observation, internal state, and step history
can be read directly from the environment object without re-running it, and
documents every useful public attribute found.
"""

from __future__ import annotations

import logging
from typing import Any

from ..utils import ProbeResult, json_safe, public_attrs, shape_of

logger = logging.getLogger(__name__)

GAME = "kaggriculture"


def run() -> ProbeResult:
    """Inspect the environment object and document accessible state."""
    try:
        import kaggle_environments
    except ImportError as exc:
        return ProbeResult(
            name="observations",
            success=False,
            summary=f"kaggle_environments is not installed: {exc}",
            details={"installed": False},
            duration_s=0.0,
            errors=(),
            mcts_verdict=False,
        )

    env = kaggle_environments.make(GAME)

    from .stepping import pass_agent

    for _ in range(3):
        try:
            env.step([pass_agent, pass_agent])
        except Exception:  # noqa: BLE001
            break

    details: dict[str, Any] = {}

    # Every public, non-callable attribute and its shape.
    described: dict[str, str] = {}
    for name in public_attrs(env):
        value = getattr(env, name)
        if callable(value):
            continue
        described[name] = shape_of(value)
    details["public_attributes"] = described

    # Current per-player state: each entry is a step record whose agent
    # observation is nested under its "observation" key.
    state = getattr(env, "state", None)
    if state is not None:
        details["state_shape"] = shape_of(state)
        try:
            details["num_players"] = len(state)
            details["state0_keys"] = list(state[0].keys())
            obs0 = state[0].get("observation", {})
            details["observation0_keys"] = list(obs0.keys()) if isinstance(obs0, dict) else None
            private = obs0.get("private", {}) if isinstance(obs0, dict) else {}
            details["observation0_private_keys"] = list(private.keys()) if isinstance(private, dict) else None
        except Exception as exc:  # noqa: BLE001
            details["state_error"] = f"{type(exc).__name__}: {exc}"

    # Step history.
    steps = getattr(env, "steps", None)
    details["history_attr"] = shape_of(steps) if steps is not None else None
    if isinstance(steps, list):
        details["history_len"] = len(steps)

    # Small scalar / list attributes worth documenting.
    for name in ("configuration", "agents", "episodeSteps", "specification", "name"):
        value = getattr(env, name, None)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            details[f"attr.{name}"] = json_safe(value)
        else:
            details[f"attr.{name}"] = shape_of(value)

    history_accessible = details.get("history_len") is not None
    summary = (
        f"found {len(described)} public attributes; "
        f"history accessible: {history_accessible}"
    )
    logger.info(summary)
    return ProbeResult(
        name="observations",
        success=True,
        summary=summary,
        details=details,
        duration_s=0.0,
        mcts_verdict=True,
    )


if __name__ == "__main__":
    from ..utils import configure_logging, run_probe

    configure_logging()
    result = run_probe("observations", run)
    logger.info("probe result: success=%s summary=%s", result.success, result.summary)
