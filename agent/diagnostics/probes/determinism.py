"""Probe 8: determinism under a fixed seed.

Runs short identical games several times under the same ``seed``
configuration and checks whether the replays are identical. Also checks that a
different seed produces different results, and (informational only) whether
the built-in ``"random"`` agent is reproducible under a fixed seed.
"""

from __future__ import annotations

import json as jsonlib
import logging
from typing import Any, Callable

from ..utils import ProbeResult

logger = logging.getLogger(__name__)

GAME = "kaggriculture"
# Short episodes keep the probe fast; seeding semantics are already visible.
DETERMINISM_STEPS = 60


def starter_agent(obs: Any, config: Any) -> dict[str, Any]:
    """A deterministic policy: buy one wheat seed on turn 0, then pass."""
    del config
    if obs.get("step", 0) == 0:
        return {"farmer": ["PASS"], "market": [["BUY_SEED", "WHEAT", 1]]}
    return {"farmer": ["PASS"], "market": []}


def _sanitize(value: Any) -> Any:
    """Recursively drop the wall-clock ``remainingOverageTime`` field.

    That field reflects real elapsed time and would make two otherwise
    identical runs compare unequal.
    """
    if isinstance(value, dict):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if key != "remainingOverageTime"
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _replay_fingerprint(env: Any) -> str:
    """A stable string fingerprint of ``env.steps`` for comparison."""
    return jsonlib.dumps(_sanitize(env.steps), sort_keys=True, default=str)


def run() -> ProbeResult:
    """Compare seeded runs for identity and sensitivity to the seed."""
    try:
        from kaggle_environments import make
    except ImportError as exc:
        return ProbeResult(
            name="determinism",
            success=False,
            summary=f"kaggle_environments is not installed: {exc}",
            details={"installed": False},
            duration_s=0.0,
            errors=(),
            mcts_verdict=False,
        )

    details: dict[str, Any] = {}

    def make_seeded(seed: int) -> Any:
        return make(GAME, configuration={"seed": seed, "episodeSteps": DETERMINISM_STEPS})

    def run_game(seed: int, agent: Callable[[Any, Any], dict[str, Any]]) -> str:
        env = make_seeded(seed)
        env.run([agent, agent])
        return _replay_fingerprint(env)

    fingerprint_a: str | None = None
    try:
        fingerprint_a = run_game(42, starter_agent)
        fingerprint_b = run_game(42, starter_agent)
        details["same_seed_identical"] = fingerprint_a == fingerprint_b
    except Exception as exc:  # noqa: BLE001
        details["same_seed_identical"] = False
        details["same_seed_error"] = f"{type(exc).__name__}: {exc}"

    if fingerprint_a is not None:
        try:
            fingerprint_c = run_game(43, starter_agent)
            details["different_seed_differs"] = fingerprint_a != fingerprint_c
        except Exception as exc:  # noqa: BLE001
            details["different_seed_differs"] = None
            details["different_seed_error"] = f"{type(exc).__name__}: {exc}"

    # Built-in "random" agent under the same seed (informational).
    try:
        e1 = make_seeded(7)
        e1.run(["random", "random"])
        e2 = make_seeded(7)
        e2.run(["random", "random"])
        details["random_agent_same_seed_identical"] = (
            _replay_fingerprint(e1) == _replay_fingerprint(e2)
        )
    except Exception as exc:  # noqa: BLE001
        details["random_agent_same_seed_identical"] = None
        details["random_agent_error"] = f"{type(exc).__name__}: {exc}"

    identical = details.get("same_seed_identical")
    summary = f"same seed identical (deterministic agent) = {identical}"
    logger.info(summary)
    return ProbeResult(
        name="determinism",
        success=identical is True,
        summary=summary,
        details=details,
        duration_s=0.0,
        mcts_verdict=bool(identical),
    )


if __name__ == "__main__":
    from ..utils import configure_logging, run_probe

    configure_logging()
    result = run_probe("determinism", run)
    logger.info("probe result: success=%s summary=%s", result.success, result.summary)
