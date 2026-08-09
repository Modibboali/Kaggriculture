"""Probe 6: arbitrary action injection.

Determines whether arbitrary actions can be supplied to the environment, and
whether player 0 and player 1 can be controlled independently and
simultaneously.
"""

from __future__ import annotations

import logging
from typing import Any

from ..utils import ProbeResult, attempt

logger = logging.getLogger(__name__)

GAME = "kaggriculture"

BUY_WHEAT: dict[str, Any] = {"farmer": ["PASS"], "market": [["BUY_SEED", "WHEAT", 1]]}
NOTHING: dict[str, Any] = {"farmer": ["PASS"], "market": []}


def run() -> ProbeResult:
    """Inject independent actions for both players and verify the effects."""
    try:
        import kaggle_environments
    except ImportError as exc:
        return ProbeResult(
            name="actions",
            success=False,
            summary=f"kaggle_environments is not installed: {exc}",
            details={"installed": False},
            duration_s=0.0,
            errors=(),
            mcts_verdict=False,
        )

    env = kaggle_environments.make(GAME)
    details: dict[str, Any] = {}

    def observation_of(player: int) -> dict[str, Any]:
        """The observation dict the environment exposes for ``player``.

        ``env.state[i]`` is a step record; the agent observation is nested
        under its "observation" key.
        """
        try:
            state = env.state[player]
        except Exception:  # noqa: BLE001
            return {}
        obs = state.get("observation", {}) if isinstance(state, dict) else {}
        return obs if isinstance(obs, dict) else {}

    def seeds_of(player: int) -> int:
        """Wheat seed count visible to ``player``, or -1 if unreadable."""
        try:
            seeds = observation_of(player).get("private", {}).get("seeds", {})
            return int(seeds.get("WHEAT", -1))
        except Exception:  # noqa: BLE001
            return -1

    # Player 0 buys wheat, player 1 passes.
    ok_p0, _, desc_p0 = attempt(
        "p0 buy / p1 pass", lambda: env.step([BUY_WHEAT, NOTHING])
    )
    details["step_accepted"] = ok_p0
    details["step_detail"] = desc_p0

    ok_p1 = False
    if ok_p0:
        details["p0_seeds_after_p0_buy"] = seeds_of(0)
        details["p1_seeds_after_p0_buy"] = seeds_of(1)

        # Player 1 buys wheat, player 0 passes.
        ok_p1, _, desc_p1 = attempt(
            "p1 buy / p0 pass", lambda: env.step([NOTHING, BUY_WHEAT])
        )
        details["p1_independent"] = ok_p1
        details["p1_detail"] = desc_p1
        if ok_p1:
            details["p1_seeds_after_p1_buy"] = seeds_of(1)
            details["p0_seeds_after_p1_buy"] = seeds_of(0)

        # Both players buy simultaneously.
        ok_both, _, _ = attempt(
            "simultaneous buys", lambda: env.step([BUY_WHEAT, BUY_WHEAT])
        )
        details["both_simultaneous"] = ok_both

    # Mixed: a callable for player 0, a raw dict for player 1.
    from .stepping import pass_agent

    ok_mixed, _, _ = attempt(
        "mixed callable+dict", lambda: env.step([pass_agent, NOTHING])
    )
    details["mixed_agent_types"] = ok_mixed

    controllable = bool(ok_p0 and ok_p1)
    details["both_players_controllable"] = controllable

    summary = (
        f"arbitrary actions accepted={ok_p0}; "
        f"both players independently controllable={controllable}; "
        f"p0 seeds after buy={details.get('p0_seeds_after_p0_buy')}"
    )
    logger.info(summary)
    return ProbeResult(
        name="actions",
        success=bool(ok_p0),
        summary=summary,
        details=details,
        duration_s=0.0,
        mcts_verdict=controllable,
    )


if __name__ == "__main__":
    from ..utils import configure_logging, run_probe

    configure_logging()
    result = run_probe("actions", run)
    logger.info("probe result: success=%s summary=%s", result.success, result.summary)
