"""Benchmark: the lightweight simulator must be faster than the full engine.

Smoke-level performance check: forwarding one turn through the simulator must
be strictly faster than stepping the official Kaggle environment (which
re-serializes observations every step). This is what makes the simulator
useful as a roll-out model for future tree search.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("kaggle_environments")

from agent.actions import TurnAction  # noqa: E402
from agent.environment.kaggle_action_serializer import to_kaggle_action  # noqa: E402
from agent.simulator import Simulator  # noqa: E402
from agent.testing.differential import DifferentialRunner, SCENARIO_PASS  # noqa: E402

pytestmark = pytest.mark.integration

SIM_STEPS = 2000
KAGGLE_STEPS = 100


def _simulator_rate(steps: int = SIM_STEPS) -> float:
    simulator = Simulator()
    state = DifferentialRunner().run_kaggle(SCENARIO_PASS).initial_state
    action = TurnAction()
    start = time.perf_counter()
    for _ in range(steps):
        state = simulator.apply(state, action)
    elapsed = time.perf_counter() - start
    return steps / elapsed


def _kaggle_rate(steps: int = KAGGLE_STEPS) -> float:
    import kaggle_environments

    env = kaggle_environments.make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": 1},
        debug=True,
    )
    action = [to_kaggle_action(TurnAction()), to_kaggle_action(TurnAction())]
    start = time.perf_counter()
    for _ in range(steps):
        env.step(action)
    elapsed = time.perf_counter() - start
    return steps / elapsed


def test_simulator_is_faster_than_kaggle() -> None:
    sim_rate = _simulator_rate()
    kaggle_rate = _kaggle_rate()
    assert sim_rate > kaggle_rate, (
        f"simulator {sim_rate:.1f} turns/s should beat Kaggle "
        f"{kaggle_rate:.1f} turns/s"
    )
