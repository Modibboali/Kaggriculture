"""Differential runner.

Executes a scenario twice: once on the official Kaggriculture environment
(ground truth) and once on a ``Simulator`` (our future lightweight transition
engine). Both sides start from the *same* initial state, derived from the
environment's step-0 observation. Each turn's resulting states are normalized
and compared path-by-path with :mod:`agent.testing.differential.state_diff`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ...actions import TurnAction
from ...environment import KaggleObservationAdapter
from ...environment.kaggle_action_serializer import to_kaggle_action
from ...state import GameState
from .observation_normalizer import CanonicalState, normalize
from .scenarios import Scenario
from .state_diff import StateDiff, diff

GAME = "kaggriculture"

# The official environment's own RNG (weeds, market, town) is non-deterministic
# unless seeded, so a fixed seed makes differential comparisons reproducible.
DEFAULT_CONFIG: dict[str, Any] = {"episodeSteps": 720, "seed": 1}


class Simulator(Protocol):
    """Transition interface a future lightweight simulator must implement.

    ``apply`` advances the state by one full step given both players' actions
    (Kaggriculture is a simultaneous-move game). The framework only calls this
    method and never inspects simulator internals, keeping the simulator fully
    environment-independent.
    """

    def apply(
        self,
        state: GameState,
        actions: tuple[TurnAction, TurnAction],
    ) -> GameState:
        """Return the state after applying one step of ``actions``."""
        ...


@dataclass(frozen=True, slots=True)
class TurnTrace:
    """One step of a run: the resulting state and its canonical form."""

    step: int
    state: GameState
    canonical: CanonicalState


@dataclass(frozen=True, slots=True)
class KaggleRun:
    """Ground-truth trace from the official Kaggriculture environment."""

    scenario: str
    initial_observation: dict[str, Any]
    initial_state: GameState
    turns: tuple[TurnTrace, ...]


@dataclass(frozen=True, slots=True)
class SimulatorRun:
    """Trace produced by the (future) simulator."""

    scenario: str
    turns: tuple[TurnTrace, ...]


@dataclass(frozen=True, slots=True)
class TurnComparison:
    """The per-turn comparison of one step."""

    step: int
    actions: tuple[TurnAction, TurnAction]
    diff: StateDiff


@dataclass(frozen=True, slots=True)
class DifferentialResult:
    """The outcome of running a scenario on both systems and comparing."""

    scenario: str
    comparisons: tuple[TurnComparison, ...]

    @property
    def has_mismatch(self) -> bool:
        """Whether any turn differed between Kaggle and the simulator."""
        return any(not comparison.diff.matches for comparison in self.comparisons)

    @property
    def mismatched_turns(self) -> tuple[int, ...]:
        """The 1-based turn numbers that differed."""
        return tuple(
            comparison.step
            for comparison in self.comparisons
            if not comparison.diff.matches
        )


class DifferentialRunner:
    """Runs a scenario on Kaggle and on a simulator, then compares the states."""

    def __init__(
        self,
        *,
        adapter: type[KaggleObservationAdapter] = KaggleObservationAdapter,
        configuration: Mapping[str, Any] | None = None,
    ) -> None:
        self._adapter = adapter
        self._configuration = dict(
            DEFAULT_CONFIG if configuration is None else configuration
        )

    def run_kaggle(self, scenario: Scenario) -> KaggleRun:
        """Execute ``scenario`` on the official environment (ground truth)."""
        import kaggle_environments

        env = kaggle_environments.make(
            GAME, configuration=self._configuration, debug=True
        )
        initial_observation: dict[str, Any] = env.state[0]["observation"]
        initial_state = self._adapter.from_observation(initial_observation)

        turns: list[TurnTrace] = []
        for step, actions in enumerate(scenario.actions, start=1):
            env.step([to_kaggle_action(actions[0]), to_kaggle_action(actions[1])])
            observation: dict[str, Any] = env.state[0]["observation"]
            state = self._adapter.from_observation(observation)
            turns.append(TurnTrace(step=step, state=state, canonical=normalize(state)))
        return KaggleRun(
            scenario=scenario.name,
            initial_observation=initial_observation,
            initial_state=initial_state,
            turns=tuple(turns),
        )

    def run_simulator(
        self,
        scenario: Scenario,
        simulator: Simulator,
        initial_state: GameState,
    ) -> SimulatorRun:
        """Execute ``scenario`` on ``simulator`` starting from ``initial_state``."""
        state = initial_state
        turns: list[TurnTrace] = []
        for step, actions in enumerate(scenario.actions, start=1):
            state = simulator.apply(state, actions)
            turns.append(TurnTrace(step=step, state=state, canonical=normalize(state)))
        return SimulatorRun(scenario=scenario.name, turns=tuple(turns))

    def compare(self, scenario: Scenario, simulator: Simulator) -> DifferentialResult:
        """Run ``scenario`` on Kaggle and on ``simulator``, then diff each turn."""
        kaggle = self.run_kaggle(scenario)
        simulated = self.run_simulator(scenario, simulator, kaggle.initial_state)

        comparisons: list[TurnComparison] = []
        for kaggle_turn, sim_turn, actions in zip(
            kaggle.turns, simulated.turns, scenario.actions
        ):
            comparisons.append(
                TurnComparison(
                    step=kaggle_turn.step,
                    actions=actions,
                    diff=diff(kaggle_turn.canonical, sim_turn.canonical),
                )
            )
        return DifferentialResult(scenario=scenario.name, comparisons=tuple(comparisons))


def compare(scenario: Scenario, simulator: Simulator) -> DifferentialResult:
    """Convenience: run ``scenario`` on Kaggle and ``simulator``, then compare."""
    return DifferentialRunner().compare(scenario, simulator)
