"""Rollout policies for MCTS simulation.

``RandomRolloutPolicy`` is the sanity baseline. ``HeuristicRolloutPolicy``
prefers obviously productive actions (harvest, collect, water, feed, plant,
sell, buy seeds, ...) using a simple priority table over the generated
candidates, falling back to a random pick among ties. Both stay lightweight
and understandable; neither encodes sophisticated strategy.

Each policy holds the :class:`ActionGenerator` it samples from, so MCTS only
sees ``choose(state, rng)``.
"""

from __future__ import annotations

import random
from typing import Protocol

from ..actions import ActionType, TurnAction
from .action_generator import ActionGenerator
from .search_state import SearchState


class RolloutPolicy(Protocol):
    """Chooses the next action during a rollout."""

    def choose(self, state: SearchState, rng: random.Random) -> TurnAction: ...


class RandomRolloutPolicy:
    """Uniformly random action from the generator."""

    def __init__(self, generator: ActionGenerator) -> None:
        self._generator = generator

    def choose(self, state: SearchState, rng: random.Random) -> TurnAction:
        return rng.choice(self._generator.generate(state))


# Heuristic preference for productive action types (higher = preferred).
_HEURISTIC_PRIORITY: dict[ActionType, int] = {
    ActionType.HARVEST: 100,
    ActionType.COLLECT_FERTILIZER: 95,
    ActionType.WATER: 90,
    ActionType.FEED: 85,
    ActionType.CARE: 80,
    ActionType.PLANT: 75,
    ActionType.SELL: 70,
    ActionType.FERTILIZE: 65,
    ActionType.BUY_SEED: 60,
    ActionType.BUY_LAND: 55,
    ActionType.HIRE: 50,
    ActionType.BUY_ANIMAL: 50,
    ActionType.PLACE: 45,
    ActionType.BUILD_COOP: 40,
    ActionType.BUILD_PASTURE: 40,
    ActionType.PICKUP: 35,
    ActionType.DROP: 35,
    ActionType.DIG: 30,
    ActionType.NORTH: 5,
    ActionType.SOUTH: 5,
    ActionType.EAST: 5,
    ActionType.WEST: 5,
    ActionType.PASS: 0,
}


class HeuristicRolloutPolicy:
    """Prefers productive actions; ties broken randomly."""

    def __init__(self, generator: ActionGenerator) -> None:
        self._generator = generator
        self._priority = _HEURISTIC_PRIORITY

    def choose(self, state: SearchState, rng: random.Random) -> TurnAction:
        actions = self._generator.generate(state)
        best_priority = -1
        best: list[TurnAction] = []
        for action in actions:
            priority = self._priority.get(self._farmer_type(action), 0)
            if priority > best_priority:
                best_priority = priority
                best = [action]
            elif priority == best_priority:
                best.append(action)
        return rng.choice(best)

    @staticmethod
    def _farmer_type(action: TurnAction) -> ActionType:
        if action.market_actions:
            return action.market_actions[0].action_type
        return action.farmer_action.action_type
