"""Agents that play Kaggriculture.

An :class:`Agent` maps a raw Kaggle observation to a raw action dict. The
observation boundary (adapter -> SearchState) and the output boundary
(action -> Kaggle dict) live here; the MCTS / rollout layers never touch the
Kaggle environment. This module also defines the classical baselines used by
the experiment harness: random, greedy-heuristic, and a minimal starter.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Protocol

from ..actions import TurnAction
from ..environment import KaggleObservationAdapter, to_kaggle_action
from ..simulator import GameConfig
from ..state import GameState
from .action_generator import ActionGenerator
from .evaluation import EvaluationConfig, Evaluator, HorizonAwareEvaluator
from .mcts import MCTS, MCTSConfig
from .rollout import HeuristicRolloutPolicy, RandomRolloutPolicy
from .search_state import SearchState
from .simulator_adapter import SimulatorAdapter
from .terminal import Terminal


class Agent(Protocol):
    """Chooses an action dict from a raw observation (and can select from a
    domain state directly for simulator-based play)."""

    def choose(self, observation: dict[str, Any]) -> dict[str, Any]: ...

    def select(self, game: GameState, player: int) -> TurnAction: ...


def _as_player(game: GameState, player: int) -> GameState:
    """The same state with ``current_player`` set to ``player`` (no copy when
    it already matches). Search transitions act on ``current_player``, so each
    agent must see the state from its own perspective."""
    if game.current_player == player:
        return game
    return replace(game, current_player=player)


class _Components:
    """Shared search components for one agent."""

    def __init__(
        self,
        config: GameConfig | None = None,
        evaluator: Evaluator | HorizonAwareEvaluator | None = None,
    ) -> None:
        self.config = config if config is not None else GameConfig()
        self.adapter = SimulatorAdapter()
        self.generator = ActionGenerator(self.config)
        self.evaluator = evaluator if evaluator is not None else Evaluator(self.config)
        self.terminal = Terminal(self.config)


class MCTSAgent:
    """Plays using UCT MCTS with a configurable rollout policy."""

    def __init__(
        self,
        mcts_config: MCTSConfig | None = None,
        *,
        config: GameConfig | None = None,
        rollout: Any = None,
        evaluator: Evaluator | HorizonAwareEvaluator | None = None,
        eval_config: EvaluationConfig | None = None,
        seed: int = 0,
    ) -> None:
        self._components = _Components(config, evaluator)
        if evaluator is None and eval_config is not None:
            self._components.evaluator = HorizonAwareEvaluator(self._components.config, eval_config)
        self._components.adapter = SimulatorAdapter(count_transitions=True)
        mcts_config = mcts_config if mcts_config is not None else MCTSConfig(seed=seed)
        if rollout is None:
            rollout = HeuristicRolloutPolicy(self._components.generator)
        self._mcts = MCTS(
            mcts_config,
            transition=self._components.adapter.transition,
            generate=self._components.generator.generate,
            is_terminal=self._components.terminal.is_terminal,
            terminal_value=self._components.terminal.value,
            evaluate=self._components.evaluator.evaluate,
            rollout=rollout.choose,
            rng=random.Random(mcts_config.seed),
        )
        self._iterations = mcts_config.iterations
        self._search_time = 0.0
        self._actions_chosen = 0

    @property
    def last_action(self) -> TurnAction | None:
        """The action chosen on the most recent ``choose`` call."""
        return self._last_action

    @property
    def stats(self) -> dict[str, float]:
        """Cumulative search statistics (time, transitions, searches)."""
        return {
            "searches": float(self._actions_chosen),
            "search_time": self._search_time,
            "simulator_transitions": float(self._components.adapter.transitions),
        }

    def select(self, game: GameState, player: int) -> TurnAction:
        """Search from a domain state (no observation parsing)."""
        import time as _time

        game = _as_player(game, player)
        state = SearchState(game)
        start = _time.perf_counter()
        action = self._mcts.search(state, player)
        self._search_time += _time.perf_counter() - start
        self._actions_chosen += 1
        self._last_action = action
        return action

    def choose(self, observation: dict[str, Any]) -> dict[str, Any]:
        game = KaggleObservationAdapter.from_observation(observation)
        return to_kaggle_action(self.select(game, game.current_player))


class RandomAgent:
    """Uniformly random action from the generator."""

    def __init__(self, *, config: GameConfig | None = None, seed: int = 0) -> None:
        self._components = _Components(config)
        self._rng = random.Random(seed)

    def select(self, game: GameState, player: int) -> TurnAction:
        game = _as_player(game, player)
        state = SearchState(game)
        return self._rng.choice(self._components.generator.generate(state))

    def choose(self, observation: dict[str, Any]) -> dict[str, Any]:
        game = KaggleObservationAdapter.from_observation(observation)
        return to_kaggle_action(self.select(game, game.current_player))


class HeuristicAgent:
    """Greedy: always take the heuristic rollout policy's preferred action."""

    def __init__(self, *, config: GameConfig | None = None, seed: int = 0) -> None:
        self._components = _Components(config)
        self._rng = random.Random(seed)
        self._policy = HeuristicRolloutPolicy(self._components.generator)

    def select(self, game: GameState, player: int) -> TurnAction:
        game = _as_player(game, player)
        state = SearchState(game)
        return self._policy.choose(state, self._rng)

    def choose(self, observation: dict[str, Any]) -> dict[str, Any]:
        game = KaggleObservationAdapter.from_observation(observation)
        return to_kaggle_action(self.select(game, game.current_player))


class StarterAgent:
    """A minimal fixed strategy: harvest, water, plant, drop, sell, buy seed."""

    def __init__(self, *, config: GameConfig | None = None, seed: int = 0) -> None:
        del seed
        self._components = _Components(config)

    def select(self, game: GameState, player: int) -> TurnAction:
        from ..actions import (
            BuySeedAction,
            HarvestAction,
            PlantAction,
            SellAction,
            TurnAction,
            WaterAction,
        )
        from ..state import CropType, EmptyTile, ItemType, PlantTile

        ps = game.players[player]
        farm = ps.farm
        tile = farm.tile_at(farm.farmer.position)

        if isinstance(tile, PlantTile):
            if not tile.plant.watered_today:
                return TurnAction(farmer_action=WaterAction())
            spec = self._components.config.crops[tile.plant.crop]
            if tile.plant.yield_units > 0 and game.day - tile.plant.planted_day >= spec.first_yield_day:
                return TurnAction(farmer_action=HarvestAction())
        if isinstance(tile, EmptyTile) and ps.seeds.get(CropType.WHEAT) > 0:
            return TurnAction(farmer_action=PlantAction(crop=CropType.WHEAT))
        if ps.inventory.get(ItemType.WHEAT) > 0:
            return TurnAction(market_actions=(SellAction(item=ItemType.WHEAT, quantity=1),))
        if farm.money >= self._components.config.crops[CropType.WHEAT].seed_cost:
            return TurnAction(market_actions=(BuySeedAction(crop=CropType.WHEAT, quantity=1),))
        return TurnAction()

    def choose(self, observation: dict[str, Any]) -> dict[str, Any]:
        game = KaggleObservationAdapter.from_observation(observation)
        return to_kaggle_action(self.select(game, game.current_player))
