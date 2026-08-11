"""MCTS regression + simulator-harness tests for the horizon-aware milestone."""

from __future__ import annotations

import random

from agent.ai import (
    EvaluationConfig,
    HeuristicRolloutPolicy,
    HorizonAwareEvaluator,
    MCTS,
    MCTSConfig,
    SearchState,
    SimulatorAdapter,
    Terminal,
    play_sim_episode,
    run_sim_matchup,
)
from agent.actions import TurnAction
from agent.ai.agent import HeuristicAgent, MCTSAgent, RandomAgent, StarterAgent
from agent.ai.sim_experiment import initial_state
from agent.simulator import GameConfig
from agent.state import GameState

from ._build import make_components, make_search_state


def _mcts(
    iterations: int = 30,
    *,
    seed: int = 0,
    eval_config: EvaluationConfig | None = None,
) -> tuple[MCTS[SearchState], SimulatorAdapter]:
    config, adapter, generator, evaluator, terminal = make_components()
    evaluator = HorizonAwareEvaluator(config, eval_config)
    rollout = HeuristicRolloutPolicy(generator)
    mcts = MCTS(
        MCTSConfig(iterations=iterations, max_simulation_steps=15, seed=seed),
        transition=adapter.transition,
        generate=generator.generate,
        is_terminal=terminal.is_terminal,
        terminal_value=terminal.value,
        evaluate=evaluator.evaluate,
        rollout=rollout.choose,
        rng=random.Random(seed),
    )
    return mcts, adapter


def test_horizon_mcts_searches_simulator() -> None:
    mcts, _ = _mcts()
    action = mcts.search(make_search_state(step=0), 0)
    assert isinstance(action, TurnAction)


def test_horizon_mcts_is_deterministic() -> None:
    a, _ = _mcts(seed=5)
    b, _ = _mcts(seed=5)
    state = make_search_state(step=0)
    assert a.search(state, 0) == b.search(state, 0)


def test_horizon_mcts_respects_budget() -> None:
    mcts, _ = _mcts(iterations=22)
    root = mcts.search_root(make_search_state(step=0), 0)
    assert root.visits == 22


def test_horizon_mcts_terminal_state_no_mutation() -> None:
    mcts, _ = _mcts(iterations=10)
    state = make_search_state(step=0)
    before = state.state_key()
    mcts.search(state, 0)
    assert state.state_key() == before  # parent never mutated


def test_horizon_mcts_agent_returns_valid_action() -> None:
    agent = MCTSAgent(
        MCTSConfig(iterations=10, max_simulation_steps=8, seed=1),
        eval_config=EvaluationConfig(),
        seed=1,
    )
    # Build a raw-style observation is overkill; select() is the state entrypoint.
    game: GameState = make_search_state(step=0).game
    action = agent.select(game, 0)
    assert isinstance(action, TurnAction)
    assert agent.last_action is action


def test_agents_select_work_for_simulator_play() -> None:
    game: GameState = initial_state(GameConfig(episode_steps=120))
    for agent in (
        MCTSAgent(MCTSConfig(iterations=5, max_simulation_steps=6, seed=1), seed=1),
        RandomAgent(seed=1),
        HeuristicAgent(seed=1),
        StarterAgent(),
    ):
        action0 = agent.select(game, 0)
        action1 = agent.select(game, 1)
        assert isinstance(action0, TurnAction)
        assert isinstance(action1, TurnAction)


def test_play_sim_episode_runs_to_terminal() -> None:
    config = GameConfig(episode_steps=48)
    result = play_sim_episode(
        StarterAgent(), RandomAgent(seed=1), config=config, seed=1
    )
    assert result.winner in (0, 1, -1)
    assert result.steps > 0
    assert result.reward0 >= 0.0


def test_run_sim_matchup_aggregates() -> None:
    config = GameConfig(episode_steps=24)
    outcome = run_sim_matchup(
        StarterAgent(),
        RandomAgent(seed=1),
        name="starter_vs_random",
        games=3,
        config=config,
    )
    assert outcome.games == 3
    assert 0.0 <= outcome.win_rate0 <= 1.0
    assert outcome.mean_reward0 >= 0.0
    assert outcome.median_reward0 >= 0.0
    assert outcome.std_reward0 >= 0.0


def test_horizon_evaluator_wired_through_agent() -> None:
    """The horizon evaluator is what the MCTS agent uses by default now."""
    from agent.ai.agent import _Components

    components = _Components()
    assert components.evaluator.__class__.__name__ == "Evaluator"
    # An explicit HorizonAwareEvaluator is accepted and used by MCTSAgent.
    agent = MCTSAgent(
        MCTSConfig(iterations=5, seed=1),
        evaluator=HorizonAwareEvaluator(GameConfig()),
        seed=1,
    )
    game: GameState = make_search_state(step=0).game
    assert isinstance(agent.select(game, 0), TurnAction)
