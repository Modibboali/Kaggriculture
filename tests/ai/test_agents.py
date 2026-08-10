"""Agent and experiment integration tests (require the real Kaggle env)."""

from __future__ import annotations

from typing import Any, cast

import pytest

pytest.importorskip("kaggle_environments")

import kaggle_environments  # noqa: E402

from agent.ai import (  # noqa: E402
    HeuristicAgent,
    MCTSConfig,
    MCTSAgent,
    RandomAgent,
    StarterAgent,
    play_episode,
    run_matchup,
)


def _initial_observation() -> dict[str, Any]:
    env = kaggle_environments.make(
        "kaggriculture", configuration={"episodeSteps": 720, "seed": 1}, debug=False
    )
    return cast(dict[str, Any], env.state[0]["observation"])


def test_mcts_agent_returns_valid_action() -> None:
    agent = MCTSAgent(MCTSConfig(iterations=10, seed=1))
    action = agent.choose(_initial_observation())
    assert isinstance(action, dict)
    assert "farmer" in action and "market" in action
    assert agent.last_action is not None


def test_baseline_agents_return_valid_actions() -> None:
    for agent in (RandomAgent(seed=1), HeuristicAgent(seed=1), StarterAgent()):
        action = agent.choose(_initial_observation())
        assert isinstance(action, dict)
        assert "farmer" in action


def test_play_episode_runs() -> None:
    result = play_episode(
        StarterAgent(), RandomAgent(seed=1), episode_steps=12, seed=1
    )
    assert result.winner in (0, 1, -1)
    assert result.reward0 >= 0.0
    assert result.steps > 0


def test_run_matchup_aggregates() -> None:
    outcome = run_matchup(
        StarterAgent(),
        RandomAgent(seed=1),
        name="starter_vs_random",
        games=2,
        episode_steps=12,
        seed=1,
    )
    assert outcome.games == 2
    assert 0.0 <= outcome.win_rate0 <= 1.0
    assert outcome.avg_reward0 >= 0.0
