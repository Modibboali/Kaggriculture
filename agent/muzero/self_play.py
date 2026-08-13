"""MuZero self-play engine.

The real loop:

    real simulator state
        -> MuZero latent MCTS  ->  selected real action
        -> Kaggriculture simulator (the ONLY place the simulator is used)
        -> real next state
        -> store transition
        -> repeat

The latent MCTS never calls the simulator (see :mod:`.puct`). MuZero plays
*both* players with the same network, so the policy / value learn from
self-play. Every transition stores numeric arrays only (observation features,
candidate embeddings, chosen action index, visit-distribution policy target,
scaled cash-delta reward) — no simulator object graphs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..ai.action_generator import ActionGenerator
from ..ai.search_state import SearchState
from ..ai.sim_experiment import initial_state
from ..ai.terminal import Terminal
from ..simulator import GameConfig, Simulator
from .config import MuZeroConfig
from .encoders import ActionEncoder, StateEncoder
from .networks import MuZeroNetwork
from .puct import Decision, MuZeroAgentMCTS
from .replay import Episode, compute_returns


class MuZeroSelfPlay:
    """Runs real-simulator self-play episodes using the latent MCTS."""

    def __init__(
        self,
        net: MuZeroNetwork,
        config: MuZeroConfig,
        game_config: GameConfig | None = None,
    ) -> None:
        self._net = net
        self._config = config
        self._game_config = game_config if game_config is not None else GameConfig(
            episode_steps=config.episode_steps, seed=config.seed
        )
        self._sim = Simulator(self._game_config)
        self._terminal = Terminal(self._game_config)
        # One agent instance serves both players (state.current_player set per view).
        self._agent = MuZeroAgentMCTS(
            net, config, self._game_config, dirichlet=True, temperature=config.temperature
        )
        self._reward_scale = float(config.reward_scale)

    def _view(self, state: SearchState, player: int) -> SearchState:
        if state.game.current_player == player:
            return state
        return SearchState(replace(state.game, current_player=player))

    def _temperature_at(self, step: int) -> float:
        horizon = self._game_config.episode_steps
        if horizon - step <= self._config.temperature_threshold:
            return 0.25
        return self._config.temperature

    def run_episode(self, seed: int | None = None) -> Episode:
        rng = np.random.RandomState(seed if seed is not None else self._config.seed)
        # Fresh search RNG per episode so a fixed seed reproduces the episode.
        self._agent.set_rng(rng)
        state = SearchState(initial_state(self._game_config))
        game_config = self._game_config

        obs0: list[np.ndarray] = []
        embs0: list[np.ndarray] = []
        acts0: list[int] = []
        pi0: list[np.ndarray] = []
        rewards0: list[float] = []
        players0: list[int] = []

        obs1: list[np.ndarray] = []
        embs1: list[np.ndarray] = []
        acts1: list[int] = []
        pi1: list[np.ndarray] = []
        rewards1: list[float] = []
        players1: list[int] = []

        opening_generator = ActionGenerator(game_config)
        state_enc = self._agent.state_encoder
        action_enc = self._agent.action_encoder

        steps = 0
        while not self._terminal.is_terminal(state) and steps < game_config.episode_steps:
            cash0_before = float(state.game.players[0].farm.money)
            cash1_before = float(state.game.players[1].farm.money)

            d0 = self._decide(state, 0, opening_generator, state_enc, action_enc, rng, steps)
            d1 = self._decide(state, 1, opening_generator, state_enc, action_enc, rng, steps)

            next_state = self._sim.apply(state.game, (d0.action, d1.action))
            r0 = (float(next_state.players[0].farm.money) - cash0_before) * self._reward_scale
            r1 = (float(next_state.players[1].farm.money) - cash1_before) * self._reward_scale

            obs0.append(d0.obs)
            embs0.append(d0.action_embs)
            acts0.append(d0.action_index)
            pi0.append(d0.probs)
            rewards0.append(r0)
            players0.append(0)

            obs1.append(d1.obs)
            embs1.append(d1.action_embs)
            acts1.append(d1.action_index)
            pi1.append(d1.probs)
            rewards1.append(r1)
            players1.append(1)

            state = SearchState(next_state)
            steps += 1

        # Returns must be per-player suffix sums (player 0's future cash, then
        # player 1's), aligned with the concatenated transition order below.
        rets0 = compute_returns(np.asarray(rewards0, dtype=np.float32), self._config.gamma)
        rets1 = compute_returns(np.asarray(rewards1, dtype=np.float32), self._config.gamma)
        returns = np.concatenate([rets0, rets1])
        rewards = np.asarray(rewards0 + rewards1, dtype=np.float32)
        return Episode(
            obs=np.asarray(obs0 + obs1, dtype=np.float32),
            candidate_embs=embs0 + embs1,
            action_indices=np.asarray(acts0 + acts1, dtype=np.int32),
            policy_targets=pi0 + pi1,
            rewards=rewards,
            returns=returns,
            players=np.asarray(players0 + players1, dtype=np.int8),
            terminal=True,
        )

    def _decide(
        self,
        state: SearchState,
        player: int,
        opening_generator: ActionGenerator,
        state_enc: StateEncoder,
        action_enc: ActionEncoder,
        rng: np.random.RandomState,
        step: int,
    ) -> Decision:
        view = self._view(state, player)
        if self._config.random_openings > 0 and step < self._config.random_openings:
            actions = list(opening_generator.generate(view))
            idx = int(rng.randint(len(actions)))
            return Decision(
                action=actions[idx],
                probs=np.ones(len(actions), dtype=np.float32) / len(actions),
                action_index=idx,
                obs=state_enc.encode(view),
                action_embs=np.stack([action_enc.encode(view, a) for a in actions]),
                actions=tuple(actions),
            )
        self._agent.set_temperature(self._temperature_at(step))
        return self._agent.decide(view)
