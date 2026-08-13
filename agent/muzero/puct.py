"""MuZero PUCT search entirely in latent space.

Hard architectural invariant: the tree stores **latent states** and search
statistics — never ``GameState`` objects — and hypothetical node expansion uses
only the learned dynamics / prediction networks. The Kaggriculture simulator
is never imported here and never called to expand a hypothetical node.

Because Kaggriculture's candidate set is variable, the search uses a
*per-search candidate vocabulary*: the root's real candidate action embeddings
are the action set for the whole latent tree. The policy head scores those
candidates at every latent node (candidate-conditioned, no fixed global action
vocabulary). This is the same mechanism used in training, self-play, MCTS and
inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..ai.search_state import SearchState
from ..actions import TurnAction
from ..simulator import GameConfig
from .config import MuZeroConfig
from .encoders import ActionEncoder, StateEncoder
from .networks import MuZeroNetwork


@dataclass
class PUCTNode:
    """One latent-tree node (search statistics, no GameState)."""

    latent: torch.Tensor | None  # latent of the state at this node
    reward: float  # reward on the edge from parent -> this node (root: 0)
    prior: float  # policy prior for this node's action (from parent expansion)
    action_index: int | None  # index into the search's candidate set
    visits: int = 0
    value_sum: float = 0.0
    children: list["PUCTNode"] | None = None  # None until expanded

    def expanded(self) -> bool:
        return self.children is not None and len(self.children) > 0

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


@dataclass(frozen=True, slots=True)
class SearchStats:
    simulations: int
    avg_depth: float
    policy_entropy: float
    root_visits: tuple[int, ...]


class MuZeroPUCT:
    """PUCT search over the learned latent dynamics (no simulator calls)."""

    def __init__(self, net: MuZeroNetwork, config: MuZeroConfig) -> None:
        self._net = net
        self._config = config
        self._c_puct = config.c_puct
        self._gamma = config.gamma

    # -- public API ---------------------------------------------------------

    def search(
        self,
        obs: np.ndarray,
        candidate_embs: np.ndarray,
        *,
        num_simulations: int | None = None,
        dirichlet: bool = False,
        rng: np.random.RandomState | None = None,
    ) -> tuple[np.ndarray, int, SearchStats]:
        """Run PUCT from ``obs`` over ``candidate_embs``.

        Returns ``(visit_distribution (N,), best_action_index, stats)``.
        ``dirichlet`` adds root exploration noise (training self-play only).
        """
        rng = rng if rng is not None else np.random.RandomState(0)
        num_simulations = num_simulations if num_simulations is not None else self._config.simulations
        N = candidate_embs.shape[0]
        if N == 0:
            raise ValueError("PUCT requires at least one candidate action")

        with torch.no_grad():
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)[None])
            aembs_t = torch.from_numpy(np.asarray(candidate_embs, dtype=np.float32)[None])
            root_latent, root_logits, root_value = self._net.initial_inference(obs_t, aembs_t)

        priors = _softmax(np.asarray(root_logits[0].cpu().numpy(), dtype=np.float32))
        if dirichlet:
            alpha = float(self._config.dirichlet_alpha)
            noise = rng.dirichlet([alpha] * N).astype(np.float32)
            eps = float(self._config.dirichlet_epsilon)
            priors = (1.0 - eps) * priors + eps * noise

        root = PUCTNode(latent=root_latent[0], reward=0.0, prior=0.0, action_index=None)
        root.children = [
            PUCTNode(latent=None, reward=0.0, prior=float(p), action_index=i)
            for i, p in enumerate(priors)
        ]

        depths: list[int] = []
        for _ in range(int(num_simulations)):
            node = root
            path: list[PUCTNode] = [root]
            while node.expanded():
                node = self._select_child(node)
                path.append(node)
            # ``node`` is now an unexpanded leaf; expand it with dynamics + prediction.
            parent = path[-2]
            parent_latent = parent.latent
            if parent_latent is None:
                raise RuntimeError("parent latent not computed before expansion")
            if node.action_index is None:
                raise RuntimeError("leaf action_index missing")
            with torch.no_grad():
                next_latent, reward = self._net.recurrent_inference(
                    parent_latent[None],
                    torch.from_numpy(np.asarray(candidate_embs[node.action_index], dtype=np.float32)[None]),
                )
                logits = self._net.policy_at(next_latent, aembs_t)
                value_t = self._net.value_at(next_latent)
            node.latent = next_latent[0]
            node.reward = float(reward[0].cpu())
            child_priors = _softmax(np.asarray(logits[0].cpu().numpy(), dtype=np.float32))
            node.children = [
                PUCTNode(latent=None, reward=0.0, prior=float(p), action_index=i)
                for i, p in enumerate(child_priors)
            ]
            leaf_value = float(value_t[0].cpu())
            self._backup(path, leaf_value)
            depths.append(len(path))

        counts = np.array([c.visits for c in root.children], dtype=np.float32)
        if counts.sum() <= 0:
            counts = np.ones_like(counts)
        probs = counts / counts.sum()
        best = int(np.argmax(counts))
        eps = 1e-8
        entropy = float(-(probs * np.log(probs + eps)).sum())
        stats = SearchStats(
            simulations=int(num_simulations),
            avg_depth=float(np.mean(depths)) if depths else 0.0,
            policy_entropy=entropy,
            root_visits=tuple(int(c.visits) for c in root.children),
        )
        return probs, best, stats

    # -- internals ----------------------------------------------------------

    def _select_child(self, node: PUCTNode) -> PUCTNode:
        best_score = float("-inf")
        best: PUCTNode | None = None
        for child in node.children or []:
            q = child.value()
            score = q + self._c_puct * child.prior * math.sqrt(node.visits) / (1.0 + child.visits)
            if score > best_score:
                best_score = score
                best = child
        if best is None:
            raise RuntimeError("select_child called on a node with no children")
        return best

    def _backup(self, path: list[PUCTNode], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = node.reward + self._gamma * value


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return np.asarray(e / e.sum(), dtype=np.float32)


class Decision:
    """A real decision from the latent MCTS (self-play / evaluation)."""

    __slots__ = ("action", "probs", "action_index", "obs", "action_embs", "actions")

    def __init__(
        self,
        action: TurnAction,
        probs: np.ndarray,
        action_index: int,
        obs: np.ndarray,
        action_embs: np.ndarray,
        actions: tuple[TurnAction, ...],
    ) -> None:
        self.action = action
        self.probs = probs
        self.action_index = action_index
        self.obs = obs
        self.action_embs = action_embs
        self.actions = actions


class MuZeroAgentMCTS:
    """Bridges the latent PUCT to a real decision for a SearchState.

    Encodes the state + candidates (the *same* encoders as training), runs
    PUCT, and returns the chosen real ``TurnAction``. Used by self-play and
    evaluation. This is where the real simulator boundary lives: the MCTS
    itself never touches a GameState.
    """

    def __init__(
        self,
        net: MuZeroNetwork,
        config: MuZeroConfig,
        game_config: GameConfig | None = None,
        *,
        dirichlet: bool = False,
        temperature: float = 1.0,
    ) -> None:
        self._net = net
        self._config = config
        self._game_config = game_config if game_config is not None else GameConfig()
        self._state_enc = StateEncoder(self._game_config)
        self._action_enc = ActionEncoder(self._game_config)
        self._puct = MuZeroPUCT(net, config)
        self._dirichlet = dirichlet
        self._temperature = temperature
        from ..ai.action_generator import ActionGenerator
        from ..ai.action_priority import ActionPriorityModel

        # Task-12 prior structure: keep the hard realizability filter (feasible
        # actions only) but NOT the hand-crafted priority ordering — the learned
        # policy determines the relative prior among feasible candidates.
        self._generator = ActionGenerator(
            self._game_config,
            action_filter=ActionPriorityModel(self._game_config).filter_realizable,
        )
        self._rng = np.random.RandomState(config.seed)
        self.last_stats: SearchStats | None = None
        self._simulations = int(config.simulations)

    @property
    def state_encoder(self) -> StateEncoder:
        return self._state_enc

    @property
    def action_encoder(self) -> ActionEncoder:
        return self._action_enc

    def set_temperature(self, temperature: float) -> None:
        self._temperature = float(temperature)

    def set_simulations(self, simulations: int) -> None:
        self._simulations = int(simulations)

    def set_rng(self, rng: np.random.RandomState) -> None:
        """Replace the search RNG (used to make each self-play episode
        deterministic given its own seed)."""
        self._rng = rng

    def decide(self, state: SearchState) -> Decision:
        """Return the chosen action plus obs / embeddings / distribution."""
        actions = list(self._generator.generate(state))
        if not actions:
            from ..actions import TurnAction as _TA

            return Decision(
                action=_TA(),
                probs=np.ones(1, dtype=np.float32),
                action_index=0,
                obs=self._state_enc.encode(state),
                action_embs=np.zeros((1, self._action_enc.width), dtype=np.float32),
                actions=(_TA(),),
            )
        embs = np.stack([self._action_enc.encode(state, a) for a in actions])
        obs = self._state_enc.encode(state)
        probs, best, stats = self._puct.search(
            obs, embs, num_simulations=self._simulations, dirichlet=self._dirichlet, rng=self._rng
        )
        self.last_stats = stats
        chosen = self._select_by_temperature(probs)
        return Decision(
            action=actions[chosen],
            probs=probs,
            action_index=chosen,
            obs=obs,
            action_embs=embs,
            actions=tuple(actions),
        )

    def _select_by_temperature(self, probs: np.ndarray) -> int:
        temp = self._temperature
        if temp <= 1e-6:
            return int(np.argmax(probs))
        logits = np.log(probs + 1e-8) / temp
        p = _softmax(logits)
        return int(self._rng.choice(len(p), p=p))
