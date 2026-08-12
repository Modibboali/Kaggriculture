"""Monte Carlo Tree Search (UCT / UCB1).

Standard four-stage MCTS:

    Selection      - descend with UCB1 while a node is fully expanded
    Expansion      - pop one untried action, create the child node
    Simulation     - rollout from the new child to a budget/terminal
    Backpropagation- accumulate the returned value up the path

The class is generic over the state type ``S`` and takes its model as simple
``Callable``s (transition / generate / terminal / evaluate / rollout). For
Kaggriculture ``S`` is :class:`~agent.ai.search_state.SearchState` and the
transition callable is the simulator (via ``SimulatorAdapter``). The generic
typing lets the algorithm be verified against a tiny synthetic environment
with no game rules at all.

``MCTSConfig`` holds the deterministic iteration budget and the exploration
constant ``c``. Transposition support is isolated behind
:class:`TranspositionTable`; the default :class:`NoTranspositionTable` keeps
the tree semantics simple and correct.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Generic, Protocol, TypeVar

from ..actions import TurnAction

S = TypeVar("S")


class MCTSConfig:
    """Search budget and exploration parameters (plain data).

    ``iterations`` is the total simulation budget for one search. ``workers``
    controls the execution strategy: ``1`` runs the canonical sequential MCTS
    in-process; ``>1`` runs root-parallel MCTS across that many worker
    processes (see :mod:`agent.ai.parallel_mcts`). The sequential
    :class:`MCTS` ignores ``workers`` entirely, so this field never changes
    the mathematical behaviour of the reference implementation.
    """

    __slots__ = (
        "iterations",
        "exploration_constant",
        "max_simulation_steps",
        "seed",
        "workers",
    )

    def __init__(
        self,
        *,
        iterations: int = 300,
        exploration_constant: float = 1.41,
        max_simulation_steps: int = 30,
        seed: int = 0,
        workers: int = 1,
    ) -> None:
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_simulation_steps = max_simulation_steps
        self.seed = seed
        self.workers = workers


class MCTSNode(Generic[S]):
    """One search node. ``state`` is the state at this node.

    Memory is a concern (millions of nodes eventually), so we use ``__slots__``
    and only materialize ``untried_actions`` lazily. ``action`` is the action
    taken from the parent to reach this node.
    """

    __slots__ = ("state", "parent", "action", "visits", "total_value", "children", "untried_actions", "terminal")

    def __init__(
        self,
        state: S,
        parent: "MCTSNode[S] | None" = None,
        action: TurnAction | None = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.visits = 0
        self.total_value = 0.0
        self.children: list[MCTSNode[S]] = []
        self.untried_actions: list[TurnAction] | None = None
        self.terminal = False


class TranspositionTable(Protocol):
    """Interface for an optional transposition table (state key -> stats).

    Not wired into the tree search yet; provided so transposition support can
    be added without changing the tree semantics.
    """

    def lookup(self, key: int) -> MCTSNode[Any] | None: ...
    def store(self, key: int, node: MCTSNode[Any]) -> None: ...


class NoTranspositionTable:
    """Default no-op table (keeps the tree fully explicit)."""

    def lookup(self, key: int) -> MCTSNode[Any] | None:
        del key
        return None

    def store(self, key: int, node: MCTSNode[Any]) -> None:
        del key, node


class MCTS(Generic[S]):
    """UCT/UCB1 Monte Carlo Tree Search over a generic forward model."""

    def __init__(
        self,
        config: MCTSConfig,
        *,
        transition: Callable[[S, TurnAction], S],
        generate: Callable[[S], tuple[TurnAction, ...]],
        is_terminal: Callable[[S], bool],
        terminal_value: Callable[[S, int], float],
        evaluate: Callable[[S, int], float],
        rollout: Callable[[S, random.Random], TurnAction],
        rng: random.Random | None = None,
        transposition: TranspositionTable | None = None,
    ) -> None:
        self._config = config
        self._transition = transition
        self._generate = generate
        self._is_terminal = is_terminal
        self._terminal_value = terminal_value
        self._evaluate = evaluate
        self._rollout = rollout
        self._rng = rng if rng is not None else random.Random(config.seed)
        self._transposition = transposition if transposition is not None else NoTranspositionTable()

    # -- public API ---------------------------------------------------------

    def search(self, root_state: S, player: int) -> TurnAction:
        """Run the configured number of iterations and return the best action."""
        root = self.search_root(root_state, player)
        return self._best_action(root)

    def search_root(self, root_state: S, player: int) -> MCTSNode[S]:
        """Run the search and return the built root node (tests/introspection)."""
        root = MCTSNode(root_state)
        self._initialize(root)
        for _ in range(self._config.iterations):
            node = self._select(root)
            value = self._simulate(node, player)
            self._backpropagate(node, value)
        return root

    def best_action(self, root: MCTSNode[S]) -> TurnAction:
        """The action the search currently favours at ``root``."""
        return self._best_action(root)

    # -- internals ----------------------------------------------------------

    def _initialize(self, node: MCTSNode[S]) -> None:
        if self._is_terminal(node.state):
            node.terminal = True
            node.untried_actions = []
            return
        actions = self._generate(node.state)
        node.untried_actions = list(actions) if actions else [TurnAction()]

    def _select(self, node: MCTSNode[S]) -> MCTSNode[S]:
        while node.children and not node.untried_actions:
            node = self._best_child(node)
        if node.untried_actions:
            action = node.untried_actions.pop()
            child = MCTSNode(self._transition(node.state, action), parent=node, action=action)
            node.children.append(child)
            self._initialize(child)
            return child
        return node

    def _best_child(self, node: MCTSNode[S]) -> MCTSNode[S]:
        log_parent = math.log(max(1, node.visits))
        c = self._config.exploration_constant

        def ucb(child: MCTSNode[S]) -> float:
            if child.visits == 0:
                return float("inf")
            return child.total_value / child.visits + c * math.sqrt(log_parent / child.visits)

        return max(node.children, key=ucb)

    def _simulate(self, node: MCTSNode[S], player: int) -> float:
        if node.terminal:
            return self._terminal_value(node.state, player)
        state = node.state
        steps = 0
        while not self._is_terminal(state) and steps < self._config.max_simulation_steps:
            action = self._rollout(state, self._rng)
            state = self._transition(state, action)
            steps += 1
        if self._is_terminal(state):
            return self._terminal_value(state, player)
        return self._evaluate(state, player)

    def _backpropagate(self, node: MCTSNode[S], value: float) -> None:
        current: MCTSNode[S] | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _best_action(self, root: MCTSNode[S]) -> TurnAction:
        if not root.children:
            return TurnAction()
        best = max(
            root.children,
            key=lambda child: (child.visits, child.total_value / max(1, child.visits)),
        )
        assert best.action is not None
        return best.action
