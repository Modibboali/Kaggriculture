"""Root-parallel MCTS via OS processes.

This module adds a clean parallel *execution* layer around the existing
sequential :class:`~agent.ai.mcts.MCTS` — it does **not** re-implement UCT and
does **not** touch the simulator, ``SearchState``, ``ActionGenerator``, the
evaluator, or the rollout policies. The sequential :class:`MCTS` remains the
canonical reference implementation and is used unchanged whenever
``workers == 1``.

Architecture::

                    MCTS interface
                         |
             +-----------+-----------+
             |                       |
       SequentialMCTS          ParallelMCTS
                                     |
                         +-----------+-----------+
                         |           |           |
                      Worker 0   Worker 1    Worker N
                         |           |           |
                         v           v           v
                     local MCTS   local MCTS   local MCTS
                         |           |           |
                         +-----------+-----------+
                                     v
                              root statistics
                                     |
                                     v
                              aggregate result

Design notes
------------

* **Processes, not threads.** MCTS is CPU-bound Python; we use a
  ``concurrent.futures.ProcessPoolExecutor`` (standard library only). No
  threading, no external distributed systems.

* **Root-parallel, not shared-tree.** Each worker builds an independent local
  search tree rooted at the same ``SearchState``, runs its share of the total
  simulation budget, and returns only the *root child statistics* (action,
  visit count, value sum). There is no shared mutable tree, no locks, no
  virtual loss, no atomic counters.

* **Small, picklable payload.** A :class:`WorkerTask` carries the compact
  immutable ``SearchState`` plus the exact model callables the sequential MCTS
  uses (transition / generate / terminal / evaluate / rollout) — which are
  bound methods of lightweight, picklable simulator-adapter / generator /
  evaluator objects. We never pickle the official Kaggle ``Environment``.

* **Exact budget accounting.** :func:`split_budget` splits ``iterations``
  into per-worker budgets that sum exactly to the requested total, and only
  ``min(workers, iterations)`` workers get a positive share.

* **Independent, reproducible RNG.** :func:`worker_seed` derives a
  deterministic per-worker seed from ``(base_seed, worker_id)``; each worker
  creates its own ``random.Random`` from that seed, so streams never overlap
  and results are reproducible for a fixed configuration.

* **Deterministic aggregation.** Workers report per-action ``(N_i, W_i)``;
  the parent sums them ``N = sum N_i``, ``W = sum W_i`` and uses
  ``Q = W / N``. We never average Q values (workers can have different visit
  counts). Aggregation is order-independent (dictionary sums), so process
  scheduling cannot change the result.

* **Canonical action identity.** ``hash(TurnAction)`` is salted per process
  (string-backed enums), so it is useless across workers. :func:`canonical_action_key`
  derives a stable, content-based key that is identical in every process.

* **Windows/spawn safety.** The worker function and task/result types are
  module-level and picklable; no process pool is created at import time; the
  ``ProcessPoolExecutor`` is created per search call with an explicit
  ``mp_context`` (default ``multiprocessing.get_context()`` → ``spawn`` on
  Windows).

* **Explicit failure.** If a worker raises, the exception is re-raised in the
  caller with the failing ``worker_id`` attached — failed simulations are
  never silently discarded or replaced by random actions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, Iterable, TypeVar, cast

from ..actions import TurnAction
from .mcts import MCTS, MCTSConfig
from .simulator_adapter import SimulatorAdapter

S = TypeVar("S")


# ---------------------------------------------------------------------------
# Canonical action identity
# ---------------------------------------------------------------------------

def _canonicalize(value: object) -> object:
    """Reduce a value to a deterministic, cross-process-stable tuple form.

    ``hash`` of strings/enums is salted per process, so it cannot be used as
    an aggregation key across workers. This normalisation converts enums to
    ``(type_name, member_name)``, dataclasses to ``(type_name, fields...)``,
    and leaves scalars/tuples/sets/dicts as stable structures.
    """
    if isinstance(value, Enum):
        return (type(value).__name__, value.name)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        obj = cast(Any, value)
        fields = tuple(
            _canonicalize(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        )
        return (type(obj).__name__, fields)
    if isinstance(value, tuple):
        return tuple(_canonicalize(v) for v in value)
    if isinstance(value, frozenset):
        return tuple(sorted((_canonicalize(v) for v in value), key=repr))
    if isinstance(value, dict):
        return tuple(
            (_canonicalize(k), _canonicalize(v))
            for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))
        )
    return value


def canonical_action_key(action: TurnAction) -> int:
    """A stable, cross-process integer identity for a ``TurnAction``.

    Equal logical actions always map to the same key in every process; the key
    does not depend on Python object identity or per-process hash salting.
    """
    data = repr(_canonicalize(action)).encode("utf-8")
    digest = hashlib.sha1(data).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Budget distribution and worker seeds
# ---------------------------------------------------------------------------

def split_budget(total: int, workers: int) -> tuple[int, ...]:
    """Split ``total`` simulations as evenly as possible across ``workers``.

    ``sum(result) == total`` and no worker runs more than
    ``ceil(total / active)`` simulations, where
    ``active = min(workers, total)``. A zero/negative budget or worker count
    yields an empty tuple.
    """
    if total <= 0 or workers <= 0:
        return ()
    active = min(workers, total)
    base, remainder = divmod(total, active)
    return tuple(base + (1 if i < remainder else 0) for i in range(active))


def worker_seed(base_seed: int, worker_id: int) -> int:
    """Deterministic per-worker seed derived from ``(base_seed, worker_id)``.

    Distinct ``worker_id``\\ s always produce distinct seeds for a fixed base
    seed (no system randomness), so parallel runs are reproducible and worker
    RNG streams never collide.
    """
    return (base_seed * 1_000_003) ^ (worker_id + 1)


# ---------------------------------------------------------------------------
# Worker protocol (module-level so it is picklable under Windows spawn)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WorkerTask(Generic[S]):
    """The minimum picklable payload sent to one worker process.

    Carries the compact root ``SearchState`` and the exact model callables the
    sequential MCTS uses. ``transition_counter`` is an optional
    :class:`SimulatorAdapter` used only to measure how many simulator
    transitions the worker performed (must be the same object whose
    ``transition`` is ``transition`` for the count to be meaningful).
    """

    root_state: S
    player: int
    iterations: int
    seed: int
    exploration_constant: float
    max_simulation_steps: int
    transition: Callable[[S, TurnAction], S]
    generate: Callable[[S], tuple[TurnAction, ...]]
    is_terminal: Callable[[S], bool]
    terminal_value: Callable[[S, int], float]
    evaluate: Callable[[S, int], float]
    rollout: Callable[[S, random.Random], TurnAction]
    worker_id: int = 0
    transition_counter: SimulatorAdapter | None = None


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """The compact result returned by one worker (never the whole tree).

    ``root_stats`` is a tuple of ``(canonical_key, action, visits, value_sum)``
    for each expanded root child, so the parent can aggregate by key and map
    the winning key back to a real ``TurnAction``. ``peak_rss_mb`` is the
    worker's peak resident memory during the search (0.0 when not measurable),
    so memory scaling can be reported per worker.
    """

    worker_id: int
    root_stats: tuple[tuple[int, TurnAction, int, float], ...]
    transitions: int
    wall_time: float
    peak_rss_mb: float = 0.0


def run_mcts_worker(task: WorkerTask[S]) -> WorkerResult:
    """Run one worker's share of the root-parallel search.

    Builds an independent local :class:`MCTS` seeded from ``task.seed``, runs
    ``task.iterations`` simulations from ``task.root_state``, and returns only
    the root child statistics (plus an optional transition count).
    """
    config = MCTSConfig(
        iterations=task.iterations,
        exploration_constant=task.exploration_constant,
        max_simulation_steps=task.max_simulation_steps,
        seed=task.seed,
    )
    mcts = MCTS(
        config,
        transition=task.transition,
        generate=task.generate,
        is_terminal=task.is_terminal,
        terminal_value=task.terminal_value,
        evaluate=task.evaluate,
        rollout=task.rollout,
        rng=random.Random(task.seed),
    )
    start = time.perf_counter()
    root = mcts.search_root(task.root_state, task.player)
    wall_time = time.perf_counter() - start
    stats = tuple(
        (canonical_action_key(child.action), child.action, child.visits, child.total_value)
        for child in root.children
        if child.action is not None
    )
    transitions = 0
    if task.transition_counter is not None:
        transitions = int(task.transition_counter.transitions)
    return WorkerResult(
        worker_id=task.worker_id,
        root_stats=stats,
        transitions=transitions,
        wall_time=wall_time,
        peak_rss_mb=_peak_rss_mb(),
    )


def _peak_rss_mb() -> float:
    """Peak resident memory of the current process in MiB (0.0 if unavailable)."""
    try:
        import psutil
    except ImportError:
        return 0.0
    mem = psutil.Process().memory_info()
    peak = getattr(mem, "peak_wset", None) or getattr(mem, "rss", 0)
    peak = peak if isinstance(peak, (int, float)) else 0
    return float(peak) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Root-statistics aggregation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RootStat:
    """Aggregated root statistics for a single action."""

    visits: int
    value_sum: float

    @property
    def q(self) -> float:
        """Mean value Q = W / N (0.0 when the action was never visited)."""
        return self.value_sum / self.visits if self.visits else 0.0


def aggregate_root_stats(
    worker_results: Iterable[WorkerResult],
) -> dict[int, tuple[int, float]]:
    """Sum per-action ``(visits, value)`` across workers, keyed canonically.

    Uses ``W_total / N_total`` semantics (never the mean of per-worker Q):
    each worker's visit count contributes its full weight.
    """
    totals: dict[int, list[float]] = {}
    for result in worker_results:
        for key, _action, visits, value in result.root_stats:
            entry = totals.setdefault(key, [0.0, 0.0])
            entry[0] += float(visits)
            entry[1] += value
    return {key: (int(v[0]), v[1]) for key, v in totals.items()}


def select_best_action_key(stats: dict[int, tuple[int, float]]) -> int | None:
    """Pick the winning canonical key by ``max(visits, Q)``.

    Mirrors the sequential ``MCTS._best_action`` selection semantics
    (most-visited child, average value as tie-break). Returns ``None`` when
    no action was visited.
    """
    if not stats:
        return None
    return max(
        stats.items(),
        key=lambda kv: (kv[1][0], kv[1][1] / max(1, kv[1][0])),
    )[0]


def select_best_action_from_stats(stats: dict[TurnAction, RootStat]) -> TurnAction:
    """Pick the winning ``TurnAction`` from an action-keyed stats dict."""
    if not stats:
        return TurnAction()
    return max(stats.items(), key=lambda kv: (kv[1].visits, kv[1].q))[0]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ParallelSearchResult:
    """The aggregated outcome of a parallel search (or its sequential fallback).

    ``worker_results`` retains each worker's compact result (root stats,
    transitions, wall time, peak RSS) so callers can inspect per-worker memory
    and timing without ever reconstructing the trees.
    """

    best_action: TurnAction
    root_stats: dict[TurnAction, RootStat]
    total_simulations: int
    wall_time: float
    worker_count: int
    worker_results: tuple[WorkerResult, ...] = ()


# ---------------------------------------------------------------------------
# ParallelMCTS
# ---------------------------------------------------------------------------

class ParallelMCTS(Generic[S]):
    """Root-parallel MCTS: N independent local UCT searches, aggregated at the root.

    ``workers == 1`` (or ``iterations <= 1``) delegates to the canonical
    sequential :class:`MCTS` in the current process — no process is spawned,
    so debugging, determinism tests and performance baselines stay exact.
    ``workers > 1`` runs root-parallel search via a per-call
    :class:`ProcessPoolExecutor`; each worker owns a private tree, runs its
    share of the simulation budget with an independent RNG, and returns only
    root child statistics.
    """

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
        process_start_method: str | None = None,
        transition_counter: SimulatorAdapter | None = None,
    ) -> None:
        self._config = config
        self._transition = transition
        self._generate = generate
        self._is_terminal = is_terminal
        self._terminal_value = terminal_value
        self._evaluate = evaluate
        self._rollout = rollout
        self._rng = rng if rng is not None else random.Random(config.seed)
        self._process_start_method = process_start_method
        self._transition_counter = transition_counter
        self._last_result: ParallelSearchResult | None = None
        self._total_transitions = 0

    # -- public API ---------------------------------------------------------

    @property
    def last_result(self) -> ParallelSearchResult | None:
        """The most recent :class:`ParallelSearchResult` (introspection/tests)."""
        return self._last_result

    @property
    def transitions(self) -> int:
        """Simulator transitions performed across the parallel search(es)."""
        return self._total_transitions

    def search(self, root_state: S, player: int) -> TurnAction:
        """Run the search and return the best root action."""
        return self.search_root(root_state, player).best_action

    def search_root(self, root_state: S, player: int) -> ParallelSearchResult:
        """Run the search; return the aggregated root statistics."""
        if self._config.workers <= 1 or self._config.iterations <= 1:
            result = self._sequential(root_state, player)
        else:
            result = self._parallel(root_state, player)
        self._last_result = result
        return result

    # -- sequential fallback (the canonical reference) ----------------------

    def _sequential(self, root_state: S, player: int) -> ParallelSearchResult:
        config = MCTSConfig(
            iterations=self._config.iterations,
            exploration_constant=self._config.exploration_constant,
            max_simulation_steps=self._config.max_simulation_steps,
            seed=self._config.seed,
        )
        mcts = MCTS(
            config,
            transition=self._transition,
            generate=self._generate,
            is_terminal=self._is_terminal,
            terminal_value=self._terminal_value,
            evaluate=self._evaluate,
            rollout=self._rollout,
            rng=self._rng,
        )
        start = time.perf_counter()
        root = mcts.search_root(root_state, player)
        wall_time = time.perf_counter() - start
        stats: dict[TurnAction, RootStat] = {}
        for child in root.children:
            if child.action is not None:
                stats[child.action] = RootStat(child.visits, child.total_value)
        return ParallelSearchResult(
            best_action=select_best_action_from_stats(stats),
            root_stats=stats,
            total_simulations=self._config.iterations,
            wall_time=wall_time,
            worker_count=1,
        )

    # -- root-parallel ------------------------------------------------------

    def _parallel(self, root_state: S, player: int) -> ParallelSearchResult:
        budgets = split_budget(self._config.iterations, self._config.workers)
        tasks = [
            WorkerTask(
                root_state=root_state,
                player=player,
                iterations=budget,
                seed=worker_seed(self._config.seed, i),
                exploration_constant=self._config.exploration_constant,
                max_simulation_steps=self._config.max_simulation_steps,
                transition=self._transition,
                generate=self._generate,
                is_terminal=self._is_terminal,
                terminal_value=self._terminal_value,
                evaluate=self._evaluate,
                rollout=self._rollout,
                worker_id=i,
                transition_counter=self._transition_counter,
            )
            for i, budget in enumerate(budgets)
        ]
        if not tasks:
            return ParallelSearchResult(
                best_action=TurnAction(),
                root_stats={},
                total_simulations=0,
                wall_time=0.0,
                worker_count=0,
            )

        context = (
            multiprocessing.get_context(self._process_start_method)
            if self._process_start_method is not None
            else multiprocessing.get_context()
        )
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=context) as pool:
            futures = [pool.submit(run_mcts_worker, task) for task in tasks]
            results: list[WorkerResult] = []
            for i, future in enumerate(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    # Surface the exception and identify the failing worker;
                    # never let partial results masquerade as a full budget.
                    raise RuntimeError(
                        f"ParallelMCTS worker {i} failed: {exc}"
                    ) from exc
        wall_time = time.perf_counter() - start

        aggregated = aggregate_root_stats(results)
        representatives: dict[int, TurnAction] = {}
        for result in results:
            for key, action, _visits, _value in result.root_stats:
                representatives.setdefault(key, action)
        best_key = select_best_action_key(aggregated)
        best_action = representatives[best_key] if best_key is not None else TurnAction()
        root_stats = {
            representatives[key]: RootStat(visits, value)
            for key, (visits, value) in aggregated.items()
        }
        if self._transition_counter is not None:
            self._total_transitions += sum(result.transitions for result in results)
        return ParallelSearchResult(
            best_action=best_action,
            root_stats=root_stats,
            total_simulations=sum(budgets),
            wall_time=wall_time,
            worker_count=len(tasks),
            worker_results=tuple(results),
        )
