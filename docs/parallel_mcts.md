# Parallel MCTS — developer guide

This document explains the root-parallel MCTS execution layer added around the
existing sequential UCT implementation, why processes are used, how budgets,
seeds and root statistics are handled, and how to configure the worker count.

## Sequential MCTS (the reference)

`agent/ai/mcts.py` defines `MCTS` (UCT/UCB1), generic over a state type `S`,
with the simulator model supplied as plain callables (transition, generate,
terminal, evaluate, rollout). `MCTSConfig` carries `iterations`,
`exploration_constant`, `max_simulation_steps`, `seed` and — since the parallel
milestone — `workers`.

```python
from agent.ai import MCTS, MCTSConfig

config = MCTSConfig(
    iterations=500,   # total simulation budget
    workers=1,        # 1 => sequential (reference) execution
    seed=42,
)
```

The sequential `MCTS` **ignores `workers` entirely** — it is mathematically
unchanged and remains the canonical reference implementation.

## Root-parallel MCTS

`agent/ai/parallel_mcts.py` adds `ParallelMCTS`, a drop-in execution strategy
with the same public `search(state, player)` / `search_root(state, player)`
API. It is **not** a rewrite of UCT and it **does not** touch the simulator,
`SearchState`, `ActionGenerator`, the evaluator, or the rollout policies.

```
                MCTS interface
                     │
         ┌───────────┴───────────┐
         │                       │
   SequentialMCTS          ParallelMCTS
                                 │
                     ┌───────────┼───────────┐
                     │           │           │
                  Worker 0   Worker 1    Worker N
                     │           │           │
                     ▼           ▼           ▼
                 local MCTS   local MCTS   local MCTS
                     │           │           │
                     └───────────┼───────────┘
                                 ▼
                          root statistics
                                 │
                                 ▼
                          aggregate result
```

### Why processes, not threads

MCTS is CPU-bound Python. A `ProcessPoolExecutor` (standard library only) runs
each worker under its own interpreter, bypassing the GIL for the Python-level
search loop. `workers == 1` (or `iterations <= 1`) **delegates to the
sequential `MCTS` in-process — no process is spawned**, so debugging and
determinism regression stay exact.

### Configuration

```python
from agent.ai import ParallelMCTS, MCTSConfig

config = MCTSConfig(
    iterations=500,
    workers=8,
    seed=42,
)
```

`iterations` is always the **total** simulation budget regardless of worker
count. `workers` only changes the execution strategy; it never changes the
sequential reference behaviour.

## How simulation budgets are distributed

`split_budget(total, workers)` splits the total as evenly as possible so the
sum is exact and no worker exceeds `ceil(total / active)`:

```python
split_budget(100, 3)   # (34, 33, 33)   sum == 100
split_budget(3, 8)     # (1, 1, 1)      active = min(8, 3)
split_budget(0, 4)     # ()
```

`active_workers = min(workers, total)` — `budget < workers` never creates
workers doing zero work.

## How random seeds are handled

`worker_seed(base_seed, worker_id)` derives a deterministic per-worker seed so
workers never share RNG state and results are reproducible:

```python
worker_seed(42, 0)  # (42 * 1_000_003) ^ 1
worker_seed(42, 1)  # (42 * 1_000_003) ^ 2
```

Each worker builds its own `random.Random(worker_seed)`. No system randomness
is used, so a fixed `(seed, workers, iterations)` configuration is
deterministic run-to-run (parallel run A == parallel run B). Because
root-parallel MCTS builds independent trees, it is **not** required (and would
be incorrect) to assert `sequential result == parallel result`.

## How root statistics are aggregated

Each worker returns only its root child statistics
`(canonical_key, action, visits, value_sum)` — never the tree. The parent sums:

```
N(a) = Σᵢ Nᵢ(a)
W(a) = Σᵢ Wᵢ(a)
Q(a) = W(a) / N(a)        # N(a) > 0
```

Q is computed from **total W over total N**, never `mean(Qᵢ)` (workers may
have different visit counts). The final action follows the sequential
`MCTS._best_action` semantics: `max(visits, Q)`.

### Action identity

`hash(TurnAction)` is salted per process (string-backed enums), so it cannot
be used across workers. `canonical_action_key(action)` derives a stable,
content-based SHA-1 key that is identical in every process; equal logical
actions always map to the same key.

## Why this is not shared-tree MCTS

Each worker owns a completely independent local tree. There are no locks, no
virtual loss, no atomic visit counters, no shared mutable nodes, no shared
memory / Manager / transposition tables. That is deliberately deferred; this
milestone is root-parallel only.

## Windows multiprocessing considerations

On Windows the default start method is `spawn`, so:

- the worker function `run_mcts_worker` and the `WorkerTask` / `WorkerResult`
  payloads are module-level and picklable;
- no process pool is created at import time (the `ProcessPoolExecutor` is
  created per search call, never at global scope);
- entry points keep their `if __name__ == "__main__":` guards.

You may pass `process_start_method` to `ParallelMCTS` to select the context
explicitly (e.g. `"spawn"` on Windows, `"fork"` where available).

## Error handling

If a worker raises, the exception is re-raised in the caller wrapped in a
`RuntimeError` identifying the failing `worker_id`. Failed simulations are
never silently discarded or replaced by random actions; partial results never
masquerade as a complete budget.

## Performance

See `docs/task13_parallel_mcts_report.md` for the measured throughput,
speedup, efficiency, serialization overhead and memory scaling, and for the
recommended worker count on the development machine.
