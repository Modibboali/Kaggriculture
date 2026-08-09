# Kaggriculture Environment Compatibility Report

- **Environment:** `kaggriculture` (official `kaggle-environments` package)
- **Generated:** 2026-08-07T13:10:39+00:00

## Verdict

**A** — Suitable for in-tree MCTS use
- Controllable stepping for both players: yes
- Environment cloning: yes
- State save/restore: yes
- Search throughput: 257.4 steps/sec (MCTS minimum assumed: 20)
- The official environment exposes controllable stepping, state save/restore, and adequate throughput for in-tree use.

## Summary

| Probe | Result | Summary |
|-------|--------|---------|
| environment_creation | PASS | created 'kaggriculture' in 0.4301s, peak 0.2 MiB; simultaneous envs: True |
| stepping | PASS | step(): dict=True, callable=True; 224.2 steps/sec @1000 |
| cloning | PASS | clone support: env.clone=True, copy.copy=True, copy.deepcopy=True, pickle=True, cloudpickle=True |
| serialization | PASS | pickle roundtrip=True, replay_json=True |
| observations | PASS | found 13 public attributes; history accessible: True |
| actions | PASS | arbitrary actions accepted=True; both players independently controllable=True; p0 seeds after buy=1 |
| replay | PASS | replay during=True, after=True; replay is env.steps (toJSON() is metadata) |
| determinism | PASS | same seed identical (deterministic agent) = True |
| benchmark | PASS | creation=22.1/s, steps=257.4/s, peak mem=1.5 MiB |

## Supported Operations

- Inject independent actions for both players, simultaneously
- Clone the environment (copy / pickle)
- Serialize and restore environment state
- Read observations / internal state / history without rerunning
- Deterministic replay under a fixed seed (with a deterministic agent)

## Unsupported Operations

- None detected

## Performance Numbers

| Metric | Value |
|--------|-------|
| environment_creation.creation_time_s | 0.430087 |
| environment_creation.peak_memory_bytes | 178149 |
| environment_creation.multiple_created | 3 |
| stepping.steps_after_step | 2 |
| stepping.run_100_steps_s | 0.411511 |
| stepping.run_100_steps_per_s | 243.0 |
| stepping.run_1000_steps_s | 4.460935 |
| stepping.run_1000_steps_per_s | 224.2 |
| stepping.avg_step_s | 0.004461 |
| observations.num_players | 2 |
| observations.history_len | 4 |
| actions.p0_seeds_after_p0_buy | 1 |
| actions.p1_seeds_after_p0_buy | 0 |
| actions.p1_seeds_after_p1_buy | 1 |
| actions.p0_seeds_after_p1_buy | 1 |
| replay.steps_during | 6 |
| replay.steps_after | 720 |
| benchmark.creation_per_sec | 22.1 |
| benchmark.creation_avg_s | 0.045282 |
| benchmark.full_run_steps | 720 |
| benchmark.full_run_s | 2.797618 |
| benchmark.steps_per_sec | 257.4 |
| benchmark.avg_step_s | 0.003886 |
| benchmark.peak_memory_run_bytes | 1525053 |
| benchmark.clone_s | 3.115195 |
| benchmark.clone_per_sec | 0.3 |
| benchmark.pickle_dump_s | 0.229863 |
| benchmark.pickle_size_bytes | 3353454 |
| benchmark.native_clone_s | 0.291109 |
| benchmark.native_clone_per_sec | 3.4 |
| benchmark.replay_json_len | 4281174 |
| benchmark.replay_json_s | 1.57987 |

## Recommendations

- Drive all search with agent.state.GameState.from_observation + agent.generators.LegalActionGenerator; these are environment-independent and hashable.
- Plan for a lightweight in-process simulator on the domain model (agent.state + agent.actions) to serve as the MCTS rollout engine, with the official environment used only for training/validation games.

## Detailed Probe Results

### environment_creation

- **Success:** True
- **Summary:** created 'kaggriculture' in 0.4301s, peak 0.2 MiB; simultaneous envs: True
- **Duration (s):** 2.3971
- **Details:**

```json
{
  "installed": true,
  "creation_time_s": 0.430087,
  "peak_memory_bytes": 178149,
  "env_type": "Environment",
  "multiple_simultaneous": true,
  "multiple_detail": "ok (list)",
  "multiple_created": 3,
  "env.name": "kaggriculture",
  "env.agents": "Struct(len=3)",
  "env.configuration": "Struct(len=15)",
  "env.specification": "Struct(len=11)"
}
```

### stepping

- **Success:** True
- **Summary:** step(): dict=True, callable=True; 224.2 steps/sec @1000
- **Duration (s):** 5.0402
- **Details:**

```json
{
  "has_step": true,
  "has_run": true,
  "has_act": false,
  "has_reset": true,
  "has_toJSON": true,
  "has_render": true,
  "has_clone": true,
  "has_play": true,
  "has_train": true,
  "step_dict_actions": true,
  "step_dict_detail": "ok (list)",
  "steps_after_step": 2,
  "step_callable_actions": true,
  "step_callable_detail": "ok (list)",
  "run_100_steps_s": 0.411511,
  "run_100_steps_per_s": 243.0,
  "run_1000_steps_s": 4.460935,
  "run_1000_steps_per_s": 224.2,
  "avg_step_s": 0.004461
}
```

### cloning

- **Success:** True
- **Summary:** clone support: env.clone=True, copy.copy=True, copy.deepcopy=True, pickle=True, cloudpickle=True
- **Duration (s):** 0.1049
- **Details:**

```json
{
  "cloudpickle_installed": true,
  "clones": {
    "env.clone": {
      "ok": true,
      "result_type": "Environment",
      "same_type_as_env": true
    },
    "copy.copy": {
      "ok": true,
      "result_type": "Environment",
      "same_type_as_env": true
    },
    "copy.deepcopy": {
      "ok": true,
      "result_type": "Environment",
      "same_type_as_env": true
    },
    "pickle": {
      "ok": true,
      "result_type": "Environment",
      "same_type_as_env": true
    },
    "cloudpickle": {
      "ok": true,
      "result_type": "Environment",
      "same_type_as_env": true
    }
  },
  "any_clone_works": true
}
```

### serialization

- **Success:** True
- **Summary:** pickle roundtrip=True, replay_json=True
- **Duration (s):** 0.0864
- **Details:**

```json
{
  "pickle": {
    "ok": true,
    "size_bytes": 24699,
    "dump_s": 0.001433,
    "load_s": 0.000525,
    "restore_ok": true,
    "throughput_bytes_per_s": 12612469.9
  },
  "cloudpickle_installed": true,
  "cloudpickle": {
    "ok": true,
    "size_bytes": 24699,
    "dump_s": 0.003511,
    "load_s": 0.000386,
    "restore_ok": true,
    "throughput_bytes_per_s": 6337464.4
  },
  "toJSON_metadata": {
    "ok": true,
    "is_dict": true,
    "s": 0.017633
  },
  "replay_json": {
    "ok": true,
    "size_bytes": 23036,
    "s": 0.00911,
    "steps": 4
  },
  "any_roundtrip_works": true
}
```

### observations

- **Success:** True
- **Summary:** found 13 public attributes; history accessible: True
- **Duration (s):** 0.0553
- **Details:**

```json
{
  "public_attributes": {
    "agents": "Struct(len=3)",
    "configuration": "Struct(len=15)",
    "debug": "bool",
    "done": "bool",
    "id": "str(len=36)",
    "info": "dict(len=1)",
    "logs": "list(len=4)",
    "name": "str(len=13)",
    "pool": "NoneType",
    "specification": "Struct(len=11)",
    "state": "list(len=2)",
    "steps": "list(len=4)",
    "version": "str(len=5)"
  },
  "state_shape": "list(len=2)",
  "num_players": 2,
  "state0_keys": [
    "action",
    "reward",
    "info",
    "observation",
    "status"
  ],
  "observation0_keys": [
    "remainingOverageTime",
    "step",
    "player",
    "farms",
    "private",
    "market",
    "town",
    "day",
    "hour"
  ],
  "observation0_private_keys": [
    "shed",
    "seeds",
    "inventories"
  ],
  "history_attr": "list(len=4)",
  "history_len": 4,
  "attr.configuration": "Struct(len=15)",
  "attr.agents": "Struct(len=3)",
  "attr.specification": "Struct(len=11)",
  "attr.name": "kaggriculture"
}
```

### actions

- **Success:** True
- **Summary:** arbitrary actions accepted=True; both players independently controllable=True; p0 seeds after buy=1
- **Duration (s):** 0.0654
- **Details:**

```json
{
  "step_accepted": true,
  "step_detail": "ok (list)",
  "p0_seeds_after_p0_buy": 1,
  "p1_seeds_after_p0_buy": 0,
  "p1_independent": true,
  "p1_detail": "ok (list)",
  "p1_seeds_after_p1_buy": 1,
  "p0_seeds_after_p1_buy": 1,
  "both_simultaneous": true,
  "mixed_agent_types": true,
  "both_players_controllable": true
}
```

### replay

- **Success:** True
- **Summary:** replay during=True, after=True; replay is env.steps (toJSON() is metadata)
- **Duration (s):** 4.6537
- **Details:**

```json
{
  "steps_during": 6,
  "replay_during": true,
  "tojson_ok": true,
  "tojson_detail": "ok (dict)",
  "tojson_is_dict": true,
  "tojson_keys": [
    "id",
    "name",
    "title",
    "description",
    "version",
    "module_version",
    "configuration",
    "specification",
    "steps",
    "rewards",
    "statuses",
    "schema_version",
    "info"
  ],
  "full_run_ok": true,
  "full_run_detail": "ran to completion",
  "steps_after": 720,
  "replay_after": true,
  "step0_keys": [
    "action",
    "reward",
    "info",
    "observation",
    "status"
  ],
  "render_json": true,
  "render_json_detail": "ok (str)"
}
```

### determinism

- **Success:** True
- **Summary:** same seed identical (deterministic agent) = True
- **Duration (s):** 1.8064
- **Details:**

```json
{
  "same_seed_identical": true,
  "different_seed_differs": true,
  "random_agent_same_seed_identical": false
}
```

### benchmark

- **Success:** True
- **Summary:** creation=22.1/s, steps=257.4/s, peak mem=1.5 MiB
- **Duration (s):** 10.7198
- **Details:**

```json
{
  "creation_per_sec": 22.1,
  "creation_avg_s": 0.045282,
  "full_run_ok": true,
  "full_run_detail": "completed",
  "full_run_steps": 720,
  "full_run_s": 2.797618,
  "steps_per_sec": 257.4,
  "avg_step_s": 0.003886,
  "peak_memory_run_bytes": 1525053,
  "clone_ok": true,
  "clone_s": 3.115195,
  "clone_per_sec": 0.3,
  "pickle_ok": true,
  "pickle_dump_s": 0.229863,
  "pickle_size_bytes": 3353454,
  "native_clone_ok": true,
  "native_clone_s": 0.291109,
  "native_clone_per_sec": 3.4,
  "replay_json_ok": true,
  "replay_json_len": 4281174,
  "replay_json_s": 1.57987
}
```
