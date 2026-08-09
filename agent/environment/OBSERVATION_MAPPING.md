# Kaggle Observation → Domain State Mapping

This document is the contract between the official Kaggriculture environment
(`kaggle_environments`) and the internal search engine. It records which
observation fields are represented in the domain model, which are
intentionally ignored, and which will need to be added before the search
simulator can predict future states.

## Adapter pipeline

```
Kaggle observation (raw dict from the environment)
        │
        ▼
observation_validation.validate_observation      → InvalidObservationError
        │
        ▼
KaggleObservationAdapter.from_observation        → agent.state.GameState
```

The adapter is the **only** module that reads the raw observation dict. It is
deterministic, never mutates the input, and copies every value into frozen
domain objects (no references to the input dicts/lists are retained).
`GameState.from_observation` is a thin, backward-compatible alias that
delegates here.

## Represented fields

| Observation field | Domain representation | Notes |
|---|---|---|
| `step` | `GameState.step` (int) | Absolute turn index; **added to the domain model in this task**. |
| `day` | `GameState.day` (int) | |
| `hour` | `GameState.hour` (int) | Turn within the day. |
| `player` | `GameState.current_player` (int) | 0 or 1. |
| `farms` | `GameState.players: tuple[PlayerState, PlayerState]` | Farm identity is **positional**: `farms[p]` → `players[p]`. |
| `farms[i].money` | `PlayerState.farm.money` (int) | Stored as int (observation uses float coin values). |
| `farms[i].tiles` | `PlayerState.farm.tiles: tuple[tuple[Tile, ...], ...]` | Row-major `tiles[y][x]`; each raw tile becomes a typed `Tile`. |
| `farms[i].farmer` | `PlayerState.farm.farmer` (`Worker`, id 0) | Main farmer, `is_main_farmer=True`. |
| `farms[i].hands` | `PlayerState.farm.workers` (`Worker` ids 1..n) | Hired hands (public positions). |
| `farms[i].unlocked_quadrants` | `PlayerState.farm.unlocked_quadrants` (`frozenset[Quadrant]`) | |
| `farms[i].hires_today` | `PlayerState.farm.hires_today` (int) | |
| tile `None` | `EmptyTile` (singleton `EMPTY_TILE`) | |
| tile `"LOCKED"` | `LockedTile` (singleton `LOCKED_TILE`) | |
| tile `{kind: "WEED"}` | `WeedTile` (singleton `WEED_TILE`) | |
| tile `{kind: "PLANT", ...}` | `PlantTile(PlantState)` | `crop/planted_day/watered_today/consecutive_unwatered/yield_units/fertilized_until_day/max_lifespan_step`. |
| tile `{kind: "COOP"|"PASTURE", ...}` | `CoopTile` / `PastureTile` with `AnimalState \| None` | `animal/placed_day/yield_units/fed_today/consecutive_unfed/cared_today/fertilizer_available/pending_care_bonus`. |
| `private` | The **observing** player's `PlayerState` | The opponent's private state is **not observable**; it is left empty (see below). |
| `private.shed` | `PlayerState.inventory` (`Inventory`) | Item counts keyed by `ItemType`. |
| `private.seeds` | `PlayerState.seeds` (`Seeds`) | Seed counts keyed by `CropType`. |
| `private.inventories` | `PlayerState.workers[].inventory` (`Inventory`) | `[0]` is the main farmer, then each hand, matching the farm's public worker order. |
| `market.inventory` | `GameState.market.inventory` (`Inventory`) | |
| `market.prices` | `GameState.market.prices` (`Mapping[ItemType, int]`) | Sell prices; used by the action generator as a buy-price proxy. |
| `town.unlocked_shops` | `GameState.town.unlocked_shops` (`frozenset[ShopType]`) | |

## Intentionally ignored fields

| Field | Why it is ignored |
|---|---|
| `remainingOverageTime` | Wall-clock per-step time budget. It is non-deterministic, irrelevant to planning/simulation, and would poison determinism checks. Validated as numeric (when present) then discarded. |

## Fields that are absent for some observations (schema uncertainty)

- **`step`** — During a real `env.run`, **both** agents receive observations
  containing `step` (verified). However, `env.state[1]["observation"]` (the
  *other* player's view exposed by the environment's `state` accessor) omits
  `step`. The adapter therefore **requires** `step` (it is always present in
  the agent-facing observations the adapter is built for). Passing an
  `env.state[1]`-style observation raises
  `InvalidObservationError: step is required`. If we later need to convert
  non-agent-facing observations, `GameState.step` would need to become
  optional (`int | None`).
- **Opponent private state** (`private`) — The observation contains only the
  observing player's `private` block. The opponent's shed/seeds/inventories
  are unobservable by design; `players[1 - current_player]` gets empty
  `Inventory`/`Seeds` and workers without inventories. The search layer must
  model this hidden information (e.g., with an opponent model or belief
  distribution) — it cannot be parsed from the observation.

## Fields the simulator will need (deferred, not in the per-step observation)

The observation is **stateless w.r.t. configuration**: it carries no static
game parameters. The simulator cannot predict future states from the
observation alone. Before implementing `GameState.apply`/a transition engine,
add a configuration object fed from `env.configuration`:

- `episodeSteps` (720), `turnsPerDay` (24), `boardSize` (10), `shedCapacity` (100)
- `startingMoney`, `maxMarketOrdersPerTurn`
- `weedSpawnChance`, `townShopUnlockInterval`, `townShopSellInterval`,
  `townCenterSellInterval`, `farmHandCostMult`
- Fixed seed/animal costs (`WHEAT 10 … MELON 80`, `GOOSE 300, COW 400, SHEEP 500`)
- Market price-curve parameters (`base`, `I0`, `T`, below/above `func`/`target`)

Additional state the simulator must track internally (not present per step):

- Turn-level market deltas (what was sold/bought this turn) to reproduce
  price moves and reward attribution.
- The town-consumption schedule (interval counters).
- Day-refresh bookkeeping (e.g., `hires_today` resets, fed/watered resets) —
  some is observable, the *transition* is not.

## Domain-model changes made in this task

1. **`GameState.step: int` added** — the smallest extension needed to
   faithfully represent the observed `step` field (required for absolute-turn
   terminal checks and scheduling).
2. **Parsing relocated out of `agent.state.game_state`** — the Kaggle-specific
   `_parse_*` helpers moved to `agent/environment/kaggle_observation_adapter.py`.
   `GameState.from_observation` remains as a delegating, backward-compatible
   shim. This honours the boundary principle: Kaggle structures no longer live
   in the domain layer. No domain value objects were changed (other than the
   `step` field).

## Correctness guarantees

- Deterministic: identical input → identical `GameState` (and identical hash).
- Immutable: mutating the input observation after conversion cannot change the
  returned state (covered by tests).
- Validated: malformed observations raise path-aware `InvalidObservationError`
  instead of silently producing corrupt state.
