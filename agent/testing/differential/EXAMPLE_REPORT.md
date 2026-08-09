# Example Differential Report

Demonstrates the differential testing framework against the **official Kaggriculture environment**. The real simulator is not built yet, so this shows:

1. A **match** (`pass`, simulator replays ground truth).
2. A **mismatch** (`buy_plant_water`, no-op simulator applies no rules).

## Match case

```
Scenario: pass

No differences detected.
```

## Mismatch case

```
Scenario: buy_plant_water

Turn: 1
Action:
    P0: PASS | BUY_SEED WHEAT 1
    P1: PASS
Differences:

    farms[0].money:
        expected: 2990
        actual: 3000

    hour:
        expected: 1
        actual: 0

    market.inventory.CARROT:
        expected: 9999
        actual: 10000

    market.inventory.EGG:
        expected: 9999
        actual: 10000

    market.inventory.MELON:
        expected: 9999
        actual: 10000

    market.inventory.MILK:
        expected: 9999
        actual: 10000

    market.inventory.STRAWBERRY:
        expected: 9999
        actual: 10000

    market.inventory.TOMATO:
        expected: 9999
        actual: 10000

    market.inventory.WHEAT:
        expected: 9999
        actual: 10000

    market.inventory.WOOL:
        expected: 9999
        actual: 10000

    market.prices.CARROT:
        expected: 36
        actual: 35

    market.prices.MELON:
        expected: 256
        actual: 250

    market.prices.MILK:
        expected: 169
        actual: 160

    market.prices.STRAWBERRY:
        expected: 128
        actual: 120

    market.prices.WHEAT:
        expected: 26
        actual: 25

    market.prices.WOOL:
        expected: 206
        actual: 200

    private.seeds.WHEAT:
        expected: 1
        actual: <missing>

    step:
        expected: 1
        actual: 0

Turn: 2
Action:
    P0: PLANT WHEAT
    P1: PASS
Differences:

    farms[0].money:
        expected: 2990
        actual: 3000

    farms[0].tiles[4][4].kind:
        expected: PLANT
        actual: EMPTY

    farms[0].tiles[4][4].plant:
        expected: {"crop": "WHEAT", "planted_day": 0, "watered_today": false, "consecutive_unwatered": 1, "yield_units": 1, "fertilized_until_day": -1, "max_lifespan_step": 120}
        actual: <missing>

    hour:
        expected: 2
        actual: 0

    market.inventory.CARROT:
        expected: 9999
        actual: 10000

    market.inventory.EGG:
        expected: 9999
        actual: 10000

    market.inventory.MELON:
        expected: 9999
        actual: 10000

    market.inventory.MILK:
        expected: 9999
        actual: 10000

    market.inventory.STRAWBERRY:
        expected: 9999
        actual: 10000

    market.inventory.TOMATO:
        expected: 9999
        actual: 10000

    market.inventory.WHEAT:
        expected: 9999
        actual: 10000

    market.inventory.WOOL:
        expected: 9999
        actual: 10000

    market.prices.CARROT:
        expected: 36
        actual: 35

    market.prices.MELON:
        expected: 256
        actual: 250

    market.prices.MILK:
        expected: 169
        actual: 160

    market.prices.STRAWBERRY:
        expected: 128
        actual: 120

    market.prices.WHEAT:
        expected: 26
        actual: 25

    market.prices.WOOL:
        expected: 206
        actual: 200

    step:
        expected: 2
        actual: 0

Turn: 3
Action:
    P0: WATER
    P1: PASS
Differences:

    farms[0].money:
        expected: 2990
        actual: 3000

    farms[0].tiles[4][4].kind:
        expected: PLANT
        actual: EMPTY

    farms[0].tiles[4][4].plant:
        expected: {"crop": "WHEAT", "planted_day": 0, "watered_today": true, "consecutive_unwatered": 1, "yield_units": 1, "fertilized_until_day": -1, "max_lifespan_step": 120}
        actual: <missing>

    hour:
        expected: 3
        actual: 0

    market.inventory.CARROT:
        expected: 9999
        actual: 10000

    market.inventory.EGG:
        expected: 9999
        actual: 10000

    market.inventory.MELON:
        expected: 9999
        actual: 10000

    market.inventory.MILK:
        expected: 9999
        actual: 10000

    market.inventory.STRAWBERRY:
        expected: 9999
        actual: 10000

    market.inventory.TOMATO:
        expected: 9999
        actual: 10000

    market.inventory.WHEAT:
        expected: 9999
        actual: 10000

    market.inventory.WOOL:
        expected: 9999
        actual: 10000

    market.prices.CARROT:
        expected: 36
        actual: 35

    market.prices.MELON:
        expected: 256
        actual: 250

    market.prices.MILK:
        expected: 169
        actual: 160

    market.prices.STRAWBERRY:
        expected: 128
        actual: 120

    market.prices.WHEAT:
        expected: 26
        actual: 25

    market.prices.WOOL:
        expected: 206
        actual: 200

    step:
        expected: 3
        actual: 0

Turn: 4
Action:
    P0: PASS | BUY_SEED CARROT 1
    P1: PASS
Differences:

    farms[0].money:
        expected: 2970
        actual: 3000

    farms[0].tiles[4][4].kind:
        expected: PLANT
        actual: EMPTY

    farms[0].tiles[4][4].plant:
        expected: {"crop": "WHEAT", "planted_day": 0, "watered_today": true, "consecutive_unwatered": 1, "yield_units": 1, "fertilized_until_day": -1, "max_lifespan_step": 120}
        actual: <missing>

    hour:
        expected: 4
        actual: 0

    market.inventory.CARROT:
        expected: 9999
        actual: 10000

    market.inventory.EGG:
        expected: 9999
        actual: 10000

    market.inventory.MELON:
        expected: 9999
        actual: 10000

    market.inventory.MILK:
        expected: 9999
        actual: 10000

    market.inventory.STRAWBERRY:
        expected: 9999
        actual: 10000

    market.inventory.TOMATO:
        expected: 9999
        actual: 10000

    market.inventory.WHEAT:
        expected: 9999
        actual: 10000

    market.inventory.WOOL:
        expected: 9999
        actual: 10000

    market.prices.CARROT:
        expected: 36
        actual: 35

    market.prices.MELON:
        expected: 256
        actual: 250

    market.prices.MILK:
        expected: 169
        actual: 160

    market.prices.STRAWBERRY:
        expected: 128
        actual: 120

    market.prices.WHEAT:
        expected: 26
        actual: 25

    market.prices.WOOL:
        expected: 206
        actual: 200

    private.seeds.CARROT:
        expected: 1
        actual: <missing>

    step:
        expected: 4
        actual: 0

```
