"""Scratch experiment: probe Kaggle observation/step/day/hour + HARVEST + lifecycle.

Not part of the shipped agent; used to confirm ground-truth rules.
"""
from __future__ import annotations

import json

import kaggle_environments

CFG = {"episodeSteps": 720, "seed": 1}


def make_env():
    return kaggle_environments.make("kaggriculture", configuration=CFG, debug=True)


def main() -> None:
    env = make_env()
    obs0 = env.state[0]["observation"]
    print("=== INITIAL OBS keys ===")
    print(sorted(obs0.keys()))
    print("has step:", "step" in obs0)
    print("day/hour:", obs0.get("day"), obs0.get("hour"))
    print("player0 obs has step after reset:", "step" in env.state[0]["observation"])
    print("player1 obs keys:", sorted(env.state[1]["observation"].keys()))
    print("private keys:", sorted(env.state[0]["observation"]["private"].keys()))
    print("private.inventories:", env.state[0]["observation"]["private"]["inventories"])
    print("shed sample:", dict(list(env.state[0]["observation"]["private"]["shed"].items())[:4]))
    print("farm farmer:", env.state[0]["observation"]["farms"][0]["farmer"])
    print("market inv WHEAT:", env.state[0]["observation"]["market"]["inventory"]["WHEAT"])
    print("town:", env.state[0]["observation"]["town"])

    # Step a few PASS turns and watch step/day/hour.
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    both = [action, action]
    for _ in range(3):
        env.step(both)
        o = env.state[0]["observation"]
        print(f"after step: step={o.get('step')} day={o['day']} hour={o['hour']}")

    # Now run a long pure-PASS episode to watch day rollover around step 23->24.
    env = make_env()
    # fast-forward to step 22
    for _ in range(22):
        env.step(both)
    o = env.state[0]["observation"]
    print(f"\nAt step ~22: step={o.get('step')} day={o['day']} hour={o['hour']}")
    for i in range(4):
        env.step(both)
        o = env.state[0]["observation"]
        print(f"  rollover step {i}: step={o.get('step')} day={o['day']} hour={o['hour']}")


if __name__ == "__main__":
    main()
