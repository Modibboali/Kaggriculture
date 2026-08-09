"""Probe 4: state serialization (save and restore).

Determines whether environment state can be saved to bytes and restored, how
fast that round-trip is, and whether the environment-native ``toJSON()``
replay format is available.
"""

from __future__ import annotations

import json as jsonlib
import logging
import pickle
import time
from typing import Any, Callable

from ..utils import ProbeResult, attempt, measure

logger = logging.getLogger(__name__)

GAME = "kaggriculture"


def _advanced_env() -> Any:
    """Return an environment with a few steps of history, for serialization."""
    import kaggle_environments

    from .stepping import pass_agent

    env = kaggle_environments.make(GAME)
    for _ in range(3):
        try:
            env.step([pass_agent, pass_agent])
        except Exception:  # noqa: BLE001
            break
    return env


def run() -> ProbeResult:
    """Try pickle / cloudpickle / toJSON round-trips and time them."""
    try:
        env = _advanced_env()
    except ImportError as exc:
        return ProbeResult(
            name="serialization",
            success=False,
            summary=f"kaggle_environments is not installed: {exc}",
            details={"installed": False},
            duration_s=0.0,
            errors=(),
            mcts_verdict=False,
        )

    details: dict[str, Any] = {}

    def try_pickle(
        name: str,
        dumps: Callable[[Any], bytes],
        loads: Callable[[bytes], Any],
    ) -> None:
        entry: dict[str, Any] = {}
        try:
            data, dump_s, _ = measure(lambda: dumps(env))
            start = time.perf_counter()
            restored = loads(data)
            load_s = time.perf_counter() - start
            entry["ok"] = True
            entry["size_bytes"] = len(data)
            entry["dump_s"] = round(dump_s, 6)
            entry["load_s"] = round(load_s, 6)
            entry["restore_ok"] = type(restored) is type(env)
            entry["throughput_bytes_per_s"] = round(len(data) / (dump_s + load_s), 1)
        except Exception as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["restore_ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        details[name] = entry

    try_pickle("pickle", pickle.dumps, pickle.loads)

    try:
        import cloudpickle
    except ImportError:
        details["cloudpickle_installed"] = False
    else:
        details["cloudpickle_installed"] = True
        try_pickle("cloudpickle", cloudpickle.dumps, cloudpickle.loads)

    # toJSON() returns environment *metadata* (not the replay); record that.
    meta_entry: dict[str, Any] = {}
    try:
        meta, meta_s, _ = measure(lambda: env.toJSON())
        meta_entry["ok"] = True
        meta_entry["is_dict"] = isinstance(meta, dict)
        meta_entry["s"] = round(meta_s, 6)
    except Exception as exc:  # noqa: BLE001
        meta_entry["ok"] = False
        meta_entry["error"] = f"{type(exc).__name__}: {exc}"
    details["toJSON_metadata"] = meta_entry

    # Serialize the actual replay (env.steps) as JSON.
    replay_entry: dict[str, Any] = {}
    try:
        js, js_s, _ = measure(lambda: jsonlib.dumps(env.steps, default=str))
        replay_entry["ok"] = True
        replay_entry["size_bytes"] = len(js)
        replay_entry["s"] = round(js_s, 6)
        replay_entry["steps"] = len(env.steps)
    except Exception as exc:  # noqa: BLE001
        replay_entry["ok"] = False
        replay_entry["error"] = f"{type(exc).__name__}: {exc}"
    details["replay_json"] = replay_entry

    any_roundtrip = bool(
        details["pickle"].get("restore_ok", False)
        or details.get("cloudpickle", {}).get("restore_ok", False)
    )
    details["any_roundtrip_works"] = any_roundtrip

    summary = (
        f"pickle roundtrip={details['pickle'].get('restore_ok')}, "
        f"replay_json={replay_entry.get('ok')}"
    )
    logger.info(summary)
    return ProbeResult(
        name="serialization",
        success=any_roundtrip,
        summary=summary,
        details=details,
        duration_s=0.0,
        mcts_verdict=any_roundtrip,
    )


if __name__ == "__main__":
    from ..utils import configure_logging, run_probe

    configure_logging()
    result = run_probe("serialization", run)
    logger.info("probe result: success=%s summary=%s", result.success, result.summary)
