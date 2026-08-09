"""Probe 3: environment cloning.

Investigates whether an advanced (stepped) environment can be copied with
``copy.copy``, ``copy.deepcopy``, ``pickle``, or ``cloudpickle``. Failures are
recorded with the exception and the most likely unsupported object.
"""

from __future__ import annotations

import copy
import logging
import pickle
import traceback
from typing import Any, Callable

from ..utils import ProbeResult, extract_unsupported

logger = logging.getLogger(__name__)

GAME = "kaggriculture"


def _advanced_env() -> Any:
    """Return an environment with a few steps of history, for cloning."""
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
    """Try every clone mechanism and record what works and what fails."""
    try:
        env = _advanced_env()
    except ImportError as exc:
        return ProbeResult(
            name="cloning",
            success=False,
            summary=f"kaggle_environments is not installed: {exc}",
            details={"installed": False},
            duration_s=0.0,
            errors=(),
            mcts_verdict=False,
        )

    details: dict[str, Any] = {}
    clones: dict[str, Any] = {}

    def try_method(name: str, fn: Callable[[], Any]) -> None:
        try:
            value = fn()
            clones[name] = {
                "ok": True,
                "result_type": type(value).__name__,
                "same_type_as_env": type(value) is type(env),
            }
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            clones[name] = {
                "ok": False,
                "exception": f"{type(exc).__name__}: {exc}",
                "unsupported": extract_unsupported(tb),
            }

    # The environment ships a native deep clone(); include it first.
    try_method("env.clone", lambda: env.clone())
    try_method("copy.copy", lambda: copy.copy(env))
    try_method("copy.deepcopy", lambda: copy.deepcopy(env))
    try_method("pickle", lambda: pickle.loads(pickle.dumps(env)))

    try:
        import cloudpickle
    except ImportError:
        details["cloudpickle_installed"] = False
    else:
        details["cloudpickle_installed"] = True
        try_method("cloudpickle", lambda: cloudpickle.loads(cloudpickle.dumps(env)))

    details["clones"] = clones
    any_clone = any(entry["ok"] for entry in clones.values())
    details["any_clone_works"] = any_clone

    summary = "clone support: " + ", ".join(f"{key}={entry['ok']}" for key, entry in clones.items())
    logger.info(summary)
    return ProbeResult(
        name="cloning",
        success=any_clone,
        summary=summary,
        details=details,
        duration_s=0.0,
        mcts_verdict=any_clone,
    )


if __name__ == "__main__":
    from ..utils import configure_logging, run_probe

    configure_logging()
    result = run_probe("cloning", run)
    logger.info("probe result: success=%s summary=%s", result.success, result.summary)
