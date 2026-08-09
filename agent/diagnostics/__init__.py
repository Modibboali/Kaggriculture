"""Environment compatibility & benchmark suite.

Reusable diagnostics for deciding whether the official Kaggle environment can
be used directly inside MCTS, or whether a domain-model simulator is required.
Entry point:

    python -m agent.diagnostics.environment_probe

The package never imports ``kaggle_environments`` at module level; probes load
it lazily so the rest of the agent runs fine without it installed. The
``environment_probe`` submodule is intentionally NOT imported here so that
``python -m`` executes it cleanly.
"""

from . import probes
from .utils import DiagnosticRun, ProbeResult

__all__ = ["DiagnosticRun", "ProbeResult", "probes"]
