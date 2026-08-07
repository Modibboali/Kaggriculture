"""The base action value object.

An action is a strongly typed, immutable statement of intent: it carries
enough information to be executed later by the simulator, but performs no
legality checking and no state changes of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from .action_type import ActionType


@dataclass(frozen=True, slots=True, kw_only=True)
class Action:
    """A single action submitted for one unit or market slot.

    ``action_type`` defaults to ``PASS`` so ``Action()`` is the canonical
    no-op action; every specialized subclass fixes it to its own value.

    ``kw_only=True`` keeps construction keyword-based (``PlantAction(crop=...)``),
    which removes the footgun of passing a parameter into the inherited
    ``action_type`` slot positionally.
    """

    action_type: ActionType = ActionType.PASS

    def __str__(self) -> str:
        """The canonical command string, e.g. ``"PASS"``.

        Subclasses override this to append their parameters, so the string
        form doubles as a compact, environment-compatible rendering that is
        useful for logging and replay.
        """
        return self.action_type.label


# Canonical no-op action. A shared immutable instance keeps search code from
# allocating a fresh object for every pass (mirrors the tile singletons in
# the state model).
PASS: Action = Action()
