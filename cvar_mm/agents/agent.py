"""
Agent base class.

Kept in a separate module with zero intra-package imports
to avoid circular dependencies between ``agents`` and ``gym``.
"""

from __future__ import annotations

import numpy as np


class Agent:
    """Minimal base following mbt-gym's ``Agent``."""

    def get_action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError
