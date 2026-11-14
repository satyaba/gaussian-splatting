"""Dataset label adapters for SGDC.

Every adapter turns a dataset's native ground-truth labels into the canonical
per-view segment map consumed by SGDC:

    L_v : int32 [H, W]   composite segment id, 0 == void

The canonical id is a *composite* ``(class_id, instance_id)``. Adapters that
have instance labels encode both; adapters without them degenerate to
class-only. The density-control code only ever sees the canonical ``L_v``, so
it never needs to know where labels come from.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class LabelAdapter:
    """Base interface. Subclasses implement :meth:`resolve` and :meth:`load`."""

    name = "base"

    def __init__(self, dataset_dir, cfg=None):
        self.dataset_dir = Path(dataset_dir)
        self.cfg = dict(cfg or {})

    def resolve(self, image_name: str):
        """Return the label file for a training image name, or ``None``."""
        raise NotImplementedError

    def load(self, image_name: str):
        """Return canonical segment ids ``int32 [H, W]`` (0 == void) or ``None``."""
        raise NotImplementedError

    def meta(self) -> dict:
        return {"name": self.name, "source": self.name}
