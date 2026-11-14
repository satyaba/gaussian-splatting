"""Per-view semantic store.

Wraps a :class:`~sgdc.datasets.base.LabelAdapter` and, for a given
``image_name`` at the camera's training resolution, returns the canonical
segment map plus its distance transform and normal field, cached on device.

The store is keyed by ``image_name`` -- exactly the field stored on
``scene.cameras.Camera`` (``cameras.py:33``) -- so **no change to the Camera
constructor is required**.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import semantic_utils as su
from .datasets.base import LabelAdapter


class SemanticStore:
    def __init__(self, adapter: LabelAdapter, device: str = "cuda", cache_dir=None):
        self.adapter = adapter
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._mem: dict = {}

    def get(self, image_name: str, H: int, W: int):
        """Return ``{"label", "dt", "normal"}`` tensors on device, or ``None``.

        ``label`` is ``int32 [H, W]`` (0 == void); ``dt`` is ``float32 [H, W]``;
        ``normal`` is ``float32 [H, W, 2]``.
        """
        key = (image_name, int(H), int(W))
        if key in self._mem:
            return self._mem[key]

        lab = self.adapter.load(image_name)
        if lab is None:
            return None

        lab = su.resize_ids(lab, H, W)
        dt, _ = su.distance_transform(lab)
        nrm = su.normal_field(dt)

        out = {
            "label": torch.as_tensor(lab, device=self.device),
            "dt": torch.as_tensor(dt, device=self.device),
            "normal": torch.as_tensor(nrm, device=self.device),
        }
        self._mem[key] = out
        return out

    def label(self, image_name: str) -> np.ndarray | None:
        """Native-resolution canonical ids, for precompute/verification."""
        return self.adapter.load(image_name)

    def meta(self) -> dict:
        return self.adapter.meta()
