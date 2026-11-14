"""Replica ground-truth label adapter.

Layout (confirmed against the ``office_0/Sequence_1`` copy):

    <DATASET_DIR>/images/rgb_<N>.png        # COLMAP RGB frames
    <DATASET_DIR>/sparse/                   # COLMAP model
    <DATASET_DIR>/semantic_class/rgb_<N>.png   # GT class labels (this adapter)
    <DATASET_DIR>/semantic_instance/rgb_<N>.png # optional; absent in this copy

Class ids are sparse Replica ids (e.g. ``0 12 17 20 ... 98``). They are remapped
to a contiguous range ``1..K``; ``void_ids`` (default ``{0}``) collapse to ``0``.
When no instance labels are present the composite degenerates to class-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import LabelAdapter


def _read_ids(path: Path) -> np.ndarray:
    """Read a label image as int32 ids.

    Uses PIL so palette (``mode='P'``) PNGs yield palette *indices*, not RGB.
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:  # defensive: collapse unexpected multi-channel labels
        arr = arr[..., 0]
    return arr.astype(np.int32)


class ReplicaLabelAdapter(LabelAdapter):
    name = "replica"

    def __init__(self, dataset_dir, cfg=None):
        super().__init__(dataset_dir, cfg)

        self.class_dir = Path(self.cfg.get("class_dir") or (self.dataset_dir / "semantic_class"))
        inst = self.cfg.get("instance_dir")
        self.instance_dir = Path(inst) if inst else (self.dataset_dir / "semantic_instance")
        if not self.instance_dir.exists():
            self.instance_dir = None
        self.void_ids = {int(v) for v in self.cfg.get("void_ids", [])}

        if not self.class_dir.exists():
            raise FileNotFoundError(f"Replica class labels not found: {self.class_dir}")

        self._index = {}
        for p in sorted(self.class_dir.iterdir()):
            if p.is_file():
                self._index.setdefault(p.stem, p)

        # Build a single, global remap from the union of ids across all frames.
        uni = set()
        for p in self._index.values():
            uni.update(int(v) for v in np.unique(_read_ids(p)))
        self._classes = sorted(c for c in uni if c not in self.void_ids)
        self._remap = {c: i + 1 for i, c in enumerate(self._classes)}

    def resolve(self, image_name: str):
        return self._index.get(Path(image_name).stem)

    def load(self, image_name: str):
        p = self.resolve(image_name)
        if p is None:
            return None

        cls = _read_ids(p)
        cmap = np.zeros(cls.shape, dtype=np.int64)
        for c, r in self._remap.items():
            cmap[cls == c] = r

        if self.instance_dir is not None:
            ip = self.instance_dir / p.name
            inst = _read_ids(ip) if ip.exists() else np.zeros_like(cls)
            seg = cmap * (int(inst.max()) + 1) + inst.astype(np.int64)
            return seg.astype(np.int32)

        return cmap.astype(np.int32)

    def meta(self) -> dict:
        return {
            "name": self.name,
            "source": "replica",
            "num_classes": len(self._classes),
            "class_ids": list(self._classes),
            "void_ids": sorted(self.void_ids),
            "has_instance": self.instance_dir is not None,
        }
