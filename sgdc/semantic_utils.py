"""Geometric utilities over canonical segment maps (cv2/numpy only)."""
from __future__ import annotations

import cv2
import numpy as np


def resize_ids(label: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbor resize of an id map. Ids must never be interpolated."""
    label = np.asarray(label)
    if label.shape == (H, W):
        return label.astype(np.int32)
    return cv2.resize(label.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def boundary_mask(label: np.ndarray) -> np.ndarray:
    """True where a pixel is adjacent to a different segment id."""
    lab = np.asarray(label)
    b = np.zeros(lab.shape, dtype=bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    return b


def distance_transform(label: np.ndarray):
    """Euclidean distance (px) to the nearest segment boundary; 0 on boundary.

    Returns ``(dt, boundary)`` with ``dt`` float32 ``[H, W]``.
    """
    b = boundary_mask(label)
    src = (~b).astype(np.uint8)  # distanceTransform measures to nearest zero pixel
    dt = cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float32)
    return dt, b


def normal_field(dt: np.ndarray) -> np.ndarray:
    """Unit boundary normal ``[H, W, 2]`` pointing away from the nearest boundary."""
    gy, gx = np.gradient(dt.astype(np.float32))
    n = np.stack([gx, gy], axis=-1)
    norm = np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6, None)
    return (n / norm).astype(np.float32)


def signed_distance(label: np.ndarray, dt: np.ndarray, seg_ids: np.ndarray) -> np.ndarray:
    """Signed distance for pixels relative to ``seg_ids`` (>=0 inside, <0 outside)."""
    inside = np.asarray(label) == np.asarray(seg_ids)
    return np.where(inside, dt, -dt).astype(np.float32)


def segment_area_perimeter(label: np.ndarray):
    """Return ``(area, perimeter)`` dicts keyed by segment id."""
    ids, counts = np.unique(label, return_counts=True)
    area = {int(i): int(c) for i, c in zip(ids, counts)}

    b = boundary_mask(label)
    bids, bcounts = np.unique(label[b], return_counts=True)
    perim = {int(i): int(c) for i, c in zip(bids, bcounts)}
    return area, perim
