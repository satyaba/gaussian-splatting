"""Label adapter registry."""
from __future__ import annotations

from .base import LabelAdapter
from .replica import ReplicaLabelAdapter

_ADAPTERS = {
    "replica": ReplicaLabelAdapter,
}


def get_adapter(name: str, dataset_dir, cfg=None) -> LabelAdapter:
    key = (name or "replica").lower()
    if key not in _ADAPTERS:
        raise KeyError(f"unknown label adapter '{name}'. available: {sorted(_ADAPTERS)}")
    return _ADAPTERS[key](dataset_dir, cfg)


__all__ = ["LabelAdapter", "ReplicaLabelAdapter", "get_adapter"]
