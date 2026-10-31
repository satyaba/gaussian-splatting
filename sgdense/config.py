"""Densification policy configuration (SGDense).

Mirrors the notebook's ``config`` dict + the ``TAU_*`` thresholds + the
geometric-support parameters, as an explicit dataclass so the densification
lifecycle on GaussianModel carries no global state.
"""

from dataclasses import dataclass, asdict


@dataclass
class DensificationConfig:
    # --- route switches (notebook config dict) ---
    lvl1_floaters: bool = True
    lvl2_floaters: bool = True
    position_based_thresh: bool = False
    texture_based_thresh: bool = False

    # --- POSITION_BASED_THRESH weights (per-regime dense_weight) ---
    interior_weight: float = 1.0
    boundary_weight: float = 0.5
    other_weight: float = 1.0

    # --- TEXTURE_BASED_THRESH ---
    texture_beta: float = 0.5

    # --- category thresholds (TAU_* in the notebook) ---
    tau_p1: float = 0.85
    tau_p2_high: float = 0.25
    tau_entropy_low: float = 0.25
    tau_entropy_high: float = 1.5
    tau_tail_high: float = 0.25
    floater_opacity_thres: float = 0.501

    # --- geometric support test (reach/height) ---
    grid_target: int = 8
    knn_k: int = 12
    knn_kmin: int = 6
    knn_m: int = 32
    holdout_frac: float = 0.10
    thin_ratio_div: float = 3.0   # thin at h / thin_ratio_div

    def to_dict(self):
        return asdict(self)
