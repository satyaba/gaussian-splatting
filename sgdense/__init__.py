"""SGDense: segmentation-guided density control for 3DGS.

Integrates the SGDense_3DGS_Experiment methodology:

  - seg_hist_readout : read-outs from the persisted per-Gaussian mass
    histogram M (consistency / structure / disagree / entropy / top2 / S).
  - generate_category_masks : 6-way taxonomy (unobserved / interior /
    ambiguous / floaters / default / boundary) from the read-outs plus a
    geometric support test.
  - geometric : 3D grid kNN + reach/height support test.
  - texture : per-class Laplacian texture-frequency solver (s_tilde).
  - config : DensificationConfig driving the densification policy.
  - densify_tracker : DensifyTracker, logs clone/split/prune events + lineage.
  - viz : post-hoc plotting of the logged events (matplotlib imported lazily).

The densification lifecycle itself lives on GaussianModel (scene/gaussian_model.py);
this package holds the separable policy/geometry/texture maths.
"""

from sgdense.config import DensificationConfig
from sgdense.readout import seg_hist_readout
from sgdense.categories import CATEGORY_NAMES, generate_category_masks
from sgdense.geometric import build_grid, pick_cell_size, grid_knn, reach_height, thin, p99
from sgdense.texture import class_freq_thres, compute_texture_freq
from sgdense.densify_tracker import DensifyTracker, TrackerLog, load_events
from sgdense.viz import (
    plot_timeline,
    plot_conditions,
    plot_prune_breakdown,
    draw_event_3d,
    plot_event_3d,
    show_event,
    LiveDensifier3D,
    replay,
    animate_replay,
    plotly_replay,
    build_family_index,
    show_lineage,
)

__all__ = [
    "DensificationConfig",
    "seg_hist_readout",
    "CATEGORY_NAMES",
    "generate_category_masks",
    "build_grid",
    "pick_cell_size",
    "grid_knn",
    "reach_height",
    "thin",
    "p99",
    "class_freq_thres",
    "compute_texture_freq",
    "DensifyTracker",
    "TrackerLog",
    "load_events",
    "plot_timeline",
    "plot_conditions",
    "plot_prune_breakdown",
    "draw_event_3d",
    "plot_event_3d",
    "show_event",
    "LiveDensifier3D",
    "replay",
    "animate_replay",
    "plotly_replay",
    "build_family_index",
    "show_lineage",
]
