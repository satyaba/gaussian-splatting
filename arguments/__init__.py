#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.seg_encoding_dim = 32
        self.num_segmentation_classes = 100  # office_0 ids reach 98 (doc 03 §6.4 — safety-critical)
        self.segmentation_path = ""  # folder of uint8 class-id PNGs (255=void); "" disables L_seg
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25_000  # SGDense standard: extended densification window
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.lambda_seg = 1.0  # weight on L_seg (segmentation cross-entropy)
        self.random_background = False
        self.optimizer_type = "default"
        self.seg_encoding_lr = 0.0025
        self.seg_warmup_iters = 2_000          # SGDense run26 default
        self.decoder_lr_init = 0.001
        self.decoder_lr_final = 0.0001
        self.decoder_lr_decay_start = 7_000   # < seg_warmup_iters: decay begins before warmup ends (open tuning item)
        self.decoder_lr_decay_iters = 15_000  # decay spans [decay_start, decay_start + decay_iters]
        # Persisted per-Gaussian segmentation mass histogram (doc 03)
        self.seg_hist_decay = 0.99             # beta: exponential decay per iteration (§6.3)
        self.seg_hist_min_evidence = 1.0       # S gate, mass not count (§6.5)
        self.seg_hist_inherit_discount = 0.5   # c: M_child = c * M_parent at densify (§6.1)
        super().__init__(parser, "Optimization Parameters")

class SGDenseParams(ParamGroup):
    """SGDense segmentation-guided density control (run26 methodology).

    Route switches, category thresholds (TAU_*), and the geometric support
    test parameters. Bools defaulting True carry explicit --no_* negations
    registered in train.py (ParamGroup only emits store_true).
    """
    def __init__(self, parser):
        # --- route switches ---
        self.lvl1_floaters = True
        self.lvl2_floaters = True
        self.position_based_thresh = False
        self.texture_based_thresh = False
        self.texture_based_scale = False
        # --- densification event logging (sgdense.densify_tracker) ---
        # Opt-in: when set, train.py attaches a DensifyTracker and writes
        # <model_path>/densify_events.pt (events + lineage + prune reasons).
        self.log_densify_events = False
        # --- accumulated-gradient statistics (sgdense.grad_stats) ---
        # Opt-in: logs the densification trigger distribution (per event) + the
        # raw per-iteration 2D-mean gradient norm to <model_path>/grad_stats.json.
        self.log_grad_stats = False
        # --- densification gradient source (Option A) ---
        # On (default) = the clone/split trigger accumulates the PHOTOMETRIC
        # (+ depth) position gradient only, decoupling densification from the
        # segmentation loss so the segmentation cross-entropy cannot inflate the
        # trigger; the full loss still drives the optimiser. Costs an extra
        # rasteriser backward pass during the densification window.
        # Pass --no_densify_photo_grad_only to restore upstream behaviour (the
        # gradient of the *total* loss w.r.t. the screen-space means drives the
        # trigger, so every loss term including L_seg can inflate it).
        self.densify_photo_grad_only = True
        # --- photo-weight ramp (seg-warmup coupling) ---
        # On (default) = the photometric term is suppressed to 0 during the seg
        # warmup and ramps back to full weight over ~10x the warmup length. The
        # densification trigger reads the photometric(+depth) objective, so the
        # ramp also damps clone/split pressure across that span. Off = full
        # photometric weight from iteration 0, decoupling the ramp from the
        # warmup gate (--seg_warmup_iters) so the two can be ablated separately.
        self.photo_weight_ramp = True
        # --- floater lifecycle ('prune' | 'freeze') ---
        self.floater_action = "prune"
        # --- POSITION_BASED_THRESH per-regime scaling-threshold weights ---
        self.interior_weight = 1.0
        self.boundary_weight = 0.5
        self.other_weight = 1.0
        # --- TEXTURE_BASED_THRESH (gradient threshold) ---
        self.texture_beta = 0.5
        # --- TEXTURE_BASED_SCALE (clone/split size threshold) ---
        self.texture_scale_beta = 0.5
        # --- SEG_UNCERTAINTY_THRESH (dynamic per-Gaussian trigger weighting, B) ---
        # When on, the clone/split trigger gradient is multiplied per-Gaussian by
        #   w = clamp(1 + alpha * (z - z_ref), w_min, w_max)
        # where z is a DYNAMIC per-Gaussian uncertainty read from the persisted
        # histogram (entropy normalized to [0,1], or disagree = 1 - top1) and z_ref
        # is the mean over observed Gaussians (so the global rate is unchanged --
        # this REDISTRIBUTES densification). alpha > 0 densifies MORE where seg is
        # uncertain; alpha < 0 suppresses it. Rows with evidence S <=
        # seg_hist_min_evidence keep a neutral weight of 1.
        self.seg_uncertainty_thresh = False
        self.seg_uncertainty_stat = "entropy"   # "entropy" | "disagree"
        self.seg_uncertainty_alpha = 0.5
        self.seg_uncertainty_wmin = 0.25
        self.seg_uncertainty_wmax = 4.0
        # --- category thresholds (TAU_*) ---
        self.tau_p1 = 0.85
        self.tau_p2_high = 0.25
        self.tau_entropy_low = 0.25
        self.tau_entropy_high = 1.5
        self.tau_tail_high = 0.25
        self.floater_opacity_thres = 0.501
        # --- geometric support test (reach/height) ---
        self.grid_target = 8
        self.knn_k = 12
        self.knn_kmin = 6
        self.knn_m = 32
        self.holdout_frac = 0.10
        self.thin_ratio_div = 3.0
        # --- segmentation-boundary placement (docs/08) ---
        # All inert by default: 'random' reproduces vanilla N(0, Sigma) split
        # placement. 'boundary_normal' places split children along the local
        # class-boundary normal derived from the persisted histogram.
        self.placement_mode = "random"        # 'random' | 'boundary_normal'
        self.placement_rho = 0.5
        self.placement_knn_k = 12
        self.placement_knn_kmin = 6
        self.placement_use_tangent_fallback = True
        self.placement_feasibility = True
        self.placement_max_retries = 2
        # --- texture-frequency solver (TEXTURE_BASED_THRESH) ---
        self.tex_erode = 3
        self.tex_min_px = 500
        self.tex_min_views = 3
        self.tex_k = -1          # <=0 means "auto" (25th pct of pixel counts)
        self.tex_step = 1
        self.tex_no_illum_norm = True
        self.tex_blur_sigma = 1.0
        super().__init__(parser, "SGDense Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
