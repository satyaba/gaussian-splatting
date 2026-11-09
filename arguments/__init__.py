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
        self.densify_until_iter = 15_000
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
