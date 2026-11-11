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

import os
import json
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import save_seg_hist
from scene.segmentation_decoder import SegmentationDecoder, save_decoder_checkpoint, load_decoder_checkpoint
from utils.general_utils import safe_state, get_expon_lr_func
from sgdense.config import DensificationConfig
from sgdense.categories import generate_category_masks
from sgdense.profiler import IterProfiler
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, SGDenseParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, sgdense, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             profile_iterations=None, profile_out=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # SGDense densification policy (run26 methodology), carried explicitly so the
    # GaussianModel lifecycle holds no global state.
    cfg = DensificationConfig(
        lvl1_floaters=sgdense.lvl1_floaters,
        lvl2_floaters=sgdense.lvl2_floaters,
        position_based_thresh=sgdense.position_based_thresh,
        texture_based_thresh=sgdense.texture_based_thresh,
        texture_based_scale=sgdense.texture_based_scale,
        floater_action=sgdense.floater_action,
        interior_weight=sgdense.interior_weight,
        boundary_weight=sgdense.boundary_weight,
        other_weight=sgdense.other_weight,
        texture_beta=sgdense.texture_beta,
        texture_scale_beta=sgdense.texture_scale_beta,
        tau_p1=sgdense.tau_p1,
        tau_p2_high=sgdense.tau_p2_high,
        tau_entropy_low=sgdense.tau_entropy_low,
        tau_entropy_high=sgdense.tau_entropy_high,
        tau_tail_high=sgdense.tau_tail_high,
        floater_opacity_thres=sgdense.floater_opacity_thres,
        grid_target=sgdense.grid_target,
        knn_k=sgdense.knn_k,
        knn_kmin=sgdense.knn_kmin,
        knn_m=sgdense.knn_m,
        holdout_frac=sgdense.holdout_frac,
        thin_ratio_div=sgdense.thin_ratio_div,
    )

    # Per-class texture score for TEXTURE_BASED_THRESH / TEXTURE_BASED_SCALE: load
    # the cached s_tilde from the model dir, else compute it from the dataset and
    # cache it.
    s_tilde = None
    if cfg.texture_based_thresh or cfg.texture_based_scale:
        s_tilde_path = os.path.join(dataset.model_path, "s_tilde.npy")
        if os.path.exists(s_tilde_path):
            s_tilde = np.load(s_tilde_path)
            print(f"[SGDense] loaded texture scores from {s_tilde_path}")
        else:
            from sgdense.texture import compute_texture_freq
            scene_dir = os.path.join(dataset.source_path, dataset.images)
            s_tilde, _k, _prior, _pc = compute_texture_freq(
                scene_dir, dataset.segmentation_path, dataset.num_segmentation_classes,
                erode=sgdense.tex_erode, min_px=sgdense.tex_min_px, min_views=sgdense.tex_min_views,
                k=None if sgdense.tex_k <= 0 else sgdense.tex_k, step=sgdense.tex_step,
                no_illum_norm=sgdense.tex_no_illum_norm, blur_sigma=sgdense.tex_blur_sigma)
            os.makedirs(dataset.model_path, exist_ok=True)
            np.save(s_tilde_path, s_tilde)
            print(f"[SGDense] computed texture scores -> {s_tilde_path}")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type,
                              seg_encoding_dim=dataset.seg_encoding_dim,
                              num_segmentation_classes=dataset.num_segmentation_classes)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    # SGDense densification event logging (opt-in via --log_densify_events):
    # attach AFTER the population is final (checkpoint restore included) so the
    # tracker's stable per-row ids stay aligned with the live rows.
    densify_tracker = None
    if getattr(sgdense, "log_densify_events", False):
        from sgdense.densify_tracker import DensifyTracker
        densify_tracker = DensifyTracker(gaussians, snapshot_max_points=5000)
        gaussians.set_densify_tracker(densify_tracker)
        print(f"[SGDense] densification event logging ON; will write "
              f"{os.path.join(dataset.model_path, 'densify_events.pt')}")

    # Accumulated-gradient diagnostics (opt-in via --log_grad_stats): records the
    # densification trigger distribution and the raw per-iteration 2D-mean gradient.
    grad_logger = None
    if getattr(sgdense, "log_grad_stats", False):
        from sgdense.grad_stats import GradStatsLogger
        grad_logger = GradStatsLogger(threshold=opt.densify_grad_threshold)
        print(f"[SGDense] gradient-stat logging ON; will write "
              f"{os.path.join(dataset.model_path, 'grad_stats.json')}")

    # Option A — densification gradient source. When on, the clone/split trigger
    # accumulates only the photometric (+ depth) position gradient, so the
    # segmentation cross-entropy (which reaches the screen-space means through the
    # rasteriser) can no longer inflate the trigger. The full loss still drives the
    # optimiser; the extra autograd pass is confined to the densification window.
    densify_photo_only = getattr(sgdense, "densify_photo_grad_only", False)
    if densify_photo_only:
        print("[SGDense] densification-gradient decoupling ON: clone/split trigger "
              "uses the photometric(+depth) position gradient only")

    # The rasterizer's channel count is a compile-time constant; keep Python-side
    # config honest about it.
    assert dataset.seg_encoding_dim == 32, "seg_encoding_dim must match NUM_SEG_CHANNELS in cuda_rasterizer/config.h (rebuild required to change)"

    # Segmentation decoder: fully decoupled from GaussianModel — own optimizer,
    # own checkpoint path.
    decoder = SegmentationDecoder(dataset.seg_encoding_dim, dataset.num_segmentation_classes).cuda()
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=opt.decoder_lr_init)
    decoder_lr_decay_args = get_expon_lr_func(opt.decoder_lr_init, opt.decoder_lr_final,
                                              max_steps=max(1, opt.decoder_lr_decay_iters))

    def get_decoder_lr(it):
        # Flat until decay_start, then exponential — decay begins BEFORE
        # seg_warmup_iters ends so confidence stabilizes early.
        if it < opt.decoder_lr_decay_start:
            return opt.decoder_lr_init
        return decoder_lr_decay_args(it - opt.decoder_lr_decay_start)

    if checkpoint:
        decoder_iter = load_decoder_checkpoint(decoder, decoder_optimizer, scene.model_path + f"/decoder{first_iter}.pth")
        assert decoder_iter == first_iter, f"decoder checkpoint iteration {decoder_iter} != gaussians checkpoint {first_iter}"

    # Warmup freeze (doc 03 §3.4): hold all Gaussian properties AND the persisted
    # seg_hist fixed so only the segmentation encoding + decoder train until
    # seg_warmup_iters. Guarded by first_iter so a resume past the boundary is
    # never re-frozen.
    gaussians.set_gaussian_properties_trainable(first_iter > opt.seg_warmup_iters)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Per-iteration phase profiler (opt-in via --profile_iterations / $SGDENSE_PROFILE_ITERS).
    # Disabled => every phase()/start()/stop() is a no-op.
    profiler = IterProfiler(
        profile_iterations,
        out_path=profile_out or os.path.join(dataset.model_path, "iter_profile.json"),
        method="sgdense",
    )
    if profiler.enabled:
        print(f"[SGDense] iteration profiling ON (spec={profiler.spec}) -> {profiler.out_path}")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_Ll1seg_for_log = 0.0
    # seg/photometric loss ratio, aggregated over a 100-iter window
    seg_ce_window_sum = 0.0
    photo_window_sum = 0.0
    seg_photo_ratio = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        profiler.begin_iteration(iteration, n_gaussians=int(gaussians.get_xyz.shape[0]))

        gaussians.update_learning_rate(iteration)

        # Warmup freeze ends here: unfreeze properties + seg_hist so density
        # control (gated below) and M accumulation resume together.
        if iteration == opt.seg_warmup_iters + 1:
            gaussians.set_gaussian_properties_trainable(True)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        with profiler.phase("render"):
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, num_segmentation_classes=dataset.num_segmentation_classes)
            image, viewspace_point_tensor, visibility_filter, radii, seg_votes = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["seg_votes"]

        with profiler.phase("loss"):
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            # SGDense photo-weight ramp: the photo term is suppressed during the seg
            # warmup, then ramps back to full weight over ~10x the warmup length.
            photo_weight = 0.0
            if opt.seg_warmup_iters <= 0:
                # Ramp disabled: keep full photometric weight from the start so
                # the objective (and the densification trigger, which reads
                # loss_no_seg below) is not damped. Also avoids // 0.
                photo_weight = 1.0
            elif iteration > opt.seg_warmup_iters:
                new_ph_weight = (iteration // opt.seg_warmup_iters) / 10
                photo_weight = new_ph_weight if new_ph_weight <= 1.0 else 1.0
            if tb_writer:
                tb_writer.add_scalar("L_photo weight", photo_weight, iteration)

            photo_loss_t = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            photo_loss_value = photo_loss_t.item()   # unweighted photometric objective, for seg/photo ratio
            loss = photo_weight * photo_loss_t

            # Depth regularization
            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            # Photometric(+depth) objective *before* the segmentation term. When
            # densify_photo_only is on, this sub-graph's position gradient is what
            # feeds the clone/split trigger (see the backward phase below).
            loss_no_seg = loss

            # Segmentation decode — AFTER rasterization, in the training loop (the
            # decoder compares decoded outputs across cameras for L_consist later).
            rendered_seg = render_pkg["rendered_seg"]                       # [NUM_SEG_CHANNELS, H, W]
            seg_logits = decoder(rendered_seg.permute(1, 2, 0))             # [H, W, num_segmentation_classes]
            gt_segmentation = getattr(viewpoint_cam, "gt_segmentation", None)
            Ll1seg = 0.0
            seg_ce_raw = None
            if gt_segmentation is not None:
                seg_ce_raw = torch.nn.functional.cross_entropy(
                    seg_logits.reshape(-1, dataset.num_segmentation_classes),
                    gt_segmentation.cuda().reshape(-1).long(),
                    ignore_index=-1)  # void rule v1: ignore pixels labeled 255
                L_seg = opt.lambda_seg * seg_ce_raw
                loss = loss + L_seg
                Ll1seg = L_seg.item()
            for param_group in decoder_optimizer.param_groups:
                param_group['lr'] = get_decoder_lr(iteration)

        with profiler.phase("backward"):
            # Densification-gradient decoupling (Option A): when enabled, extract
            # the photometric(+depth) gradient w.r.t. the screen-space means with a
            # non-accumulating autograd.grad pass, so the clone/split trigger can be
            # fed that instead of the total (photo + seg) gradient. autograd.grad
            # does NOT write .grad and retain_graph keeps the graph alive for the
            # full backward below, so parameter updates are unchanged.
            densify_point_grad = None
            if (densify_photo_only and loss_no_seg.requires_grad
                    and gt_segmentation is not None
                    and iteration < opt.densify_until_iter
                    and iteration > opt.seg_warmup_iters):
                densify_point_grad = torch.autograd.grad(
                    loss_no_seg, viewspace_point_tensor,
                    retain_graph=True, allow_unused=True)[0]
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Raw per-iteration 2D-mean gradient norm (the "gradient each Gaussian
            # sees"); logged before the optimizer zeroes the grads below. When the
            # densification gradient is decoupled, log the gradient that actually
            # drives the trigger so the stats reflect clone/split pressure.
            densify_grad_for_log = (densify_point_grad
                                    if densify_point_grad is not None
                                    else viewspace_point_tensor.grad)
            if grad_logger is not None and densify_grad_for_log is not None:
                grad_logger.observe_iter(
                    torch.norm(densify_grad_for_log[visibility_filter, :2], dim=-1),
                    iteration)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_Ll1seg_for_log = 0.4 * Ll1seg + 0.6 * ema_Ll1seg_for_log

            # seg/photometric loss ratio: raw (unweighted) CE vs the unweighted
            # photometric objective, aggregated then reported every 100 iters.
            if seg_ce_raw is not None:
                seg_ce_window_sum += seg_ce_raw.item()
                photo_window_sum += photo_loss_value
            if iteration % 100 == 0 and photo_window_sum > 0:
                seg_photo_ratio = seg_ce_window_sum / photo_window_sum
                if tb_writer:
                    tb_writer.add_scalar("loss/seg_photo_ratio", seg_photo_ratio, iteration)
                seg_ce_window_sum = 0.0
                photo_window_sum = 0.0

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "Seg Loss": f"{ema_Ll1seg_for_log:.{7}f}", "S/P": f"{seg_photo_ratio:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_decoder_checkpoint(decoder, decoder_optimizer, iteration, scene.model_path + f"/decoder{iteration}.pth")
                save_seg_hist(gaussians, iteration, scene.model_path + f"/seg_hist{iteration}.npz")

            # Densification — suppressed during the warmup freeze (doc 03 §3.4):
            # also disables prune, add_densification_stats and reset_opacity,
            # whose optimizer rewrites re-enable requires_grad and mutate values.
            # Profiling note: the per-iteration add_densification_stats below is
            # common to vanilla and SGDense, so the "densify" phase times only the
            # periodic densification EVENT (readout -> masks -> prune) and the
            # opacity reset — matching the vanilla shim.
            densify_ran = False
            if iteration < opt.densify_until_iter and iteration > opt.seg_warmup_iters:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter,
                                                  point_grad=densify_point_grad)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    densify_ran = True
                    profiler.start("densify")
                    if densify_tracker is not None:
                        densify_tracker.step = iteration
                    # Read M BEFORE any lifecycle call — densification mutates the
                    # population and M in lockstep, so this must come first.
                    consistency, structure, disagree, entropy, top2, S = gaussians.seg_hist_readout()
                    if tb_writer:
                        tb_writer.add_histogram("seg/consistency", consistency, iteration)
                        tb_writer.add_histogram("seg/structure", structure, iteration)
                        tb_writer.add_histogram("seg/disagree", disagree, iteration)
                        tb_writer.add_histogram("seg/entropy", entropy, iteration)
                        tb_writer.add_histogram("seg/evidence_S", S, iteration)
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    opacities = gaussians.get_opacity.squeeze(1)
                    gauss_xyz = gaussians.get_xyz

                    interior_mask = boundary_mask = floaters_mask = other_mask = None
                    if cfg.lvl1_floaters or cfg.lvl2_floaters or cfg.position_based_thresh:
                        (unobserved_mask, default_mask, interior_mask, floaters_mask,
                         ambiguous_mask, boundary_mask) = generate_category_masks(
                            opacities.shape[0], top2, S, entropy, gauss_xyz, opacities,
                            cfg.floater_opacity_thres, opt.seg_hist_min_evidence, cfg)
                        other_mask = default_mask | ambiguous_mask

                    # Floaters are applied only when BOTH floater levels are on (run26
                    # gating); POSITION_BASED_THRESH supplies the clone/split regimes.
                    floaters_mask_t = None
                    interior_mask_t = boundary_mask_t = other_mask_t = None
                    if cfg.lvl1_floaters and cfg.lvl2_floaters:
                        floaters_mask_t = torch.tensor(floaters_mask, device='cuda')
                        if cfg.position_based_thresh:
                            interior_mask_t = torch.tensor(interior_mask, device='cuda')
                            boundary_mask_t = torch.tensor(boundary_mask, device='cuda')
                            other_mask_t = torch.tensor(other_mask, device='cuda')

                    # Snapshot the accumulated trigger BEFORE densify_and_prune
                    # resets xyz_gradient_accum/denom.
                    if grad_logger is not None:
                        grow = grad_logger.snapshot(gaussians.xyz_gradient_accum,
                                                    gaussians.denom, iteration)
                        if tb_writer:
                            tb_writer.add_scalar("grad/acc_mean", grow["mean"], iteration)
                            tb_writer.add_scalar("grad/acc_p99", grow["p99"], iteration)
                            tb_writer.add_scalar("grad/acc_frac_over", grow["frac_over"] or 0.0, iteration)
                            tb_writer.add_scalar("grad/acc_thresh_pct", grow["thresh_pct"] or 0.0, iteration)

                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                size_threshold, radii, cfg=cfg, s_tilde=s_tilde,
                                                interior_mask=interior_mask_t, boundary_mask=boundary_mask_t,
                                                floater_mask=floaters_mask_t, other_mask=other_mask_t)
                    profiler.stop("densify")

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    densify_ran = True
                    profiler.start("densify")
                    if densify_tracker is not None:
                        densify_tracker.step = iteration
                    gaussians.reset_opacity()
                    profiler.stop("densify")

                # Close the event once per iteration (no-op when nothing happened);
                # also verifies the tracked ids still match the live population.
                if densify_tracker is not None:
                    densify_tracker.flush(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                with profiler.phase("optimizer"):
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    decoder_optimizer.step()
                    decoder_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                save_decoder_checkpoint(decoder, decoder_optimizer, iteration, scene.model_path + f"/decoder{iteration}.pth")

            profiler.end_iteration(densify=densify_ran,
                                   warmup=iteration <= opt.seg_warmup_iters)

    # Persist the densification policy for reproducibility / downstream scripts.
    with open(os.path.join(dataset.model_path, "experiment_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1)
    if densify_tracker is not None:
        events_path = os.path.join(dataset.model_path, "densify_events.pt")
        densify_tracker.save(events_path)
        print(f"[SGDense] wrote densification events -> {events_path}")
    if grad_logger is not None:
        grad_path = os.path.join(dataset.model_path, "grad_stats.json")
        grad_logger.save(grad_path)
        print(f"[SGDense] wrote gradient stats -> {grad_path}")
    if profiler.enabled:
        prof_path = profiler.save()
        if prof_path:
            print(f"[SGDense] wrote iteration profile -> {prof_path}")
    if tb_writer:
        tb_writer.flush()
        tb_writer.close()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    sp = SGDenseParams(parser)
    # Negations for the SGDense bool flags (ParamGroup only emits store_true, so
    # without these a flag can never be turned off from the command line/extra).
    # The lvl*/photo-grad flags default True; the route flags default False but
    # still need a negation so `run_by_id(extra=...)` can switch them off.
    parser.add_argument('--no_lvl1_floaters', action='store_false', dest='lvl1_floaters')
    parser.add_argument('--no_lvl2_floaters', action='store_false', dest='lvl2_floaters')
    parser.add_argument('--no_densify_photo_grad_only', action='store_false',
                        dest='densify_photo_grad_only')
    parser.add_argument('--no_position_based_thresh', action='store_false',
                        dest='position_based_thresh')
    parser.add_argument('--no_texture_based_thresh', action='store_false',
                        dest='texture_based_thresh')
    parser.add_argument('--no_texture_based_scale', action='store_false',
                        dest='texture_based_scale')
    parser.add_argument('--no_tex_no_illum_norm', action='store_false', dest='tex_no_illum_norm')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Per-iteration phase profiling (see sgdense/profiler.py). Off by default;
    # default also honours $SGDENSE_PROFILE_ITERS / $SGDENSE_PROFILE_OUT so the
    # experiment harness can enable it without editing the command.
    parser.add_argument("--profile_iterations", type=str,
                        default=os.environ.get("SGDENSE_PROFILE_ITERS"),
                        help="iterations to profile: 'a:b' range, 'a,b,c' set, or 'all'")
    parser.add_argument("--profile_out", type=str,
                        default=os.environ.get("SGDENSE_PROFILE_OUT"),
                        help="where to write iter_profile.json (default <model_path>/iter_profile.json)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), sp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
             profile_iterations=args.profile_iterations, profile_out=args.profile_out)

    # All done
    print("\nTraining complete.")
