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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, num_segmentation_classes : int = -1):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # Per-view GT segmentation for the segmentation-guided density-control pass:
    # int64 [H, W] on CUDA, -1 = void. An empty tensor signals "no segmentation"
    # for this view; the kernel then receives a null pointer and casts no votes.
    gt_segmentation = getattr(viewpoint_camera, "gt_segmentation", None)
    if gt_segmentation is None:
        gt_segmentation = torch.empty(0, dtype=torch.long, device="cuda")
    else:
        gt_segmentation = gt_segmentation.to(device="cuda", dtype=torch.long).contiguous()

    # Per-Gaussian segmentation vote buffer [P, C], filled in-kernel as
    # sum(alpha * T) at [gaussian_id, gt_label]; read back via render_pkg["seg_votes"].
    # Empty (0 elements) when segmentation is off, which disables voting.
    # doc 03 §3.4: persist M on the model. Decay in place, then hand the SAME
    # tensor to the kernel so its atomicAdd lands directly in pc.seg_hist. Decay AND
    # the pass-through stay INSIDE this guard, so validation/GUI renders
    # (num_segmentation_classes = -1) never touch M (correctness, doc 03 §4.4).
    # Decay-on-observation (doc 03 §3.3/§6.2): this mul_ is decay-ALL; the rows it
    # wrongly decayed are restored below, once the rasterizer reports which
    # Gaussians were visible. Net effect: observed rows = decay-then-add (§3.1);
    # unobserved rows keep their evidence (no erosion for being out of frustum).
    if (num_segmentation_classes > 0 and gt_segmentation.numel() > 0
            and pc.seg_hist.numel() > 0 and not pc.seg_hist_frozen):
        pc.seg_hist.mul_(pc.seg_hist_decay)
        vote_buffer = pc.seg_hist
    else:
        vote_buffer = torch.empty(0, dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        gt_segmentation=gt_segmentation,
        num_segmentation_classes=num_segmentation_classes,
        vote_buffer=vote_buffer,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if separate_sh:
        rendered_image, rendered_seg, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            seg_encoding = pc.get_segmentation_encoding,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, rendered_seg, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            seg_encoding = pc.get_segmentation_encoding,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

    # Decay-on-observation (doc 03 §3.3/§6.2): the mul_ above decayed EVERY row,
    # but only Gaussians the rasterizer actually rendered cast a vote. radii == 0
    # is the culled / zero-radius set -- exactly the rows that received no
    # atomicAdd -- so undo the decay there (x / beta ~= x; ~1 ulp/step, not
    # bit-exact). Must run here: the visible set only exists after the render.
    if vote_buffer.numel() > 0:
        unobs = (radii == 0).nonzero(as_tuple=False).squeeze(1)
        vote_buffer[unobs] = vote_buffer[unobs] * (1.0 / pc.seg_hist_decay)

    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "rendered_seg": rendered_seg,
        "seg_votes": vote_buffer,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }

    return out
