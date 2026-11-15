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

import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        # --- SGDC semantic signature buffers (see docs/sgdc_spec.md §3) ---
        self.seg_top1 = torch.empty(0)
        self.seg_top1_cnt = torch.empty(0)
        self.seg_tot_cnt = torch.empty(0)
        self.seg_void_cnt = torch.empty(0)
        self.seg_dt_sum = torch.empty(0)
        self.seg_norm_sum = torch.empty(0)
        self.semantic_store = None
        self.sgdc_enable = False
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # --- SGDC config + semantic buffers (docs/sgdc_spec.md §3.2) ---
        self.sgdc_enable = getattr(training_args, "sgdc_enable", False)
        self.sgdc_f_b = getattr(training_args, "f_b", 0.5)
        self.sgdc_f_int = getattr(training_args, "f_int", 2.0)
        self.sgdc_eta = getattr(training_args, "eta", 0.3)
        self.sgdc_dt_norm = getattr(training_args, "sgdc_dt_norm", 100.0)
        _n = self.get_xyz.shape[0]
        self.seg_top1 = torch.zeros((_n,), device="cuda", dtype=torch.long)
        self.seg_top1_cnt = torch.zeros((_n, 1), device="cuda")
        self.seg_tot_cnt = torch.zeros((_n, 1), device="cuda")
        self.seg_void_cnt = torch.zeros((_n, 1), device="cuda")
        self.seg_dt_sum = torch.zeros((_n, 1), device="cuda")
        self.seg_norm_sum = torch.zeros((_n, 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
        if self.sgdc_enable:
            self.seg_top1 = self.seg_top1[valid_points_mask]
            self.seg_top1_cnt = self.seg_top1_cnt[valid_points_mask]
            self.seg_tot_cnt = self.seg_tot_cnt[valid_points_mask]
            self.seg_void_cnt = self.seg_void_cnt[valid_points_mask]
            self.seg_dt_sum = self.seg_dt_sum[valid_points_mask]
            self.seg_norm_sum = self.seg_norm_sum[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_seg_top1=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))

        if self.sgdc_enable and new_seg_top1 is not None:
            self.seg_top1 = torch.cat((self.seg_top1, new_seg_top1))

        _n = self.get_xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((_n, 1), device="cuda")
        self.denom = torch.zeros((_n, 1), device="cuda")
        self.max_radii2D = torch.zeros((_n), device="cuda")

        if self.sgdc_enable:
            self.seg_top1_cnt = torch.zeros((_n, 1), device="cuda")
            self.seg_tot_cnt = torch.zeros((_n, 1), device="cuda")
            self.seg_void_cnt = torch.zeros((_n, 1), device="cuda")
            self.seg_dt_sum = torch.zeros((_n, 1), device="cuda")
            self.seg_norm_sum = torch.zeros((_n, 1), device="cuda")

    def densify_and_split(self, grads, tau, theta, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the per-point gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.reshape(-1)
        padded_tau = torch.zeros((n_init_points), device="cuda")
        padded_tau[:tau.shape[0]] = tau
        padded_theta = torch.zeros((n_init_points), device="cuda")
        padded_theta[:theta.shape[0]] = theta
        selected_pts_mask = torch.where(padded_grad >= padded_tau, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > padded_theta)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_seg_top1 = self.seg_top1[selected_pts_mask].repeat(N) if self.sgdc_enable else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_seg_top1)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, tau, theta):
        # Extract points that satisfy the per-point gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= tau, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= theta)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_seg_top1 = self.seg_top1[selected_pts_mask] if self.sgdc_enable else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_seg_top1)

    def compute_sgdc_thresholds(self, grad_threshold, scene_extent):
        """Per-Gaussian densification threshold tau_i and split gate theta_i (spec §4A)."""
        cnt = self.seg_tot_cnt.squeeze(-1)
        d = (self.seg_dt_sum.squeeze(-1) / cnt.clamp_min(1.0)) / max(self.sgdc_dt_norm, 1e-6)
        d = d.clamp(0.0, 1.0)
        unseen = cnt <= 0
        d = torch.where(unseen, torch.ones_like(d), d)  # unseen -> conservative (interior)

        tau = grad_threshold * (self.sgdc_f_b * (1.0 - d) + self.sgdc_f_int * d)
        theta = self.percent_dense * scene_extent * (1.0 - self.sgdc_eta * (1.0 - d))

        if getattr(self, "_sgdc_dbg", 0) < 6:
            self._sgdc_dbg = getattr(self, "_sgdc_dbg", 0) + 1
            raw = self.seg_dt_sum.squeeze(-1) / cnt.clamp_min(1.0)
            rawv = raw[cnt > 0]
            p50 = rawv.median().item() if rawv.numel() else -1.0
            p90 = torch.quantile(rawv, 0.9).item() if rawv.numel() else -1.0
            print(f"[SGDC-diag] dt_raw px(mean/p50/p90)="
                  f"{rawv.mean().item():.1f}/{p50:.1f}/{p90:.1f} | "
                  f"d(min/mean/max)={d.min().item():.3f}/{d.mean().item():.3f}/{d.max().item():.3f} | "
                  f"unseen={unseen.float().mean().item():.3f} "
                  f"near(d<0.1)={(d < 0.1).float().mean().item():.3f} "
                  f"far(d>0.9)={(d > 0.9).float().mean().item():.3f} | "
                  f"tau[{tau.min().item():.2e},{tau.max().item():.2e}] theta_mean={theta.mean().item():.2e}")
        return tau, theta

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii

        if self.sgdc_enable:
            tau, theta = self.compute_sgdc_thresholds(max_grad, extent)
        else:
            _n = self.get_xyz.shape[0]
            tau = torch.full((_n,), max_grad, device="cuda")
            theta = torch.full((_n,), self.percent_dense * extent, device="cuda")

        self.densify_and_clone(grads, tau, theta)
        self.densify_and_split(grads, tau, theta, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_semantic_stats(self, camera, visibility_filter, store):
        """Accumulate per-Gaussian semantic signature from visible projections (spec §3.3)."""
        if store is None:
            return
        vis = visibility_filter.squeeze(-1)
        if vis.numel() == 0:
            return
        H, W = int(camera.image_height), int(camera.image_width)
        sem = store.get(camera.image_name, H, W)
        if sem is None:
            return
        label, dt = sem["label"], sem["dt"]
        with torch.no_grad():
            mu = self.get_xyz[vis]
            P = camera.full_proj_transform
            ones = torch.ones((mu.shape[0], 1), device=mu.device, dtype=mu.dtype)
            # Match CUDA's transformPoint4x4 (column-major read): p = [mu;1] @ P.
            p = torch.cat([mu, ones], dim=1) @ P
            ndc = p[:, :3] / p[:, 3:4].clamp_min(1e-6)
            # ndc2Pix(v, S) = ((v + 1) * S - 1) / 2   (rasterizer/auxiliary.h)
            u = ((ndc[:, 0] + 1) * W - 1) / 2
            v = ((ndc[:, 1] + 1) * H - 1) / 2
            ui = u.round().long().clamp(0, W - 1)
            vi = v.round().long().clamp(0, H - 1)
            lab = label[vi, ui]
            d = dt[vi, ui].unsqueeze(-1)

            void = lab == 0
            labeled = ~void

            self.seg_tot_cnt[vis] += labeled[:, None].float()
            self.seg_void_cnt[vis] += void[:, None].float()
            self.seg_dt_sum[vis] += d
            self.seg_norm_sum[vis] += 1.0

            prev = self.seg_top1[vis]
            hit = labeled & (lab == prev)
            self.seg_top1_cnt[vis] += hit[:, None].float()
            first = labeled & (~hit) & (self.seg_top1_cnt[vis, 0] == 0)
            self.seg_top1[vis] = torch.where(first, lab, prev)
            cnt = self.seg_top1_cnt[vis]
            self.seg_top1_cnt[vis] = torch.where(first[:, None], torch.ones_like(cnt), cnt)

    def boundary_loss(self, camera, visibility_filter, store, band_px=32.0, eps=1e-6):
        """Cross-boundary scale regularizer (spec §4C).

        Penalises projected Gaussian mass that crosses a semantic boundary:
            psi_iv ~= 0.5 * erfc(z),  z = sd / sigma_perp,  sigma_perp = sqrt(n^T Sigma' n)
        Minimising it elongates Gaussians tangentially and shrinks them normally at
        boundaries, without removing any capacity (so it cannot regress interiors).
        Only boundary-adjacent, labelled Gaussians are regularised.
        """
        zero = torch.zeros((), device=self.get_xyz.device)
        if store is None:
            return zero
        vis = visibility_filter.squeeze(-1)
        if vis.numel() == 0:
            return zero
        H, W = int(camera.image_height), int(camera.image_width)
        sem = store.get(camera.image_name, H, W)
        if sem is None:
            return zero
        label, dt, normal = sem["label"], sem["dt"], sem["normal"]

        mu = self.get_xyz[vis]
        ones = torch.ones((mu.shape[0], 1), device=mu.device, dtype=mu.dtype)

        # Pixel coordinates (rasterizer convention: ndc2Pix, row-vector product).
        P = camera.full_proj_transform
        p = torch.cat([mu, ones], dim=1) @ P
        ndc = p[:, :3] / p[:, 3:4].clamp_min(1e-6)
        ui = (((ndc[:, 0] + 1) * W - 1) / 2).round().long().clamp(0, W - 1)
        vi = (((ndc[:, 1] + 1) * H - 1) / 2).round().long().clamp(0, H - 1)
        dt_pix = dt[vi, ui]
        seg = self.seg_top1[vis]

        near = (dt_pix < band_px) & (seg > 0)
        if near.sum() == 0:
            return zero
        idx = vis[near]

        # Projected 2D covariance for the boundary-adjacent subset.
        mu_n = self.get_xyz[idx]
        ones_n = ones[near]
        Wv = camera.world_view_transform
        t = (torch.cat([mu_n, ones_n], dim=1) @ Wv)[:, :3]
        tz = t[:, 2].clamp_min(1e-6)
        tanx = math.tan(camera.FoVx * 0.5)
        tany = math.tan(camera.FoVy * 0.5)
        fx = 0.5 * W / tanx
        fy = 0.5 * H / tany
        tx = torch.clamp(t[:, 0] / tz, -1.3 * tanx, 1.3 * tanx) * tz
        ty = torch.clamp(t[:, 1] / tz, -1.3 * tany, 1.3 * tany) * tz
        n_pts = mu_n.shape[0]
        J = torch.zeros((n_pts, 2, 3), device=mu_n.device, dtype=mu_n.dtype)
        J[:, 0, 0] = fx / tz
        J[:, 0, 2] = -(fx * tx) / (tz * tz)
        J[:, 1, 1] = fy / tz
        J[:, 1, 2] = -(fy * ty) / (tz * tz)

        R = build_rotation(self._rotation[idx])
        S = self.get_scaling[idx]
        Sigma = R @ torch.diag_embed(S * S) @ R.transpose(1, 2)      # world cov
        W3 = Wv[:3, :3]                                              # rasterizer W
        Sigma_cam = W3.t() @ Sigma @ W3                              # = W^T Sigma W (CUDA: T^T Vrk T, T = W J)
        Sig2 = J @ Sigma_cam @ J.transpose(1, 2) + eps * torch.eye(2, device=mu_n.device, dtype=mu_n.dtype)

        lab_pix = label[vi[near], ui[near]]
        sd = torch.where(lab_pix == seg[near], dt_pix[near], -dt_pix[near])
        nrm = normal[vi[near], ui[near]]
        nx, ny = nrm[:, 0], nrm[:, 1]
        sig_perp = torch.sqrt(torch.clamp(
            Sig2[:, 0, 0] * nx * nx + 2.0 * Sig2[:, 0, 1] * nx * ny + Sig2[:, 1, 1] * ny * ny,
            min=eps))
        z = sd / sig_perp
        psi = 0.5 * torch.erfc(z / math.sqrt(2.0))
        alpha = self.get_opacity[idx].squeeze(-1)
        return (alpha * psi).sum() / max(vis.numel(), 1)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
