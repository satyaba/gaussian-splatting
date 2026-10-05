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


    def __init__(self, sh_degree, optimizer_type="default", seg_encoding_dim=32, num_segmentation_classes=0):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self.seg_encoding_dim = seg_encoding_dim
        self.num_segmentation_classes = num_segmentation_classes
        self.seg_hist_decay = 1.0                  # beta; overwritten in training_setup
        self.seg_hist_inherit_discount = 1.0       # c; overwritten in training_setup
        self.seg_hist_frozen = False               # warmup: suppress M writes (set via set_gaussian_properties_trainable)
        self.seg_split_deterministic = True        # T0: overwritten in training_setup
        self.seg_split_offset_ratio = 0.5          # T0: overwritten in training_setup
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._segmentation_encoding = torch.empty(0)
        self.seg_l1 = torch.empty(0, dtype=torch.long)
        self.seg_n1 = torch.empty(0, dtype=torch.int32)
        self.seg_l2 = torch.empty(0, dtype=torch.long)
        self.seg_n2 = torch.empty(0, dtype=torch.int32)
        self.seg_n_tot = torch.empty(0, dtype=torch.int32)
        self.seg_hist = torch.empty(0)  # [N, C] fp32 persisted mass histogram (doc 03)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
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
            self._segmentation_encoding,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.seg_l1,
            self.seg_n1,
            self.seg_l2,
            self.seg_n2,
            self.seg_n_tot,
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
        self._segmentation_encoding,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        seg_l1,
        seg_n1,
        seg_l2,
        seg_n2,
        seg_n_tot,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.seg_l1 = seg_l1
        self.seg_n1 = seg_n1
        self.seg_l2 = seg_l2
        self.seg_n2 = seg_n2
        self.seg_n_tot = seg_n_tot
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
    def get_segmentation_encoding(self):
        # Raw passthrough, no activation (locked decision) — the decoder
        # applies the only nonlinearity.
        return self._segmentation_encoding

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
        seg_encoding = 0.01 * torch.randn((fused_point_cloud.shape[0], self.seg_encoding_dim), dtype=torch.float, device="cuda")
        self._segmentation_encoding = nn.Parameter(seg_encoding.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Segmentation consistency counters (integer counts; labels long, -1 = empty slot).
        self.seg_l1 = torch.full((self.get_xyz.shape[0], 1), -1, dtype=torch.long, device="cuda")
        self.seg_n1 = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.int32, device="cuda")
        self.seg_l2 = torch.full((self.get_xyz.shape[0], 1), -1, dtype=torch.long, device="cuda")
        self.seg_n2 = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.int32, device="cuda")
        self.seg_n_tot = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.int32, device="cuda")
        # Persisted [N, C] mass histogram (doc 03 §4.3); decay/inherit knobs live on the model.
        self.seg_hist_decay = training_args.seg_hist_decay
        self.seg_hist_inherit_discount = training_args.seg_hist_inherit_discount
        self.seg_split_deterministic = training_args.seg_split_deterministic
        self.seg_split_offset_ratio = training_args.seg_split_offset_ratio
        self.seg_hist = torch.zeros((self.get_xyz.shape[0], self.num_segmentation_classes),
                                    dtype=torch.float32, device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._segmentation_encoding], 'lr': training_args.seg_encoding_lr, "name": "seg_encoding"}
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

    def set_gaussian_properties_trainable(self, flag: bool):
        """Warmup freeze: toggle trainability of the Gaussian properties EXCEPT
        the segmentation encoding, and freeze/unfreeze the persisted seg_hist.

        When frozen (flag=False) the six photometric/geometric groups carry no
        gradient and the renderer skips every write to M, so only the
        segmentation encoding and the decoder warm up (doc 03 §3.4).
        """
        for p in (self._xyz, self._features_dc, self._features_rest,
                  self._opacity, self._scaling, self._rotation):
            p.requires_grad_(flag)
        self.seg_hist_frozen = not flag

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
        for i in range(self._segmentation_encoding.shape[1]):
            l.append('seg_{}'.format(i))
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
        seg = self._segmentation_encoding.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, seg), axis=1)
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

        # Segmentation encoding columns; legacy PLYs (pre-seg) fall back to zeros
        # so render.py/metrics.py keep working on old checkpoints.
        seg_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("seg_")]
        if len(seg_names) == self.seg_encoding_dim:
            seg_names = sorted(seg_names, key = lambda x: int(x.split('_')[-1]))
            segs = np.zeros((xyz.shape[0], len(seg_names)))
            for idx, attr_name in enumerate(seg_names):
                segs[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            print(f"Warning: found {len(seg_names)} seg_ columns in PLY (expected {self.seg_encoding_dim}); zero-initializing segmentation encoding.")
            segs = np.zeros((xyz.shape[0], self.seg_encoding_dim), dtype=np.float32)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._segmentation_encoding = nn.Parameter(torch.tensor(segs, dtype=torch.float, device="cuda").requires_grad_(True))

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
        self._segmentation_encoding = optimizable_tensors["seg_encoding"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

        self.seg_l1 = self.seg_l1[valid_points_mask]
        self.seg_n1 = self.seg_n1[valid_points_mask]
        self.seg_l2 = self.seg_l2[valid_points_mask]
        self.seg_n2 = self.seg_n2[valid_points_mask]
        self.seg_n_tot = self.seg_n_tot[valid_points_mask]
        self.seg_hist = self.seg_hist[valid_points_mask]


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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_seg_encoding, new_tmp_radii, new_seg_hist=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "seg_encoding" : new_seg_encoding}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._segmentation_encoding = optimizable_tensors["seg_encoding"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Fresh observation window: same dtypes as training_setup (labels long, -1 slot).
        self.seg_l1 = torch.full((self.get_xyz.shape[0], 1), -1, dtype=torch.long, device="cuda")
        self.seg_n1 = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.int32, device="cuda")
        self.seg_l2 = torch.full((self.get_xyz.shape[0], 1), -1, dtype=torch.long, device="cuda")
        self.seg_n2 = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.int32, device="cuda")
        self.seg_n_tot = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.int32, device="cuda")

        # doc 03 §6.1: children INHERIT identity (c * M_parent). This is an APPEND,
        # not a zero-reset like the counters above — the histogram is persistent.
        if new_seg_hist is not None:
            self.seg_hist = torch.cat((self.seg_hist, new_seg_hist), dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if self.seg_split_deterministic:
            sel_scaling = self.get_scaling[selected_pts_mask]
            K = sel_scaling.shape[0]
            major_len, major_idx = sel_scaling.max(dim=1)
            fracs = (torch.linspace(-1.0, 1.0, N, device="cuda") if N > 1 else torch.zeros(1, device="cuda"))
            offs = torch.zeros((K,N, 3), device="cuda")
            offs.scatter_(2, major_idx[:, None, None].expand(K, N, 1), (self.seg_split_offset_ratio * major_len)[:, None].mul(fracs[None, :]).unsqueeze(-1))
            # Match the runtime tiling: base/rotation use .repeat(N,1) so row i
            # belongs to parent (i % K). Tile the offsets the same way (transpose,
            # NOT reshape) or children inherit the wrong parent's offset vector.
            samples = offs.transpose(0, 1).reshape(K * N, 3)
        else:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)

        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_seg_encoding = self._segmentation_encoding[selected_pts_mask].repeat(N,1)
        new_seg_hist = self.seg_hist[selected_pts_mask].repeat(N,1) * self.seg_hist_inherit_discount
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_seg_encoding, new_tmp_radii, new_seg_hist)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_seg_encoding = self._segmentation_encoding[selected_pts_mask]
        new_seg_hist = self.seg_hist[selected_pts_mask] * self.seg_hist_inherit_discount
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_seg_encoding, new_tmp_radii, new_seg_hist)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    @torch.no_grad()
    def seg_hist_readout(self, eps=1e-8):
        """Read-outs from the persisted mass histogram M (doc 03 §3.2).

        Returns (consistency, structure, disagree, S), each [N]:
          consistency = max_c P[g, c]     (replaces the old top1/tot proxy)
          structure   = P_(1) + P_(2)     (boundary-ness)
          disagree    = 1 - P_(1)
          S           = sum_c M[g, c]     (evidence = mass, NOT count)
        Rows with S == 0 give P == 0, so their 'disagree' looks maximal — gate
        them out with S (`S > seg_hist_min_evidence`), never with disagree.
        """
        if self.seg_hist.numel() == 0:
            z = torch.zeros((self.get_xyz.shape[0],), device="cuda")
            return z, z, z, z
        S = self.seg_hist.sum(dim=1)
        P = self.seg_hist / S.clamp_min(eps).unsqueeze(1)
        k = min(2, P.shape[1])
        top2 = torch.topk(P, k=k, dim=1).values
        consistency = top2[:, 0]
        structure = top2[:, 0] + top2[:, 1] if k == 2 else top2[:, 0]
        disagree = 1.0 - top2[:, 0]
        return consistency, structure, disagree, S

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


def save_seg_hist(gaussians, iteration, path):
    """Persist the [N, C] segmentation mass histogram M as a compressed NumPy
    .npz (doc 03 §4.3). Also pandas-friendly:
        arr = np.load(path)["seg_hist"]; pd.DataFrame(arr)

    No-op when the histogram is unallocated (num_segmentation_classes <= 0).
    """
    M = gaussians.seg_hist
    if M.numel() == 0:
        return
    np.savez_compressed(
        path,
        seg_hist=M.detach().to("cpu", torch.float32).numpy(),
        iteration=np.int64(iteration),
        num_gaussians=np.int64(M.shape[0]),
        num_segmentation_classes=np.int64(gaussians.num_segmentation_classes),
        seg_hist_decay=np.float32(gaussians.seg_hist_decay),
    )
