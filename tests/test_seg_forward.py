# Forward smoke test for the segmentation-encoding rendering path.
# Run on a GPU machine with the rebuilt extension installed:
#   python tests/test_seg_forward.py
# (pytest-compatible: pytest tests/test_seg_forward.py)
#
# Verifies design doc 01 §5 criterion 2: rendered_seg has shape
# (NUM_SEG_CHANNELS, H, W), is finite, and zero encodings -> zero output
# (no background compositing term for segmentation).
import torch
from argparse import ArgumentParser

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.segmentation_decoder import SegmentationDecoder
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import BasicPointCloud

H, W = 64, 64


class FakeCam:
    """Minimal viewpoint duck-type matching what render() touches."""

    def __init__(self, device="cuda"):
        self.image_height = H
        self.image_width = W
        self.FoVx = 0.9
        self.FoVy = 0.9
        self.znear = 0.01
        self.zfar = 100.0
        # Identity world->view: camera at origin looking down +z
        # (this codebase's convention: in_frustum culls p_view.z <= 0.2).
        self.world_view_transform = torch.eye(4, device=device)
        proj = get_projection().to(device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def get_projection(znear=0.01, zfar=100.0, fovX=0.9, fovY=0.9):
    from utils.graphics_utils import getProjectionMatrix
    return getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovX, fovY=fovY).transpose(0, 1)


def build_scene(seed=42, P=32, encoding_std=0.05):
    torch.manual_seed(seed)
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, seg_encoding_dim=dataset.seg_encoding_dim)
    points = torch.rand((P, 3)) * torch.tensor([1.0, 1.0, 1.0]) + torch.tensor([-0.5, -0.5, 3.5])
    pcd = BasicPointCloud(
        points.numpy(),
        torch.rand((P, 3)).numpy(),
        torch.zeros((P, 3)).numpy(),
    )
    gaussians.create_from_pcd(pcd, [], spatial_lr_scale=1.0)
    gaussians.training_setup(opt)
    # Push opacity/scale up so the synthetic splats are actually visible.
    with torch.no_grad():
        gaussians._opacity.copy_(torch.full_like(gaussians._opacity, 2.0))     # sigmoid -> ~0.88
        gaussians._scaling.copy_(torch.full_like(gaussians._scaling, -1.0))    # exp(-1) ~ 0.37
    return dataset, pipe, opt, gaussians


def test_forward_shape_and_finiteness():
    dataset, pipe, opt, gaussians = build_scene()
    cam = FakeCam()
    bg = torch.zeros(3, device="cuda")
    out = render(cam, gaussians, pipe, bg)
    rendered_seg = out["rendered_seg"]

    assert rendered_seg.shape == (32, H, W), f"bad shape {tuple(rendered_seg.shape)}"
    assert torch.isfinite(rendered_seg).all(), "non-finite values in rendered_seg"

    # Decoding sanity: decoder maps [H, W, 32] -> [H, W, num_classes]
    decoder = SegmentationDecoder(dataset.seg_encoding_dim, dataset.num_semantic_classes).cuda()
    logits = decoder(rendered_seg.permute(1, 2, 0))
    assert logits.shape == (H, W, dataset.num_semantic_classes)
    print("PASS forward shape/finiteness/decode")


def test_zero_encoding_gives_zero_seg():
    dataset, pipe, opt, gaussians = build_scene(seed=43)
    with torch.no_grad():
        gaussians._segmentation_encoding.zero_()
    cam = FakeCam()
    bg = torch.zeros(3, device="cuda")
    out = render(cam, gaussians, pipe, bg)
    rendered_seg = out["rendered_seg"]
    assert rendered_seg.abs().max().item() == 0.0, (
        f"zero encodings must render to zero seg output (no bg term); got max |value| = "
        f"{rendered_seg.abs().max().item()}"
    )
    print("PASS zero-encoding -> zero rendered_seg (no background term)")


def test_background_does_not_bleed_into_seg():
    # With a bright background the COLOR image gets + T * bg, but the seg
    # channels must not. Compare seg renders with bg=0 vs bg=1: identical.
    dataset, pipe, opt, gaussians = build_scene(seed=44)
    cam = FakeCam()
    out0 = render(cam, gaussians, pipe, torch.zeros(3, device="cuda"))
    out1 = render(cam, gaussians, pipe, torch.ones(3, device="cuda"))
    diff = (out0["rendered_seg"] - out1["rendered_seg"]).abs().max().item()
    assert diff == 0.0, f"background bled into seg channels: max diff {diff}"
    # Sanity: the color image DOES differ (bg term exists for color).
    assert (out0["render"] - out1["render"]).abs().max().item() > 0.0, \
        "color should differ between backgrounds (sanity check)"
    print("PASS background excluded from seg compositing")


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    test_forward_shape_and_finiteness()
    test_zero_encoding_gives_zero_seg()
    test_background_does_not_bleed_into_seg()
    print("ALL FORWARD TESTS PASSED")