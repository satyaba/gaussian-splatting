# MANDATORY dL_dalpha unit test (design doc 01 §2 / §5 criterion 3).
# Run on a GPU machine with the rebuilt extension installed:
#   python tests/test_seg_backward.py
#
# Freezes the color loss to zero: the loss touches ONLY the decoded
# segmentation logits, so grad_out_color is zero. Then asserts that the
# segmentation gradient flows into GEOMETRY parameters (xyz, opacity,
# scaling — via the shared dL_dalpha scalar) while SH/color parameters
# receive EXACTLY zero.
#
# If segmentation's dL_dalpha contribution were dropped from the backward
# kernel, every geometry assertion below would fail (silently, at training
# time — this test is the guard).
import torch
from argparse import ArgumentParser
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.segmentation_decoder import SegmentationDecoder
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import BasicPointCloud

H, W = 64, 64


class FakeCam:
    def __init__(self, device="cuda"):
        from utils.graphics_utils import getProjectionMatrix
        self.image_height = H
        self.image_width = W
        self.FoVx = 0.9
        self.FoVy = 0.9
        self.znear = 0.01
        self.zfar = 100.0
        self.world_view_transform = torch.eye(4, device=device)
        proj = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).to(device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def build_scene(seed=42, P=32):
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
    pcd = BasicPointCloud(points.numpy(), torch.rand((P, 3)).numpy(), torch.zeros((P, 3)).numpy())
    gaussians.create_from_pcd(pcd, [], spatial_lr_scale=1.0)
    gaussians.training_setup(opt)
    with torch.no_grad():
        gaussians._opacity.copy_(torch.full_like(gaussians._opacity, 2.0))
        gaussians._scaling.copy_(torch.full_like(gaussians._scaling, -1.0))
    return dataset, pipe, gaussians


def test_seg_only_backprop_drives_geometry():
    dataset, pipe, gaussians = build_scene()
    cam = FakeCam()
    bg = torch.zeros(3, device="cuda")

    # Zero any parameter grads from setup, then render.
    gaussians.optimizer.zero_grad(set_to_none=True)
    out = render(cam, gaussians, pipe, torch.zeros(3, device="cuda"))
    rendered_seg = out["rendered_seg"]

    # Decode and build a seg-ONLY loss (color image is untouched by the loss
    # => its gradient contribution to dL_dalpha is exactly zero).
    decoder = SegmentationDecoder(dataset.seg_encoding_dim, dataset.num_segmentation_classes).cuda()
    logits = decoder(rendered_seg.permute(1, 2, 0))                     # [H, W, C]
    targets = torch.randint(0, dataset.num_segmentation_classes, (H * W,), device="cuda")
    L_seg = torch.nn.functional.cross_entropy(logits.reshape(-1, dataset.num_segmentation_classes), targets)
    L_seg.backward()

    # 1) Segmentation encoding receives gradient.
    seg_grad = gaussians._segmentation_encoding.grad
    assert seg_grad is not None and seg_grad.abs().sum().item() > 0.0, \
        "no gradient reached _segmentation_encoding"

    # 2) THE THESIS CHECK: geometry parameters receive gradient from a
    #    segmentation-only loss. This is only possible if seg's contribution
    #    was folded into the shared dL_dalpha in the backward kernel.
    for name, param in (("xyz", gaussians._xyz), ("opacity", gaussians._opacity), ("scaling", gaussians._scaling)):
        assert param.grad is not None, f"{name}.grad is None"
        assert param.grad.abs().sum().item() > 0.0, \
            f"L_seg produced ZERO gradient on {name} — segmentation is not driving geometry (doc 01 §2 failure mode)"

    # 3) Color parameters receive EXACTLY zero from the seg-only loss.
    for name, param in (("features_dc", gaussians._features_dc), ("features_rest", gaussians._features_rest)):
        assert param.grad is not None and param.grad.abs().max().item() == 0.0, \
            f"seg-only backprop leaked gradient into {name} (expected exactly zero)"

    print("PASS seg-only backprop drives geometry (dL_dalpha folding verified)")
    print(f"  |grad| sums: seg={gaussians._segmentation_encoding.grad.abs().sum().item():.4e} "
          f"xyz={gaussians._xyz.grad.abs().sum().item():.4e} "
          f"opacity={gaussians._opacity.grad.abs().sum().item():.4e} "
          f"scaling={gaussians._scaling.grad.abs().sum().item():.4e} "
          f"f_dc={gaussians._features_dc.grad.abs().max().item():.4e}")


def test_color_only_backprop_unchanged():
    # Control: a photometric-only loss must produce exactly zero seg-encoding
    # gradients (guards against accidentally mixing channels).
    dataset, pipe, gaussians = build_scene(seed=45)
    gaussians.optimizer.zero_grad(set_to_none=True)
    cam = FakeCam()
    out = render(cam, gaussians, pipe, torch.zeros(3, device="cuda"))
    image = out["render"]
    target = torch.rand_like(image)
    (torch.nn.functional.l1_loss(image, target)).backward()

    seg_grad = gaussians._segmentation_encoding.grad
    assert seg_grad is None or seg_grad.abs().max().item() == 0.0, \
        "photometric-only backprop produced nonzero seg-encoding gradient"
    print("PASS color-only backprop leaves seg encoding untouched")


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    test_seg_only_backprop_drives_geometry()
    test_color_only_backprop_unchanged()
    print("ALL BACKWARD TESTS PASSED")