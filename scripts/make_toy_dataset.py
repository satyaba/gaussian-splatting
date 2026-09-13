# Generate a tiny synthetic Blender-format scene for GPU smoke tests
# (Task 16 resume-parity check, and generally for validating the seg path
# end-to-end without touching Replica data).
#
# Usage: python scripts/make_toy_dataset.py /path/to/out [--frames 12] [--size 64]
#
# Layout produced:
#   out/transforms_train.json, out/transforms_test.json
#   out/train/r_XX.png   (RGB)
#   out/test/r_XX.png    (RGB)
#   out/train_seg/r_XX.png  (uint8 class ids: 0=wall,1=red blob,2=green blob, 255=void/ignore)
#   out/train_depth/r_XX.png (float32 inverse depth, 0 = invalid)
#
# NOTE: the segmentation/depth folders are NOT wired into the loader yet (Task 12
# deferred); they are written for future use and for manual inspection.
import json
import os
import sys
import argparse

import numpy as np
from PIL import Image
import cv2

FOVX = 0.9  # radians, ~51 degrees
RADIUS = 2.8
SIZE_DEFAULT = 64
NUM_CLASSES = 3  # 0=background/wall, 1=red blob, 2=green blob (255 = void)


def look_at_c2w(eye, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0])):
    """Blender/NeRF convention: camera looks along -z of its local frame."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up2 = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up2
    c2w[:3, 2] = -forward   # blender camera z points backwards
    c2w[:3, 3] = eye
    return c2w


def render_frame(azimuth, height, size, seed=0):
    """Pure-numpy 'renderer': colored blobs on a wall, with class map + depth."""
    rng = np.random.RandomState(seed)
    img = np.full((size, size, 3), 0.15, dtype=np.float32)          # dark wall
    seg = np.full((size, size), 0, dtype=np.uint8)                  # class 0
    depth = np.zeros((size, size), dtype=np.float32)                # 0 = invalid

    # Red blob (class 1) and green blob (class 2), roughly centered,
    # with slight per-frame jitter so training has signal.
    cx = int(size * (0.35 + 0.06 * np.sin(azimuth)))
    cy = int(size * 0.45)
    r1 = size // 5
    cv2.circle(img, (cx, cy), r1, (0.85, 0.15, 0.15), -1)
    cv2.circle(seg, (cx, cy), r1, 1, -1)
    cv2.circle(depth, (cx, cy), r1, (1.0 / (RADIUS - 0.3)), -1)

    gx = int(size * (0.65 + 0.06 * np.cos(azimuth)))
    gy = int(size * 0.55)
    r2 = size // 7
    cv2.circle(img, (gx, gy), r2, (0.2, 0.8, 0.25), -1)
    cv2.circle(seg, (gx, gy), r2, 2, -1)
    cv2.circle(depth, (gx, gy), r2, (1.0 / (RADIUS + 0.2)), -1)

    noise = (rng.rand(size, size, 1) * 0.02).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)
    return img, seg, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--size", type=int, default=SIZE_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for sub in ("train", "test", "train_seg", "train_depth"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    def cam_entry(azimuth, height, file_path):
        eye = np.array([RADIUS * np.sin(azimuth), height, RADIUS * np.cos(azimuth)])
        c2w = look_at_c2w(eye)
        return {"file_path": file_path, "transform_matrix": c2w.tolist()}

    train_frames, test_frames = [], []
    for i in range(args.frames):
        az = 2.0 * np.pi * i / args.frames
        h = 0.3 * np.sin(az * 2.0)
        name = f"train/r_{i:02d}"
        train_frames.append(cam_entry(az, h, name))
        img, seg, dep = render_frame(az, h, args.size, seed=i)
        stem = os.path.join(args.out, name)
        Image.fromarray((img * 255).astype(np.uint8)).save(stem + ".png")
        cv2.imwrite(os.path.join(args.out, "train_seg", f"r_{i:02d}.png"), seg)
        cv2.imwrite(os.path.join(args.out, "train_depth", f"r_{i:02d}.png"), dep)

    for i in range(2):
        az = 2.0 * np.pi * (i + 0.5) / 4.0
        name = f"test/r_{i:02d}"
        test_frames.append(cam_entry(az, 0.0, name))
        img, _, _ = render_frame(az, 0.0, args.size, seed=100 + i)
        Image.fromarray((img * 255).astype(np.uint8)).save(
            os.path.join(args.out, name + ".png"))

    transforms = {"camera_angle_x": FOVX}
    with open(os.path.join(args.out, "transforms_train.json"), "w") as f:
        json.dump({**transforms, "frames": train_frames}, f, indent=2)
    with open(os.path.join(args.out, "transforms_test.json"), "w") as f:
        json.dump({**transforms, "frames": test_frames}, f, indent=2)

    n_points_note = "points3d.ply intentionally absent -> loader generates 100k random points"
    print(f"Wrote toy scene to {args.out} ({args.frames} train, 2 test, {args.size}px). {n_points_note}")


if __name__ == "__main__":
    main()