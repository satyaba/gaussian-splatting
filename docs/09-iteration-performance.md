# Per-iteration performance: removing the segmentation-rendering tax

Branch: `perf/seg-render` (parent) + `perf/seg-render` (rasterizer submodule
`submodules/diff-gaussian-rasterization`). Nothing is merged; nothing committed.

## 1. The observed problem

SGDense iterations ran ~4–5x vanilla 3DGS. The suspected cause was the
segmentation rendering (`NUM_SEG_CHANNELS = 32` compositing in the CUDA kernels).
That is part of it, but three independent costs stack up:

| # | Cost | Where | Effect |
|---|---|---|---|
| 1 | A **second full rasteriser backward** every iteration (`autograd.grad(loss_no_seg, means2D)`) for the decoupled densification trigger, active for `iters (2000, 25000)` | `train.py` | ~1.4–1.6x |
| 2 | Backward seg block held `accum_rec_seg[32]` + `last_seg[32]` (64 registers) plus `dL_pixel_seg[32]`; 32 global `atomicAdd`s per (Gaussian, pixel) pair | `cuda_rasterizer/backward.cu` | register spills / occupancy + atomic contention |
| 3 | Forward seg block: 32-channel accumulate + 1 vote atomic + 32-channel `out_seg` write per pair | `cuda_rasterizer/forward.cu` | the remaining seg tax |

## 2. What changed

### 2a. Decoupled trigger gradient produced by the same backward (cost #1)

The trigger only ever needs `∂loss_photo/∂(means2D).xy` (see `sgdense/densify_grad.py`;
the 3DGS trigger is the screen-space 2D-mean gradient norm). The rasteriser
backward already computes `dL_dalpha` as color + seg + depth. We now keep a
second accumulator `dL_dalpha_photo` = color + depth + background (seg excluded)
and write its 2D-mean contribution into a new buffer `dL_dmean2D_photo[P,3]`.

- `backward.cu`: new `float3* dL_dmean2D_photo` output; `dL_dG_photo` feeds two
  extra `atomicAdd`s.
- Plumbed through `backward.h`, `rasterizer.h`, `rasterizer_impl.cu`,
  `rasterize_points.{h,cu}`, `diff_gaussian_rasterization/__init__.py`
  (`GaussianRasterizationSettings.dL_dmean2D_photo`).
- `gaussian_renderer/__init__.py` allocates the zeroed buffer when
  `num_segmentation_classes > 0` and returns it as `render_pkg["dmean2D_photo"]`.
- `train.py`: `autograd.grad(...)` is gone; after `loss.backward()` the trigger
  reads `render_pkg["dmean2D_photo"]`. The full photo+seg gradient still drives
  the optimiser.

This is **exact** (equal to the old `autograd.grad(loss_no_seg, means2D)` up to
atomic summation order), so the trigger semantics are unchanged.

### 2b. Backward seg `dL_dalpha` reduced to two scalars (cost #2)

The per-channel recurrence feeds `dL_dalpha` only via the dot products
`A = Σ_ch a[ch]·accum_rec_seg[ch]`, `B = Σ_ch a[ch]·s[ch]` (`a = dL_pixel_seg`).
Tracking `A`, `B` is algebraically identical and removes two 32-float register
arrays from the hottest kernel. `dL_pixel_seg[32]` stays (it feeds the per-Gaussian
encoding gradient).

### 2c. Seg-render interval `K` (cost #3) — opt-in, default off

`--seg_render_interval K` (default `1` = current behaviour). On iterations where
`iteration % K != 0`, the forward kernel skips the vote atomic, the 32-channel `S`
accumulation and the `out_seg` write (new runtime `do_seg` flag), and the loop
skips the decoder / `L_seg` (so the backward skips the seg block automatically,
because `grad_out_seg` is `None`). Those iterations cost ~vanilla; average cost is
~`vanilla + seg_extra/K`.

`do_seg` is threaded through `forward.{h,cu}`, `rasterizer.{h,_impl.cu}`,
`rasterize_points.{h,cu}` and `GaussianRasterizationSettings`.

**This is a training-dynamics tradeoff, not a free win:** it reduces decoder
updates, seg→geometry gradients, and histogram evidence growth (the
`seg_hist_min_evidence` gate is crossed ~K× slower). Ablate it; do not adopt
blindly.

## 3. Correctness notes

- Trigger gradient is photo(+depth) only, matching the previous `loss_no_seg`
  semantics; the optimiser keeps the full gradient.
- Seg's contribution to `dL_dalpha` (the doc 01 §2 thesis constraint) is
  preserved exactly — `tests/test_seg_backward.py` still applies.
- Buffer is zeroed per render (fresh `torch.zeros`), so `atomicAdd` accumulation
  is clean. Empty buffer ⇒ null pointer ⇒ kernel skips (eval/GUI paths).

## 4. Follow-up: backward register/occupancy (measured)

A/B at matched populations (44k vs 329k → `--photo_weight_ramp`/`--no_photo_weight_ramp`
to reach 262k vs 325k; the ramp is now **OFF by default**) put ~3× of the gap in
`backward`. `-Xptxas -v` (sm_75) showed the
cause: the backward `renderCUDA<3>` used **100 registers** vs the stock **64**, so
it ran at **2 blocks/SM (50% occupancy)** vs 4 blocks (100%). No spills — purely an
occupancy cap. (The forward was 64 vs 33 registers, but both are thread-capped at
4 blocks, so it was unaffected.)

Fix (this branch): stop holding `dL_pixel_seg[32]` live across the j-loop — read
`a[ch] = dL_pixels_seg[ch*H*W + pix_id]` from global inside the seg loop (per-pixel
constant, L1-cached) — and pin `__launch_bounds__(BLOCK_X*BLOCK_Y, 4)` so ptxas
targets ≤64 registers. Verify register count/spills after rebuild with
`NVCC_PREPEND_FLAGS="-Xptxas -v"`.

## 5. Not done yet (candidates)

- Warp/block reduction of the `dL_dseg_encoding` atomics (all threads in a tile
  write the same `global_id`); hardware may already aggregate same-address
  atomics, so measure first.
- Kernel-side `seg_hist` decay (removes the `[N,C]` `mul_` pass and the
  `nonzero()` sync/restore in `gaussian_renderer/__init__.py`).
- Decoder as a 1x1 conv on the contiguous `[32,H,W]` (avoids the permute copy).
- `template<bool DO_SEG>` forward instantiation so skipped iterations also drop
  the `S[32]` register footprint, not just the executed work.

## 6. Build & measure

```
pip install -e submodules/diff-gaussian-rasterization --break-system-packages -v
python -m experiments.profile --iterations 4200 --profile-iters 3999:4099
```

Expect the `backward` phase to drop most for the removed second pass; then
re-profile with `--seg_render_interval 4` / `8` to see the render-phase drop.
