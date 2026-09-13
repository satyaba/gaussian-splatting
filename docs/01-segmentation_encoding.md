# Proposal: Segmentation Encoding Rendering Pipeline

**Purpose of this document.** This is an implementation proposal for the segmentation-encoding rendering pipeline of the SGDense-3DGS project. It specifies, file by file, what must be built to integrate a per-Gaussian `_segmentation_encoding` (32-dim, locked) into the Inria 3DGS rasterizer — from the Python-side `GaussianModel` through the CUDA forward/backward kernels — and how to verify correctness.

It is **not** a record of completed work. Treat every item below as a task to implement and verify unless it is explicitly listed under "Existing partial work in the repo." Where an item says "must," it is a correctness or safety constraint, not a suggestion.

**Context.** See `docs/00-design_decisions.md` for the project thesis (segmentation confidence drives Gaussian lifecycle, not passively attached), the two-stage training schedule, the loss term set, and the densification mechanism. This document is the rendering-pipeline implementation spec that supports those decisions.

---

## 1. Locked design decisions (do not re-litigate without reason)

These are architectural decisions already made. An implementer should follow them; only revisit if a concrete profiling or correctness result forces it, and flag the conflict explicitly.

| Decision | Resolution |
|---|---|
| Encoding dimensionality | Fixed at **32**, compile-time constant `NUM_SEG_CHANNELS` in `config.h`. Not swept as a hyperparameter without a rebuild. |
| `num_segmentation_classes` vs `seg_encoding_dim` | Both thread through the same config path (`ModelParams` in `arguments/__init__.py`), but serve different layers: `seg_encoding_dim` is Gaussian/rasterizer-side (compile-time), `num_segmentation_classes` is decoder-side (runtime, dataset-dependent). |
| Decoder checkpoint lifecycle | Fully decoupled from `GaussianModel.capture()`/`restore()`. Own optimizer (`decoder_optimizer`), own save/load functions, own `.pth` file saved alongside the existing `chkpnt{iteration}.pth`. |
| Decoder LR schedule | Flat at `decoder_lr_init` until a decay-start point, then exponential decay via `get_expon_lr_func`. Decay starts **before** `seg_warmup_iters` ends, so the decoder is already slowing down when confidence-gated splitting activates. Exact offset is an open tuning item (§6), not decided analytically. |
| Seg encoding activation | None — raw passthrough. Composited via alpha-blending, same as color. The decoder applies the only nonlinearity. |
| Shared-memory staging for seg in `renderCUDA` | **Read `seg_encoding` directly via `collected_id`; do not add a `collected_seg` shared array.** A 32-channel shared cache would cost ~32 KB/block (vs. ~7 KB for existing `collected_xy`/`collected_conic_opacity`/`collected_id` combined), risking a 4–5× shared-memory footprint increase and tanked occupancy. Direct global read is the same tradeoff already made implicitly for the existing color channels; make it explicit here given the larger channel count. |
| `preprocessCUDA` staging buffer (`geomState.seg_features`) | **Removed (2026-09-13).** The staging was a pure copy with no per-Gaussian computation (~244 MiB redundant VRAM at P≈2M plus bandwidth in a bandwidth-bound kernel); `renderCUDA` (forward and backward) now reads the raw `seg_encoding` input tensor directly, same access pattern as the color feature pointer. `Rasterizer::backward` gained a `seg_encoding` parameter because the staged buffer was previously its only seg data source. Supersedes the original "keep for consistency" resolution. |
| Background compositing term for segmentation | **None.** Color has `+ T * bg_color[ch]`; segmentation does not — unfilled space naturally decodes to near-zero encoding. This affects how `L_seg` must be masked at image boundaries / thin structures (open item, §6). |

---

## 2. `dL_dalpha` — the core correctness constraint

**This is the single most safety-critical piece of the implementation, directly tied to the thesis's central claim.**

`alpha` (and derived transmittance `T`) is shared between color and segmentation compositing. In the backward kernel, `dL_dalpha` must accumulate contributions from **both** loss sources before it flows downstream into `dL_dopacity`, `dL_dconic2D`, `dL_dmean2D`:

```cpp
dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;       // color's contribution
dL_dalpha += (s - accum_rec_seg[ch]) * dL_dseg_ch;    // segmentation's contribution — MUST be summed into the same scalar
```

If segmentation's contribution to `dL_dalpha` is dropped (e.g. only `dL_dseg_encoding` is computed, without folding into `dL_dalpha`), `L_seg` would still update the seg-encoding parameter correctly, but would **silently fail to drive position, scale, rotation, or opacity** — undermining the thesis claim that segmentation actively drives geometry rather than sitting as a passive add-on. This does not throw an error; it fails silently and surfaces only as a confusing "segmentation loss doesn't seem to affect density" symptom after a full training run.

**Required verification:** an isolated unit test — freeze color loss to zero, backprop only `L_seg` through a synthetic scene, assert `dL_dmean2D`/`dL_dopacity`/`dL_dconic2D` are nonzero while `dL_dsh`/`dL_dcolors` are exactly zero. See §5.

---

## 3. Full chain of files to implement (Python → CUDA)

### 3.1 `scene/gaussian_model.py`
- `__init__`: add `self._segmentation_encoding = torch.empty(0)`; store `self.seg_encoding_dim`.
- `create_from_pcd`: initialize `_segmentation_encoding` as an `nn.Parameter`, zeros or small noise, shape `[P, 32]`.
- `training_setup`: add as its own optimizer param group with dedicated `seg_encoding_lr`.
- `capture` / `restore`: add to the state tuple (Gaussian-side only — decoder excluded per the decoupling decision).
- `cat_tensors_to_optimizer` / `densification_postfix`: add the `"seg_encoding"` key to the dict `d` and the resulting assignment.
- `densify_and_clone` / `densify_and_split`: slice (`[selected_pts_mask]`) or repeat (`.repeat(N,1)`) `_segmentation_encoding` alongside other attributes; pass through to `densification_postfix`. **This is also the intended hook point for the confidence-gated 4-way split and discounted-confidence child inheritance** (see §4 — deferred until the rendering path is verified correct).
- `_prune_optimizer` / `prune_points`: works automatically once registered as an optimizer param group; add the `self._segmentation_encoding = optimizable_tensors["seg_encoding"]` assignment in `prune_points`.
- `construct_list_of_attributes` / `save_ply` / `load_ply`: add new `seg_{i}` columns.
- New `get_segmentation_encoding` property: raw passthrough, no activation.

### 3.2 `arguments/__init__.py`
- `ModelParams`: add `seg_encoding_dim` (default 32) and `num_segmentation_classes` (default 88 for Replica).
- `OptimizationParams`: add `decoder_lr_init`, `decoder_lr_final`, `decoder_lr_decay_start`, `decoder_lr_decay_iters`, `seg_encoding_lr`, and `seg_warmup_iters`.

### 3.3 `scene/segmentation_decoder.py`
- `SegmentationDecoder(nn.Module)`: a single `nn.Linear(seg_encoding_dim, num_segmentation_classes)`. Own optimizer, own checkpoint save/load functions (`save_decoder_checkpoint` / `load_decoder_checkpoint`), separate `.pth` file.
- **Fix the existing file** (see §4): it is currently broken — `torch.save`/`torch.load` are called but `torch` is never imported.

### 3.4 `train.py`
- Instantiate `GaussianModel(dataset.sh_degree, dataset.seg_encoding_dim, ...)` and `SegmentationDecoder(dataset.seg_encoding_dim, dataset.num_segmentation_classes)` from the same `dataset` config object.
- Separate `decoder_optimizer = torch.optim.Adam(decoder.parameters(), ...)`.
- Per-iteration: apply the decoder LR schedule (flat-then-decay); zero/step `decoder_optimizer` alongside `gaussians.optimizer`.
- Resume path: load both `gaussians.restore(...)` and `load_decoder_checkpoint(...)`, matched by iteration number.
- Warmup gating (`if iteration > opt.seg_warmup_iters: ...`) for confidence-based splitting — logic to be written; cleanly separable into `train.py` or a dedicated tracker class since the decoder is independent of `GaussianModel`.
- Compute `L_seg` (cross-entropy on decoded `rendered_seg`) and the other loss terms per `docs/00-design_decisions.md`; decode `rendered_seg` with the decoder **after** rasterization, in the training loop (not inside the CUDA kernel), because `L_consist` compares decoded outputs across different camera poses of the same Gaussians.

### 3.5 CUDA rasterizer — `submodules/diff-gaussian-rasterization/`

**`cuda_rasterizer/config.h`**
- `#define NUM_SEG_CHANNELS 32` alongside the existing `NUM_CHANNELS 3`.

**`cuda_rasterizer/forward.cu`**
- `preprocessCUDA`: does **not** touch segmentation at all (updated 2026-09-13 — the earlier raw-copy staging into `out_seg_features` was removed; the encoding needs no per-Gaussian preprocessing).
- `renderCUDA` (forward): new accumulator `S[NUM_SEG_CHANNELS]`, composited via the same per-Gaussian `alpha`/`T` loop as color; written to `out_seg[ch*H*W + pix_id]` with **no background term**. Reads the raw `seg_encoding` input tensor directly from global memory via `collected_id` (no staging, no shared-memory caching — see §1; implemented 2026-09-13).

**`cuda_rasterizer/backward.cu`**
- `renderCUDA` (backward): new `accum_rec_seg[NUM_SEG_CHANNELS]`, `last_seg[NUM_SEG_CHANNELS]`, `dL_dpixel_seg[NUM_SEG_CHANNELS]`. Segmentation's `dL_dalpha` contribution summed into the **same scalar** used by color (see §2). `dL_dseg_encoding` written via `atomicAdd`, same pattern as `dL_dcolors`. No `BACKWARD::preprocess` step needed for segmentation (raw passthrough forward → no inverse transform needed, unlike color's SH backward).

**`cuda_rasterizer/rasterizer_impl.h` / `.cu`**
- `GeometryState`: no segmentation field (updated 2026-09-13 — the `seg_features` staging was removed; `renderCUDA` reads `seg_encoding` directly). General buffer-sizing rule still applies to any *future* `GeometryState` field: it must appear in `fromChunk()`, which is the single size-calculation path in this codebase (`required<GeometryState>` routes through it), or buffer under-allocation will occur with a delayed, unrelated-looking crash.
- `out_seg` (`[H,W,32]`) and `dL_dseg_encoding` (`[P,32]`) are **not** part of `GeometryState`/`ImageState` — they are direct PyTorch output tensors allocated in `rasterize_points.cu`.

**`rasterize_points.h` / `.cu`**
- `RasterizeGaussiansCUDA` / `RasterizeGaussiansBackwardCUDA`: signatures extended with `seg_encoding` (forward in), `dL_dout_seg` (backward in), `out_seg` (forward out), `dL_dseg_encoding` (backward out).
- **Verify declaration/definition match.** The header's forward return-tuple arity currently disagrees with the `.cu` (see §4); reconcile before compiling.

**`ext.cpp`**
- No signature changes needed directly (forwards whatever `rasterize_points.h` declares) — correctness is contingent on `rasterize_points.h` being correct.

### 3.6 `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`
- `_RasterizeGaussians.forward`: add `seg_encoding` as an explicit input; return `rendered_seg` alongside `rendered_image`.
- `_RasterizeGaussians.backward`: accept `grad_out_seg`; return `grad_seg_encoding` in the **exact positional slot** matching `forward`'s input order (PyTorch assigns gradients positionally — easy to get wrong).
- `GaussianRasterizer` / `GaussianRasterizationSettings`: thread `seg_encoding` through as an explicit param, same pattern as `colors_precomp`.

### 3.7 `gaussian_renderer/__init__.py`
- `render()`: pass `seg_encoding=pc.get_segmentation_encoding` into the rasterizer call; receive `rendered_seg` (`[32, H, W]`) as an additional return value. The decode step (`SegmentationDecoder(rendered_seg)`) happens **after** rasterization, in the training loop.

### 3.8 Segmentation GT dataloader (implemented 2026-09-13 — Task 12)

Per-view ground-truth class maps thread through `arguments → dataset_readers → camera_utils → cameras`:

- `ModelParams.segmentation_path` (default `""` = off): folder of **uint8 class-id PNGs**, one per view, stems matching the RGB frames. Resolution rule: `<dir>/<split>/<stem>.png` preferred (split ∈ {train, test}), falling back to `<dir>/<stem>.png`; a missing file yields `""` → that camera carries `gt_segmentation=None` and `L_seg` is skipped for it.
- Void pixels: class id **255** in the PNG → `-1` in the tensor (`ignore_index=-1` in `train.py`). Maps are resized with `cv2.INTER_NEAREST` (never blend class ids) and validated against `num_segmentation_classes` at load (taxonomy guard).
- `Camera.gt_segmentation`: int64 `[H, W]` tensor on `data_device`, or `None`.
- `OptimizationParams.lambda_seg` (default 1.0) weights `L_seg`; training logs an EMA "Seg Loss".
- Colmap and Blender readers share the same wiring (`_segmentation_path_for` helper).
- Dataset note: replica maps come from the Semantic-NeRF pre-rendered release (proper noun — see terminology note in docs/00); the loader itself only reads a user-supplied folder, no fetching.

---

## 4. Existing partial work in the repo (baseline state — verify before building)

A partial, **non-compiling** stub already exists in two files. An implementer must fix these rather than assume they work:

- **`submodules/diff-gaussian-rasterization/rasterize_points.cu`** — `out_seg` tensor allocation and `seg_encoding`/`out_seg` threading into `CudaRasterizer::Rasterizer::forward` are present, but:
  - Line 71: `torch:full(...)` — single colon, syntax error; must be `torch::full(...)`.
  - Line ~121: missing comma between `out_seg.contiguous().data<float>()` and the following `out_invdepthptr` argument.
- **`submodules/diff-gaussian-rasterization/rasterize_points.h`** — forward and backward signatures already include `seg_encoding` / `dL_dout_seg`, but the forward return-tuple arity in the header (7-tuple) does not match the `.cu` (returns 8-tuple including `out_invdepth`). Reconcile.
- **`scene/segmentation_decoder.py`** — `SegmentationDecoder` class and `save_decoder_checkpoint` / `load_decoder_checkpoint` exist, but the file only does `from torch import nn` and then calls `torch.save` / `torch.load` — `torch` is unbound at runtime. Add `import torch`.
- **Everything else in §3 has no segmentation code yet** — `scene/gaussian_model.py`, `arguments/__init__.py`, `train.py`, `gaussian_renderer/__init__.py`, `diff_gaussian_rasterization/__init__.py`, and the `cuda_rasterizer/` kernels (`config.h`, `forward.cu`, `backward.cu`, `rasterizer_impl.h/.cu`) contain zero segmentation references. Implement them from scratch per §3.

> Note: an earlier draft of this doc claimed `save_ply`/`load_ply` and the CUDA kernels were already implemented. They are not. Trust the repo, not that draft.

---

## 5. Acceptance criteria & verification

1. **Build:** `pip install -e submodules/diff-gaussian-rasterization --break-system-packages -v` completes with a shared object produced (`find submodules/diff-gaussian-rasterization -name "*.so"`). A bare `256`-style output is inconclusive — use `-v` and confirm the `.so`.
2. **Forward smoke test:** synthetic Gaussians → assert `rendered_seg.shape == (32, H, W)` and no NaNs.
3. **`dL_dalpha` unit test (§2, mandatory):** freeze color loss to zero, backprop only `L_seg` through a synthetic scene; assert `dL_dmean2D`/`dL_dopacity`/`dL_dconic2D` are nonzero while `dL_dsh`/`dL_dcolors` are exactly zero.
4. **Signature consistency:** `rasterize_points.h` declarations match `.cu` definitions (param order, types, return-tuple arity).
5. **Buffer sizing:** `GeometryState::fromChunk` allocates exactly what it parses — it is the single sizing path in this codebase; the segmentation staging buffer no longer exists (removed 2026-09-13), so nothing segmentation-related is sized here anymore.
6. **Resume parity:** `gaussians.restore(...)` + `load_decoder_checkpoint(...)` restore both models matched by iteration; decoder optimizer state restored.

Note: do not run the verification step without permission of the user.

---

## 6. Open design questions (resolve or flag before/while implementing)

- **Decoder LR decay offset vs. `seg_warmup_iters`.** Intended to be tuned empirically via W&B (confidence-histogram spread vs. the warmup boundary vs. the LR curve), not decided analytically. Pick a concrete starting offset and document the choice.
- **~~`geomState.seg_features` staging buffer.~~** Resolved 2026-09-13: removed; direct read of `seg_encoding` implemented in forward and backward `renderCUDA` (see §1 decision table). Original resolution ("keep for consistency, profile before changing") superseded.
- **`L_seg` background masking.** Segmentation has no background compositing term, so `L_seg` must mask background pixels / thin structures. Decide the mask rule and document it.
  - **Implemented (v1, 2026-09-13):** void rule = ignore pixels whose GT class id is 255 (remapped to `ignore_index=-1` at load time, `utils/camera_utils.py`). Boundary/thin-structure masking beyond void pixels remains open, deferred to `L_edge`.
- **Confidence-gated 4-way split.** Deferred until the rendering path is verified correct (§5 criteria 2–3 pass). Implement in `densify_and_split` with discounted-confidence child inheritance; gating lives behind `seg_warmup_iters` in `train.py`.
