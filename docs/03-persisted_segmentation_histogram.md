# Proposal: Persisted Per-Gaussian Segmentation Histogram (`[N, C]`)

**Purpose of this document.** This specifies the **cross-iteration aggregation** layer of SGDense-3DGS: how the per-iteration `α·T` vote masses produced by the rasterizer's forward pass (doc 01) accumulate into a persistent, per-Gaussian class histogram, and how that histogram is read out as the consistency signals that density control (doc 02) consumes. It supersedes the aggregation half of `docs/02-segmentation_guided_density_control.md`.

It is **not** a record of completed work. It is a design proposal with a frozen *starting design* (§4) and a deferred, measurement-gated *optimization stage* (§5). Where an item says "must," it is a correctness or safety constraint.

**Context.** `docs/00-design_decisions.md` holds the thesis; `docs/01-segmentation_encoding.md` holds the rendering-pipeline spec; `docs/02-segmentation_guided_density_control.md` holds the density-control policy spec. This document **revises doc 02 §2 (the "Counter scheme" and "transient" decisions) and doc 02 §3 (the statistic and its §3.3 count-based update)**, replacing the two-slot streaming counter with a persisted `[N, C]` mass histogram, and **revises doc 02 §2's "GT class at the projected image centre" label source** (superseded by per-pixel in-kernel label sampling — see §4.1).

---

## 0. Status and scope

| Phase | What it is | Status in this doc | Compute risk |
|---|---|---|---|
| **Starting design** | Persistent `[N, C]` fp32 mass histogram; per-iteration exponential decay; read `argmax` + row-sum. | Frozen (§4). **Build this first.** | low (rides the existing forward pass) |
| **Optimization stage** | Measurement-gated cost/robustness improvements (fused read-out, visible-set decay, bf16, compaction, kernel-level atomics, adaptive β). | Deferred roadmap (§5). **Nothing implemented until Stage 1 measurement justifies it.** | varies |

No `git commit` until explicitly authorized (parent repo + `diff-gaussian-rasterization` submodule would be two separate commits).

---

## 1. Context: why a persisted histogram, not a streaming counter

The rasterizer's forward pass already computes, at every pixel, the compositing weight `α·T` for each contributing Gaussian — the same quantity the colour accumulation uses (`C[ch] += features[ch]·α·T`). That means **per-pixel, per-Gaussian class evidence is available for free**; no extra render, no point-sampling of projected means, no separate footprint routine. The only question is how those per-iteration masses become a class belief that survives across iterations.

Doc 02 answered that with a **two-slot streaming heavy-hitter** (`l1, n1, l2, n2, tot`) that keeps only the top-two classes and updates by `tot[g] += 1` per observation. Three problems emerged:

1. **It is lossy by construction.** It discards everything but two classes and collapses each iteration's multi-pixel evidence to an unweighted `+1`. The soft split (51/49 vs 99/1), the runner-up's *mass*, and any ambiguity/entropy signal all vanish.
2. **Its memory rationale does not hold.** Doc 02 §2 rejected a full `[P, K]` histogram (~700 MB at P≈2M) — but the per-iteration vote buffer is `[N, C]` in **both** schemes: a streaming design must still allocate the `[N, C]` buffer every iteration to do the per-class `atomicAdd`. So persisting it costs **nothing extra**, while streaming adds the five scalars on top. At `C≈100` the `[N, C]` buffer dominates either way (measured §3.5); streaming is the *more* expensive option here, and only pays off if `C` grows large enough that `[N, C]` must be avoided — which it is not here.
3. **A stacked streaming tracker is fragile.** A decay-adapted, weighted, `k=2` eviction rule **locks out new classes at the decay constant one would actually use**: comparing a fresh vote's single-iteration mass `w` against a slot's decayed multi-iteration accumulation `≈ w/(1−β)`, a challenger can only enter once the incumbent has decayed below one vote. Verified by simulation — with `β=0.99`, a third class arriving after two established classes **never becomes the leader**, even after 30 consecutive iterations. (The formal Misra–Gries/Space-Saving error bounds do not carry over once decay and weights are layered in — a limitation the streaming proposal correctly disclaims but does not avoid.)

A persisted, decayed, full-row histogram removes all three: it is exact per class, costs no more peak memory, and needs no eviction rule (the class ordering falls out of `argmax`).

---

## 2. Variable glossary (every symbol used below)

| Symbol | Meaning |
|---|---|
| `N` | Number of Gaussians in the model (`P` in some docs; both mean the same). ~2M on office_0. |
| `C` | Number of segmentation classes = `num_segmentation_classes`. Must be **100** for office_0, not 88 (§6.4). |
| `M` | The persisted histogram, shape `[N, C]`. `M[g,c]` = decayed total `α·T` mass Gaussian `g` has cast on class-`c` pixels. |
| `g` | Gaussian index. `c` = class index. `p` = a pixel. `t` = current iteration; `k` = a past iteration. |
| `α` | Gaussian opacity at a pixel (blend factor, 0–1). |
| `T` | Transmittance at a pixel when reaching this Gaussian (1 at front, →0 behind opaque geometry). |
| `w = α·T` | The vote weight — the Gaussian's actual contribution at that pixel. |
| `B_t[g,c]` | This iteration's per-class mass: `Σ_{p : label(p)=c} w` over pixels `p` the Gaussian covers in view `t`. |
| `s_t[g]` | Row total for the iteration: `Σ_c B_t[g,c]` = the Gaussian's total compositing mass in view `t`. |
| `β` | Decay factor, just under 1 (e.g. 0.99). Old evidence fades geometrically; effective memory ≈ `1/(1−β)` iterations. |
| `S[g]` | Evidence: `Σ_c M[g,c]` (row sum) — how much mass accumulated lately. |
| `P[g,c]` | Read-out distribution: `M[g,c] / S[g]` (row-normalized). `c*` = `argmax_c`. |

---

## 3. The design

### 3.1 Per-iteration update

The histogram is **never zeroed between iterations**. Each iteration, decay first, then add the new masses:

```
M ← β · M                     # decay the accumulated evidence (one pass)
M[g, label(p)] += α_g(p)·T_g(p)   # for every contributing pixel, in-kernel (atomicAdd)
```

Written out per Gaussian and class, over past iterations `k` with weights `β^{t−k}`:

```
M[g,c] = Σ_k β^{t−k} · B_k[g,c]        S[g] = Σ_c M[g,c] = Σ_k β^{t−k} · s_k[g]
```

So `M` is a **geometrically-decayed pooled mass histogram** — the α·T evidence pooled over the recent past. Two consequences (both verified numerically):

- **Read-out is normalization-invariant.** `argmax_c M[g,c] = argmax_c P[g,c]` and `max_c M / Σ_c M = max_c P[g,c]`. Storing `M` unnormalized loses nothing for decisions.
- **`S` is redundant as storage.** Because the EMA is linear and `Σ_c B_k[g,c] = s_k[g]`, the row sum `Σ_c M[g,c]` equals the EMA of the row sums exactly — compute it on the fly, do not store a second buffer.
- **`β = 1` ⇒ exact.** With no decay, `M = Σ_k B_k` and `P` is the true pooled class distribution over all iterations — no approximation. `β<1` only bounds how far back the pool reaches; it is the *only* knob that replaces the old "window vs forever" fork.

### 3.2 Read-outs

| Quantity | Expression | doc-02 mapping |
|---|---|---|
| consistency / confidence | `max_c P[g,c]` | replaces `top1/tot`; doc-00 "confidence ≥ 0.75" |
| structure | `P_(1) + P_(2)` | replaces `(n1+n2)/tot` |
| disagreement | `1 − P_(1)` | replaces `1 − n1/tot` |
| evidence / maturity | `S[g]` | replaces `tot` (but see §6.5 — `S` is mass, not a count) |
| ambiguity (free extra) | `−Σ_c P log P` | new; an ordinal density-control input |

All read-outs lie in `[0,1]` **by construction** (`P` sums to 1), so the old `n1 + n2 ≤ tot` invariant hazard disappears.

### 3.3 Decay policy

- **Decay-all** (`M ← βM` over all `N` rows) is simplest. It cannot move `P` (the ratio is scale-invariant) — it only shrinks `S`. So decay policy *cannot corrupt the class estimate*; it governs only evidence lifetime.
- **Decay-on-observation** (decay only the current view's visible set, known from culling) avoids penalizing a Gaussian merely for being outside the frustum this step ("view-sampling luck"). The streaming proposal raised this correctly; it is worth adopting, at the cost of one indexing step. **Open — see §6.2.** Either way, a Gaussian never observed has `S → 0`, which is exactly the "don't trust this" signal.

### 3.4 Live-in-the-kernel write

Because `M` persists, the kernel can `atomicAdd` **directly into `M`** after the decay pass — the old zero-seeded scratch buffer (`gaussian_renderer/__init__.py:48`) is **removed rather than duplicated**. The existing `vote_buffer` plumbing (`float*` through the rasterizer chain) is unchanged; only the Python side changes from `torch.zeros((P, C))` to a persistent `M.mul_(β)`.

### 3.5 Cost (measured; 5090, ~1.2 TB/s effective)

| `N` | `M` size | decay pass | read-out pass |
|---|---|---|---|
| 2 M (office_0) | 763 MB | 1.33 ms | 0.67 ms |
| 5 M | 1.91 GB | 3.33 ms | 1.67 ms |
| 10 M | 3.82 GB | 6.67 ms | 3.33 ms |

≈2 ms/iter at 2M (**≈5–10 % of a render step**), and `M` is ≈36 % of the per-Gaussian training state (~1.07 KB/Gaussian). Comfortable on a 32 GB 5090 up to ~10–20 M Gaussians; tight on Colab beyond ~5 M. Peak memory is **identical** to a streaming design (§1.2).

---

## 4. Starting design — file by file

Legend: **[done]** already in the repo; **[todo]** to build; **[user]** user-authored (kernel algorithm); **[agent]** plumbing only, on explicit request.

### 4.1 Kernel / rasterizer (user-authored body)

- **[done]** `gt_segmentation` reaches the forward kernel (`int64 [H,W]`, `-1 = void`), `vote_buffer` threaded as `float*` with per-pixel label sampling and `atomicAdd` of `α·T`.
- **[todo · user]** Three outstanding safety/correctness items in the kernel:
  1. **Bound guard** `label < num_segmentation_classes` before the `atomicAdd` — currently absent, and with `num_segmentation_classes = 88` versus office_0 ids up to 98 this is a live out-of-bounds write (§6.4).
  2. **`(int64_t)` cast** on `g * num_segmentation_classes + label` — `int` overflow at `N·C ≥ 2³¹` (≈21 M Gaussians at C=100).
  3. **`vote_buffer != nullptr` guard** so the empty-buffer path (segmentation off) is safe.

### 4.2 `gaussian_renderer/__init__.py`

- **[todo]** Replace the per-call zero seed at `:48` with a persistent, decayed histogram owned by the model: pass `pc.seg_hist` (decayed in place via `pc.seg_hist.mul_(β)` before the call), return it as `render_pkg["seg_votes"]` (`:145`). Guard with `num_segmentation_classes > 0 and gt_segmentation.numel() > 0` as today.

### 4.3 `scene/gaussian_model.py`

- **[todo]** Add `self.seg_hist` `[N, C]` fp32 on `"cuda"`; allocate zero at `training_setup` (alongside the existing counters, `:215–219`); **resize in lockstep at every population change** — append at `densification_postfix` (children inherit or start fresh, §6.1) and mask at `prune_points` (`:406–410`); persist via `capture()`/`restore()` (`:88–92`/`:109–113`) **alongside** the five counters already persisted there.
- **[todo]** Read-out method returning `(consistency, structure, disagree, S)` from `M` — replaces doc 02 §4.1's `add_segmentation_consistency_stats` projected-centre proxy (superseded: labels are now sampled per pixel in the kernel).
- **[done]** The five two-slot counters and their checkpoint persistence remain (retained for the consecutive-iteration streak rule; §6.5).

### 4.4 `train.py`

- **[todo]** At the densification step, compute read-outs from `M` **before** any lifecycle call (ordering is a correctness constraint — the histogram is mutated in place each iteration). Keep the existing `gt_segmentation` in scope.
- **[done]** `capture()`/`restore()` call sites (`:240` save, `:55–56` load) need no change beyond the widening tuple (§4.3).

### 4.5 `arguments/__init__.py`

- **[todo]** `num_segmentation_classes = 88 → 100` (`:60`).
- **[todo]** Add `seg_hist_decay` (β, default `0.99`) and `seg_hist_min_evidence` (S gate).

### 4.6 No changes

`scene/segmentation_decoder.py`, `utils/camera_utils.py`, `scene/cameras.py`, and the rest of the rasterizer are untouched.

---

## 5. Optimization stage (deferred, measurement-gated)

**Invariant:** every optimization below preserves D2's *semantics* (decayed α·T mass sum, normalized read) so results stay comparable across stages. Each stage is triggered by a measured threshold, not adopted speculatively.

**Stage 1 — Instrument, don't optimize.** Profile peak VRAM, Δt/iter with and without the decay+read passes, and the atomic-contention signal. *Acceptance:* a written baseline. No optimization lands before this exists.

**Stage 2 — Low-risk wins (semantics-preserving).**
| Item | Mechanism | Gain |
|---|---|---|
| 2a fused read-out | one kernel: row-sum + argmax together, over the touched set only | halves read-out traffic |
| 2b visible-set decay | `index_select` on the cull list instead of global `mul_` | traffic ∝ V, not N; gives decay-on-observation |
| 2c division guard | `clamp_min(eps)` / skip `S=0` rows | kills 0/0 NaN |
| 2d "lock" stable Gaussians | once `max(P) ≥ θ` for K consecutive iters, stop updating that row | perf + late-stage noise removal |

*Acceptance:* measured Δt/iter drop with **bit-identical** class assignments on a short reference run.

**Stage 3 — Memory scaling (trigger: N > ~10 M, or Colab).**
| Item | Mechanism | Gain |
|---|---|---|
| 3a bf16 storage | accumulate fp32, store bf16 (D2 is linear ⇒ rounding-safe) | ×0.5 |
| 3b compacted `[V, C]` | allocate for the visible set via the cull list | large scenes: `V ≪ N` (near-zero in a room) |
| 3c chunked reduction | process `M` in row-chunks to bound peak | bounds peak at very large N |

**Stage 4 — Kernel-level atomics (trigger: Stage 1 shows contention).** Within a tile, threads whose pixels share a label serialize on the same `(g,c)` cell. Mitigations: shared-memory per-tile aggregation then one global `atomicAdd` per `(g,c)`; or warp aggregation via `__match_any_sync`. *Kernel-body — user-authored.*

**Stage 5 — Adaptive / algorithmic refinements.**
| Item | Mechanism |
|---|---|
| 5a β schedule | tie decay to `xyz_gradient_accum` (geometry-stability proxy) instead of a fixed constant |
| 5b child inheritance | at densify, `M_child = c·M_parent` ⇒ identity inherited, confidence discounted (§6.1) |
| 5c entropy signal | free `−ΣP log P` from the same row |
| 5d streak counter | keep a separate `[P]` observation counter for the "≥3 consecutive iterations" test (§6.5) |

**Non-goals for the optimization stage:** streaming top-2 / Space-Saving (memory-identical, eviction lockout at β=0.99); per-iteration normalization (D1, discards mass weighting); point-sampling projected means.

---

## 6. Open questions & conflicts (resolve or flag before implementing)

- **6.1 Child inheritance: doc 00 vs doc 02.** doc 00 `:15` says children "inherit parent Segmentation identity with discounted confidence"; doc 02 `:135` says they "start a fresh observation window". Under D2 these are one line apart: inherit **identity** = `M_child = c·M_parent` (same `P`, lower `S`); fresh = `M_child = 0`. **Must be settled** — they are opposite choices.
- **6.2 Decay-all vs decay-on-observation.** §3.3. Only affects evidence `S`, never `P`. Adopting decay-on-observation avoids frustum-luck erosion of eligibility; decay-all is simpler and ~1 ms/iter.
- **6.3 β value.** Starting hypothesis `0.99` (≈ one ~100-iteration densification window). Fixed constant first; escalate to a schedule only if empirically insufficient (§5a).
- **6.4 `num_segmentation_classes = 88 → 100`.** office_0 has ids up to 98; 88 is a live out-of-bounds risk in the kernel (`arguments/__init__.py:60`). Safety-critical.
- **6.5 `S` is mass, not a count.** The per-iteration row sums vary (measured `[2.09, 1.13, 1.16, 1.03, 0.29, 1.84]` in one step) and scale with footprint, so `S` cannot stand in for "number of iterations observed". Gate the abstain rule on mass (`S > τ`), and keep a separate `[P]` counter for the "≥3 consecutive" streak rule (EMA smooths that criterion away).
- **6.6 Doc 02 rows superseded.** §2 "Counter scheme" (two-slot), §2 "transient, not saved to checkpoint", §3.3 (`tot[g] += 1`), and §2/§3 "GT class at the projected image centre" are all superseded by this document; doc 02 should be amended or marked.
- **6.7 Checkpoint back-compat.** Widening the checkpoint tuple (13 → 18 entries) breaks resume from old `chkpnt*.pth`. Accepted for a research repo; checkpoints must be regenerated.
- **6.8 0/0 NaN.** Guard the `M / S` division (§5 2c); required, not optional.

---

## 7. Acceptance criteria & verification

> **Permission-gated.** Do not run GPU-heavy steps without explicit user approval; the CUDA build check and any training run need a separate go.

1. **fp64 reference.** On a short run, accumulate all per-iteration rows in fp64 *once*; D2's `M`/`P` must match within tolerance. Catches decay/offset errors a "looks reasonable" check misses.
2. **Semantic invariance.** Class assignments must be unchanged before/after any optimization (or differ only by documented bf16 rounding).
3. **Invariants.** `P` sums to 1; `argmax`/confidence invariant to normalization; `Σ_c M = S`.
4. **Memory + time.** Peak VRAM and Δt/iter match §3.5 at office_0 (`≤10 %` Δt, `≤40 %` VRAM proposed budgets).
5. **No NaN.** No `0/0` in read-outs for never-observed Gaussians.
6. **Downstream.** The consistency statistic correlates with GT class purity on a held-out view.
7. **THE SPIKE (decides the thesis spine).** Dump each Gaussian's `(disagree, structure)` (doc 02 §7.6) and colour by whether its 3D position lies on a GT class boundary. Boundary Gaussians must occupy the high-`structure` region, floaters the low-`structure` region, interior the low-`disagree` region — visibly separable.

---

## 8. Non-goals (starting design)

- No streaming top-2 / Space-Saving (rejected, §1.3).
- No per-iteration normalization (D1).
- No extra renders or M-view probes.
- No kernel atomic micro-optimization (deferred to §5 Stage 4).
- No multi-scene sweep; validation remains single-scene (`office_0`) until the policy is settled.
- No optimization-stage work until Stage 1's baseline exists.
