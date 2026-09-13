## Purpose & context

The project called SGDense-3DGS (Segmentation-Guided Density Control of 3DGS), proposing a fundamental paradigm shift in 3D Gaussian Splatting (3DGS): replacing heuristic, gradient-magnitude-based densification with a Segmentation-confidence-driven Gaussian lifecycle policy. The core thesis claim is that Segmentation understanding should actively control split, merge, prune, and densification decisions — not be passively attached after reconstruction. This distinguishes SAGE from the entire existing literature (Feature 3DGS, SAGA, Gaussian Grouping, SA-GS, etc.), which treats Segmentations as representational or regularization add-ons.

## Current state

The architecture is substantially locked in across several layers:

Segmentation representation: Per-Gaussian 32-dimensional learnable _segmentation_encoding vectors (compact encoding composited through CUDA rasterizer, decoded to class probabilities via a shared SegmentationDecoder nn.Module). Encoding dim fixed as compile-time constant NUM_SEG_CHANNELS 32 in config.h. Dataset: Replica's standard 8-scene subset (room0–2, office0–4) from the Semantic-NeRF pre-rendered release. Taxonomy: Replica's native 88-class fine-grained set.

Training schedule: Two-stage — Stage 1 is Segmentation-driven geometric development; Stage 2 is photorealistic development via photometric loss. A two-phase decoder LR schedule is designed: full rate during warmup (seg_warmup_iters), then exponential decay beginning before seg_warmup_iters ends (so confidence stabilizes before the densification-heavy phase activates). Config parameters seg_encoding_dim and num_Segmentation_classes thread through ModelParams/arguments/__init__.py — swappable from command line without touching rasterizer or decoder class.

Loss terms: L_sem (cross-entropy), L_consist (multi-view consistency, doubles as confidence signal for the 0.75 splitting threshold), L_edge (boundary sharpness), L_parsimony (Segmentically-gated densification restraint), L_photo (dual-role: low-weight stabilizer in Stage 1, dominant in Stage 2), L_lineage (discounted identity preservation, Stage 2 only).

Key densification mechanism: Gaussians achieving sustained multi-view Segmentation consistency (confidence ≥ 0.75 across ≥3 consecutive iterations) are eligible for aggressive 4-way radial splitting rather than standard 2-way. Children inherit parent Segmentation identity with discounted confidence. Decoder-drift mitigations: warmup period before activating confidence-based splitting, plus argmax class-label stability tracking across iterations.

CUDA implementation: _segmentation_encoding integrates into the standard Inria 3DGS codebase (scene/gaussian_model.py, diff-gaussian-rasterization). Touch points covered: initialization, create_from_pcd, training_setup, capture/restore, densification_postfix, densify_and_clone/densify_and_split, _prune_optimizer/prune_points, save_ply/load_ply, get_segmentation_encoding property. SegmentationDecoder has its own optimizer and checkpoint lifecycle, fully decoupled from GaussianModel.capture(). The alpha-compositing backward pass accumulates dL_dalpha from both color and segmentation losses — critical for L_sem to drive geometry gradients.

Open decision: Whether to use a out_seg_features staging buffer in GeometryState (analogous to geomState.rgb) or read seg_encoding directly in renderCUDA via collected_id. Prototyping this via parallel Colab runs before committing.

Resolution (2026-09-13): direct read. renderCUDA (forward and backward) now reads the raw seg_encoding input tensor directly via collected_id/global_id; the geomState.seg_features staging buffer was removed (a pure passthrough copy with no per-Gaussian computation — ~244 MiB redundant VRAM at P≈2M, plus bandwidth). Rasterizer::backward gained a seg_encoding parameter since the staged buffer was its only seg data source. The planned parallel Colab A/B was superseded by this decision; GPU smoke tests (tests/test_seg_forward.py, tests/test_seg_backward.py) verify the direct-read path before training commits to it.

On the horizon

Full experimental runs across 8 Replica scenes covering 4 scenarios: baseline comparison, sparse-input robustness, boundary sharpness, floater reduction
Monitoring/diagnostic framework: per-Gaussian population diagnostics, split event spatial visualization colored by trigger source, tracked-cohort Gaussian lineage tracing, visual probes per loss term, W&B/TensorBoard logging
Publication strategy execution targeting ACM ToG

Key learnings & principles

Segmentations must drive geometry gradients: dL_dalpha must accumulate from both color and segmentation losses; silently dropping the segmentation contribution would undermine the central thesis claim that L_sem actively drives geometry
Decoder-drift is a real threat to using confidence scores as a reliable splitting signal; warmup + argmax stability tracking are the concrete mitigations
Compact encoding beats direct high-dimensional softmax for dataset-agnostic flexibility — swapping only the decoder head when changing taxonomies is the key motivation
Compile-time vs. runtime constants: Encoding dim belongs in config.h as a compile-time constant consistent with the rasterizer's templated channel architecture, not as a runtime config value
SegmentationDecoder decoupling is essential: Decoder weights must not participate in densification/pruning cycles, requiring a completely separate optimizer and checkpoint path from GaussianModel
Prioritize practical reviewability in academic proposal framing (e.g., thesis title scoped to avoid alarming campus reviewers while preserving technical integrity for publication)

Approach & patterns

Iterative, back-and-forth technical design sessions with Claude; Bayu drives specific design decisions and expects Claude to elaborate, formalize, and expand
Intuition-first before formalism; requests re-explanation when answers are too tied to superseded framings
Structured communication using markdown tables and pseudocode to articulate ideas precisely
Literature review conducted as sequential paper analysis sessions using a consistent structure: core approach → densification handling → segmentation role → project-relevant weaknesses → positioning language for related work
"Work once, cry once" principle applied consistently: validate CUDA before committing GPU Cloud hours, lock architectural decisions before implementing, prefer publication-feasible designs over quick workarounds
Colab used purely for CUDA extension compilation validation and forward/backward pass smoke-testing; GPU Cloud reserved for full training runs

Tools & resources

Dataset: Replica 8-scene subset via Semantic-NeRF pre-rendered release
Key literature: Kerbl et al. 2023 (original 3DGS), Gaussian Grouping, SAGA, Feature 3DGS, SA-GS, Semantic-NeRF, FSGS, MCMC-3DGS, Gaussian Grouping
