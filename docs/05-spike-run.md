# 05 — Running THE SPIKE (doc 04 §7.5 item 5 / §7.6 calibration)

**Purpose.** The decisive, GPU-gated test that decides whether T1
(segmentation-gated densification routing) may be enabled at all. It answers one
question: *does the model's persisted belief `(disagree, structure)` actually
separate the three regimes that ground truth says exist?* Until this passes,
`seg_route_enabled` stays `False` (doc 04 §7).

Tool: `spike_route.py` (repo root). It loads a trained model, computes the
read-out `(disagree, structure, S)`, independently labels every Gaussian a GT
regime by projecting its centre into the training views and sampling the GT
segmentation (`floater` = void pixel, `boundary` = valid pixel on a GT class
edge, `interior` = valid pixel in a uniform region; majority vote across views),
then scatters and grid-searches the thresholds. It does not train, enable
routing, or mutate the model.

## Step 1 — produce a saved iteration inside the densification window

Densification runs `iteration > seg_warmup_iters (10 000)` and
`< densify_until_iter (15 000)`, so the SPIKE must read a state from ~10k–15k.
Train with an explicit save (the default `--save_iterations` are 7000/30000):

```bash
python train.py -s datasets/office_0 -m output/office_0 \
    --segmentation_path <gt_seg_dir> \
    --save_iterations 12000
```

## Step 2 — run the SPIKE

```bash
python spike_route.py -s datasets/office_0 -m output/office_0 \
    --segmentation_path <gt_seg_dir> \
    --load_iteration 12000 \
    --out spike_12000.png --dump spike_12000.npz
```

Useful flags: `--num_views 100` (default; evenly-spaced training views to
project into), `--stride`, `--min_evidence 1.0` (the `S` evidence gate).

## Pass criterion (doc 04 §7.5)

The three colours occupy **distinct regions** of the `(disagree, structure)`
plane, in this arrangement:

| regime   | disagree | structure |
|----------|----------|-----------|
| boundary | high     | high      |
| interior | low      | (any)     |
| floater  | high     | low       |

**Pass** → proceed to Step 3. **Fail** (regions overlap) → `(disagree,
structure)` is not a usable disambiguator; T1 must **not** be enabled and the
design needs rework. The saved PNG is the decisive evidence; the printed
per-regime centroids / nearest-centroid score are supporting.

## Step 3 — calibrate thresholds (doc 04 §7.6)

The script grid-searches `(q_disagree, q_structure)` against the GT labels and
prints the best pair + its agreement. **Set `seg_q_disagree` / `seg_q_structure`
in `arguments/__init__.py` from that pair** (or read them off the scatter). No
hand-tuning — the current `0.75 / 0.50` are placeholders.

## Caveats (documented approximations)

- **No occlusion test.** A Gaussian behind a wall votes on the front pixel's GT;
  multi-view voting absorbs most of it.
- **Labels depend on current geometry.** GT labels come from projecting the
  model's (training-dependent) centres, so noisy early geometry → noisy labels.
  Treat the scatter as the read, not a single accuracy number.
- CPU-only self-check of the labeling/calibration logic:
  `python spike_route.py --selftest`.
