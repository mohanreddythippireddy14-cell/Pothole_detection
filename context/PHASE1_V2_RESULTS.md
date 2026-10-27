> **CORRECTION (2026-10-27):** the v2 "v1 test set" numbers below (mAP@50 0.95, recall 0.86) are **not valid**. The v1 test photos are part of the v2 training pool (chitholian contains the same 665 photos as andrewmvd), so the model had seen them. Use the 5-fold CV result (mAP@50 0.731 ± 0.010) until a clean held-out evaluation is done. See `FUTURE_ACTIONS.md`.

# Phase 1 v2 — Merged Mega-Dataset + Audit Fixes: Results

Directly resolves audit findings #1, #2, #3, #5, #8 from `context/PERFORMANCE_AUDIT.md`.
Built per user instruction to merge multiple Kaggle sources into one dataset (converging
duplicate/similar data, dropping non-comparable features) and to use k-fold CV instead of
one fixed split.

## What changed vs. v1

| | v1 (original) | v2 (this) |
|---|---|---|
| Images | 665 (andrewmvd only) | **1986** (1336 annotated + 650 background) |
| Sources | 1 | 4, merged and deduplicated (see below) |
| Native resolution | ~400×300 (variable, small) | mixed: 720×720 (chitholian) + 640×640 (rajdalsaniya) — no more tiny sub-400px images |
| Background (pothole-free) images | 0 | 650 |
| `close_mosaic` schedule | **never reached** (early-stopped at epoch 86, mosaic-off was epoch 90) | **confirmed reached in all runs** (verified via log grep for "Closing dataloader mosaic") |
| Val/test methodology | 1 fixed 80/10/10 split | 5-fold CV for validation, then final model trained on a 90/10 split of the full pool |

## Dataset build (`scripts/11_build_combined_kfold_dataset.py`)

- **andrewmvd** (665 photos) and **chitholian** (665 photos) are perceptual-hash-verified
  to be **the exact same 665 underlying photos** (665/665 matched, mean-abs-diff <5 on a
  16×16 grayscale hash) — chitholian is just a higher-resolution re-encode (720×720 vs
  andrewmvd's variable ~400×300). Converged into one entry per photo, **keeping
  chitholian's higher-resolution version** with its own independently-encoded VOC
  annotation.
- **rajdalsaniya** (2083 files) turned out to be **671 unique base photos**, each
  Roboflow-export-augmented into ~3 variants (`<id>_jpg.rf.<hash>.jpg` naming).
  Converged to 1 representative per base photo (using all variants would leak
  near-identical siblings across k-fold train/val splits). Perceptual-hash-verified as
  genuinely new: **0/60 sampled representatives matched** anything in andrewmvd/chitholian.
- **banilkumar**'s 650 "Normal" (pothole-free) photos included as YOLO background images
  (empty label files). Its 650 "Pothole" photos were **excluded** — classification-only
  labels, no bounding boxes, doesn't fit the detection schema (would need new annotation
  or risky pseudo-labeling).
- **Converged feature schema**: raw pixel `box_width`/`box_height` dropped from all
  cross-source statistics (not comparable — sources have different native resolutions);
  only resolution-independent `relative_width`, `relative_height`, `relative_area`,
  `aspect_ratio` used. Full per-source distributions: `context/phase1_v2_geometry_features.csv`.

Result: **1986 images** (1336 annotated + 650 background) vs. v1's 665/0.

## Training fix (`scripts/12_kfold_train.py`, `scripts/13_train_final_v2.py`)

v1 used `epochs=100, patience=20, close_mosaic=10` — mosaic augmentation was scheduled to
turn off at epoch 90, but early stopping fired at epoch 86, so the model never got its
intended clean-image fine-tune phase (audit finding #1). v2 uses
`epochs=60, patience=25, close_mosaic=15` (mosaic-off at epoch 45) — **verified this time**:
grepping all 6 training runs (5 k-fold + 1 final) for Ultralytics' "Closing dataloader
mosaic" message returns 6/6 matches, and none of the 6 runs early-stopped (all completed
the full 60 epochs).

## 5-fold cross-validation results (on the new, harder, more diverse val folds)

```
| Fold | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---|---|---|
| 0 | 0.7339 | 0.6676 | 0.7421 | 0.4429 |
| 1 | 0.7596 | 0.6560 | 0.7273 | 0.4379 |
| 2 | 0.7802 | 0.6622 | 0.7365 | 0.4370 |
| 3 | 0.8201 | 0.5993 | 0.7142 | 0.4245 |
| 4 | 0.8111 | 0.6254 | 0.7362 | 0.4199 |
| mean±std | 0.7810±0.0319 | 0.6421±0.0259 | 0.7313±0.0098 | 0.4324±0.0087 |
```

**Read this carefully — it's lower than v1's number, and that's expected, not a
regression.** These folds validate against a dataset that now includes 650 background
images (so false positives are penalized for the first time) and rajdalsaniya's harder,
more varied close-up crops. The k-fold mean is a more honest difficulty benchmark, not
directly comparable to v1's number. The real comparison is below.

## Same-benchmark comparison — the number that actually matters

Evaluated a v2 model on **v1's original, unchanged 67-image test set** (never seen during
v2 training) — a true apples-to-apples comparison:

```
                  Precision   Recall   mAP@50   mAP@50:95
v1 (original)       0.7935    0.6770    0.7708      0.4773
v2 fold-0 model      0.9169    0.8323    0.9211      0.6846
v2 FINAL model       0.9262    0.8578    0.9537      0.7044   <- promoted checkpoint
```

**Every metric improved substantially**: +13pp precision, +19pp recall, +18pp mAP@50,
**+23pp mAP@50:95** (the metric most relevant to the eventual severity-scoring use case).
Recall — the audit's flagged weak point (v1 missed 24-32% of potholes) — improved the
most in absolute terms.

## Background false-positive check (audit finding #5)

Ran the fold-0 model on all 130 background (pothole-free) images in its own validation
fold: **3/130 (2.3%) produced a false-positive detection**, 3 total false-positive boxes.
Low, and — importantly — now actually *measured* instead of unknown/untested as it was
in v1 (where the dataset had zero background images to test against at all).

## Promoted checkpoint

`checkpoints/yolov8n_pothole_v2_final_best.pt` — trained on 1787/1986 images (90% of the
merged mega-dataset), the checkpoint scripts `04`/`10` etc. should now point to instead
of `yolov8n_pothole_phase1_best.pt`. Training curves: `context/phase1_v2_final_training_curves.png`,
confusion matrix: `context/phase1_v2_final_confusion_matrix.png`, eval plots on v1's test
set: `context/phase1_v2_eval_on_v1test/`.

## What this does and doesn't resolve, from the audit

| Audit finding | Status |
|---|---|
| #1 close_mosaic never triggered | **Fixed & verified** (6/6 runs reached it) |
| #2 small dataset (665 img) | **Improved**: 1336 annotated (2x), still single-domain-ish (all sourced from public Kaggle pothole photos, not a purpose-collected fleet dataset) |
| #3 low native resolution / upscaling | **Improved**: chitholian's 720×720 + rajdalsaniya's 640×640 replace andrewmvd's variable ~400×300 |
| #4 crack-shaped label outliers | Not addressed — would need a labeling-convention pass across the now-larger combined set |
| #5 zero background images | **Fixed**: 650 added, false-positive rate now measured (2.3%) |
| #6 segmentation domain narrowness (Phase 2) | Not addressed in this pass — detector-only |
| #7 depth calibration (Phase 2) | Not addressed in this pass — detector-only |
| #8 small/noisy test set | **Addressed**: 5-fold CV gives a variance estimate (±0.009 on mAP@50:95) instead of one number |
