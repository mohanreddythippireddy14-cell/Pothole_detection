> **CORRECTION (2026-10-27):** the v2 "v1 test set" numbers below (mAP@50 0.95, recall 0.86) are **not valid**. The v1 test photos are part of the v2 training pool (chitholian contains the same 665 photos as andrewmvd), so the model had seen them. Use the 5-fold CV result (mAP@50 0.731 ± 0.010) until a clean held-out evaluation is done. See `FUTURE_ACTIONS.md`.

> **UPDATE (Phase 1 v2, `context/PHASE1_V2_RESULTS.md`)**: findings #1, #2, #3, #5, #8
> below were acted on — merged mega-dataset (1986 img, 4 sources, deduplicated),
> `close_mosaic` fix, 5-fold CV, 650 background images added. Result on v1's original
> unchanged test set: mAP@50 0.77→0.95, mAP@50:95 0.48→0.70, recall 0.68→0.86. See that
> doc for the full before/after. Findings #4, #6, #7 remain open.

# Performance Audit — why the model isn't scoring higher

Evidence-based audit of Phase 1 (detector) and Phase 2 (segmenter/pipeline), each finding
checked against actual logs/data rather than assumed. Ranked by how much it's actually
costing accuracy and how easy it is to fix.

## 1. Detector training never reached its planned "clean fine-tune" phase (config bug, confirmed)

`scripts/03_train_yolov8n.py` sets `epochs=100, patience=20, close_mosaic=10`.
`close_mosaic=10` means Ultralytics disables mosaic augmentation for the **last 10
epochs of the requested schedule** — i.e. epoch 90 onward for a 100-epoch run — so the
model gets a final stretch of training on clean, non-mosaiced images. This is the exact
mechanism your notes describe (`ARCHITECTURE.md`: "mosaic augmentation is disabled for
the final 10 epochs so the model fine-tunes on more natural image distributions").

**Early stopping fired at epoch 86** (best epoch 66 + patience 20) — *before* epoch 90.
Confirmed directly: grepping the training log for Ultralytics' `"Closing dataloader
mosaic"` message returns **0 matches** for the detector run, vs **1 match** for the
segmenter run (which trained the full 100 epochs and did reach epoch 90). So `best.pt`
was selected while the model was still training under heavy mosaic augmentation, and it
never got the clean-image fine-tune stage the schedule was designed to give it.

**Fix**: lower `close_mosaic` (e.g. to 20-30) so the transition happens well before
patience is likely to trigger, or raise `patience` so early stopping doesn't cut the run
short of epoch 90, or set a shorter total `epochs` (e.g. 60) with `close_mosaic=15` so the
schedule matches how long the model actually trains for on this dataset.

## 2. Small objects, small dataset — a genuinely hard combination for a nano model

- Pothole boxes are **small relative to the image**: median `relative_area` = 0.021
  (~2% of the frame), 25th percentile = 0.007. Small-object detection is a known weak
  point for YOLO in general, and YOLOv8n (the smallest variant, 3M params) has the least
  capacity to compensate.
- Only **532 training images** (665 total, minus val/test). At batch=16, that's ~34
  steps/epoch — the 86 epochs that actually ran amount to only ~2,900 gradient steps.
  That's a thin training budget for a nano model learning small, visually-subtle objects
  (a pothole often just reads as "different texture/color patch on similar-colored road"
  — much lower signal than typical COCO objects).
- **Extreme scale variance within the single class**: `relative_area` ranges from ~0 to
  0.85 across the dataset (std 0.116, larger than the mean) — the model has to learn to
  detect both tiny cracks and potholes filling most of the frame from the same 532
  examples, with no per-scale curriculum or oversampling.

**Fix (not attempted here)**: more training data (even unlabeled-then-pseudo-labeled, or
combining with the second Kaggle pothole dataset seen in the notebook,
`rajdalsaniya/pothole-detection-dataset`, which was downloaded/explored in the source
notebook but never actually used for training); a slightly larger backbone (YOLOv8s) if
latency budget allows; tiling/higher native resolution for small-object recall.

## 3. Native images are small and get ~1.5-2x upscaled to the 640 training resolution

Median native resolution is 400×300–400×400 (`PHASE1_STEP1_AUDIT.txt`: width mean 428,
height mean 344, both well under 640). Resizing up to 640×640 for training doesn't add
real detail, just interpolation — this plausibly hurts the model's ability to precisely
localize pothole edges (small blurred boundaries), which is consistent with the observed
gap between mAP@50 (0.77, "found it roughly") and mAP@50:95 (0.48, "found it tightly") —
this Phase1-Phase2 gap-size difference (0.77→0.48 vs Phase 2's 0.91→0.59) was already
flagged as unexplained in `PHASE2_RESULTS.md`; the native-resolution difference between
the two source datasets (Phase 1's ~400px photos vs PothRGBD's close-up, higher-detail
captures) is a plausible cause worth testing directly.

## 4. 5.4% of ground-truth boxes are extreme, thin, crack-shaped (up to 13.6:1 aspect ratio)

94/1740 boxes have aspect ratio >5:1 or <0.2:1 (e.g. 150px wide × 28px tall). These may be
legitimately crack-like pothole damage, or may reflect inconsistent labeling of what
counts as a "pothole" vs a crack in the source dataset (single annotator/dataset,
no stated labeling guideline). Either way, this widens the shape distribution the model
has to learn from very few examples, and a detector tuned toward compact/blob-shaped
potholes (the majority of the data, median aspect 2.08:1) is a plausible source of some
of the missed detections (recall = 0.68-0.76 depending on run).

## 5. Zero background (pothole-free) training images

Every one of the 665 images contains at least one pothole (`0 backgrounds` in the
ultralytics dataset scan log) — the detector never sees a clean-road negative example
during training. This dataset's own test set is curated the same way so it doesn't show
up as a recall/precision problem *in this project's own eval numbers*, but it means the
model's false-positive behavior on genuinely pothole-free road images (real deployment
conditions) is untested and likely underestimated by these metrics.

## 6. Segmentation generalizes only within its own narrow domain (confirmed by the demo run)

Already documented in `PHASE2_RESULTS.md`, reconfirmed live in the 10-image demo you just
reviewed: 2 of 10 detections came back `UNKNOWN` (segmenter found no mask) specifically
on Phase-1-domain (open road scene) crops, never on PothRGBD-domain crops. Root cause:
PothRGBD's 1000 training images come from only **2 capture dates/sessions**
(2025-02-27, 2025-03-05) — same camera, similar lighting/distance conventions throughout.
The segmenter has essentially only ever seen one "shooting style" and doesn't reliably
transfer to Phase 1's more varied road photography.

**Fix (not attempted here)**: train the segmenter on a mix of PothRGBD + a
segmentation-labeled subset closer to Phase 1's domain (none currently exists — would
need new annotation), or fall back to a classical segmentation method (GrabCut/watershed
seeded from the detection box) for out-of-domain crops instead of returning `UNKNOWN`.

## 7. Depth signal is real but unreliable on ~22% of frames, and isn't metrically calibrated

From `scripts/09`'s evaluation: mean correlation with true depth is good (r≈-0.73,
correctly accounting for Depth-Anything-V2's inverse-depth convention) but weak or
wrong-signed on 22% of sampled frames, and even the "good" frames are only affine-
calibrated using ground truth that won't exist at real inference time (see
`PHASE2_RESULTS.md`'s "diagnostic-only" section). This isn't currently a severity-score
issue only because the current pipeline deliberately excludes depth from the severity
decision (area-only) — but it does mean depth is not yet usable as a scoring input
without fine-tuning or a different calibration strategy.

## 8. Small, single-source test sets → metrics carry real variance

Phase 1 test set: 67 images. Phase 2 test set: 100 images (from the same 2 capture
dates as everything else). Neither is large enough to pin down precision/recall/mAP to
better than a few points of noise — the ~2-8 point gap between this project's numbers
and the original notebook's numbers (`PHASE1_NOTEBOOK_REFERENCE.md`) is plausibly just
this, not a real regression.

## What's NOT a likely cause (checked and ruled out)

- **The degenerate zero-width box** (1/1740, found and fixed in `scripts/02`) — too rare
  (0.06%) to move aggregate metrics measurably.
- **Overfitting** — checked directly against the notes' own diagnostic criterion
  (train/val loss trend); both decrease together through early stopping, no divergence.
- **PothRGBD label format bug** — verified 0/1000 malformed polygon lines.
- **Random-seed/environment nondeterminism** — the `_py311_compat.py` shim only patches
  an unrelated stdlib introspection function used by `torch._dynamo`'s import chain; it
  has no effect on training math.

## Priority ranking (impact × ease of fix)

1. **#1 (close_mosaic never triggered)** — free to fix, just a training-arg change,
   re-run and compare directly against the current mAP@50:95=0.477 (test-set) baseline.
2. **#6 (segmentation domain narrowness)** — already has a workaround path (fallback
   classical segmentation) that doesn't require new data collection.
3. **#2/#3 (small dataset + small/upscaled images)** — real ceiling-setting factors but
   require new data or a bigger backbone, not a quick fix.
4. **#4/#5 (label shape variance / no negatives)** — worth a labeling-quality pass if
   pursuing this further, lower priority than #1.
5. **#7 (depth calibration)** — already correctly scoped out of the severity decision;
   only matters if depth-based severity becomes a requirement.
