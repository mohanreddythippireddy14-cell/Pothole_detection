# Phase 2 — Segmentation, Depth, Severity: Local Implementation Results

Continues Phase 1 (`context/PHASE1_RESULTS.md`). Implements the architecture in
`context/ARCHITECTURE.md`'s "Phase 2" section: detection → padded crop → pixel
segmentation → geometry → depth → severity → annotated visualization. Since the source
notebook (`pothole_detection (1).py`) stops at *exploring* Depth-Anything-V2 and PothRGBD
without training a segmentation model or assembling the end-to-end severity pipeline, all
of Phase 2's code here is new (not ported), built directly from the architecture the
handwritten notes describe.

## Step 1 — PothRGBD dataset audit (`scripts/06_download_pothrgbd.py`)

Downloaded `mahyeks/pothrgbd-rgb-and-depth-images-of-potholes` via kagglehub.

- **1000 fully-matched RGB + depth + segmentation-polygon triples** (`.jpg` + `.npy` +
  `.txt`, matched by `YYYYMMDD_HHMMSS` timestamp / stem — 100% match rate, 0 unmatched).
- Depth arrays: `uint16`, shape (480, 640), range 0–1470, mean ≈1098 (units not stated
  by the dataset; consistent with millimeters for a handheld RGBD sensor at close range).
  94.1% of pixels have valid (>0) depth in the sampled array.
- The `.txt` files are **already in YOLO-segmentation format**
  (`class_id x1 y1 x2 y2 ... xn yn`, normalized 0–1) — confirmed on 1000/1000 files
  (`scripts/07`'s sanity check: 0 malformed polygon lines). No conversion needed.
- **Caveat found and documented, not in the original notebook**: 998 unique timestamps
  across 1000 images (median gap 19s between shots, only 1.8% of gaps ≤2s) — this is
  *not* dense video, so little literal frame duplication. But all 1000 images come from
  only **2 capture dates** (2025-02-27 and 2025-03-05), meaning train/val/test splits
  drawn from this dataset are not domain-independent — they share lighting, road surface
  type, and camera setup across all three splits. Segmentation numbers below should be
  read as "in-domain" performance, not a generalization estimate.

## Step 2 — YOLOv8n-seg training (`scripts/08_train_yolov8n_seg.py`)

80/10/10 split (seed=42, consistent with Phase 1): train 800 / val 100 / test 100.
100 epochs, imgsz=640, batch=16, patience=20 — ran the full 100 epochs without
early-stopping (unlike Phase 1's detector, which stopped at 86). ~0.2 hours on the
Quadro RTX 6000.

**Held-out test-set results** (100 images, 112 instances, never used for training or
early-stopping decisions):

```
Box  — Precision: 0.901  Recall: 0.866  mAP@50: 0.907  mAP@50:95: 0.589
Mask — Precision: 0.919  Recall: 0.884  mAP@50: 0.908  mAP@50:95: 0.592
```

Best validation-split epoch (85): Mask mAP@50=0.943, mAP@50:95=0.664 — reported here for
reference since it's the checkpoint-selection criterion, but the test-split numbers above
are the honest generalization estimate (still in-domain per the caveat above).

Notably, Box and Mask metrics are nearly identical (mask mAP@50:95 0.592 vs box 0.589) —
unlike Phase 1's detector where mAP@50:95 (0.477) trailed mAP@50 (0.771) by ~30 points,
here the gap is much smaller. Plausible reason: PothRGBD's potholes are photographed
close-up and fill much more of the frame than Phase 1's road-scene detection dataset, so
both box and mask localization have less room for the pixel-level slack that widens the
mAP@50-to-mAP@50:95 gap in Phase 1.

Checkpoint: `checkpoints/yolov8n_seg_pothrgbd_best.pt`. Plots: `context/phase2_seg_eval_plots/`
(test-set), `context/phase2_seg_training_curves.png`, `context/phase2_seg_val_confusion_matrix.png`.

## Step 3 — Depth-Anything-V2 evaluation (`scripts/09_evaluate_depth_anything.py`)

The notes flagged (qualitatively, from eyeballing outputs) that pretrained
Depth-Anything-V2 "not performing well" on real pothole RGBD pairs. Quantified this
against 40 randomly sampled PothRGBD triples with ground-truth sensor depth.

**Important interpretation point, not in the original notes**: Depth-Anything-V2's
`predicted_depth` output is a *relative inverse depth* map (MiDaS/DPT convention — higher
value = nearer, not farther), while PothRGBD's sensor depth is direct metric depth
(higher = farther). A well-functioning model should therefore show **negative**
correlation with the raw sensor values, not positive — comparing the two without
accounting for this convention would make even a perfect model look "wrong."

```
N=40 images
Full-frame Pearson r:  mean=-0.726  std=0.346  (min=-0.993, max=+0.302)
Full-frame RMSE after per-image affine fit (a*pred+b), sensor units: mean=69.0, std=96.5
  (sensor depth mean ≈1098 → ~6.3% relative error on average, once scale+shift are fit)

Pothole-mask-region only:
Pearson r: mean=-0.729  std=0.437
RMSE (reusing whole-image affine fit): mean=35.1, std=64.5

Fraction of images with WRONG-SIGN (positive) correlation: 5% (2/40)
Fraction of images with WEAK (|r|<0.3) correlation: 17% (7/40)
```

**Revised conclusion**: once the inverse-depth convention is accounted for, pretrained
Depth-Anything-V2-Small performs **reasonably well on the majority of PothRGBD images**
(mean |r|≈0.73, ~6% relative RMSE after affine calibration) — this is a more favorable
picture than the notes' blanket "not performing well." However, it is **inconsistent**:
22% of sampled images (17% weak + 5% wrong-sign) show poor-to-no agreement, so it cannot
be trusted uncalibrated on every frame. The notes' concern is valid as a reliability
concern, just not as a blanket failure — worth stating precisely for anyone deciding
whether to fine-tune. Full numbers: `context/PHASE2_DEPTH_ANYTHING_EVAL.md`.

**Fine-tuning Depth-Anything-V2 on PothRGBD was not attempted in this pass** (out of
scope for this session — flagged, not silently skipped). If pursued, PothRGBD's 1000
matched RGB/depth pairs would support it directly.

## Step 4 — Integrated severity pipeline (`scripts/10_full_pipeline.py`)

Implements the "Final Architecture" diagram from the notes:
`Input Image → YOLOv8n (Phase 1) detect → padded crop → YOLOv8n-seg (Phase 2) mask →
geometry + depth → severity function → annotated image`.

### Bug found and fixed during first run (not caught by design review, only by looking
at actual output numbers)

First run: `relative_area` was computed as `mask_pixels / padded_crop_pixels`. Since the
crop is deliberately sized to snugly contain the detected box (15% padding), the mask
*always* fills a large fraction of its own crop — relative_area came out 0.2–0.9 for
essentially every detection regardless of the pothole's true size relative to the road,
driving nearly every detection to "HIGH" severity. **Fixed** by normalizing against the
full original image instead (`mask_pixels / full_image_pixels`), with thresholds
(0.01 / 0.05) set from Phase 1's own box-relative-area quartiles
(`context/phase1_annotations.csv`: 25%=0.0072, 50%=0.0214, 75%=0.0663) rather than
arbitrary numbers. Re-running produced a sane LOW/MEDIUM/HIGH spread — see below.

Also fixed: a detection whose crop the segmenter finds **no mask** in was silently
defaulting to "LOW" severity through the area formula (0 area < 0.01 threshold). That's
wrong — it's a segmentation failure, not evidence of a small pothole. Now reported as a
distinct `"UNKNOWN"` severity.

### Depth is diagnostic-only, not fed into severity — and this is a real limitation, not a simplification

`scripts/09`'s affine calibration (`a*pred+b` fit to sensor depth) requires ground-truth
depth to solve for `a` (the scale factor). At inference time on a genuinely new image
there is no ground truth, so only the *offset* can be estimated (via a road-plane-ring
heuristic — median relative depth in a ring of pixels around the mask, used as a
zero-reference). The resulting `depth_p95` numbers are in an **uncalibrated relative
unit**, not millimeters, and folding them into a fixed-threshold severity score would be
fabricating precision that doesn't exist. The pipeline reports depth_p95 alongside
severity for visibility but the severity decision is area-only in this implementation.
Recovering real metric depth would need either a metric (not relative) depth model, or a
camera-intrinsics-based scale-recovery step — neither is in scope for this pass.

### Demo run — 3 images from each domain

Ran on 3 Phase 1 test images (open road scenes) and 3 PothRGBD test images (close-up
macro shots) — see `context/phase2_pipeline_demo/REPORT.md` for full per-detection
numbers and `context/phase2_pipeline_demo/*_annotated.png` for visuals.

- **Phase 1 domain** (potholes103/120/122.png): 12 detections total, severities spread
  LOW (1) / MEDIUM (7) / HIGH (3) / UNKNOWN (1). Visual check (`potholes122_annotated.png`)
  shows box+mask alignment on the actual road potholes is accurate.
- **PothRGBD domain** (3 test images): 6 detections, MEDIUM (1) / HIGH (4) / UNKNOWN (1).
  Visual check (`20250227_162635_..._annotated.png`) shows near-pixel-perfect mask
  boundary tracking on the pothole edge — as expected, since both detector and segmenter
  cross a domain boundary here (segmenter trained on PothRGBD's own domain).
- **Cross-domain segmentation gap, found not assumed**: both UNKNOWN cases (1 in each
  domain) are the segmenter failing to find *any* mask in a detector-cropped region that
  visually does contain a pothole. The segmenter was trained only on PothRGBD's close-up,
  single-capture-setup domain (see Step 1 caveat) — this is early, concrete evidence that
  it does not fully generalize to Phase 1's open-road-scene crops, consistent with the
  2-capture-date domain-narrowness flagged in Step 1.

## Summary of what's real vs. what's a known limitation

| Component | Status |
|---|---|
| Detection (Phase 1 YOLOv8n) | Trained & evaluated, test mAP@50=0.771 |
| Segmentation (YOLOv8n-seg on PothRGBD) | Trained & evaluated, test Mask mAP@50=0.908, **in-domain only** (2 capture dates) |
| Geometry (width/height/area from mask) | Implemented, bug-fixed (crop-relative → image-relative) |
| Depth (Depth-Anything-V2 pretrained) | Quantified: ~73% mean correlation once sign convention is accounted for, but inconsistent (22% of frames weak/wrong-sign); **not fine-tuned** |
| Severity function | Area-only (depth excluded — no valid metric calibration at inference time); thresholds from Phase 1's empirical area distribution |
| End-to-end pipeline | Runs on both domains; cross-domain segmentation generalization gap observed directly, not just anticipated |

## Next steps (not done in this pass)

1. Fine-tune Depth-Anything-V2 on PothRGBD's 1000 RGB/depth pairs — the dataset needed
   is already downloaded and cached.
2. Broaden the segmentation training domain (more capture sessions/road types) if
   cross-domain generalization matters for the target deployment scenario.
3. If metric depth becomes available (fine-tuned or a metric depth model), revisit
   folding it into the severity function with real calibration.
