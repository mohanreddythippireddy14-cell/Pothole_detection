# Future Actions & Current Limitations

What still needs doing, ordered by priority. Each item says **what you need** (datasets, GPU,
weights) and **concrete steps**, so it can be picked up as soon as the resources are available.

Last reviewed: 2026-10-27.

---

## A. Do as soon as the trained weights are available

### A1. Publish the weights · *needs: the two `.pt` files*
1. Create a GitHub Release (e.g. `v1.0`) and attach `yolov8n_pothole_v2_final_best.pt` and `yolov8n_seg_pothrgbd_best.pt`.
2. In `README.md` → *Get the trained weights*, replace "will be published" with direct download links.
3. Optional: add a small `scripts/download_weights.py` that fetches them into `checkpoints/`.

### A2. Visually check the segmentation-mask fix · *needs: weights*
`scripts/10` now uses `retina_masks=True` (masks at crop resolution) instead of resizing the letterboxed mask.
1. Run `python scripts/10_full_pipeline.py --images docs/assets/samples/potholes103_raw.png --out-dir outputs/`.
2. Temporarily draw `mask_crop` onto the crop and confirm it lines up with the pothole.
3. Re-generate `docs/assets/samples/*_annotated.png` and `REPORT.md` with the fixed code, **with depth on** (the current samples were made with `--no-depth`).

### A3. Better README example · *needs: weights*
The README sample photo (`potholes103`) is from the training data. Replace it with a photo the
model has never seen (e.g. your own road photo), shown with depth enabled and an example of depth escalation.

---

## B. Do when you have the datasets and a GPU

### B1. Clean held-out test set for the detector · *needs: Kaggle datasets, GPU (~1–2 h)* · **highest priority**
**Problem:** v2 has no untouched test set. The old 0.95 mAP@50 was measured on v1 test photos that were also in v2 training.
1. In `scripts/11`, after building the pool, assign a **group id** to every image:
   chitholian/andrewmvd photo id, rajdalsaniya Roboflow base stem, and a near-duplicate
   cluster for banilkumar backgrounds (see B2).
2. Hold out **15% of groups** as `test/` *before* any training, using `sklearn.model_selection.GroupShuffleSplit` (seed 42). Never use it for early stopping or checkpoint selection.
3. Switch `scripts/12` to `GroupKFold` on the remaining 85%.
4. Retrain the final model (`scripts/13`) on the 85%, then evaluate **once** on the test set with `scripts/04 --weights ...` (point it at the new `data.yaml` with a `test:` entry).
5. Update the README Results table: replace the struck-out row with the clean test numbers.

### B2. De-duplicate the background images · *needs: banilkumar dataset*
banilkumar is an *augmented* dataset, so near-copies of the same road photo may sit in both training and validation, making the 2.3% false-positive rate optimistic.
1. Compute a perceptual hash (e.g. `imagehash.phash`) for all 650 "Normal" images.
2. Cluster images within a small Hamming distance (≤ 6) and keep one per cluster, or use the cluster id as the group in B1.
3. Re-measure the background false-positive rate on the clean test set.

### B3. Make the duplicate detection more robust · *needs: datasets*
`scripts/11`'s "perceptual hash" is a 16×16 grayscale mean-difference with a hand-picked threshold, and the rajdalsaniya-vs-chitholian overlap was checked on only 60 sampled images.
1. Use a real perceptual hash (`imagehash.phash`) and check **all** images, not a sample.
2. Log how many cross-source duplicates are found and drop them.

### B4. Segmentation split by recording session · *needs: PothRGBD, GPU (~30 min)*
The 0.908 mask mAP@50 is in-domain: random split across ~1,000 frames from 2 sessions (2025-02-27 and 2025-03-05).
1. In `scripts/07`, split by timestamp instead of randomly. Either train on one date and test on the other, or use contiguous time blocks so neighbouring frames stay together.
2. Retrain with `scripts/08` and report the new (more honest) mask mAP.
3. Optionally add open-road segmentation labels (e.g. a few hundred hand-annotated crops) to close the domain gap behind the `UNKNOWN` results.

### B5. Severity: fix the box-vs-mask area mismatch · *needs: idanbaru dataset + weights*
Area thresholds are fitted on **box** area (idanbaru only has boxes) but applied to **mask** area (smaller), so ratings lean low.
Pick one:
- **Option 1 (simple):** compute severity area from the *detection box* in `scripts/10` (`relative_area = box_w*box_h / image_area`), matching the calibration exactly. Keep the mask for the depth factor.
- **Option 2:** run the segmenter on idanbaru crops, measure the average mask/box area ratio, and rescale the thresholds.

### B6. Severity: hold-out calibration and end-to-end accuracy · *needs: idanbaru + weights*
1. In `scripts/14`, fit thresholds on 80% of idanbaru **images** (grouped by image) and report balanced accuracy on the other 20%.
2. Note: idanbaru shares its photos with chitholian/andrewmvd (detector training data), and its first 665 labels were produced by a script, not by hand. Prefer evaluating on the 52 hand-labelled images, or on new labelled photos.
3. **End-to-end check:** run the full pipeline on held-out idanbaru images, match predicted potholes to labelled boxes (IoU ≥ 0.5), and report a confusion matrix of predicted vs. labelled severity, plus how often depth escalation fired and whether it helped.

### B7. Depth math · *needs: PothRGBD + GPU*
1. Depth-Anything-V2 outputs **relative inverse depth** (disparity). Instead of negating it (`-pred`), work in disparity space: estimate the road-plane disparity from the ring, then compute the relative depth as `(1/d_inside − 1/d_road)` after a scale/shift fit, or keep everything in disparity and define steepness on disparity.
2. In `scripts/09`, fit the affine alignment in inverse-depth space (`1/gt`) rather than linearly to metric depth, then recompute correlation and RMSE.
3. **Recalibrate the escalation threshold the way it is used:** on detector crops with *predicted* masks (not full images with ground-truth masks), and check it against labelled severity (B6) instead of simply taking the 90th percentile.
4. Consider fitting a road plane (RANSAC on the ring pixels) instead of a single median level, to handle sloped roads.

---

## C. Product features (planned, not built)

| Feature | Notes |
|---|---|
| JSON output | Emit per-image JSON (count, boxes, severity, confidence, features) next to `REPORT.md`; prerequisite for any dashboard. |
| Number-plate / face blurring | Add a fast blurring step before saving outputs (privacy). |
| Video / real-time | Process video frames; track potholes across frames to avoid duplicates. |
| Edge deployment | Export to ONNX/TensorRT and quantise; Depth-Anything-V2 is the slow part. |
| Metric scale | Use a known reference (e.g. lane width) or camera intrinsics to convert relative depth and area to cm. |
| Standing water | Reported (not yet measured) that water inside a pothole can look flat to the depth model; collect examples and measure. |

---

## D. Current limitations (summary)

| Area | Limitation |
|---|---|
| Detection | No untouched v2 test set; 5-fold score (mAP@50 0.731) is slightly optimistic because the scored fold also picks the checkpoint. Recall ~0.64. |
| Data | Background images come from an augmented dataset (possible near-duplicates); duplicate check is approximate. |
| Segmentation | In-domain only (2 recording sessions); can miss potholes on open-road crops (`UNKNOWN`). |
| Depth | Relative, uncalibrated inverse depth; sign-flip approximation; threshold set on a different setup than inference. |
| Severity | Rule-based approximation of ASTM D6433; area thresholds fitted on box area but applied to mask area; in-sample accuracy (52.5%); no end-to-end accuracy. |
| Weights | Not yet published, so the pipeline can't be run without retraining. |
| Samples | README/demo images come from the training data and were generated without depth. |
| Repo | Old history still contains the removed `runs/` images (~85 MB). Rewriting history would shrink the repo but isn't required. |
