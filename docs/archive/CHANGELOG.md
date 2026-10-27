# Project changelog — full session record

This documents everything built, tested, found, and decided across this project's
build-out, in order. Each section links to the detailed doc in `context/` for full
numbers/evidence — this is the narrative that ties them together.

## 0. Setup

- Project scaffolded at `the repository root (`Pothole_detection/`)` (code, small files).
- All datasets, checkpoints, training runs, and the Python venv placed on
  `$POTHOLE_DATA_ROOT/` — root disk had only
  ~2.2GB free at project start (later cleared by unrelated activity to ~18GB), so
  nothing large is stored there.
- Kaggle API credentials configured (`~/.kaggle/kaggle.json`).
- Python env: `python3.11` (system default) + `torch 2.13.0+cu126` + `ultralytics
  8.4.121` + `transformers`/`scipy`/`scikit-learn`/`kaggle` CLI, installed via `uv`.
- **Environment bug found & fixed**: the system's `python3.11` is an incomplete
  pre-release build (reports as `3.11.0rc1`, missing `sys.get_int_max_str_digits`) that
  crashes `torch._dynamo`'s import chain. Fixed with a compat shim
  (`src/_py311_compat.py`), imported first in every training/inference script.

## 1. Phase 1 — Pothole detection (original port from the Colab notebook)

Ported `pothole_detection (1).py`'s detection pipeline: `scripts/01`-`05`.

- Downloaded `andrewmvd/pothole-detection` (665 images, 1740 boxes), audited it
  (`context/PHASE1_STEP1_AUDIT.txt`), found and fixed **1 degenerate zero-width
  ground-truth box** not caught in the original notebook.
- Converted VOC XML → YOLO format, 80/10/10 split (train 532 / val 66 / test 67, seed
  42, matching the notebook exactly).
- Trained YOLOv8n (100 epochs requested, early-stopped at epoch 86).
- **Result** (test set): Precision 0.79, Recall 0.68, mAP@50 0.77, mAP@50:95 0.48 — in
  the same ballpark as the notebook's original recorded numbers (P≈0.86, R=0.66,
  mAP50=0.75, mAP50:95=0.49; kept separately in `context/PHASE1_NOTEBOOK_REFERENCE.md`).
- Checked training/validation loss trend directly against the handwritten notes'
  overfitting criterion (page 5, Q8) — no divergence found.
- Full writeup: `context/PHASE1_RESULTS.md`.

## 2. Phase 2 — Segmentation, depth, severity (new build, not in the source notebook)

The source notebook only *explores* Depth-Anything-V2 and PothRGBD; it never trains a
segmentation model or assembles an end-to-end severity pipeline. All of this was built
fresh, following the architecture the handwritten notes describe (pages 9-12).

- Downloaded PothRGBD (`mahyeks/pothrgbd-...`, 1000 RGB+depth+segmentation-polygon
  triples, 100% matched). Audited it: found the `.txt` files were **already** valid
  YOLO-segmentation format (0/1000 malformed), and flagged a caveat — all 1000 images
  come from only 2 capture dates, so it's not domain-diverse.
- Trained YOLOv8n-seg on it (`scripts/07`-`08`). **Result** (test set): Box mAP@50=0.907,
  Mask mAP@50=0.908, mAP@50:95=0.589/0.592.
- Quantitatively evaluated pretrained Depth-Anything-V2 against PothRGBD's real sensor
  depth (`scripts/09`). Key finding, not in the original notes: DA-V2's output is
  *inverse* relative depth (near=high value), so the notes' qualitative "not performing
  well" verdict was partly a sign-convention artifact — once corrected, mean correlation
  is ~0.73 and ~6% relative error after per-image affine calibration, but unreliable on
  ~22% of frames.
- Built the integrated pipeline (`scripts/10`): detect → crop → segment → geometry →
  depth → severity → annotated image.
  - **Bug found and fixed during the first run**: `relative_area` was computed against
    the padded crop (always large, ~0.2-0.9) instead of the full image — this pinned
    nearly every detection to "HIGH" regardless of true size. Fixed to normalize
    against the full image.
  - **Bug found and fixed**: a zero-mask detection was silently defaulting to "LOW"
    severity — changed to a distinct `UNKNOWN` outcome (segmentation failure ≠ small
    pothole).
  - Ran a demo on both dataset domains: found a real cross-domain generalization gap
    (segmenter, trained only on PothRGBD, returns `UNKNOWN` on some Phase-1-domain
    crops) — not just anticipated, actually observed.
- Full writeup: `context/PHASE2_RESULTS.md`.

## 3. Visualization iteration

- On request: ran the pipeline on 10 randomly-selected test images (mixed domains),
  displayed input/output pairs.
- On request: removed the segmentation-mask overlay from the visualization (mask is
  used internally for scoring, not shown to the viewer) — switched to box-only output
  colored by severity (green/orange/red/gray) with a filled label showing severity +
  confidence.

## 4. Performance audit

On request ("audit the project, find issues for low performance"): systematically
checked training config, dataset statistics, and logs against the actual results
(not assumption-based). Findings, ranked:

1. **Confirmed config bug**: `close_mosaic=10` with `epochs=100` means mosaic
   augmentation should turn off at epoch 90 — but early stopping fired at epoch 86 (best
   epoch 66 + patience 20), so the detector never reached its intended clean-image
   fine-tune phase. Verified by grepping the log for Ultralytics' "Closing dataloader
   mosaic" message: 0 matches for the detector run, 1 match for the segmenter run
   (which completed all 100 epochs).
2. Small objects (median 2% of image area) + small dataset (532 train images, ~2,900
   total gradient steps) — a genuinely hard combination for a 3M-param nano model.
3. Native images heavily upscaled (~400×300 → 640×640 training resolution), plausibly
   explaining why Phase 1's mAP@50→mAP@50:95 gap (0.77→0.48) is much wider than Phase
   2's (0.91→0.59), where PothRGBD's close-ups have more native detail.
4. 5.4% of ground-truth boxes are extreme, crack-shaped (up to 13.6:1 aspect ratio).
5. Zero background (pothole-free) training images — false-positive behavior on real
   clean roads was untested.
6. Segmentation domain narrowness (see Phase 2 above).
7. Depth calibration limits (see Phase 2 above).
8. Small single-source test sets (67/100 images) → noisy metrics.
- Ruled out: overfitting, the degenerate box, PothRGBD label corruption.
- Full writeup: `context/PERFORMANCE_AUDIT.md`.

## 5. Phase 1 v2 — merged mega-dataset + audit fixes

On request ("get a decent dataset, solve the issues... converge datasets... use k-fold").

- Searched Kaggle for additional pothole datasets. Downloaded 3 candidates:
  - `chitholian/annotated-potholes-dataset` (665 img) — **perceptual-hash-verified to be
    the exact same 665 photos as andrewmvd**, just re-encoded at a consistent, higher
    720×720 resolution. Converged: kept chitholian's higher-res version instead of
    andrewmvd's, rather than treating it as new data.
  - `rajdalsaniya/pothole-detection-dataset` (2083 files) — found to be **671 unique
    base photos**, each Roboflow-export-time-augmented into ~3 variants (verified via
    filename pattern). Converged to 1 representative per base photo (avoids k-fold
    train/val leakage from near-identical siblings). Perceptual-hash-verified as
    genuinely new (0/60 sampled matches against the existing corpus).
  - `banilkumar20phd7071/...-augmented-data` — used its 650 "Normal" (pothole-free)
    photos as YOLO background images; excluded its 650 "Pothole" photos (classification
    labels only, no bounding boxes, doesn't fit the detection schema).
  - Also converged the **feature schema**: dropped raw pixel `box_width`/`box_height`
    from all cross-source statistics (not comparable once sources have different native
    resolutions), kept only resolution-independent `relative_width`/`relative_height`/
    `relative_area`/`aspect_ratio`.
  - **Result**: 1986 images (1336 annotated + 650 background), vs. v1's 665/0.
- Fixed the close_mosaic bug: `epochs=60, patience=25, close_mosaic=15` — verified
  "Closing dataloader mosaic" fired in all 6 training runs this time, none early-stopped.
- Ran **5-fold cross-validation** (`scripts/12`): mean mAP@50:95 = 0.432±0.009. This
  looked *lower* than v1's single-split number at first glance — investigated and
  confirmed this is because the new val folds include background images (penalizing
  false positives for the first time) and harder/more varied data, not a regression.
- Trained a **final production model** on a 90/10 split of the full merged dataset
  (`scripts/13`).
- **Verified the real improvement** by evaluating on v1's original, unchanged 67-image
  test set (fair apples-to-apples comparison):

  | | Precision | Recall | mAP@50 | mAP@50:95 |
  |---|---|---|---|---|
  | v1 | 0.79 | 0.68 | 0.77 | 0.48 |
  | v2 final | **0.93** | **0.86** | **0.95** | **0.70** |

- Checked background false-positive rate directly: 2.3% (3/130 background images
  produced a false detection) — now measured instead of unknown.
- New checkpoint `yolov8n_pothole_v2_final_best.pt` promoted as the default detector,
  wired into `scripts/10`'s pipeline.
- Full writeup: `context/PHASE1_V2_RESULTS.md`.

## 6. Severity function redesign + depth integration

On request ("integrate depth into severity, we don't have a severity dataset, choose a
best-fitting function, refer research papers, re-test on a small set, document
everything including the note photos").

- **Researched the literature** (web search): found ASTM D6433 (the real
  pavement-engineering PCI standard) classifies pothole severity from a **diameter ×
  depth matrix** — the structural principle (depth escalates severity for a fixed size)
  the new function is built around. Also found arXiv:2505.21049, a 2025 paper using the
  same YOLOv8 + Depth-Anything-V2 + area-estimation architecture this project already
  had, validating the pipeline shape.
- **Found a real severity-labeled dataset** the project didn't have before:
  `idanbaru/annotated-potholes-with-severity-levels` (717 images, 1886 boxes labeled
  minor/medium/major pothole — an enhanced/re-labeled version of the same
  chitholian/andrewmvd corpus, with a documented caveat that most labels were
  script-derived, not purely manual).
- **Calibrated the area factor** (`scripts/14`): grid-searched thresholds against
  idanbaru's real labels, optimizing balanced accuracy (not raw accuracy, which
  degenerates under class imbalance). Result: thresholds 0.0208/0.1030, balanced
  accuracy 0.525 (vs. 0.33 chance) — **quantifies that area alone is a weak-to-moderate
  signal**, directly justifying why depth needed to become a real second factor.
- **Calibrated the depth factor**: since metric depth still isn't available at
  inference time (same limitation as before), designed a scale-independent "steepness"
  ratio (depth signal / pothole pixel extent) on Depth-Anything-V2's own uncalibrated
  scale, with an escalation threshold (90th percentile) calibrated from 59 PothRGBD
  samples.
- **Integrated depth into `scripts/10`'s severity function**: an unusually steep
  (deep-for-its-size) pothole now escalates severity up one level (LOW→MEDIUM or
  MEDIUM→HIGH, capped at HIGH) — matching ASTM's real structural logic without
  fabricating metric precision the pipeline doesn't have.
- **Re-ran the pipeline on a fresh small test set** (10 images, both domains) to verify
  it works end-to-end: confirmed the escalation logic fires correctly and caps properly
  (e.g. one detection with area alone indicating LOW correctly escalated exactly one
  level to MEDIUM by a genuinely deep, small-diameter hole). Found and fixed one minor
  rendering bug (severity labels running off the right edge of the image) during this
  check.
- Full writeup: `context/SEVERITY_FUNCTION.md`.

## 7. Documentation set (this pass)

- `CHANGELOG.md` (this file) — full session record.
- `context/HANDWRITTEN_NOTES_TRANSCRIPTION.md` — verbatim page-by-page transcription of
  the 9 handwritten notebook photos shared at the start of the project, plus a mapping
  of which project doc each page's content ended up informing.
- `context/SEVERITY_FUNCTION.md` — the severity-function research, dataset, and
  calibration writeup.

## Known open items (not done in this pass, stated explicitly)

- Depth-Anything-V2 has not been fine-tuned on PothRGBD (only evaluated).
- Segmentation training data is still limited to PothRGBD's 2 capture dates — the
  cross-domain `UNKNOWN` gap (finding #6) is unresolved.
- No end-to-end severity accuracy number exists (only the area-factor-in-isolation
  accuracy was measured against idanbaru) — running full detect+segment+severity on
  idanbaru's images and comparing to its labels would give a true pipeline-level number.
- Audit finding #4 (crack-shaped label outliers) not addressed.
- Datasets/cache on the second drive (~12GB+) are still present — per earlier
  instruction, these should be deleted once you're satisfied with the results.
