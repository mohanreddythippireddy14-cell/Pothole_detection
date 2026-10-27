# Phase 1 — YOLOv8n Pothole Detection: Local Reproduction Results

Full local re-run of the Colab notebook's Phase 1 pipeline (see `context/PROJECT.md` for
storage layout, `context/ARCHITECTURE.md` for the underlying concepts, and
`context/PHASE1_NOTEBOOK_REFERENCE.md` for the original notebook's recorded numbers).

## Pipeline run

1. `scripts/01_download_and_audit.py` — downloaded `andrewmvd/pothole-detection` via
   kagglehub (cached on second drive). **665 images, 665 XML annotations, 1740 annotated
   pothole boxes, 0 corrupt images.** Audit flagged **1 degenerate ground-truth box**
   (`potholes58.xml`, `xmin==xmax`, zero pixel width) — not present in the notebook's
   printed stats (`box_width` min was reported but not flagged there).
2. `scripts/02_convert_voc_to_yolo.py` — VOC XML → YOLO txt, 80/10/10 split (seed=42):
   **train 532 / val 66 / test 67**, matching the notebook's split sizes exactly. The one
   degenerate box was dropped rather than converted to a zero-width YOLO label (fix
   applied here, not present in the original notebook — see code comment in the script).
3. `scripts/03_train_yolov8n.py` — YOLOv8n, 100 epochs requested, imgsz=640, batch=16,
   patience=20, seed=42, deterministic=True. **Early-stopped at epoch 86** (best epoch:
   66), total training time **0.091 hours (~5.5 min)** on a Quadro RTX 6000.
4. `scripts/04_evaluate.py` — best.pt evaluated on the held-out **test** split (67 images,
   161 ground-truth boxes after dropping the degenerate one).
5. `scripts/05_visualize_samples.py` — sample training-box overlays and test predictions.

## Environment note (not in the original Colab run)

The system `python3.11` on this machine is a leftover pre-release build (reports as
`3.11.0rc1`, missing `sys.get_int_max_str_digits`/`set_int_max_str_digits`, which
`torch._dynamo`'s polyfill loader imports unconditionally). Fixed with a compat shim
(`src/_py311_compat.py`) that stubs the two functions with real-CPython-3.11+ default
behavior (default limit 4300) before `torch` is imported anywhere. This has no bearing on
training correctness — it only unblocks `torch.use_deterministic_algorithms()`'s internal
import chain.

## Results — held-out test set (67 images, 161 boxes)

```
Precision      : 0.7935
Recall         : 0.6770
F1             : 0.7307
mAP@50         : 0.7708
mAP@50:95      : 0.4773
```

Plots: `context/phase1_eval_plots/` (confusion_matrix.png, BoxPR_curve.png, BoxF1_curve.png,
val_batch*_pred.jpg vs val_batch*_labels.jpg side-by-side comparisons — note these val_batch
images are actually from the val split used during `model.val()`'s internal plotting, while
the printed numbers above are the test-split numbers).

## Results — best-epoch validation set (66 images, epoch 66, from training)

```
Precision      : 0.798
Recall         : 0.760
mAP@50         : 0.842
mAP@50:95      : 0.535
```

## Comparison to original notebook run

| Metric | Notebook (original) | This local run (test split) |
|---|---|---|
| Precision | ~0.86 | 0.7935 |
| Recall | 0.6647 | 0.6770 |
| mAP@50 | 0.7549 | 0.7708 |
| mAP@50:95 | 0.4941 | 0.4773 |

Same ballpark as the original notebook, not identical — expected given different library
versions (Ultralytics 8.4.121 here vs. whatever Colab had at notebook-authoring time),
different torch/CUDA versions, and RNG draws in augmentation that differ across library
versions even with the same seed. **Directionally the same conclusions hold**: precision
is the strongest metric (model rarely hallucinates potholes), recall is the weak point
(~24-32% of potholes missed depending on run), and mAP@50:95 sits well below mAP@50
(confirms boxes are found more reliably than they are tightly localized) — this is the
number Phase 2's severity estimation needs to watch.

## Training/validation loss trend — overfitting check (notes Q&A item 8)

Checked directly against the notes' claim ("if training loss keeps decreasing and
validation loss decreases then increases, conclude overfitting"):

- `train/box_loss`: 1.68 (epoch 1) → 0.87 (epoch 86), monotonic decrease.
- `val/box_loss`: 1.46 (epoch 1) → 1.16 (epoch 86), **also decreasing overall**, with its
  minimum (1.15) around epoch 80, i.e. near the end of training, not diverging upward.

**No overfitting signature in this run** — both curves trend down together through the
early-stop point. Early stopping triggered because `mAP@50:95` plateaued for 20 epochs
(best at epoch 66), not because of validation loss divergence. Full curves:
`context/phase1_training_curves.png`, raw per-epoch numbers:
`context/phase1_training_results.csv`.

## Sample outputs

- `context/phase1_training_samples.png` — 6 random training images with YOLO-format boxes
  drawn from the converted labels (sanity check on the VOC→YOLO conversion).
- `context/phase1_test_predictions.png` — 6 test-set images with the trained model's
  predictions overlaid (conf=0.25).
- `context/phase1_val_confusion_matrix.png` — confusion matrix from the training run's
  final validation pass.

## Checkpoint

`checkpoints/yolov8n_pothole_phase1_best.pt` (also at
`$POTHOLE_DATA_ROOT/checkpoints/yolov8n_pothole_phase1_best.pt`,
copied automatically by `scripts/03_train_yolov8n.py`). This is the checkpoint Phase 2's
detection stage should load.

## Data-quality finding not present in the original notebook

One ground-truth annotation (`potholes58.xml`) has `xmin == xmax` (a zero-pixel-width
box) — geometrically invalid, likely an annotation tool artifact (single click without
drag, or an off-by-one in the source labeling). It normalizes to `width=0.0` in YOLO
format, which is degenerate for IoU-based matching and box-regression loss. This local
pipeline drops it during VOC→YOLO conversion (`scripts/02_convert_voc_to_yolo.py`); the
original notebook's `convert_voc_to_yolo()` did not filter it, so if the notebook trained
directly on this dataset without downstream YOLO validation-time filtering, that run
would have carried one degenerate label silently. Not expected to materially change
results given it's 1 of 1740 boxes (0.06%), but worth fixing for correctness.
