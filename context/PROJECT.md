# Pothole Detection & Severity Estimation

**Origin**: ported from a Google Colab notebook (`pothole_detection (1).py`) + handwritten
project notes documenting Phase 1 (detection) and Phase 2 (segmentation + depth + severity)
architecture. This repo re-runs those experiments locally, outside Colab, with results
logged in `context/`.

## Storage layout (root disk only has ~2.2GB free — DO NOT write datasets/checkpoints there)

All datasets, model checkpoints, and training runs live on the second drive:
`$POTHOLE_DATA_ROOT/`
```
pothole_detection_data/
├── datasets/     # kagglehub-downloaded datasets (raw)
├── checkpoints/  # trained weights (best.pt, last.pt, depth checkpoints)
├── runs/         # ultralytics training run directories
├── cache/        # uv + kagglehub cache dirs
└── venv/         # python venv (symlinked as .venv in this repo)
```
This is a scratch/experiment area — datasets get deleted after training+validation is
complete and results are documented (per user instruction). Code, configs, and docs live
in `the repository root (`Pothole_detection/`)` (this repo) which stays on root but is small
(no large binaries checked in).

## Architecture (from notes, see context/ARCHITECTURE.md)

**Phase 1 (this pass)**: YOLOv8n object detection on potholes.
- Dataset: Kaggle `andrewmvd/pothole-detection` (Pascal VOC XML annotations)
- Convert VOC XML -> YOLO txt format (single class: pothole)
- Train YOLOv8n (transfer learning from COCO pretrained, 80->1 class head)
- Evaluate: confusion matrix, precision, recall, mAP@50, mAP@50:95, F1
- Document results, confusion-matrix-derived metrics, and answers to the Q&A list in the
  notes (IoU, mAP, DFL loss, class loss w/ 1 class, etc.)

**Phase 2 (future pass)**: segmentation + depth + severity.
- Pixel-level pothole segmentation (YOLOv8-seg on cropped detections)
- Geometric analysis (width/length/area from mask)
- Depth estimation (Depth-Anything-V2) for relative depth inside the pothole
- PothRGBD dataset explored as ground-truth depth source; notes say pretrained
  Depth-Anything-V2 "not performing well" on real pothole RGBD pairs -> may need
  fine-tuning on PothRGBD
- Severity function: combine geometry + depth stats (mean/P95/max) -> LOW/MEDIUM/HIGH
- Final annotated image output

## Status (2026-08-19)

**Phase 1 — complete.** YOLOv8n detector trained on `andrewmvd/pothole-detection`
(665 images, 1740 boxes). Test-set: Precision 0.79, Recall 0.68, mAP@50 0.77,
mAP@50:95 0.48. Checkpoint: `checkpoints/yolov8n_pothole_phase1_best.pt`.
Full writeup: `context/PHASE1_RESULTS.md`.

**Phase 2 — complete (first pass).** YOLOv8n-seg trained on PothRGBD (1000 RGB/depth/
polygon triples). Test-set: Mask mAP@50 0.908, mAP@50:95 0.592. Depth-Anything-V2
quantitatively evaluated against PothRGBD sensor depth (not fine-tuned). Integrated
detect→segment→geometry→depth→severity pipeline built and demoed on both dataset
domains, with one real bug found+fixed (crop-relative vs image-relative area) and one
real cross-domain segmentation generalization gap observed. Checkpoint:
`checkpoints/yolov8n_seg_pothrgbd_best.pt`. Full writeup: `context/PHASE2_RESULTS.md`.

**Phase 1 v2 — complete.** Performance audit (`context/PERFORMANCE_AUDIT.md`) identified
why Phase 1 was underperforming; acted on it by merging 4 Kaggle sources into one
deduplicated 1986-image mega-dataset (1336 annotated + 650 background), fixing a
`close_mosaic` training-schedule bug, and switching to 5-fold CV. Result on the
**original, unchanged v1 test set**: mAP@50 0.77→0.95, mAP@50:95 0.48→0.70, recall
0.68→0.86. New checkpoint: `checkpoints/yolov8n_pothole_v2_final_best.pt` — this is now
the recommended detector, superseding `yolov8n_pothole_phase1_best.pt`. Full writeup:
`context/PHASE1_V2_RESULTS.md`.

**Severity function v2 — complete.** Depth is now genuinely integrated into severity
scoring (previously diagnostic-only), via a two-factor function structurally modeled on
ASTM D6433 (the real pavement-engineering pothole-severity standard). Area thresholds
calibrated against a real severity-labeled dataset found on Kaggle
(`idanbaru/annotated-potholes-with-severity-levels`); depth-escalation threshold
calibrated from PothRGBD. See `context/SEVERITY_FUNCTION.md`. Full session history:
`docs/archive/CHANGELOG.md`. Handwritten notes verbatim transcription:
`context/HANDWRITTEN_NOTES_TRANSCRIPTION.md`.

**Not yet done**: fine-tuning Depth-Anything-V2 on PothRGBD; broadening segmentation
training data beyond PothRGBD's 2 capture dates; end-to-end severity accuracy
measurement (only the area-factor-in-isolation was measured against idanbaru).

**Cleanup reminder (per user instruction)**: datasets on the second drive
(`$POTHOLE_DATA_ROOT/datasets/`,
`cache/kagglehub/`) should be deleted once all training/validation is finished and
results are confirmed satisfactory — checkpoints and context/ docs are the durable
artifacts, the raw datasets are not.
