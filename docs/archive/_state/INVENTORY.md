# Project Inventory

## Folder Map
- `/`: Project root. Contains main configuration files (e.g. `README.md`, `requirements.txt`, `.gitignore`).
- `/src/`: Core python modules, including `config.py` for path resolution.
- `/scripts/`: Python pipeline scripts (01 to 14) for data processing, training, and inference.
- `/context/`: Original documentation, results, evaluations, and handwritten notes.
- `/checkpoints/`: Local directory containing trained YOLO `.pt` model weights.
- `/runs/`: Ultralytics output directories containing logs, config YAMLs, and evaluation plots.
- `/docs/`: Destination for the final compiled world-class documentation.
- `/docs/_state/`: Ledger and state-tracking files (this directory).
- `/docs/_original/`: Untouched copies of original context docs (to be populated).
- `/docs/assets/`: Final destination for diagrams and sample images.

## Results & Graphs
- `context/phase1_training_curves.png`: Training curve plots for initial Phase 1 YOLOv8n training.
- `context/phase1_val_confusion_matrix.png`: Validation confusion matrix for Phase 1.
- `context/phase1_test_predictions.png`: Visual predictions from Phase 1 test set.
- `context/phase1_v2_final_training_curves.png`: Training curve plots for improved Phase 1 v2.
- `context/phase1_v2_final_confusion_matrix.png`: Confusion matrix for improved Phase 1 v2 on test set.
- `context/phase2_seg_training_curves.png`: Training curves for YOLOv8n-seg model on PothRGBD.
- `context/phase2_seg_val_confusion_matrix.png`: Confusion matrix for Phase 2 segmentation model.
- `runs/pothole_v2_final/final90_10/`: Full Ultralytics logs for the final v2 detection model.
- `runs/pothole_seg_training/yolov8n_seg_pothrgbd/`: Full Ultralytics logs for the segmentation model.

## Data Files
- `context/phase1_annotations.csv`: Original bounding box annotations log.
- `context/phase1_training_results.csv`: Log of loss and metrics for Phase 1.
- `context/phase1_v2_geometry_features.csv`: Extracted geometric features for severity scoring.
- `context/phase2_seg_training_results.csv`: Log of loss and metrics for Phase 2 segmentation.
- `context/idanbaru_severity_geometry.csv`: Ground truth vs geometry mapping for the idanbaru dataset.
- `context/severity_calibration.json`: JSON storing the calibrated thresholds for area and steepness.

## Weights
- `checkpoints/yolov8n_pothole_phase1_best.pt`: Original Phase 1 weights.
- `checkpoints/yolov8n_pothole_v2_final_best.pt`: Final recommended detection weights (Phase 1 v2).
- `checkpoints/yolov8n_seg_pothrgbd_best.pt`: Final recommended segmentation weights (Phase 2).
- `yolo26n.pt`: Stray YOLO weight file in root.

## Sample Images
- Path: External Drive (`$POTHOLE_DATA_ROOT/datasets/pothole_yolo/images/test/`).
- Examples found previously: `potholes89.png`, `potholes411.png`.
- Output artifacts from tests: `/tmp/pothole_demo_audit/REPORT.md`.
