# Model Card: Pothole Detection & Severity Engine

## Model Details
- **Architecture:** Ensemble Pipeline (YOLOv8n + YOLOv8n-Seg + Depth-Anything-V2)
- **Task:** Object Detection, Instance Segmentation, Monocular Depth Estimation
- **Framework:** PyTorch, Ultralytics, Transformers
- **Primary Developer:** Mohan Reddy Tippireddy

## Intended Use
- **Primary Use Case:** Automated visual inspection of road surfaces from vehicle-mounted cameras to triage municipal repair schedules.
- **Out of Scope:** Absolute metric depth estimation; autonomous vehicle navigation.

## Training Data
- **Corpus:** 2,083 images curated and merged from 4 public datasets (andrewmvd, v2-2022, RDD2022, PothRGBD).
- **Split:** 70% Train, 15% Val, 15% Test.

## Evaluation Results (Test Set)
- **Detection (Phase 1 v2):** mAP@50 = 0.950, Recall = 0.860
- **Segmentation (Phase 2):** Mask mAP@50 = 0.908, Mask mAP@50:95 = 0.592

## Ethical Considerations & Caveats
- **Privacy:** Images collected from public roads may inadvertently capture license plates or pedestrians. Production deployment must include a PII obfuscation pre-processing step.
- **Bias:** The dataset is heavily biased towards daylight, clear weather conditions. Performance degrades in low-light, rain, or snow.
