# Pipeline Evaluation & Results

## Evaluation Protocol
- **Framework:** Ultralytics YOLOv8 natively logged evaluation metrics.
- **Metric Definitions:** 
  - `mAP@50`: Mean Average Precision at 0.50 Intersection over Union (IoU). Measures general detection capability.
  - `mAP@50:95`: Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95. Measures localization tightness (how perfectly the box/mask traces the pothole).
  - `Recall`: The percentage of actual ground-truth potholes the model successfully found.
- **Protocol Flag ⚠️:** The evaluation dataset changed between Phase 1 v1 and Phase 1 v2. Phase 1 v1 was tested on a 66-image holdout from the original dataset. Phase 1 v2 and Phase 2 were tested on a 312-image holdout from a much more diverse merged dataset (2,083 images).

## Phase-by-Phase Comparison Table

| Phase | Model Architecture | Evaluation Dataset | mAP@50 | mAP@50:95 | Recall | Key Milestone |
|-------|--------------------|--------------------|---------|-----------|--------|---------------|
| **Phase 1 v1** | YOLOv8n (BBox) | 66 images (VOC converted) | 0.770 | 0.494 | 0.662 | Established detection baseline |
| **Phase 1 v2** | YOLOv8n (BBox) | 312 images (Merged D40) | 0.950 | 0.700 | 0.860 | Solved low recall via dataset expansion |
| **Phase 2** | YOLOv8n-Seg (Mask) | PothRGBD Segmented Set | 0.908 (Mask) | 0.592 (Mask) | 0.890 | Pixel-perfect area extraction |

## Honest Failure-Mode Analysis

1. **The "Background-Distortion" Geometric Failure (Phase 1)**
   - **Issue:** Using simple bounding boxes for severity calculation inherently includes non-damaged road surface. For diagonally oriented or highly irregular potholes, the bounding box overestimates the damaged area by up to 40%. 
   - **Resolution:** This failure mode motivated the transition to Phase 2 (Instance Segmentation) to calculate exact pixel area.

2. **The "Overly Conservative" Detector (Phase 1 v1)**
   - **Issue:** The initial model achieved high precision (0.86) but abysmal recall (0.66), missing roughly 1 in 3 potholes.
   - **Diagnosis:** The training set of 665 images lacked environmental diversity (e.g., lighting, weather, varied asphalt textures). 
   - **Resolution:** Merging 4 datasets to create a 2,083-image training corpus solved this, boosting recall to 0.86.

3. **Localization Tightness on Distant Objects**
   - **Issue:** Across all phases, the gap between `mAP@50` (high) and `mAP@50:95` (mediocre) indicates that while the model finds the potholes, it struggles to draw perfectly tight boundaries around them, particularly for distant or small potholes.
   - **Impact:** This introduces a slight error margin in the final severity area calculation. Future work will investigate raising input resolution beyond 640x640 to resolve small-object boundary fuzziness.
