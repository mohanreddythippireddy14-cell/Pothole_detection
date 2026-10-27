# Pipeline Evaluation & Results

## Metric definitions
- **mAP@50:** mean average precision at IoU 0.50. Measures whether potholes are found.
- **mAP@50:95:** averaged over IoU 0.50–0.95. Measures how tightly boxes or masks fit.
- **Recall:** share of real potholes found.

## Results by phase

| Phase | Model | Evaluated on | mAP@50 | mAP@50:95 | Recall |
|---|---|---|---|---|---|
| Phase 1 v1 | YOLOv8n | andrewmvd 67-image held-out test split | 0.771 | 0.477 | 0.677 |
| Phase 1 v2 | YOLOv8n | 5-fold CV on 1,986 merged images (mean) | **0.731** | **0.432** | **0.642** |
| Phase 1 v2 | YOLOv8n | ~~v1 test split~~ | ~~0.954~~ | ~~0.704~~ | ~~0.858~~ |
| Phase 2 | YOLOv8n-seg (mask) | PothRGBD 100-image test split (in-domain) | 0.908 | 0.592 | 0.884 |

**The struck-out row is invalid:** the v1 test photos are part of the v2 training pool (as chitholian's higher-resolution copies). v1 and v2 rows use different evaluation sets and are not directly comparable.

## Failure modes observed
1. **Low recall:** about a third of potholes are missed (v2 recall 0.64).
2. **Loose localisation:** the gap between mAP@50 and mAP@50:95 shows boxes and masks are found more reliably than they fit tightly. Small or distant potholes are hardest.
3. **Cross-domain segmentation:** the segmenter, trained on two PothRGBD sessions, sometimes returns no mask on open-road crops, giving `UNKNOWN` severity.
