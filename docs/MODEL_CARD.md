# Model Card: Pothole Detection & Severity Pipeline

## Model details
- **Pipeline:** YOLOv8n (detection) → YOLOv8n-seg (segmentation, per crop) → Depth-Anything-V2-Small (relative depth, per crop, pretrained) → rule-based severity engine
- **Tasks:** object detection, instance segmentation, relative monocular depth, rule-based severity rating
- **Frameworks:** PyTorch, Ultralytics 8.4.x, Hugging Face Transformers
- **Developer:** Mohan Reddy Tippireddy

## Intended use
- **Primary use:** offline triage of road photos (dashcam frames, citizen uploads), producing per-pothole LOW / MEDIUM / HIGH severity to help prioritise repairs.
- **Out of scope:** metric depth or size measurement, safety-critical or autonomous-driving decisions, real-time video (not implemented).

## Training data
| Model | Data | Size | Split |
|---|---|---|---|
| Detector v2 | chitholian (same photos as andrewmvd, 720×720) + rajdalsaniya (1 image per original photo) + banilkumar "Normal" backgrounds | 1,986 images (1,336 annotated + 650 background) | 5-fold CV; final model on a 90/10 split. **No separate test set.** |
| Segmenter | PothRGBD | 1,000 images from 2 recording sessions | Random 80/10/10 |
| Severity calibration | idanbaru severity-labelled boxes; PothRGBD for depth threshold | 1,886 boxes; 59 images | Fitted on all data (no hold-out) |

## Evaluation results
| Model | Metric | Value | Caveat |
|---|---|---|---|
| Detector v2 | mAP@50 (5-fold mean ± std) | **0.731 ± 0.010** | Fold used for checkpoint selection is also the scored fold (slightly optimistic) |
| Detector v2 | Recall (5-fold) | 0.642 ± 0.026 | |
| Detector v2 | mAP@50 on v1 test set | ~~0.954~~ | **Invalid:** v1 test photos are in v2 training data |
| Segmenter | Mask mAP@50 / mAP@50:95 | 0.908 / 0.592 | In-domain only (same 2 recording sessions) |
| Depth-Anything-V2 | Pearson r vs. sensor depth | −0.73 (40 images) | Inverse relative depth; not metric |
| Severity (area factor only) | Balanced accuracy vs. idanbaru | 52.5% (chance 33%) | In-sample; fitted on box area, applied to mask area |

No end-to-end severity accuracy has been measured for the full pipeline.

## Limitations
- Severity is a rule-based approximation of ASTM D6433 logic, not a validated measurement.
- Depth is relative and uncalibrated; the depth escalation threshold was set on a different setup (full images, true masks) from how it is used (crops, predicted masks).
- The segmenter is trained on two recording sessions and can fail on other road scenes (reported as `UNKNOWN`).
- See `FUTURE_ACTIONS.md` for planned fixes.

## Ethical considerations
- **Privacy:** road photos may contain number plates or people. Blur them before storing or sharing outputs (not implemented yet).
- **Bias:** training photos are mostly daylight, dry conditions; expect worse results at night, in rain or in snow.
