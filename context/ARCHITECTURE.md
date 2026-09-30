# Architecture (transcribed from project notebook)

## Confusion matrix basics

A confusion matrix shows what the model predicted vs. what was actually present. YOLO is
doing object detection, not simple image classification — so the confusion matrix is based
on whether predicted detections can be matched to ground-truth objects, usually via an IoU
threshold.

|              | Predicted +ve | Predicted -ve |
|--------------|----------------|----------------|
| **Actual +ve** | True +ve (TP) | False -ve (FN) |
| **Actual -ve** | False +ve (FP) | True -ve (TN) |

Derived metrics:
- **Accuracy** = (TP + TN) / (TP + FP + TN + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **Specificity** = TN / (TN + FP)
- **F1 score** = (2 × TP) / (2×TP + FP + FN)

## IoU (Intersection over Union)

Used to decide whether a predicted box is actually close enough to the ground truth.

```
IoU = Area of Intersection / Area of Union
```
computed between the predicted box (YOLO) and the actual ground-truth box.

## mAP@50 vs mAP@50:95

- **mAP@50**: a detection counts as correct if its IoU with the ground-truth box is ≥ 0.50.
  Answers "how good is the detector when a predicted box only needs 50% overlap with the
  ground truth to count as correctly localized?"
- **mAP@50:95**: evaluates across IoU thresholds 0.50, 0.55, ..., 0.95 and averages the
  resulting AP values. Much stricter — a model can have high mAP@50 but much lower
  mAP@50:95 if its boxes are found but not tightly localized (relevant here because
  imprecise boxes will bias relative depth/severity estimation in Phase 2 — see the
  "for severity calculation" note below).

Example interpretation used in the notes: mAP@50 = 0.92, mAP@50:95 = 0.58 → the model is
probably finding potholes, but its boxes aren't especially precise. This matters for
severity because if the box doesn't tightly fit the pothole, the relative width/height/
depth estimate used downstream will be affected.

## Why class loss (cls_loss) is nonzero with only one class

Even with one class, the model still makes class-confidence predictions for candidate
detections, and the loss function measures whether those predictions agree with the
assigned target class. Example: candidate box predicted as "class 0? confidence 0.72" but
the target says "definitely class 0" → still a classification error (confidence should
have been ~1.0, wasn't), captured by BCE-style class loss even in a single-class setting.

## DFL loss (Distribution Focal Loss)

Related to bounding-box localization — associated with precise localization of bounding
box edges. Measures how well the model is learning the distribution of bounding-box
coordinates (YOLOv8's box regression represents each box edge as a discrete probability
distribution rather than a single regressed scalar; localization loss combines this
distribution-based mechanism with IoU-related geometry).

## Overfitting / model health signals

- If training loss keeps decreasing and validation loss decreases then increases →
  overfitting.
- Generally: precision↑, recall↑, mAP↑, validation loss↓ together = model learning
  increasingly useful discriminative & localization features for pothole detection. But
  need to confirm all four move together, not just training loss dropping.
- If recall & precision curves visually coincide, that does NOT mean FP = FN — many
  different (FP, FN) combinations can produce the same computed precision/recall values.
  Coincidence of the curves is suggestive, not proof.

## Ultralytics checkpoints

Ultralytics saves checkpoints during training (`last.pt` every epoch, `best.pt` = best
validation performance so far), but the "complete weights" are not retrained from scratch
indefinitely every epoch — training resumes/continues incrementally. The two checkpoints
that matter: `weights/last.pt` and `weights/best.pt`.

## VOC XML → YOLO txt pipeline (no separate XML→TXT converter tool used)

```
PNG image
XML annotation ──> xml.etree.ElementTree parser (import xml.etree.ElementTree as ET)
                 ──> extract xmin/ymin/xmax/ymax
                 ──> coordinate conversion (VOC absolute px -> YOLO normalized cx,cy,w,h)
                 ──> YOLO .txt label
```
A coordinate-transformation program was written in Python (this repo:
`scripts/02_convert_voc_to_yolo.py`) — there is no separate XML→TXT converter tool
involved; parsing + coordinate math is done directly.

## Model computation (YOLOv8n)

- YOLOv8n: ~8.2 GFLOPs, ~3.0M parameters — lightweight detector.
- GFLOPs roughly describes the amount of floating-point computation required for one
  forward inference pass under the stated configuration (e.g. 640×640 input).
- During training all images are resized to 640×640 as part of preprocessing.

```
IMAGE INPUT
  -> preprocessing (resize 640x640 / normalization)
  -> YOLOv8n backbone (feature extraction)
       -> neck (feature fusion)
  -> detection head
       -> box localization
       -> classification (1 class)
       -> DFL localization
  -> NMS / post-processing
  -> detection results (bounding box, confidence, class = pothole)
```

Transfer learning note: ~319/355 weight tensors were transferred from COCO-pretrained
YOLOv8n weights even though the head is overridden from 80 classes to 1 class
(class 0 = pothole). Most of the useful feature-extraction machinery is inherited from
the pretrained backbone; only the class-specific detection machinery is adapted to the
pothole task. The frozen/mismatched-shape tensors (the ones not transferred) correspond
to layers whose output dimensionality depends on the number of classes — including the
DFL convolution weight, which is why gradient/parameter diffs show up there.

The training dataset is small relative to the model's capacity, so data augmentation is
used every epoch (flip, scale, mosaic). `close_mosaic=10` was set — mosaic augmentation
is disabled for the final 10 epochs so the model fine-tunes on more natural (non-mosaiced)
image distributions after the early epochs used strong augmentation to prevent overfitting
on a small dataset.

## Phase 2 architecture (planned — not yet implemented in this pass)

**Objective**: increase recall (FN = missed potholes; for road safety, missed potholes are
a serious failure mode), improve mAP@50:95 (severity prediction depends on getting a
reasonably accurate pothole region), keep precision from collapsing, and add severity
detection.

```
Detected pothole -> segmentation model -> pixel-level pothole mask
  -> bounding box + mask -> geometric analysis (width, length, area)
  -> relative geometry -> severity rules -> LOW / MEDIUM / HIGH -> final annotated image
```

Depth branch (explored, not finalized):
```
RGB image -> Depth-Anything-V2 -> depth map
           -> apply pothole mask -> depth statistics (mean, P95, max)
           -> feature vector -> severity function -> LOW / MEDIUM / HIGH
```

Notes flagged: pretrained Depth-Anything-V2 was **not performing well** on real pothole
RGBD pairs when checked qualitatively against the PothRGBD dataset (RGB + raw sensor
depth + polygon segmentation annotations, matched by `YYYYMMDD_HHMMSS` timestamp in
filenames). Planned mitigation: fine-tune Depth-Anything-V2 on PothRGBD's paired RGB/depth
data before trusting its depth output for severity scoring.

Final architecture sketched in notes:
```
Input Image -> YOLOv8 -> detections -> bounding boxes
  -> segmentation model -> masks
  -> [geometry] + [Depth-Anything-V2 -> depth map] -> feature extraction
  -> severity function -> final visualization
```

Also considered: FN (missed potholes) can potentially be reduced by training on more
small-pothole images specifically, since small/distant potholes are the likely blind spot.
