# Handwritten notes — verbatim transcription

You shared 9 photos of handwritten notebook pages at the start of this project (plus the
Colab `.py` export). This document transcribes their text content page-by-page, as a
durable record separate from the synthesized/reorganized version in
`context/ARCHITECTURE.md`. Minor illegible fragments are marked `[?]`; obvious shorthand
is expanded in `[brackets]` only where needed for clarity, otherwise transcribed as
written (including the author's own arrows, abbreviations, and informal phrasing).

---

## Page 1 — Confusion matrix basics

> • A confusion matrix:—
> shows what the model predicted v/s what was actually present.
> → YOLO is doing object detection, not simple image classification. So confusion
> matrix is based on weather [whether] predicted detections can be matched to ground
> truth objects, usually using an IoU threshold.
>
> [Table: rows = Actual +ve / Actual -ve, columns = Predicted +ve / Predicted -ve]
> Actual +ve, Predicted +ve → True +ve
> Actual +ve, Predicted -ve → False -ve
> Actual -ve, Predicted +ve → False +ve
> Actual -ve, Predicted -ve → True -ve
> "Confusion matrix"
>
> ① Accuracy: (TP + TN) / (TP + FP + TN + FN)
> ② Precision: (TP) / (TP + FP)
> ③ Recall: (TP) / (TP + FN)
> ④ Specificity: (TN) / (TN + FP)
> ⑤ F1 score: (2×TP) / (2TP + FP + FN)

## Page 2 — IoU and open questions

> → What is IoU:— Intersection over Union,
> → for deciding whether a predicted box is actually close enough to the ground truth.
> → Predicted box by YOLO & actual ground truth box,
> IoU = Area of Intersection / Area of Union
>
> ① How the .xml is converted to .txt ?
> ② What are mAP@50 & mAP@50:95 ?
> ③ How class loss can happen, when there is only one class?
> ④ What is dfl loss ?
> ⑤ Recall = (TP) / (TP + FN) → Recall ↑ means, better model.
> ⑥ → Isn't precision should also be high.
> ⑥ → Are we storing the weights of every epoch & loading the best performed epoch and
>    actually during the testing phase.
> ⑦ → How is box_loss actually calculated?
> ⑧ → If validation loss is on the top of the training loss in all epochs, what can we
>    conclude from that?
> ⑨ → When recall & precision graphs are coinciding, we can conclude that FP=FN, what
>    does this tell us about the model?
> ⑩ → With one class available, how can the classification loss even happen?
> ⑪ → for what purpose, xml.etree.ElementTree is used in pothole detection project?

## Page 3 — Answers, part 1

> ① There is no XML → TXT converter involved.
> → We used a coordinate transformation program using Python.
> ↳ We parse the xml using, import xml.etree.ElementTree as ET
> Then, extract the variable stored, & then we calculate the variables that are
> actually needed in YOLO format.
>
> ② mAP = mean Average Precision.
> ↳ IoU: Intersection / Union {actual box, predicted box}
> ↳ mAP@50:— We consider a detection is correct if its IoU with the ground truth box
>   is atleast 0.50, subject to the matching rules.
> → How good is the detector when a predicted box only needs 50% overlap with the
>   ground truth to count as correctly localized?
> → mAP@50:95:—
> ↳ It evaluates across IoU = 0.50, 0.55 . . . 0.95 & averages the resulting AP values.

## Page 4 — Answers, part 2 (severity implication, class loss, DFL)

> → Suppose, mAP@50 = 0.92
>            mAP@50:95 = 0.58.
> we can conclude that,
> — the model probably finding potholes, but its boxes aren't especially precise.
> ## → for severity calculation, this is imp[ortant], if the box is not exactly
> fitting in it, relative width/height/depth estimation will be effected.
>
> ③②⑩ Even with one class, model can't have non-zero cls_loss, because,
> — the model still makes class predictions for candidate detections, & the loss func
> measures whether those predictions agree with the assigned target class.
> e.g:— Candidate box ↓ Is cls 0? ↓ Model says = 0.72 confidence
>       But target says, definitely class 0.
>       → So, still there is a classification error.
>
> ④ DFL = Distribution focal loss.
> ↳ related to bounding box localization → associated with precise localisation of
>   bounding box edges.
> → dfl_loss:— How well is the model learning the distribution of bounding-box
>   co-ordinates?

## Page 5 — Answers, part 3 (overfitting, checkpoints, box loss, precision/recall)

> ⑤ Generally yes, but we need to ensure that:—
>   precision ↑
>   recall ↑        } model learning increasingly useful discriminative &
>   mAP ↑            } localization features for pothole detection.
>   validation loss ↓ }
>
> ⑥ Ultralytics does save checkpoints during training, but you should not assume that
>   the complete weights for every single epoch are all retained indefinitely.
>   → imp[ortant] checkpoints are: → last.py [sic, means last.pt]
>                                    → best.py [sic, means best.pt]
>
> ⑦ GT says (xmin, ymin, xmax, ymax) & prediction (x'min, y'min, x'max, y'max)
>   → Here, MSE is not simply implemented.
>   → In YOLO, localization loss is based on IoU-related geometry, together with the
>     distribution-based localization mechanism represented by DFL.
>
> ⑧ If training loss continues to decrease & validation loss ↓'es & then ↑'es. Then we
>   can conclude that model is overfitting.
>
> ⑨ If, recall & precision graph coincides that doesn't mean FP=FN. There'll be many
>   combinations available that gives the same result.
>   → May be but not for sure.

## Page 6 — XML→TXT pipeline diagram + Phase 1 conclusions

> XML → TXT conversion
> PNG → XML → ElementTree parser → xmin/ymin/xmax/ymax → co-ordinate conversion →
> YOLO TXT LABELS
>
> Phase 1, conclusion after results:—
> ① → precision high, meaning model is correct 86% of the time, ⇒ model is not
>   hallucinating potholes everywhere.
>   → FP is not an issue.
> ② Recall = 0.6647     100 actual potholes
>                        ↓
>                        66 detected
>                        ↓
>                        34 missed
>   → FN is a concern.
>   → The confidence threshold & learned confidence distribution also influence
>     precision/recall.
> ③ mAP@50 = 0.7549. for IoU≥50, there is only 75.5% of localization criteria met.
>   → mAP@50:95 = 0.4941
>   → this is critical for severity calculation.

## Page 7 — Phase 1 training details

> → During phase 1 training:—
> * Transferred 319/355 items from pretrained weights.
> * We used YOLOv8 nano with 3.1 million parameters.
> * 80 classes are overridden with 1 class, class 0 = pothole.
> * for detecting only 1 class, we almost used most of the pretrained weights, which
>   are trained for 80 classes.
> * Most of the useful feature extraction machinery is inherited from pretrained
>   YOLOv8n, while the class specific detection machinery is adapted to pothole task.
> * Diff b/w parameters & gradients correspond to the frozen weights. Here that is
>   related to frozen DFL convolution weight.
> * The used training dataset is far small relative to the model's capacity.
> * for every epoch we did data manipulation:— Data Augmentation.
> * We used close_mosaic = 10.
>   This is because, only [early] training gets strong augmentation, while the final
>   stage lets the model fine-tune on more natural image distributions.
>   [diagram] Original image → Augmentation → flip / scale / mosaic → Training images

## Page 8 — Model computation & Phase 1 architecture diagram

> Model computation:
> → YOLOv8n → 8.2 GFLOPs → 3.0M parameters — lightweight detector.
> → GFLOPs roughly describes the amount of floating point computation required for one
>   forward inference under the stated configuration.
> * During training all images are resized to 640×640 as part of pre-processing.
> → By the end of phase 1:—
> IMAGE INPUT
>   ↓
> Preprocessing → resize 640×640 / Normalization
>   ↓
> YOLOv8n
> 3,091,043 params
> 8.2 GFLOPs
>   ↓                              ↓
> Backbone                        Neck
> feature extraction              feature fusion
>   ↓                              ↓
>          Detection head
>   ↓              ↓                  ↓
> Box localization  Classification    DFL
>                   (1 class)         localization
>   ↓
> NMS / post-processing
>   ↓
> Detection Results
> — Bounding box
> — confidence
> — class = pothole

## Page 9 — Phase 2 architecture plan

> Phase 2
> Main objective:— ① Increase recall.
>   FN = missed potholes.
>   and for road safety, missed potholes are a serious failure mode.
> ② ↑ mAP@50:95 because, severity prediction depends on getting a reasonably accurate
>   pothole region.
> ③ Precision shouldn't collapse.
> ④ Severity detection.
>   Detected pothole → Segmentation model → Pixel level pothole mask
>   ↓
>   Bounding box            Mask
>   ↓                        ↓
>              Geometric Analysis
>   ↓             ↓            ↓
> width          length        Area
>   ↓
> Relative Geometry
>   ↓
> Severity rules
>   ↓
> LOW / MEDIUM / HIGH
>   ↓
> Final annotated image.
>
> [box diagram] Pothole
>   Bounding box
>     [Segmentation mask]
>   confidence: 91%
>   severity: High

## Page 10 — Depth branch exploration + FN mitigation

> → FN:— can overcome by training model on more small pothole images.
>
> RGB Image
>   ↓
> YOLO Detection
>   ↓
> b-box + confidence
>   ↓                    ↓
> geometry           padded crop
>                         ↓
>                   yolov8-segmentation
>                         ↓
>                       mask
>   ↓                     ↓
>        geometric features
>
> RGB → Depth Anything V2 → depth map
>   ↓
> Apply pothole mask
>   ↓
> depth statistics
>   ↓          ↓        ↓
> mean        P95       max
>   ↓
> Feature vector
>   ↓
> Severity function
>   ↓        ↓         ↓
> LOW      MEDIUM      HIGH

## Page 11 — Present vs. Final architecture comparison

> Original RGB image
>   ↓                                    ↓
> YOLO                          Depth-Anything V2
>   ↓                                    ↓
> Bounding box                       Depth map
>   ↓                                    ↑
> Segmentation      ←───────────────────┘
>   ↓
> Mask
>   ↓
> Depth inside pothole
>
> → Depth-Anything v2 pretrained, not performing well.
>
> pothRGBD (dataset)
>   ↓              ↓
> RGB images     Depth maps
>   ↓
> Depth-Anything V2 pretrained
>   ↓
> Finetune on pothRGBD
>   ↓
> depth map.

## Page 12 — Final architecture (present vs. final side-by-side)

> Present:
> Input RGB
>   ↓            ↓
> YOLOv8      Depth-Anything V2
>   ↓            ↓
> Bounding    Depth map
> Box
>   ↓
> Segmentation
> using YOLO box
>   ↓
> Pothole mask
>   ↓            ↓
> Geometry + depth features
>   ↓
> Severity function
>   ↓
> Final annotated image.
>
> Final Architecture:
> Input Image
>   ↓
> YOLOv8
>   ↓
> Detections
>   ↓
> Bounding boxes
>   ↓
> Segmentation model
>   ↓
> Masks
>   ↓                          ↓
> Geometry              Depth-Anything V2
>                               ↓
>                         Depth map
>   ↓                          ↓
>      Feature extraction
>          ↓
>   Severity function
>          ↓
>   Final visualisation.

---

## How this transcription was used in the project

Every technical claim in these notes was either implemented, tested against real data,
or explicitly flagged where the project deviated from the plan:

- Pages 1-5 (confusion matrix, IoU, mAP, DFL, checkpoints, overfitting Q&A) →
  `context/ARCHITECTURE.md`'s "Confusion matrix basics" through "Overfitting / model
  health signals" sections, each claim checked against actual training logs
  (`context/PHASE1_RESULTS.md`'s overfitting check, `context/PHASE1_V2_RESULTS.md`'s
  close_mosaic verification).
- Page 6-8 (Phase 1 conclusions, transfer-learning stats, architecture diagram) →
  `context/PHASE1_NOTEBOOK_REFERENCE.md` (kept as the original run's reference numbers)
  and reproduced independently in `context/PHASE1_RESULTS.md` (319/355 transferred
  weights, 3.0M params, 8.2 GFLOPs all matched exactly in the local re-run).
- Page 9-12 (Phase 2 severity architecture) → implemented in full in
  `scripts/07`-`scripts/10` and `scripts/14`; see `context/PHASE2_RESULTS.md` and
  `context/SEVERITY_FUNCTION.md` for what was built, what deviated from the notes (depth
  fine-tuning on PothRGBD was evaluated quantitatively rather than done, per page 11's
  own flag that pretrained Depth-Anything-V2 "not performing well"), and why.
