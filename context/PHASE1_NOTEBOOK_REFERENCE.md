# Phase 1 — original Colab run (reference numbers, from notes)

These are the numbers recorded in the handwritten notes from the **original Colab run**
(`pothole_detection (1).py`, `andrewmvd/pothole-detection` dataset, YOLOv8n, 100 epochs,
imgsz=640, batch=16). They are a reference point only — our local re-run
(`context/PHASE1_RESULTS.md`) may differ due to different hardware, package versions,
and augmentation RNG draws even with the same seed, and should be reported separately
rather than overwritten on top of these.

## Conclusion after results (from notes)

1. **Precision high** → model correct ~86% of the time → model is not hallucinating
   potholes everywhere. **FP is not an issue.**
2. **Recall = 0.6647** → of 100 actual potholes, ~66 detected, ~34 missed.
   **FN is a concern.** Confidence threshold and the learned confidence distribution
   also influence precision/recall trade-off.
3. **mAP@50 = 0.7549** → for IoU ≥ 0.50, only 75.5% of the localization criteria are met.
4. **mAP@50:95 = 0.4941** → critical number for severity calculation (Phase 2), since
   severity depends on tight, accurate box/mask localization, not just "found it".

## Training characteristics (from notes)

- 319/355 weight tensors transferred from COCO-pretrained YOLOv8n weights.
- YOLOv8 **nano**, ~3.1M parameters (3,091,043 exact count noted), ~8.2 GFLOPs.
- 80 COCO classes overridden to 1 class (class 0 = pothole); most of the useful
  feature-extraction machinery is inherited from the pretrained backbone, only the
  class-specific detection machinery is adapted.
- Dataset judged "far small relative to the model's capacity" → heavy augmentation
  used every epoch (flip, scale, mosaic), with `close_mosaic=10` (mosaic disabled for
  the final 10 epochs so the model fine-tunes on more natural, non-mosaiced images).

These numbers are what our local reproduction should be compared against — not treated
as a target to hit exactly, since dataset shuffling/splits and library versions differ.
