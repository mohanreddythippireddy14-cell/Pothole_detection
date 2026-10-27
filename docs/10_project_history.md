# 10. Project History & Evolution

```mermaid
timeline
    title Project Phase Evolution
    Phase 1 v1 : YOLOv8n on andrewmvd (665 images)
               : Test mAP@50 0.77, recall 0.68 (low recall)
    Phase 1 v2 : Merged 4 sources into 1,986 images, added backgrounds
               : 5-fold CV mAP@50 0.73, recall 0.64 (harder validation folds)
    Phase 2    : YOLOv8n-seg on PothRGBD for pixel masks
               : Mask mAP@50 0.91 (in-domain)
    Severity   : Depth-Anything-V2 + ASTM D6433-style rules
               : Area thresholds + depth escalation, calibrated on idanbaru and PothRGBD
    Audit      : Found v1-test leakage in the 0.95 v2 figure
               : Corrected docs, added tests and CI
```

Earlier documentation reported a v2 mAP@50 of 0.95. That figure was measured on photos that were also in the training data, and has been withdrawn. Historical notes are in `archive/`.
