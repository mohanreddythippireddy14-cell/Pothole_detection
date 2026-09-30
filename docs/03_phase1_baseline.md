# 03. Phase 1: Bounding Box Baseline

## Methodology
The project began by establishing a robust 2D object detection baseline using `YOLOv8n` (nano). The objective was to determine if standard bounding boxes could suffice for severity estimation based purely on 2D area (width $\times$ height).

## Results
After the dataset expansion (Phase 1 v2), the baseline model achieved:
- **mAP@50:** 0.950
- **Recall:** 0.860

## The Geometric Failure Mode
While detection accuracy was exceptionally high, we discovered a fatal flaw in using bounding boxes for severity assessment. Potholes are inherently irregular and often oriented diagonally relative to the camera frame. A rectangular bounding box inevitably includes significant portions of non-damaged road surface. 

This "background distortion" led the heuristic engine to overestimate damaged area by up to 40%, rendering the severity ratings unusable for strict municipal triage. This necessitated the architectural shift to Phase 2.
