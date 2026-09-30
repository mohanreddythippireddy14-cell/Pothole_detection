# Inference Example

This visual demonstrates the pipeline transitioning from a raw image frame to a processed, severity-annotated dashboard output.

| Raw Input (`potholes103_raw.png`) | Pipeline Output (`potholes103_annotated.png`) |
|---|---|
| ![](../samples/potholes103_raw.png) | ![](../samples/potholes103_annotated.png) |

## Pipeline Log
```text
- pothole 0: severity=HIGH, det_conf=0.904, relative_area=0.2095, box_px=210x188
- pothole 1: severity=LOW, det_conf=0.809, relative_area=0.0117, box_px=57x30
- pothole 2: severity=LOW, det_conf=0.281, relative_area=0.0062, box_px=43x21
```
