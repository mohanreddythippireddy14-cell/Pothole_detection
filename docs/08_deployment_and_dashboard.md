# 08. Deployment & Dashboard Integration (planned)

**Status: not implemented.** This repository is the offline ML component. Today `scripts/10_full_pipeline.py` writes an annotated PNG per image and a markdown `REPORT.md` with per-pothole severity, confidence and features.

## Intended integration
- Expose the pipeline as a service that accepts photo uploads or video frames.
- Return a structured JSON payload per image: pothole count, boxes, severity, confidence, features.
- Store results in a geographic database (e.g. PostGIS) so a municipal dashboard can cluster duplicate reports of the same pothole by GPS location and rank repairs.

## Prerequisites
JSON output, blurring of number plates and faces, faster inference (ONNX/quantisation, since Depth-Anything-V2 is slow on edge devices), and a clean held-out evaluation. See `../FUTURE_ACTIONS.md`.
