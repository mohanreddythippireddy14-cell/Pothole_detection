# 01. System Overview

## The problem
Road departments must find and prioritise road damage across large areas. Manual inspection is slow and subjective, and citizen reports usually say *where* a pothole is but not *how bad* it is.

## The objective
Turn an ordinary RGB road photo into a per-pothole severity rating (LOW / MEDIUM / HIGH) that could feed a repair-prioritisation workflow. A dashboard integration is a planned use case, not part of this repository.

## Core contribution
The project does not invent a new model architecture. Its contribution is:

1. **Integration:** chaining detection (YOLOv8n), per-crop segmentation (YOLOv8n-seg) and monocular depth (Depth-Anything-V2) so size *and* depth cues come from one photo.
2. **A transparent severity engine:** decision rules that reproduce the structure of ASTM D6433 (severity = f(size, depth); deeper ⇒ more severe for the same size) using scale-free proxies, calibrated against the data that exists.

See `05_phase3_severity.md` for the engine and `07_failure_modes.md` for its limits.
