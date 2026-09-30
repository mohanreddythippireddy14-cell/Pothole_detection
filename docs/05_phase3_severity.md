# 05. Phase 3: Monocular Depth & Severity

## The 3D Constraint
Civil engineering standards (like ASTM D6433) dictate that pothole severity is a function of both surface area and depth. Because municipal dashcams do not have LiDAR or stereo capabilities, we lacked ground-truth depth data.

## Depth-Anything-V2 Integration
We integrated `Depth-Anything-V2` into a parallel branch of the pipeline. While it cannot provide absolute metric depth (e.g., "5 centimeters"), it provides a robust continuous relative depth map.

By overlaying the Phase 2 segmentation mask onto the DA-V2 depth map, we extract a statistical feature vector (e.g., $95^{th}$ percentile depth delta between the mask interior and the surrounding padded road surface).

## Heuristic Engine
The combined feature vector (Mask Area + Relative Depth Delta) is fed into a rules-based heuristic engine that aligns with adapted municipal standards, outputting a final classification of:
- `LOW`
- `MEDIUM`
- `HIGH`
