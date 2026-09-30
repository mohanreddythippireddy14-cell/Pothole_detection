# Phase 2 — Depth-Anything-V2 (pretrained) vs PothRGBD sensor depth

```
DEPTH-ANYTHING-V2 vs POTHRGBD SENSOR DEPTH
============================================================
N images evaluated (full-frame): 40
Full-frame Pearson r: mean=-0.726 std=0.346 min=-0.993 max=0.302
Full-frame RMSE (post affine-align, sensor units): mean=69.0 std=96.5
Full-frame GT depth typical scale: sensor mean~1097 (see PHASE2_STEP1 audit)

N images evaluated (pothole-mask only): 40
Mask-region Pearson r: mean=-0.729 std=0.437 min=-0.995 max=0.596
Mask-region RMSE (whole-image affine reused): mean=35.1 std=64.5

Fraction of images with NEGATIVE full-frame correlation: 0.95
Fraction of images with WEAK (|r|<0.3) full-frame correlation: 0.17
```
