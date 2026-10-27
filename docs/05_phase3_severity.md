# 05. Depth & the Severity Engine

## Why rules, not a model
There is no ground-truth severity for arbitrary road photos, so severity cannot simply be learned. Instead, the pipeline uses **transparent decision rules** that reproduce the structure of **ASTM D6433** (Table X1.1): severity depends on diameter *and* depth, and a deeper pothole of the same size is rated more severe.

## Depth signal
Depth-Anything-V2-Small runs on each padded crop. Its output is *relative inverse depth* (near = high), so the pipeline negates it. It takes the median of a ring of pixels just outside the mask as the road level, and measures the 95th-percentile depth inside the mask relative to that level. Dividing by the pothole's pixel extent gives **steepness**, a scale-free depth-to-size ratio.

## The rules (`scripts/10_full_pipeline.py`)
```text
area_score = LOW if relative_area < 0.0208, MEDIUM if < 0.1030, else HIGH
if steepness > 0.00381: raise area_score by one level (max HIGH)
no mask -> UNKNOWN
```
- **Area thresholds:** grid-searched for balanced accuracy (52.5%, chance 33%) on 1,886 idanbaru minor/medium/major boxes (`scripts/14`).
- **Depth threshold:** the 90th percentile of steepness over 59 PothRGBD photos.
- Values live in `context/severity_calibration.json`. Design notes: `context/SEVERITY_FUNCTION.md`.

## Known approximations
- Thresholds were fitted on **box** area but are applied to **mask** area, so area scores lean low.
- Calibration accuracy is in-sample.
- Negated inverse depth is only a rough stand-in for depth, and the depth threshold was set on full images with true masks.

All of these are tracked in `FUTURE_ACTIONS.md`.
