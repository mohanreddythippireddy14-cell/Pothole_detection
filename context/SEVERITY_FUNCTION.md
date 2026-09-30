# Severity function — design, literature, and calibration

Prompted by the user's instruction: "check the severity predictions, I don't think we've
a dataset for severity, choose a best fitting function, refer research papers." This doc
records what was found and how the function was built, so the choice isn't opaque.

## What we didn't have (before this pass)

`context/PHASE2_RESULTS.md`'s original severity function was **area-only** — depth was
computed but deliberately excluded from the decision (documented reason: no metric
calibration at inference time). Thresholds (0.01 / 0.05) were picked from Phase 1's own
box-relative-area quartiles — not validated against any actual severity judgment, human
or otherwise. There was no severity-labeled dataset in the project at all.

## Literature found

**ASTM D6433** (Standard Practice for Roads and Parking Lots Pavement Condition Index
Surveys) — the real pavement-engineering standard used in PCI surveys — classifies
pothole severity from a **diameter × depth matrix** (Table X1.1), not diameter alone:

| Diameter | Depth 13-25mm | Depth 25-50mm | Depth >50mm |
|---|---|---|---|
| 100-200mm | Low | Low | Medium |
| 200-450mm | Low | Medium | High |
| 450-750mm | Low | Medium | High |

The structural insight: **for a fixed diameter, a deeper pothole is rated more severe.**
This is the real-world engineering logic our function should reproduce, even without
metric measurements.
([source](https://www.geocadra.com/en/standards/astm-d6433-pci) via web search, Aug 2026)

**Computer-vision literature** confirms detection + Depth-Anything-V2 + area estimation
is an actively-used, published pattern for exactly this problem — e.g. "Robust
Video-Based Pothole Detection and Area Estimation for Intelligent Vehicles with Depth
Map and Kalman Smoothing" (arXiv:2505.21049, 2025) uses YOLOv8 + Depth Anything V2 + a
custom area-estimation method (their "MBTP" — Minimum Bounding Triangulated Pixel) for
the same detect→depth→area pipeline shape this project already had. This validates the
architecture direction (`context/ARCHITECTURE.md`'s Phase 2 diagram), not a specific
formula — the paper's exact math isn't public in the abstract.
([arXiv:2505.21049](https://arxiv.org/abs/2505.21049))

## The dataset we didn't have — found one

`idanbaru/annotated-potholes-with-severity-levels` (Kaggle): 717 images, 1886
bounding boxes, each hand-labeled `minor_pothole` / `medium_pothole` / `major_pothole`.
**This is the same underlying photo corpus as chitholian/andrewmvd** (confirmed in the
dataset's own README: "enhanced version of chitholian/annotated-potholes-dataset"), plus
52 new images. Important caveat, stated by the dataset's own author: *"the severity
classification for the initial 665 images was performed using a custom Python script"* —
only the 52 new images and corrections used manual labelImg review. So this is not pure
human ground truth throughout, but it's the best available proxy, and it's the first
severity-labeled data this project has had.

## Calibration (`scripts/14_calibrate_severity.py`)

**Area factor**: grid-searched threshold pairs on `relative_area` (box area / image
area) against idanbaru's 3-class labels, optimizing balanced accuracy (not raw accuracy
— raw accuracy degenerates to "always predict medium" under this class imbalance,
868/600/418 medium/major/minor).

```
best thresholds: LOW/MEDIUM boundary = 0.0208, MEDIUM/HIGH boundary = 0.1030
balanced accuracy: 0.525   (33% = chance for 3 balanced classes)
overall accuracy:  0.479
per-class recall:  minor=0.80, medium=0.39, major=0.39
```

**Read this honestly**: area alone explains real severity judgments only moderately
well. This is the direct, measured justification for why depth needed to become a real
second input rather than staying diagnostic-only — a single scalar (area) is
demonstrably insufficient, exactly matching why ASTM's real standard uses two factors
instead of one.

**Depth factor**: since we lack metric depth at inference time (same limitation as
before — Depth-Anything-V2's output is scale-ambiguous without ground truth to fit
against), the depth signal is expressed as a **steepness ratio**: `depth_p95 / pixel
extent of the pothole` — analogous to a depth-to-diameter ratio, but on
Depth-Anything-V2's own uncalibrated relative scale rather than real mm. The escalation
threshold (fire when a pothole's steepness exceeds this, bumping severity up one level)
is the **90th percentile** of this ratio measured across 59 PothRGBD images (real
close-up pothole photos with genuine sensor depth, though the ratio itself uses
DA-V2's predicted depth, not the sensor depth, to stay on the same scale used at
inference):

```
steepness reference (PothRGBD, n=59): median=0.00242, p75=0.00297, p90=0.00381 (escalation threshold)
```

## The function (`scripts/10_full_pipeline.py`)

```
area_score  = 0 (LOW) if relative_area < 0.0208
            = 1 (MEDIUM) if < 0.1030
            = 2 (HIGH) otherwise
if steepness > 0.00381:
    area_score = min(area_score + 1, 2)   # depth can push severity up, never down or past HIGH
severity = [LOW, MEDIUM, HIGH][area_score]
```

Verified working on the small pipeline test (`context/phase2_v2_pipeline_test/`):
example `potholes642.png` — a small-area (0.0095, area-only says LOW) but visibly deep
hole — correctly escalated to MEDIUM by the depth factor, matching the ASTM-style logic
this was designed to reproduce.

## Honest limitations, stated plainly

1. **Area-only calibration accuracy is moderate (~48-52%)**, and the idanbaru labels
   themselves are partly script-derived, not fully independent human judgment — so this
   calibration is better than nothing but should not be read as "validated against
   ground truth" in the strict sense.
2. **Depth escalation is a relative, uncalibrated-scale heuristic**, not a metric
   depth/diameter ratio in real units. It reproduces ASTM's structural logic (depth can
   raise severity) without reproducing its literal mm thresholds, because no camera
   calibration is available for arbitrary uploaded photos.
3. **No end-to-end severity accuracy number exists for the full pipeline** (detection →
   segmentation → geometry → depth → severity) against idanbaru or any other labeled
   set — only the area-factor-in-isolation accuracy was measured. Getting a true
   pipeline-level severity accuracy would require running full detection+segmentation on
   idanbaru's images and comparing predicted vs. labeled severity per detected pothole,
   which was not done in this pass (flagged as a natural next step, not silently skipped).
