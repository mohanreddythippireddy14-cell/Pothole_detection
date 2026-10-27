# 07. Failure Modes & Limitations

1. **Missed potholes.** Detector recall is about 0.64 (5-fold CV); small and distant potholes are missed most often.
2. **Loose boundaries.** mAP@50:95 is 0.43 for detection and 0.59 for masks, so area estimates carry noticeable error, especially for distant potholes.
3. **Segmentation outside its training domain.** The segmenter learned from two PothRGBD recording sessions; on other scenes it can return no mask, which the pipeline reports as `UNKNOWN`.
4. **No metric depth.** Depth-Anything-V2 gives relative inverse depth. Without camera calibration, the system cannot output real depths (e.g. "4 cm"), and the depth factor is a heuristic.
5. **Approximate severity calibration.** The area thresholds were fitted on box area and applied to mask area, with in-sample accuracy. The depth threshold comes from a different setup than inference.
6. **Evaluation gaps.** There is no untouched v2 test set, and no end-to-end severity accuracy has been measured.

Planned fixes for each: `../FUTURE_ACTIONS.md`.
