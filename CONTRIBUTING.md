# Contributing

Thanks for your interest in improving the pothole detection and severity pipeline!

## Where help is most useful
The prioritised list is in [`FUTURE_ACTIONS.md`](FUTURE_ACTIONS.md). Highlights:
1. A clean, group-aware held-out test set and re-evaluation of the detector.
2. An end-to-end severity evaluation of the full pipeline.
3. Better depth handling (proper inverse-depth treatment, recalibration on detector crops).
4. JSON output and number-plate/face blurring.

## Pull requests
1. Fork the repository and create a branch.
2. Keep ML pipeline code (`scripts/`, `src/`) separate from any serving or dashboard code.
3. Run the unit tests: `pip install -r requirements-dev.txt && pytest`.
4. If a change affects model results, include before/after metrics and say which evaluation set they come from.
