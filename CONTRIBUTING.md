# Contributing

Thank you for your interest in contributing to the AI Road Damage Detection System! 

## Focus Areas
As outlined in our `OPEN_ITEMS.md`, we are currently prioritizing:
1. Handling standing water (which tricks the depth estimator).
2. Implementing PII blurring (faces/license plates) in the pre-processing step.
3. Edge deployment optimizations (TensorRT/ONNX exports).

## Pull Request Protocol
1. Fork the repository.
2. Ensure your code strictly separates the ML pipeline logic from the SIH dashboard routing logic.
3. Test your changes against the holdout test set to verify `mAP` has not degraded.
4. Submit a PR with a clear summary of changes and before/after metrics.
