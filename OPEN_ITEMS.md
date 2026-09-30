# Open Items & Future Research

While the current pipeline successfully executes 2D segmentation and pseudo-3D severity mapping, the following areas require future research:

- [ ] **Standing Water Reflections:** Depth-Anything-V2 occasionally interprets standing water inside a pothole as a flat surface, leading to false negatives in the severity engine.
- [ ] **Edge Optimization:** The inclusion of DA-V2 pushes inference times above real-time limits on edge devices (e.g., Jetson Nano). An ONNX export and quantization strategy is required.
- [ ] **PII Obfuscation:** Integrate a fast face and license plate blurring model into the pre-processing loader to ensure municipal privacy compliance.
- [ ] **Metric Depth Calibration:** Research methods to derive absolute metric depth using average lane width as a scaling anchor in the frame.
