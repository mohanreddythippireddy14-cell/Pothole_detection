# 07. Failure Modes & Limitations

As a senior research artifact, it is critical to document known boundaries of the system.

## 1. Small Object Localization Tightness
While general detection is exceptional (`mAP@50` > 0.90), the pipeline struggles to draw perfectly tight boundaries around small or highly distant potholes, evidenced by the drop in `mAP@50:95` (0.592 for masks). This slight boundary fuzziness introduces marginal error into the area calculations for distant objects.

## 2. Lack of Metric Depth Calibration
The severity engine relies on *relative* depth deltas. Without a physical camera intrinsic matrix or a reference object of known scale in the frame, the system cannot output absolute metric depths (e.g., 4.2 cm). Extreme lighting changes (e.g., hard shadows or standing water) can occasionally trick the monocular depth estimator into false depth readings.
