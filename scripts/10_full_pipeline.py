"""Phase 2, step 5: the integrated severity pipeline.

Input Image -> YOLOv8n (Phase 1) -> detections -> bounding boxes
  -> padded crop -> YOLOv8n-seg (Phase 2) -> pixel mask
  -> [geometry: width/length/area from mask] + [Depth-Anything-V2 -> depth map -> masked stats]
  -> feature extraction -> severity function -> final annotated visualization

Matches the "Final Architecture" diagram in the handwritten notes (context/ARCHITECTURE.md).
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401
from config import CHECKPOINTS_DIR, CONTEXT_DIR, ensure_dirs  # noqa: E402

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DET_WEIGHTS = CHECKPOINTS_DIR / "yolov8n_pothole_v2_final_best.pt"
SEG_WEIGHTS = CHECKPOINTS_DIR / "yolov8n_seg_pothrgbd_best.pt"

# Two-factor severity, matching the STRUCTURE of ASTM D6433's real pothole-severity
# matrix (Table X1.1: severity = f(diameter, depth) -- a deep pothole outranks a
# shallow one of the same size). We can't reproduce the standard's literal mm bins (no
# camera calibration at inference time -> no metric diameter/depth), so both factors
# use scale-independent proxies, each calibrated against real data:
#   - area factor: thresholds grid-searched for best balanced accuracy against
#     idanbaru/annotated-potholes-with-severity-levels (717 images, 1886
#     minor/medium/major-labeled boxes). Area-only reaches only ~52% balanced accuracy
#     (33% = chance for 3 classes) -- informative but far from sufficient alone, which
#     is why depth is now a real second factor rather than diagnostic-only.
#     NOTE: those thresholds were fitted on BOUNDING-BOX area (the only geometry the
#     idanbaru labels provide) but are applied below to segmentation-MASK area, which
#     is smaller than the box for the same pothole -- so area scores lean low. Known
#     limitation, tracked in FUTURE_ACTIONS.md.
#   - depth factor: a "steepness" ratio (depth-vs-road-plane signal / pothole pixel
#     extent) on Depth-Anything-V2's own uncalibrated relative-depth scale, with an
#     escalation threshold (90th percentile) calibrated from a PothRGBD sample. If a
#     detected pothole's steepness exceeds this, severity is bumped up one level --
#     the same "depth pushes severity up for a given size" logic ASTM uses, without
#     pretending we have metric depth. See context/SEVERITY_FUNCTION.md.
_calib_path = CONTEXT_DIR / "severity_calibration.json"
if _calib_path.exists():
    _CALIBRATION = json.loads(_calib_path.read_text())
else:
    warnings.warn(
        f"{_calib_path} not found -- falling back to uncalibrated area thresholds and "
        "DISABLING depth escalation. Run scripts/14_calibrate_severity.py to create it.",
        stacklevel=1,
    )
    _CALIBRATION = {
        "area_threshold_low_medium": 0.01, "area_threshold_medium_high": 0.05,
        "depth_steepness_escalation_threshold": float("inf"),
    }
SEVERITY_AREA_THRESHOLDS = (
    _CALIBRATION["area_threshold_low_medium"], _CALIBRATION["area_threshold_medium_high"]
)
DEPTH_STEEPNESS_ESCALATION_THRESHOLD = _CALIBRATION["depth_steepness_escalation_threshold"]


class SeverityPipeline:
    def __init__(self, det_weights=DET_WEIGHTS, seg_weights=SEG_WEIGHTS, use_depth=True, device=None):
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.detector = YOLO(str(det_weights))
        self.segmenter = YOLO(str(seg_weights))
        self.use_depth = use_depth
        self.depth_pipe = None
        if use_depth:
            from transformers import pipeline
            self.depth_pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=self.device if isinstance(self.device, int) else -1,
            )

    def _geometry_from_mask(self, mask: np.ndarray, full_image_shape: tuple[int, int]) -> dict:
        """relative_area is normalized against the FULL image, not the padded crop the
        segmenter saw -- normalizing against the crop would make relative_area nearly
        constant (~0.4-0.9) regardless of how big the pothole actually is relative to the
        road, since the crop is deliberately sized to snugly contain the detection box."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return {"width_px": 0, "height_px": 0, "area_px": 0, "relative_area": 0.0, "mask_found": False}
        width_px = int(xs.max() - xs.min() + 1)
        height_px = int(ys.max() - ys.min() + 1)
        area_px = int(mask.sum())
        full_h, full_w = full_image_shape
        relative_area = area_px / (full_h * full_w)
        return {"width_px": width_px, "height_px": height_px, "area_px": area_px,
                "relative_area": relative_area, "mask_found": True}

    def _depth_stats(self, image_rgb: np.ndarray, mask: np.ndarray) -> dict:
        """Relative-depth-inside-pothole-vs-surrounding-road-plane, in DA-V2's own units.
        The road level is the median of a ring of pixels just outside the mask (no
        affine fit -- there is no ground-truth depth at inference time on new images)."""
        from PIL import Image as PILImage

        result = self.depth_pipe(PILImage.fromarray(image_rgb))
        pred = np.array(result["predicted_depth"].squeeze().cpu())
        if pred.shape != mask.shape:
            pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        # DA-V2 output is inverse-relative-depth (near=high). Flip sign so "higher = farther",
        # matching physical depth intuition, then reference against a ring of pixels just
        # outside the mask (proxy for the surrounding road surface) since we have no metric GT.
        inv_depth = -pred
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8))
        ring = (dilated > 0) & (mask == 0)
        if ring.sum() < 20:
            ring = mask == 0  # fallback: whole non-mask region

        road_level = float(np.median(inv_depth[ring]))
        pothole_depth = inv_depth[mask > 0] - road_level  # positive = deeper than road
        depth_p95 = float(np.percentile(pothole_depth, 95))

        ys, xs = np.where(mask > 0)
        pixel_extent = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        steepness = depth_p95 / pixel_extent if pixel_extent > 0 else 0.0

        return {
            "depth_mean": float(np.mean(pothole_depth)),
            "depth_p95": depth_p95,
            "depth_max": float(np.max(pothole_depth)),
            "steepness": steepness,
        }

    @staticmethod
    def _severity_from_features(mask_found: bool, relative_area: float, steepness: float | None) -> str:
        if not mask_found:
            return "UNKNOWN"  # segmentation failed on this crop -- not the same as LOW
        area_lo, area_hi = SEVERITY_AREA_THRESHOLDS
        area_score = 0 if relative_area < area_lo else (1 if relative_area < area_hi else 2)

        # depth escalation: an unusually steep (deep-for-its-size) pothole bumps severity
        # up one level, same structural logic as ASTM D6433's diameter x depth matrix
        if steepness is not None and steepness > DEPTH_STEEPNESS_ESCALATION_THRESHOLD:
            area_score = min(area_score + 1, 2)

        return ["LOW", "MEDIUM", "HIGH"][area_score]

    def run(self, image_bgr: np.ndarray, conf=0.25, pad_frac=0.15) -> list[dict]:
        h, w = image_bgr.shape[:2]
        det_result = self.detector.predict(image_bgr, conf=conf, device=self.device, verbose=False)[0]

        detections = []
        for box in det_result.boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
            det_conf = float(box.conf[0].cpu())

            bw, bh = xmax - xmin, ymax - ymin
            pad_x, pad_y = bw * pad_frac, bh * pad_frac
            cxmin = max(0, int(xmin - pad_x))
            cymin = max(0, int(ymin - pad_y))
            cxmax = min(w, int(xmax + pad_x))
            cymax = min(h, int(ymax + pad_y))
            crop = image_bgr[cymin:cymax, cxmin:cxmax]
            if crop.size == 0:
                continue

            # retina_masks=True makes Ultralytics return masks at the crop's own resolution;
            # without it masks.data is at the letterboxed inference size (incl. stride
            # padding), and resizing that straight to the crop shifts the mask slightly.
            seg_result = self.segmenter.predict(crop, conf=0.25, device=self.device, verbose=False,
                                                retina_masks=True)[0]
            if seg_result.masks is None or len(seg_result.masks.data) == 0:
                mask_crop = np.zeros(crop.shape[:2], dtype=np.uint8)
            else:
                areas = [m.sum().item() for m in seg_result.masks.data]
                best_idx = int(np.argmax(areas))
                mask_crop = (seg_result.masks.data[best_idx].cpu().numpy() > 0.5).astype(np.uint8)
                if mask_crop.shape != crop.shape[:2]:  # defensive; retina masks should match
                    mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

            geometry = self._geometry_from_mask(mask_crop, image_bgr.shape[:2])

            depth_stats = None
            if self.use_depth and mask_crop.sum() > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                depth_stats = self._depth_stats(crop_rgb, mask_crop)

            steepness = depth_stats["steepness"] if depth_stats else None
            severity = self._severity_from_features(geometry["mask_found"], geometry["relative_area"], steepness)
            escalated = (
                steepness is not None and steepness > DEPTH_STEEPNESS_ESCALATION_THRESHOLD
                and geometry["mask_found"]
            )

            detections.append({
                "box": (int(xmin), int(ymin), int(xmax), int(ymax)),
                "crop_box": (cxmin, cymin, cxmax, cymax),
                "det_confidence": det_conf,
                "mask_crop": mask_crop,
                "geometry": geometry,
                "depth": depth_stats,
                "severity": severity,
                "depth_escalated": escalated,
            })

        return detections

    def annotate(self, image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Box-only visualization, colored by severity. The segmentation mask is used
        internally (geometry/severity scoring) but intentionally not drawn -- it's a
        modeling aid, not something the end viewer needs to see."""
        colors = {"LOW": (0, 200, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255), "UNKNOWN": (128, 128, 128)}
        out = image_bgr.copy()
        for det in detections:
            xmin, ymin, xmax, ymax = det["box"]
            color = colors[det["severity"]]
            cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 3)

            escalated_tag = " (depth+)" if det.get("depth_escalated") else ""
            label = f"{det['severity']}{escalated_tag} conf={det['det_confidence']:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_y = ymin - 8 if ymin - 8 - th > 0 else ymin + th + 8
            label_x = min(xmin, out.shape[1] - tw - 2)  # keep the label inside the frame
            cv2.rectangle(out, (label_x, label_y - th - baseline), (label_x + tw, label_y + baseline), color, -1)
            cv2.putText(out, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return out


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, default=str(CONTEXT_DIR / "phase2_pipeline_demo"))
    parser.add_argument("--no-depth", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SeverityPipeline(use_depth=not args.no_depth)

    report_lines = ["# Phase 2 pipeline demo — per-image results\n"]
    for img_path in args.images:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print("Could not read", img_path)
            continue
        detections = pipeline.run(image_bgr)
        annotated = pipeline.annotate(image_bgr, detections)

        out_path = out_dir / f"{Path(img_path).stem}_annotated.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"{img_path}: {len(detections)} pothole(s) -> {out_path}")

        report_lines.append(f"\n## {Path(img_path).name}\n")
        for i, det in enumerate(detections):
            depth_str = f"{det['depth']['depth_p95']:.1f}" if det["depth"] else "n/a"
            steep_str = f"{det['depth']['steepness']:.5f}" if det["depth"] else "n/a"
            esc_str = " [DEPTH-ESCALATED]" if det.get("depth_escalated") else ""
            report_lines.append(
                f"- pothole {i}: severity={det['severity']}{esc_str}, det_conf={det['det_confidence']:.3f}, "
                f"relative_area={det['geometry']['relative_area']:.4f}, "
                f"box_px={det['geometry']['width_px']}x{det['geometry']['height_px']}, "
                f"depth_p95={depth_str}, steepness={steep_str}"
            )

    (out_dir / "REPORT.md").write_text("\n".join(report_lines))
    print("\nWrote", out_dir / "REPORT.md")


if __name__ == "__main__":
    main()
