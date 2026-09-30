"""Phase 2 v2: calibrate the two-factor (area + depth) severity function.

Design rationale (see context/SEVERITY_FUNCTION.md for the full writeup):
  - ASTM D6433 (the real pavement-engineering standard for pothole severity, Table
    X1.1) classifies severity from a DIAMETER x DEPTH matrix -- for a given diameter
    bin, a deeper pothole is a HIGHER severity. We don't have metric (real-mm)
    diameter or depth available at inference time on an arbitrary uploaded photo (no
    camera calibration), so we can't reproduce the ASTM table's literal mm bins -- but
    we CAN reproduce its two-factor STRUCTURE using scale-independent proxies:
      * area factor:  relative_area (mask_pixels / full_image_pixels), thresholds
        calibrated empirically against idanbaru/annotated-potholes-with-severity-levels
        (717 images, human/semi-automated minor/medium/major labels -- a real, if
        imperfect, severity-labeled dataset; see calibration below).
      * depth factor: a "steepness" ratio (depth signal / pothole pixel extent),
        analogous to ASTM's depth-for-a-given-diameter logic, computed on
        Depth-Anything-V2's own output scale (uncalibrated relative units) so it's
        usable at inference without requiring metric depth.
  - The area-only classifier's accuracy against idanbaru's real labels is measured
    here too (~48-52% balanced accuracy across 3 classes, vs 33% chance) -- this
    quantifies WHY depth needs to be a genuine second factor, not just diagnostic
    output as it was before this pass.

This script:
  1. Grid-searches area thresholds (t1, t2) maximizing balanced accuracy against
     idanbaru's minor/medium/major labels.
  2. Runs Depth-Anything-V2 over a PothRGBD sample to get a reference distribution of
     the depth-steepness ratio, and picks a high-percentile escalation threshold from
     it (used to bump severity up one level when a pothole is unusually deep for its
     size, on DA-V2's own uncalibrated scale).
  3. Writes both to context/severity_calibration.json for scripts/10 to load.
"""

import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401
from config import KAGGLEHUB_CACHE, CONTEXT_DIR  # noqa: E402

os.environ.setdefault("KAGGLEHUB_CACHE", str(KAGGLEHUB_CACHE))

import cv2
import kagglehub
import numpy as np
import pandas as pd
from PIL import Image


def calibrate_area_thresholds():
    root = Path(kagglehub.dataset_download("idanbaru/annotated-potholes-with-severity-levels"))
    rows = []
    for xml_path in sorted((root / "annotations").glob("*.xml")):
        tree = ET.parse(xml_path)
        r = tree.getroot()
        size = r.find("size")
        w, h = int(size.findtext("width")), int(size.findtext("height"))
        for obj in r.findall("object"):
            name = obj.findtext("name")
            bbox = obj.find("bndbox")
            xmin, ymin = float(bbox.findtext("xmin")), float(bbox.findtext("ymin"))
            xmax, ymax = float(bbox.findtext("xmax")), float(bbox.findtext("ymax"))
            if xmax <= xmin or ymax <= ymin:
                continue
            bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
            rows.append({"severity": name, "relative_area": bw * bh})

    df = pd.DataFrame(rows)
    df.to_csv(CONTEXT_DIR / "idanbaru_severity_geometry.csv", index=False)

    order = {"minor_pothole": 0, "medium_pothole": 1, "major_pothole": 2}
    y_true = df["severity"].map(order).values
    area = df["relative_area"].values

    best = None
    candidates = np.geomspace(0.001, 0.5, 300)
    for t1 in candidates:
        for t2 in candidates:
            if t2 <= t1:
                continue
            pred = np.where(area < t1, 0, np.where(area < t2, 1, 2))
            per_class_recall = [np.mean(pred[y_true == c] == c) for c in [0, 1, 2]]
            balanced_acc = float(np.mean(per_class_recall))
            if best is None or balanced_acc > best[0]:
                best = (balanced_acc, float(t1), float(t2), per_class_recall)

    bacc, t1, t2, recalls = best
    pred = np.where(area < t1, 0, np.where(area < t2, 1, 2))
    overall_acc = float((pred == y_true).mean())
    print(f"Area thresholds: t1={t1:.5f} t2={t2:.5f} | balanced_acc={bacc:.4f} "
          f"overall_acc={overall_acc:.4f} | per-class recall (minor/med/major)={recalls}")
    return {
        "area_threshold_low_medium": t1,
        "area_threshold_medium_high": t2,
        "area_calibration_balanced_accuracy": bacc,
        "area_calibration_overall_accuracy": overall_acc,
        "area_calibration_n_boxes": len(df),
        "area_calibration_source": "idanbaru/annotated-potholes-with-severity-levels",
    }


def calibrate_depth_steepness(n_samples=60, seed=1):
    from transformers import pipeline

    import torch
    device = 0 if torch.cuda.is_available() else -1
    depth_pipe = pipeline(task="depth-estimation",
                           model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

    root = Path(kagglehub.dataset_download("mahyeks/pothrgbd-rgb-and-depth-images-of-potholes"))
    images = sorted(root.rglob("*.jpg"))
    txts = {p.stem: p for p in root.rglob("*.txt")}

    random.seed(seed)
    sample = random.sample(images, min(n_samples, len(images)))

    steepness_values = []
    for img_path in sample:
        txt_path = txts.get(img_path.stem)
        if txt_path is None:
            continue
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        for line in txt_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = list(map(float, line.split()[1:]))
            pts = np.array([[parts[i] * w, parts[i + 1] * h] for i in range(0, len(parts), 2)], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        if mask.sum() < 30:
            continue

        result = depth_pipe(image)
        pred = np.array(result["predicted_depth"].squeeze().cpu())
        if pred.shape != mask.shape:
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

        inv_depth = -pred
        dilated = cv2.dilate(mask, np.ones((25, 25), np.uint8))
        ring = (dilated > 0) & (mask == 0)
        if ring.sum() < 20:
            ring = mask == 0
        road_level = float(np.median(inv_depth[ring]))
        pothole_depth = inv_depth[mask > 0] - road_level
        depth_p95 = float(np.percentile(pothole_depth, 95))

        ys, xs = np.where(mask > 0)
        pixel_extent = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        steepness = depth_p95 / pixel_extent
        steepness_values.append(steepness)

    steepness_values = np.array(steepness_values)
    p75 = float(np.percentile(steepness_values, 75))
    p90 = float(np.percentile(steepness_values, 90))
    print(f"Depth steepness (DA-V2 relative units / px) over {len(steepness_values)} PothRGBD samples: "
          f"median={np.median(steepness_values):.5f} p75={p75:.5f} p90={p90:.5f}")
    return {
        "depth_steepness_escalation_threshold": p90,
        "depth_steepness_p75_reference": p75,
        "depth_steepness_median_reference": float(np.median(steepness_values)),
        "depth_calibration_n_samples": len(steepness_values),
        "depth_calibration_source": "mahyeks/pothrgbd-rgb-and-depth-images-of-potholes",
    }


def main():
    area_cal = calibrate_area_thresholds()
    depth_cal = calibrate_depth_steepness()
    calibration = {**area_cal, **depth_cal}
    out_path = CONTEXT_DIR / "severity_calibration.json"
    out_path.write_text(json.dumps(calibration, indent=2))
    print("\nWrote", out_path)
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
