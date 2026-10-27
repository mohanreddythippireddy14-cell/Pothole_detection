"""Phase 1, step 4: evaluate the trained YOLOv8n checkpoint on the held-out test split.

Mirrors the notebook's test-set evaluation + prediction cells. Prints P/R/mAP50/
mAP50-95 and dumps them to context/PHASE1_RESULTS.md-ready text.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401 - must run before torch
from config import YOLO_DATASET_DIR, CONTEXT_DIR, RUNS_DIR, ensure_dirs  # noqa: E402

from ultralytics import YOLO
import torch


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    model = YOLO(str(weights))
    data_yaml = YOLO_DATASET_DIR / "data.yaml"

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(RUNS_DIR / "pothole_eval"),
        name="val",
        exist_ok=True,
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    report = [
        "TEST SET PERFORMANCE",
        "=" * 50,
        f"Precision      : {precision:.4f}",
        f"Recall         : {recall:.4f}",
        f"F1             : {f1:.4f}",
        f"mAP@50         : {map50:.4f}",
        f"mAP@50:95      : {map5095:.4f}",
        f"Weights        : {weights}",
    ]
    print("\n".join(report))

    out = CONTEXT_DIR / "PHASE1_RESULTS.md"
    out.write_text("# Phase 1 — YOLOv8n Pothole Detection Results\n\n```\n" + "\n".join(report) + "\n```\n")
    print("\nWrote", out)


if __name__ == "__main__":
    main()
