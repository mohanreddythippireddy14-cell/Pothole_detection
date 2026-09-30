"""Phase 2, step 3: train YOLOv8n-seg on the PothRGBD segmentation dataset.

Pixel-level pothole mask model -- feeds the geometric-analysis stage of the Phase 2
severity pipeline (width/length/area from mask, per context/ARCHITECTURE.md).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401
from config import DATASETS_DIR, RUNS_DIR, CHECKPOINTS_DIR  # noqa: E402

from ultralytics import YOLO
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", type=str, default="yolov8n_seg_pothrgbd")
    args = parser.parse_args()

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    data_yaml = DATASETS_DIR / "pothrgbd_yolo_seg" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found — run scripts/07 first")

    pretrained = CHECKPOINTS_DIR / "pretrained" / "yolov8n-seg.pt"
    model = YOLO(str(pretrained) if pretrained.exists() else "yolov8n-seg.pt")
    print("YOLOv8n-seg model loaded (COCO-pretrained).")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu",
        patience=args.patience,
        project=str(RUNS_DIR / "pothole_seg_training"),
        name=args.name,
        plots=True,
        seed=42,
        deterministic=True,
    )

    run_dir = RUNS_DIR / "pothole_seg_training" / args.name
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dest = CHECKPOINTS_DIR / f"{args.name}_best.pt"
        dest.write_bytes(best.read_bytes())
        print("Copied best checkpoint to", dest)

    print("Training complete. Run dir:", run_dir)


if __name__ == "__main__":
    main()
