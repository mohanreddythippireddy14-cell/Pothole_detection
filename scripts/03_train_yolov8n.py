"""Phase 1, step 3: train YOLOv8n on the converted pothole dataset.

Mirrors the Colab notebook's training cell (epochs=100, imgsz=640, batch=16,
patience=20) but writes runs to the second drive, not /content.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401 - must run before torch
from config import YOLO_DATASET_DIR, RUNS_DIR, CHECKPOINTS_DIR  # noqa: E402

from ultralytics import YOLO
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", type=str, default="yolov8n_pothole")
    args = parser.parse_args()

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: GPU not detected, training will be slow")

    data_yaml = YOLO_DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found — run scripts/02_convert_voc_to_yolo.py first"
        )

    pretrained = CHECKPOINTS_DIR / "pretrained" / "yolov8n.pt"
    model = YOLO(str(pretrained) if pretrained.exists() else "yolov8n.pt")
    print("YOLOv8n model loaded (COCO-pretrained).")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu",
        patience=args.patience,
        project=str(RUNS_DIR / "pothole_training"),
        name=args.name,
        plots=True,
        seed=42,
        deterministic=True,
    )

    run_dir = RUNS_DIR / "pothole_training" / args.name
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dest = CHECKPOINTS_DIR / f"{args.name}_best.pt"
        dest.write_bytes(best.read_bytes())
        print("Copied best checkpoint to", dest)

    print("Training complete. Run dir:", run_dir)


if __name__ == "__main__":
    main()
