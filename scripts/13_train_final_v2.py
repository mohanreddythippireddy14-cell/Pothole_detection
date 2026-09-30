"""Phase 1 v2: train the final production model on a 90/10 split of the full merged
mega-dataset (1986 images), after the 5-fold CV (scripts/12) validated that the
combined-dataset + close_mosaic fix genuinely improves over v1 (see
context/PHASE1_V2_KFOLD_RESULTS.md). Standard practice: k-fold estimates generalization,
then the deployed model trains on as much data as possible.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401
from config import DATASETS_DIR, RUNS_DIR, CHECKPOINTS_DIR  # noqa: E402

import numpy as np
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO
import torch

COMBINED_DIR = DATASETS_DIR / "pothole_combined_v2"
FINAL_DIR = DATASETS_DIR / "pothole_combined_v2_final90_10"


def main():
    manifest = np.array((COMBINED_DIR / "manifest.txt").read_text().splitlines())
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(manifest))  # first split only = 90/10

    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)
    for split, idx in [("train", train_idx), ("val", val_idx)]:
        (FINAL_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (FINAL_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem in manifest[idx]:
            img_src = next((COMBINED_DIR / "images").glob(f"{stem}.*"))
            lbl_src = COMBINED_DIR / "labels" / f"{stem}.txt"
            (FINAL_DIR / "images" / split / img_src.name).symlink_to(img_src)
            (FINAL_DIR / "labels" / split / lbl_src.name).symlink_to(lbl_src)
    print(f"train={len(train_idx)} val={len(val_idx)}")

    data_yaml = FINAL_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump({"path": str(FINAL_DIR), "train": "images/train", "val": "images/val",
                   "names": {0: "pothole"}}, f, sort_keys=False)

    pretrained = CHECKPOINTS_DIR / "pretrained" / "yolov8n.pt"
    model = YOLO(str(pretrained) if pretrained.exists() else "yolov8n.pt")
    model.train(
        data=str(data_yaml), epochs=60, imgsz=640, batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        patience=25, close_mosaic=15,
        project=str(RUNS_DIR / "pothole_v2_final"), name="final90_10",
        plots=True, seed=42, deterministic=True,
    )

    run_dir = RUNS_DIR / "pothole_v2_final" / "final90_10"
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dest = CHECKPOINTS_DIR / "yolov8n_pothole_v2_final_best.pt"
        dest.write_bytes(best.read_bytes())
        print("Copied best checkpoint to", dest)
    print("Done. Run dir:", run_dir)


if __name__ == "__main__":
    main()
