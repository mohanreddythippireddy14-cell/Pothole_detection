"""Phase 2, step 2: build a YOLOv8-seg training set from PothRGBD.

PothRGBD's .txt annotations are ALREADY in YOLO-segmentation format (verified by
inspection: `class_id x1 y1 x2 y2 ... xn yn`, all coordinates normalized 0-1, single
class 0 = pothole, one polygon line per image in the sampled files) -- no conversion
needed, just split + reorganize into images/labels train/val/test directories,
mirroring scripts/02's 80/10/10 split (seed=42) for consistency with Phase 1.

Depth .npy files are NOT copied here -- Phase 2's depth work (scripts/08+) reads them
directly from the kagglehub cache by matching timestamp, no need to duplicate ~1000
uint16 480x640 arrays (~600MB) into the YOLO dataset tree.
"""

import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401
from config import KAGGLEHUB_CACHE, DATASETS_DIR, ensure_dirs  # noqa: E402

os.environ.setdefault("KAGGLEHUB_CACHE", str(KAGGLEHUB_CACHE))

import kagglehub
import yaml

POTHRGBD_YOLO_SEG_DIR = DATASETS_DIR / "pothrgbd_yolo_seg"


def main():
    ensure_dirs()
    root = Path(kagglehub.dataset_download("mahyeks/pothrgbd-rgb-and-depth-images-of-potholes"))
    print("Source:", root)

    images = sorted(p for p in root.rglob("*.jpg"))
    txts = {p.stem: p for p in root.rglob("*.txt")}
    pairs = [(img, txts[img.stem]) for img in images if img.stem in txts]
    print("Matched image/label pairs:", len(pairs))

    # sanity check: every label really is YOLO-seg format (class + >=6 coords, even count)
    bad = 0
    for _, label_path in pairs:
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            n_coords = len(parts) - 1
            if n_coords < 6 or n_coords % 2 != 0:
                bad += 1
    print("Labels with malformed polygon lines:", bad)

    if POTHRGBD_YOLO_SEG_DIR.exists():
        shutil.rmtree(POTHRGBD_YOLO_SEG_DIR)
    for split in ["train", "val", "test"]:
        (POTHRGBD_YOLO_SEG_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (POTHRGBD_YOLO_SEG_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(0.80 * total)
    val_end = int(0.90 * total)
    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    for split, split_pairs in splits.items():
        img_dir = POTHRGBD_YOLO_SEG_DIR / "images" / split
        lbl_dir = POTHRGBD_YOLO_SEG_DIR / "labels" / split
        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(label_path, lbl_dir / f"{img_path.stem}.txt")
        print(f"{split:5} | images: {len(split_pairs):4}")

    data_yaml = {
        "path": str(POTHRGBD_YOLO_SEG_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pothole"},
    }
    yaml_path = POTHRGBD_YOLO_SEG_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print("Wrote", yaml_path)


if __name__ == "__main__":
    main()
