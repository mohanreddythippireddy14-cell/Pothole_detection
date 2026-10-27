"""Phase 1, step 2: convert the VOC-XML pothole dataset into YOLO format.

Ported from the Colab notebook's VOC->YOLO cells: match image/xml pairs by stem,
80/10/10 train/val/test split (seed=42), convert bboxes to normalized YOLO txt,
write data.yaml. Single class: pothole (id 0).
"""

import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import KAGGLEHUB_CACHE, YOLO_DATASET_DIR, ensure_dirs  # noqa: E402

os.environ.setdefault("KAGGLEHUB_CACHE", str(KAGGLEHUB_CACHE))

import kagglehub
import yaml

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def convert_voc_to_yolo(xml_path: Path) -> tuple[list[str], int]:
    """Returns (yolo_lines, n_degenerate_boxes_skipped).

    One annotation in the source dataset (potholes58.png) has xmin==xmax (zero
    pixel width) -- a degenerate ground-truth box that would normalize to
    width=0.0 in YOLO format. Zero-area boxes contribute nothing to IoU-based
    matching and can destabilize the box-regression loss, so they're dropped
    here rather than passed through silently.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    image_width = int(size.findtext("width"))
    image_height = int(size.findtext("height"))

    yolo_annotations = []
    n_skipped = 0
    for obj in root.findall("object"):
        class_id = 0  # single class: pothole
        bbox = obj.find("bndbox")
        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        if xmax <= xmin or ymax <= ymin:
            n_skipped += 1
            continue

        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return yolo_annotations, n_skipped


def process_split(pairs, split, yolo_dir: Path) -> int:
    image_output_dir = yolo_dir / "images" / split
    label_output_dir = yolo_dir / "labels" / split
    total_skipped = 0
    for image_path, xml_path in pairs:
        shutil.copy2(image_path, image_output_dir / image_path.name)
        yolo_annotations, n_skipped = convert_voc_to_yolo(xml_path)
        total_skipped += n_skipped
        if n_skipped:
            print(f"  skipped {n_skipped} degenerate box(es) in {xml_path.name}")
        label_path = label_output_dir / f"{image_path.stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_annotations))
    return total_skipped


def main():
    ensure_dirs()
    source_dir = Path(kagglehub.dataset_download("andrewmvd/pothole-detection"))
    print("Source dataset:", source_dir)
    print("YOLO dataset target:", YOLO_DATASET_DIR)

    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)
    for split in ["train", "val", "test"]:
        (YOLO_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    annotation_files = sorted(p for p in source_dir.rglob("*.xml") if p.is_file())
    print("Images:", len(image_files), "| XML annotations:", len(annotation_files))

    image_stems = {p.stem for p in image_files}
    annotation_stems = {p.stem for p in annotation_files}
    images_without_annotations = image_stems - annotation_stems
    annotations_without_images = annotation_stems - image_stems
    print("Images without annotations:", len(images_without_annotations))
    print("Annotations without images:", len(annotations_without_images))

    annotation_map = {p.stem: p for p in annotation_files}
    pairs = [
        (img, annotation_map[img.stem])
        for img in image_files
        if img.stem in annotation_map
    ]
    print("Matched image/XML pairs:", len(pairs))

    random.seed(42)
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(0.80 * total)
    val_end = int(0.90 * total)
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}  Test: {len(test_pairs)}")

    n_skipped = 0
    n_skipped += process_split(train_pairs, "train", YOLO_DATASET_DIR)
    n_skipped += process_split(val_pairs, "val", YOLO_DATASET_DIR)
    n_skipped += process_split(test_pairs, "test", YOLO_DATASET_DIR)
    print("Dataset conversion completed. Degenerate (zero-area) boxes skipped:", n_skipped)

    for split in ["train", "val", "test"]:
        n_img = len(list((YOLO_DATASET_DIR / "images" / split).glob("*")))
        n_lbl = len(list((YOLO_DATASET_DIR / "labels" / split).glob("*.txt")))
        print(f"{split:5} | images: {n_img:3} | labels: {n_lbl:3}")

    data_yaml = {
        "path": str(YOLO_DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pothole"},
    }
    yaml_path = YOLO_DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print("Wrote", yaml_path)


if __name__ == "__main__":
    main()
