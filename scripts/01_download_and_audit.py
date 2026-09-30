"""Phase 1, step 1: download the Kaggle pothole-detection dataset and audit it.

Ported from the Colab notebook's early cells (dataset download, file-type census,
image dimension stats, VOC XML parsing, bbox statistics). Redirects the kagglehub
cache to the second drive (root disk has ~2.2GB free).
"""

import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import KAGGLEHUB_CACHE, CONTEXT_DIR  # noqa: E402

os.environ.setdefault("KAGGLEHUB_CACHE", str(KAGGLEHUB_CACHE))

import kagglehub
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_EXTENSIONS = {".txt", ".xml", ".json"}


def parse_voc_annotation(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    objects = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        bbox = obj.find("bndbox")
        objects.append({
            "class": class_name,
            "xmin": int(float(bbox.findtext("xmin"))),
            "ymin": int(float(bbox.findtext("ymin"))),
            "xmax": int(float(bbox.findtext("xmax"))),
            "ymax": int(float(bbox.findtext("ymax"))),
        })
    return {"filename": filename, "width": width, "height": height, "objects": objects}


def main():
    log_lines = []

    def log(*args):
        line = " ".join(str(a) for a in args)
        print(line)
        log_lines.append(line)

    dataset_path = Path(kagglehub.dataset_download("andrewmvd/pothole-detection"))
    log("Dataset downloaded to:", dataset_path)
    log("Exists:", dataset_path.exists(), "Is dir:", dataset_path.is_dir())

    file_extensions = Counter()
    for item in dataset_path.rglob("*"):
        if item.is_file():
            file_extensions[item.suffix.lower() or "[no extension]"] += 1
    log("\nFILE TYPE SUMMARY")
    for ext, count in file_extensions.most_common():
        log(f"{ext:15} {count}")

    image_files = sorted(
        p for p in dataset_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    log("\nTotal images:", len(image_files))

    image_info, corrupt_images = [], []
    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                image_info.append({
                    "path": image_path, "filename": image_path.name,
                    "width": img.width, "height": img.height,
                    "format": img.format, "mode": img.mode,
                })
        except Exception as e:
            corrupt_images.append({"path": image_path, "error": str(e)})

    log("Valid images:", len(image_info))
    log("Corrupt images:", len(corrupt_images))
    for c in corrupt_images:
        log("  CORRUPT:", c["path"], c["error"])

    image_df = pd.DataFrame(image_info)
    log("\nWidth stats:\n", image_df["width"].describe())
    log("\nHeight stats:\n", image_df["height"].describe())

    resolution_counts = image_df.groupby(["width", "height"]).size().sort_values(ascending=False)
    log("\nMOST COMMON RESOLUTIONS (top 20)")
    log(resolution_counts.head(20).to_string())

    annotation_files = sorted(
        p for p in dataset_path.rglob("*")
        if p.is_file() and p.suffix.lower() in ANNOTATION_EXTENSIONS
    )
    log("\nAnnotation files found:", len(annotation_files))

    annotation_records = []
    for xml_path in annotation_files:
        try:
            ann = parse_voc_annotation(xml_path)
            for obj in ann["objects"]:
                annotation_records.append({
                    "filename": ann["filename"], "image_width": ann["width"],
                    "image_height": ann["height"], "class": obj["class"],
                    "xmin": obj["xmin"], "ymin": obj["ymin"],
                    "xmax": obj["xmax"], "ymax": obj["ymax"],
                })
        except Exception as e:
            log("XML parse error:", xml_path, e)

    annotation_df = pd.DataFrame(annotation_records)
    log("\nTotal annotated objects:", len(annotation_df))
    log("\nCLASS DISTRIBUTION")
    log(annotation_df["class"].value_counts().to_string())

    annotation_df["box_width"] = annotation_df["xmax"] - annotation_df["xmin"]
    annotation_df["box_height"] = annotation_df["ymax"] - annotation_df["ymin"]
    annotation_df["box_area"] = annotation_df["box_width"] * annotation_df["box_height"]
    annotation_df["relative_width"] = annotation_df["box_width"] / annotation_df["image_width"]
    annotation_df["relative_height"] = annotation_df["box_height"] / annotation_df["image_height"]
    annotation_df["relative_area"] = annotation_df["box_area"] / (
        annotation_df["image_width"] * annotation_df["image_height"]
    )

    log("\nBOUNDING BOX STATISTICS")
    log("\nPixel width:\n", annotation_df["box_width"].describe())
    log("\nPixel height:\n", annotation_df["box_height"].describe())
    log("\nRelative width:\n", annotation_df["relative_width"].describe())
    log("\nRelative height:\n", annotation_df["relative_height"].describe())
    log("\nRelative area:\n", annotation_df["relative_area"].describe())

    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    out_log = CONTEXT_DIR / "PHASE1_STEP1_AUDIT.txt"
    out_log.write_text("\n".join(log_lines))
    log("\nWrote audit log to", out_log)

    out_csv = CONTEXT_DIR / "phase1_annotations.csv"
    annotation_df.to_csv(out_csv, index=False)
    log("Wrote annotation table to", out_csv)


if __name__ == "__main__":
    main()
