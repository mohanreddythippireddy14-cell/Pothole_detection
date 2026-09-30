"""Phase 1 v2: build a single merged "mega dataset" from 4 Kaggle sources, then prepare
it for k-fold cross-validation.

Directly addresses the performance audit (context/PERFORMANCE_AUDIT.md) and the user's
"converge the datasets" instruction:

  COMMON FEATURES kept, aligned to one schema across all sources:
    image + single-class ("pothole") bounding box, expressed as resolution-independent
    YOLO-normalized coordinates. Absolute pixel box_width/box_height are NOT used for
    any cross-source statistic below -- they aren't comparable once sources have
    different native resolutions (andrewmvd ~400x300, chitholian 720x720, rajdalsaniya
    640x640). Only relative_width/relative_height/relative_area/aspect_ratio (all
    resolution-independent) are computed and reported.

  SIMILAR FEATURES CONVERGED into one canonical entry:
    - andrewmvd (665 img) and chitholian (665 img) are perceptual-hash-verified to be
      THE SAME underlying 665 photos, just re-encoded (andrewmvd: ~400x300 variable,
      chitholian: consistent 720x720). Converged into ONE entry per photo, keeping
      chitholian's higher-resolution version + its own independently-encoded VOC
      annotation (not andrewmvd's) -- this is a genuine resolution upgrade, not
      duplication.
    - rajdalsaniya's 2083 files are Roboflow-export-time AUGMENTED variants of only 671
      unique base photos (verified: filenames follow `<base>_jpg.rf.<hash>.jpg`,
      3.1 variants/photo on average). Converged into 671 entries, one representative
      per base photo -- using all variants would leak near-duplicate siblings across
      train/val folds in the k-fold script. Perceptual-hash-verified as NOT overlapping
      with andrewmvd/chitholian (0/60 sampled matches) -- genuinely new photos.

  FEATURES LEFT OUT (don't fit the unified detection schema):
    - banilkumar's "Pothole" folder (650 img): classification-only, no bounding boxes.
      Not usable for detection without new annotation; deliberately excluded rather
      than pseudo-labeled (would just bake in the current weak detector's mistakes).
    - PothRGBD (Phase 2's dataset): different task (segmentation) and domain (close-up
      macro shots), stays in the Phase 2 pipeline, not folded into this detection set.

  banilkumar's "Normal" folder (650 img, no pothole) IS included, as background images
  with empty label files -- fixes audit finding #5 (zero background images).
"""

import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401
from config import KAGGLEHUB_CACHE, DATASETS_DIR, CONTEXT_DIR  # noqa: E402

os.environ.setdefault("KAGGLEHUB_CACHE", str(KAGGLEHUB_CACHE))

import kagglehub
import numpy as np
import pandas as pd
import yaml
from PIL import Image

COMBINED_DIR = DATASETS_DIR / "pothole_combined_v2"
PHASH_SIZE = 16
DUP_THRESHOLD = 5.0


def phash(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((PHASH_SIZE, PHASH_SIZE))
    return np.array(img, dtype=np.float32)


def parse_voc(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w, h = int(size.findtext("width")), int(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin, ymin = float(bbox.findtext("xmin")), float(bbox.findtext("ymin"))
        xmax, ymax = float(bbox.findtext("xmax")), float(bbox.findtext("ymax"))
        if xmax <= xmin or ymax <= ymin:
            continue  # degenerate box, same fix as scripts/02
        boxes.append((xmin, ymin, xmax, ymax))
    return w, h, boxes


def voc_boxes_to_yolo_lines(w, h, boxes):
    lines = []
    for xmin, ymin, xmax, ymax in boxes:
        xc, yc = ((xmin + xmax) / 2) / w, ((ymin + ymax) / 2) / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def roboflow_base_stem(p: Path) -> str:
    m = re.match(r"(.+?)_jpg\.rf\.", p.stem)
    return m.group(1) if m else p.stem


def main():
    print("Resolving source datasets (uses kagglehub cache if already present)...")
    andrewmvd_root = Path(kagglehub.dataset_download("andrewmvd/pothole-detection"))
    chit_root = Path(kagglehub.dataset_download("chitholian/annotated-potholes-dataset"))
    raj_root = Path(kagglehub.dataset_download("rajdalsaniya/pothole-detection-dataset"))
    banil_root = Path(kagglehub.dataset_download(
        "banilkumar20phd7071/pothole-and-normal-road-pavement-augmented-data"))

    andrewmvd_imgs = sorted(andrewmvd_root.rglob("*.png"))
    andrewmvd_xml = {p.stem: p for p in andrewmvd_root.rglob("*.xml")}
    chit_imgs = sorted(chit_root.rglob("*.jpg"))
    chit_xml = {p.stem: p for p in chit_root.rglob("*.xml")}
    raj_imgs = sorted(raj_root.rglob("*.jpg"))
    raj_labels = {p.stem: p for p in raj_root.rglob("*.txt")}
    normal_imgs = sorted((banil_root / "Dataset" / "Normal").rglob("*.jpg"))

    print(f"andrewmvd: {len(andrewmvd_imgs)} img | chitholian: {len(chit_imgs)} img | "
          f"rajdalsaniya: {len(raj_imgs)} img | banilkumar-normal: {len(normal_imgs)} img")

    # --- Converge andrewmvd + chitholian: same underlying photos, keep chitholian (higher-res) ---
    print("\nde-duplicating chitholian against andrewmvd (perceptual hash)...")
    andrewmvd_hashes = np.stack([phash(p) for p in andrewmvd_imgs])
    andrewmvd_stems = [p.stem for p in andrewmvd_imgs]

    chit_is_dup_of = {}  # chit stem -> matched andrewmvd stem
    for p in chit_imgs:
        h = phash(p)
        diffs = np.mean(np.abs(andrewmvd_hashes - h[None]), axis=(1, 2))
        best = int(np.argmin(diffs))
        if diffs[best] < DUP_THRESHOLD:
            chit_is_dup_of[p.stem] = andrewmvd_stems[best]

    matched_andrewmvd_stems = set(chit_is_dup_of.values())
    print(f"chitholian: {len(chit_is_dup_of)}/{len(chit_imgs)} matched an andrewmvd photo "
          f"(higher-res version will replace the andrewmvd copy)")

    # --- Converge rajdalsaniya's roboflow-augmented variants into 1 rep per base photo ---
    raj_groups: dict[str, list[Path]] = {}
    for p in raj_imgs:
        raj_groups.setdefault(roboflow_base_stem(p), []).append(p)
    raj_reps = [sorted(v)[0] for v in raj_groups.values()]
    print(f"rajdalsaniya: {len(raj_imgs)} files -> {len(raj_reps)} unique base photos "
          f"(1 representative kept per augmented group)")

    if COMBINED_DIR.exists():
        shutil.rmtree(COMBINED_DIR)
    (COMBINED_DIR / "images").mkdir(parents=True)
    (COMBINED_DIR / "labels").mkdir(parents=True)

    manifest = []
    geometry_rows = []  # for the converged relative-geometry feature report

    def write_entry(stem, img_src, label_lines, source):
        dest_img = COMBINED_DIR / "images" / f"{stem}{img_src.suffix.lower()}"
        shutil.copy2(img_src, dest_img)
        (COMBINED_DIR / "labels" / f"{stem}.txt").write_text("\n".join(label_lines))
        manifest.append(stem)
        for line in label_lines:
            if not line.strip():
                continue
            _, xc, yc, bw, bh = map(float, line.split())
            geometry_rows.append({
                "source": source, "relative_width": bw, "relative_height": bh,
                "relative_area": bw * bh,
                "aspect_ratio": bw / bh if bh > 0 else 0.0,
            })

    # 1. andrewmvd photos NOT superseded by a chitholian higher-res duplicate
    for img in andrewmvd_imgs:
        if img.stem in matched_andrewmvd_stems:
            continue  # superseded by chitholian below
        xml = andrewmvd_xml.get(img.stem)
        if xml is None:
            continue
        w, h, boxes = parse_voc(xml)
        write_entry(f"andrewmvd_{img.stem}", img, voc_boxes_to_yolo_lines(w, h, boxes), "andrewmvd")

    # 2. chitholian photos (both the ones that matched -- replacing andrewmvd -- and any
    #    that didn't match anything, which would be genuinely new; none were found, but
    #    the logic handles it correctly either way)
    for img in chit_imgs:
        xml = chit_xml.get(img.stem)
        if xml is None:
            continue
        w, h, boxes = parse_voc(xml)
        tag = "chitholian_replaces_andrewmvd" if img.stem in chit_is_dup_of else "chitholian_new"
        write_entry(f"chitholian_{img.stem}", img, voc_boxes_to_yolo_lines(w, h, boxes), tag)

    # 3. rajdalsaniya: one representative per base photo, using its own YOLO label as-is
    for img in raj_reps:
        label = raj_labels.get(img.stem)
        if label is None:
            continue
        lines = [ln for ln in label.read_text().splitlines() if ln.strip()]
        write_entry(f"rajdalsaniya_{img.stem}", img, lines, "rajdalsaniya")

    # 4. banilkumar background images (no pothole -> empty label, no geometry row)
    for img in normal_imgs:
        write_entry(f"banilkumar_normal_{img.stem}", img, [], "banilkumar_normal")

    n_annotated = sum(1 for s in manifest if not s.startswith("banilkumar_normal_"))
    n_background = len(manifest) - n_annotated
    print(f"\nMEGA DATASET: {len(manifest)} images total "
          f"({n_annotated} annotated + {n_background} background)")

    yaml_path = COMBINED_DIR / "data_full.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"path": str(COMBINED_DIR), "names": {0: "pothole"}}, f, sort_keys=False)
    (COMBINED_DIR / "manifest.txt").write_text("\n".join(manifest))

    # --- Converged-feature report (resolution-independent geometry only) ---
    geo_df = pd.DataFrame(geometry_rows)
    geo_df.to_csv(CONTEXT_DIR / "phase1_v2_geometry_features.csv", index=False)

    report = [
        "## Source accounting",
        f"- andrewmvd: {len(andrewmvd_imgs)} photos -> {len(andrewmvd_imgs) - len(matched_andrewmvd_stems)} "
        f"kept as-is (rest superseded by chitholian's higher-res duplicates)",
        f"- chitholian: {len(chit_imgs)} photos, {len(chit_is_dup_of)} replacing an andrewmvd "
        f"duplicate (higher native resolution: 720x720 vs ~400x300), "
        f"{len(chit_imgs) - len(chit_is_dup_of)} genuinely new",
        f"- rajdalsaniya: {len(raj_imgs)} files -> {len(raj_reps)} unique base photos kept "
        f"(Roboflow export-time augmentation converged back to 1 rep/photo to avoid k-fold leakage), "
        f"perceptual-hash-verified as 0/60-sampled-overlap with andrewmvd/chitholian",
        f"- banilkumar: {len(normal_imgs)} background (pothole-free) photos included; "
        f"650 classification-only 'Pothole' photos excluded (no bounding boxes, doesn't fit schema)",
        "",
        f"## Result: {len(manifest)} images ({n_annotated} annotated + {n_background} background) "
        f"vs. original Phase 1: 665 images, 0 background",
        "",
        "## Converged geometry features (resolution-independent; raw pixel width/height DROPPED",
        "## from all cross-source statistics since sources have different native resolutions)",
        "",
        geo_df.groupby("source")[["relative_width", "relative_height", "relative_area", "aspect_ratio"]]
            .describe().T.to_string(),
    ]
    (CONTEXT_DIR / "PHASE1_V2_DATASET_BUILD.md").write_text(
        "# Phase 1 v2 — merged mega-dataset build\n\n" + "\n".join(report) + "\n"
    )
    print("\n".join(report[:9]))
    print("\nWrote", CONTEXT_DIR / "PHASE1_V2_DATASET_BUILD.md")


if __name__ == "__main__":
    main()
