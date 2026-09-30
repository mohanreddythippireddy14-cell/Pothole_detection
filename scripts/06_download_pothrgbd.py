"""Phase 2, step 1: download and audit the PothRGBD dataset (RGB + sensor depth +
segmentation polygon annotations), matching by YYYYMMDD_HHMMSS timestamp in filenames.

Ported from the notebook's PothRGBD exploration cells.
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401
from config import KAGGLEHUB_CACHE, CONTEXT_DIR  # noqa: E402

os.environ.setdefault("KAGGLEHUB_CACHE", str(KAGGLEHUB_CACHE))

import kagglehub
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEPTH_EXTENSIONS = {".npy", ".npz", ".tif", ".tiff"}
ANNOTATION_EXTENSIONS = {".txt", ".xml", ".json", ".yaml", ".yml"}


def extract_timestamp(path: Path):
    match = re.search(r"(\d{8}_\d{6})", path.name)
    return match.group(1) if match else None


def main():
    log_lines = []

    def log(*args):
        line = " ".join(str(a) for a in args)
        print(line)
        log_lines.append(line)

    root = Path(kagglehub.dataset_download("mahyeks/pothrgbd-rgb-and-depth-images-of-potholes"))
    log("PothRGBD location:", root)
    log("Exists:", root.exists())

    file_counts = Counter()
    for p in root.rglob("*"):
        if p.is_file():
            file_counts[p.suffix.lower()] += 1
    log("\nFILE TYPE COUNTS")
    for ext, count in sorted(file_counts.items()):
        log(f"{ext or '[no ext]':10s}: {count}")

    rgb_candidates = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    depth_candidates = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in DEPTH_EXTENSIONS)
    txt_files = sorted(root.rglob("*.txt"))

    log("\nRGB-like files:", len(rgb_candidates))
    log("Depth candidates:", len(depth_candidates))
    log("TXT annotation files:", len(txt_files))

    depth_by_ts = {}
    for dp in depth_candidates:
        ts = extract_timestamp(dp)
        if ts is not None:
            depth_by_ts[ts] = dp

    matched, unmatched = [], []
    for rp in rgb_candidates:
        ts = extract_timestamp(rp)
        if ts in depth_by_ts:
            matched.append((rp, depth_by_ts[ts]))
        else:
            unmatched.append(rp)

    log("\nRGB<->DEPTH MATCHING (by timestamp)")
    log("Matched pairs:", len(matched))
    log("Unmatched RGB:", len(unmatched))

    # segmentation polygon txt files: match by stem to rgb images
    txt_by_stem = {p.stem: p for p in txt_files}
    seg_matched = [(rp, txt_by_stem[rp.stem]) for rp in rgb_candidates if rp.stem in txt_by_stem]
    log("\nRGB<->SEGMENTATION-TXT MATCHING (by stem)")
    log("Matched pairs:", len(seg_matched))

    triple_matched = []
    for rp, dp in matched:
        if rp.stem in txt_by_stem:
            triple_matched.append((rp, dp, txt_by_stem[rp.stem]))
    log("\nRGB<->DEPTH<->SEG-TXT triple matches:", len(triple_matched))

    if depth_candidates:
        sample_depth = depth_candidates[0]
        if sample_depth.suffix.lower() == ".npy":
            d = np.load(sample_depth)
            log("\nSample depth array:", sample_depth.name, "shape", d.shape, "dtype", d.dtype)
            log("min/max/mean:", float(np.nanmin(d)), float(np.nanmax(d)), float(np.nanmean(d)))
            valid = d[d > 0]
            if valid.size:
                log("valid(>0) pixels:", valid.size, "of", d.size,
                    f"({100*valid.size/d.size:.1f}%)")

    if txt_files:
        sample_txt = txt_files[0]
        content = sample_txt.read_text().strip()
        parts = content.split()
        log("\nSample seg-txt:", sample_txt.name)
        log("n values:", len(parts), "-> class_id=", parts[0] if parts else None,
            "n_points=", (len(parts) - 1) // 2 if parts else 0)

    out = CONTEXT_DIR / "PHASE2_STEP1_POTHRGBD_AUDIT.txt"
    out.write_text("\n".join(log_lines))
    log("\nWrote", out)

    return root


if __name__ == "__main__":
    main()
