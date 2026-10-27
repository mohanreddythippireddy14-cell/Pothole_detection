"""Phase 2, step 4: quantify how well pretrained Depth-Anything-V2 agrees with
PothRGBD's real sensor depth.

The notes flagged (qualitatively) that pretrained Depth-Anything-V2 "not performing
well" on real pothole RGBD pairs. This script makes that claim quantitative: monocular
depth models like DA-V2 predict *relative* (affine-ambiguous) depth, not metric depth,
so a fair comparison requires per-image least-squares affine alignment
(pred_aligned = a*pred + b, fit on valid GT pixels) before computing agreement --
comparing raw DA-V2 output directly against metric sensor depth would be meaningless
by construction, not just "not performing well".

Reports, per image and aggregated: Pearson correlation (scale/shift-invariant by
nature, no alignment needed) and post-alignment RMSE/MAE in the sensor's native units,
both over the whole valid-depth region and restricted to the annotated pothole mask
(the region that actually matters for severity scoring).
"""

import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401
from config import KAGGLEHUB_CACHE, CONTEXT_DIR, ensure_dirs  # noqa: E402

os.environ.setdefault("KAGGLEHUB_CACHE", str(KAGGLEHUB_CACHE))

import numpy as np
import kagglehub
import torch
from PIL import Image
from scipy.stats import pearsonr
import cv2


def polygon_mask_from_yolo_seg_txt(txt_path: Path, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in txt_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = list(map(float, line.split()[1:]))
        pts = np.array(
            [[parts[i] * width, parts[i + 1] * height] for i in range(0, len(parts), 2)],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 1)
    return mask


def fit_affine(pred: np.ndarray, gt: np.ndarray):
    """Least-squares a,b minimizing ||a*pred + b - gt||^2."""
    A = np.stack([pred, np.ones_like(pred)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, gt, rcond=None)
    return a, b


def main(n_samples=40, seed=0):
    ensure_dirs()
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    print("Depth-Anything-V2-Small loaded. Device:", "GPU" if device == 0 else "CPU")

    root = Path(kagglehub.dataset_download("mahyeks/pothrgbd-rgb-and-depth-images-of-potholes"))
    images = sorted(root.rglob("*.jpg"))
    txts = {p.stem: p for p in root.rglob("*.txt")}

    def ts(p):
        m = re.search(r"(\d{8}_\d{6})", p.name)
        return m.group(1) if m else None

    depth_by_ts = {}
    for p in root.rglob("*.npy"):
        t = ts(p)
        if t:
            depth_by_ts[t] = p

    triples = []
    for img in images:
        t = ts(img)
        if t in depth_by_ts and img.stem in txts:
            triples.append((img, depth_by_ts[t], txts[img.stem]))

    random.seed(seed)
    sample = random.sample(triples, min(n_samples, len(triples)))
    print(f"Evaluating on {len(sample)} PothRGBD triples")

    full_corrs, full_rmses = [], []
    mask_corrs, mask_rmses = [], []

    for img_path, depth_path, txt_path in sample:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        gt_depth = np.load(depth_path).astype(np.float32)  # (H,W), uint16 source, mm presumably
        if gt_depth.shape != (h, w):
            # sensor depth resolution may differ from RGB; resize GT to match RGB grid
            gt_depth = cv2.resize(gt_depth, (w, h), interpolation=cv2.INTER_NEAREST)

        result = depth_pipe(image)
        pred_depth = np.array(result["predicted_depth"].squeeze().cpu())
        if pred_depth.shape != (h, w):
            pred_depth = cv2.resize(pred_depth, (w, h), interpolation=cv2.INTER_LINEAR)

        valid = gt_depth > 0
        if valid.sum() < 100:
            continue

        gt_valid = gt_depth[valid]
        pred_valid = pred_depth[valid]

        # DA-V2 outputs are *inverse*-depth-like (higher = closer); sensor depth is
        # direct metric depth (higher = farther). Correlate against -gt to check sign,
        # then let the affine fit absorb the sign either way.
        corr, _ = pearsonr(pred_valid, gt_valid)
        a, b = fit_affine(pred_valid, gt_valid)
        aligned = a * pred_valid + b
        rmse = float(np.sqrt(np.mean((aligned - gt_valid) ** 2)))
        full_corrs.append(corr)
        full_rmses.append(rmse)

        pothole_mask = polygon_mask_from_yolo_seg_txt(txt_path, w, h)
        mvalid = valid & (pothole_mask > 0)
        if mvalid.sum() >= 30:
            gt_m = gt_depth[mvalid]
            pred_m = pred_depth[mvalid]
            corr_m, _ = pearsonr(pred_m, gt_m)
            aligned_m = a * pred_m + b  # reuse whole-image affine fit, as a real pipeline would
            rmse_m = float(np.sqrt(np.mean((aligned_m - gt_m) ** 2)))
            mask_corrs.append(corr_m)
            mask_rmses.append(rmse_m)

    lines = []

    def log(*args):
        line = " ".join(str(a) for a in args)
        print(line)
        lines.append(line)

    log("DEPTH-ANYTHING-V2 vs POTHRGBD SENSOR DEPTH")
    log("=" * 60)
    log(f"N images evaluated (full-frame): {len(full_corrs)}")
    log(f"Full-frame Pearson r: mean={np.mean(full_corrs):.3f} std={np.std(full_corrs):.3f} "
        f"min={np.min(full_corrs):.3f} max={np.max(full_corrs):.3f}")
    log(f"Full-frame RMSE (post affine-align, sensor units): mean={np.mean(full_rmses):.1f} "
        f"std={np.std(full_rmses):.1f}")
    log(f"Full-frame GT depth typical scale: sensor mean~1097 (see PHASE2_STEP1 audit)")
    log("")
    log(f"N images evaluated (pothole-mask only): {len(mask_corrs)}")
    if mask_corrs:
        log(f"Mask-region Pearson r: mean={np.mean(mask_corrs):.3f} std={np.std(mask_corrs):.3f} "
            f"min={np.min(mask_corrs):.3f} max={np.max(mask_corrs):.3f}")
        log(f"Mask-region RMSE (whole-image affine reused): mean={np.mean(mask_rmses):.1f} "
            f"std={np.std(mask_rmses):.1f}")
    log("")
    frac_negative = float(np.mean(np.array(full_corrs) < 0))
    frac_weak = float(np.mean(np.abs(full_corrs) < 0.3))
    log(f"Fraction of images with NEGATIVE full-frame correlation: {frac_negative:.2f}")
    log(f"Fraction of images with WEAK (|r|<0.3) full-frame correlation: {frac_weak:.2f}")

    out = CONTEXT_DIR / "PHASE2_DEPTH_ANYTHING_EVAL.md"
    out.write_text("# Phase 2 — Depth-Anything-V2 (pretrained) vs PothRGBD sensor depth\n\n```\n" + "\n".join(lines) + "\n```\n")
    log("\nWrote", out)


if __name__ == "__main__":
    main()
