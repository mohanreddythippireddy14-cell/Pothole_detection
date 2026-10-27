"""Phase 1 v2: k-fold cross-validated YOLOv8n training on the combined dataset.

Addresses two audit findings at once:
  #1 (close_mosaic never triggered): epochs/patience/close_mosaic are chosen together
     here so the mosaic-off transition happens well before patience could plausibly
     fire -- see the comment below for the arithmetic.
  #8 (small/noisy single test set): standard YOLO k-fold recipe (Ultralytics docs) --
     split the combined image pool into K folds, train K independent models (each
     holding out one fold as validation), evaluate each on its own held-out fold, then
     report mean +/- std across folds. This gives an actual variance estimate instead
     of one lucky/unlucky split.

Background images (banilkumar Normal, empty label files) are distributed across folds
using the same KFold split as annotated images, so every fold's train/val mix keeps the
pothole:background ratio roughly constant.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401
from config import DATASETS_DIR, RUNS_DIR, CHECKPOINTS_DIR, CONTEXT_DIR, ensure_dirs  # noqa: E402
from fs_utils import link_or_copy  # noqa: E402

import numpy as np
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO
import torch

COMBINED_DIR = DATASETS_DIR / "pothole_combined_v2"
KFOLD_DIR = DATASETS_DIR / "pothole_combined_v2_kfold"


def build_fold_dirs(manifest: list[str], k: int, seed: int) -> list[Path]:
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    manifest = np.array(manifest)

    if KFOLD_DIR.exists():
        shutil.rmtree(KFOLD_DIR)

    fold_yaml_paths = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(manifest)):
        fold_dir = KFOLD_DIR / f"fold{fold_idx}"
        for split, idx in [("train", train_idx), ("val", val_idx)]:
            (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            for stem in manifest[idx]:
                img_src = next((COMBINED_DIR / "images").glob(f"{stem}.*"))
                lbl_src = COMBINED_DIR / "labels" / f"{stem}.txt"
                link_or_copy(img_src, fold_dir / "images" / split / img_src.name)
                link_or_copy(lbl_src, fold_dir / "labels" / split / lbl_src.name)

        data_yaml = fold_dir / "data.yaml"
        with open(data_yaml, "w") as f:
            yaml.dump({
                "path": str(fold_dir),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "pothole"},
            }, f, sort_keys=False)
        fold_yaml_paths.append(data_yaml)
        print(f"fold {fold_idx}: train={len(train_idx)} val={len(val_idx)}")

    return fold_yaml_paths


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    # close_mosaic fixed at 15 with epochs=60: mosaic turns off at epoch 45. patience=25
    # means early stopping can't fire before epoch (best_epoch + 25); as long as the
    # best epoch found in practice is not absurdly early (it wasn't in the v1 run --
    # best was epoch 66/86), this keeps the clean-image fine-tune phase reachable,
    # unlike v1's epochs=100/patience=20/close_mosaic=10 (mosaic-off at 90, but
    # patience could and did fire at 86).
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--close_mosaic", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = (COMBINED_DIR / "manifest.txt").read_text().splitlines()
    print(f"Total combined pool: {len(manifest)} images, k={args.k}")

    fold_yaml_paths = build_fold_dirs(manifest, args.k, args.seed)

    pretrained = CHECKPOINTS_DIR / "pretrained" / "yolov8n.pt"
    pretrained_arg = str(pretrained) if pretrained.exists() else "yolov8n.pt"

    fold_metrics = []
    for fold_idx, data_yaml in enumerate(fold_yaml_paths):
        print(f"\n{'='*60}\nFOLD {fold_idx}/{args.k}\n{'='*60}")
        model = YOLO(pretrained_arg)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=0 if torch.cuda.is_available() else "cpu",
            patience=args.patience,
            close_mosaic=args.close_mosaic,
            project=str(RUNS_DIR / "pothole_kfold_v2"),
            name=f"fold{fold_idx}",
            plots=True,
            seed=args.seed,
            deterministic=True,
        )
        run_dir = RUNS_DIR / "pothole_kfold_v2" / f"fold{fold_idx}"
        best = run_dir / "weights" / "best.pt"
        val_model = YOLO(str(best))
        metrics = val_model.val(data=str(data_yaml), split="val", imgsz=args.imgsz,
                                 device=0 if torch.cuda.is_available() else "cpu")
        fold_result = {
            "fold": fold_idx,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
        fold_metrics.append(fold_result)
        print(f"Fold {fold_idx} result: {fold_result}")

        if best.exists():
            dest = CHECKPOINTS_DIR / f"yolov8n_pothole_v2_fold{fold_idx}_best.pt"
            dest.write_bytes(best.read_bytes())

    # Aggregate
    keys = ["precision", "recall", "map50", "map50_95"]
    summary_lines = [f"# Phase 1 v2 — {args.k}-fold cross-validation results\n",
                      f"Combined dataset: {len(manifest)} images "
                      f"({args.epochs} epochs/fold, patience={args.patience}, "
                      f"close_mosaic={args.close_mosaic}, imgsz={args.imgsz})\n",
                      "| Fold | Precision | Recall | mAP@50 | mAP@50:95 |",
                      "|---|---|---|---|---|"]
    for fm in fold_metrics:
        summary_lines.append(
            f"| {fm['fold']} | {fm['precision']:.4f} | {fm['recall']:.4f} | "
            f"{fm['map50']:.4f} | {fm['map50_95']:.4f} |"
        )
    means = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
    stds = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in keys}
    summary_lines.append(
        f"| **mean±std** | {means['precision']:.4f}±{stds['precision']:.4f} | "
        f"{means['recall']:.4f}±{stds['recall']:.4f} | "
        f"{means['map50']:.4f}±{stds['map50']:.4f} | "
        f"{means['map50_95']:.4f}±{stds['map50_95']:.4f} |"
    )

    print("\n" + "\n".join(summary_lines))
    out = CONTEXT_DIR / "PHASE1_V2_KFOLD_RESULTS.md"
    out.write_text("\n".join(summary_lines) + "\n")
    print("\nWrote", out)


if __name__ == "__main__":
    main()
