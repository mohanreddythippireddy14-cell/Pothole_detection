"""Phase 1, step 5: visualize a handful of converted-YOLO training samples with their
boxes drawn, and (if a trained checkpoint is given) run+plot predictions on test images.

Ported from the notebook's sample-visualization + test-prediction-plotting cells.
Saves PNGs instead of plt.show() (no notebook display here).
"""

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import _py311_compat  # noqa: F401 - must run before torch
from config import YOLO_DATASET_DIR, CONTEXT_DIR, ensure_dirs  # noqa: E402

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def make_grid(n: int, cols: int = 3):
    """A rows x cols grid with room for n panels (at least one row)."""
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    return fig, axes.flatten()


def plot_training_samples(n=6, seed=0, out_path=None):
    train_images = list((YOLO_DATASET_DIR / "images" / "train").glob("*"))
    if not train_images:
        raise FileNotFoundError("No training images found — run script 02 first.")
    random.seed(seed)
    samples = random.sample(train_images, min(n, len(train_images)))

    fig, axes = make_grid(len(samples))

    for ax, image_path in zip(axes, samples):
        image = Image.open(image_path)
        image_width, image_height = image.size
        ax.imshow(image)

        label_path = YOLO_DATASET_DIR / "labels" / "train" / f"{image_path.stem}.txt"
        if label_path.exists():
            lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
            for line in lines:
                _, x_center, y_center, box_width, box_height = map(float, line.split())
                box_width_px = box_width * image_width
                box_height_px = box_height * image_height
                x_center_px = x_center * image_width
                y_center_px = y_center * image_height
                xmin = x_center_px - box_width_px / 2
                ymin = y_center_px - box_height_px / 2
                rect = patches.Rectangle(
                    (xmin, ymin), box_width_px, box_height_px, linewidth=2, fill=False,
                    edgecolor="red",
                )
                ax.add_patch(rect)
                ax.text(xmin, ymin, "pothole", fontsize=10, color="red")

        ax.set_title(image_path.name)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = out_path or (CONTEXT_DIR / "phase1_training_samples.png")
    plt.savefig(out_path, dpi=120)
    print("Wrote", out_path)


def plot_predictions(weights: str, n=6, conf=0.25, out_path=None):
    from ultralytics import YOLO
    import torch

    model = YOLO(weights)
    test_images = sorted((YOLO_DATASET_DIR / "images" / "test").glob("*"))[:n]
    if not test_images:
        raise FileNotFoundError("No test images found.")

    results = model.predict(
        source=[str(p) for p in test_images],
        imgsz=640, conf=conf,
        device=0 if torch.cuda.is_available() else "cpu",
        save=False,
    )

    num_to_show = min(n, len(results))
    fig, axes = make_grid(num_to_show)
    for i in range(num_to_show):
        annotated = results[i].plot()
        axes[i].imshow(annotated[..., ::-1])
        axes[i].set_title(test_images[i].name)
        axes[i].axis("off")
    for i in range(num_to_show, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    out_path = out_path or (CONTEXT_DIR / "phase1_test_predictions.png")
    plt.savefig(out_path, dpi=120)
    print("Wrote", out_path)


if __name__ == "__main__":
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="if given, also plot test predictions")
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()

    plot_training_samples(n=args.n)
    if args.weights:
        plot_predictions(args.weights, n=args.n)
