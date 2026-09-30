"""Central paths for the pothole-detection project.

Everything large (datasets, checkpoints, training runs) lives on the second drive —
root only has ~2.2GB free. Do not write large files under DATA_ROOT's parent.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Datasets and cache remain on external drive by default due to size constraints
DATA_ROOT = Path(os.environ.get("POTHOLE_DATA_ROOT", str(REPO_ROOT / "data")))
DATASETS_DIR = DATA_ROOT / "datasets"
CACHE_DIR = DATA_ROOT / "cache"

# Checkpoints and runs are now kept locally in the project repository
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RUNS_DIR = REPO_ROOT / "runs"

# kagglehub / ultralytics need their caches redirected off the root disk too
KAGGLEHUB_CACHE = CACHE_DIR / "kagglehub"

CONTEXT_DIR = REPO_ROOT / "context"

# Phase 1 YOLO dataset (VOC -> YOLO converted)
YOLO_DATASET_DIR = DATASETS_DIR / "pothole_yolo"

for d in [DATASETS_DIR, CHECKPOINTS_DIR, RUNS_DIR, CACHE_DIR, KAGGLEHUB_CACHE]:
    d.mkdir(parents=True, exist_ok=True)
