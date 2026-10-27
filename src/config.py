"""Central paths for the pothole-detection project.

Large data (datasets, kagglehub cache) lives under DATA_ROOT, which defaults to
<repo>/data and can be pointed at another disk with the POTHOLE_DATA_ROOT environment
variable. Checkpoints and training runs are kept inside the repository.

Importing this module has no side effects; scripts that write to these folders call
ensure_dirs() first.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("POTHOLE_DATA_ROOT", str(REPO_ROOT / "data")))
DATASETS_DIR = DATA_ROOT / "datasets"
CACHE_DIR = DATA_ROOT / "cache"

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RUNS_DIR = REPO_ROOT / "runs"

# kagglehub's download cache, kept under DATA_ROOT with the datasets
KAGGLEHUB_CACHE = CACHE_DIR / "kagglehub"

CONTEXT_DIR = REPO_ROOT / "context"

# Phase 1 YOLO dataset (VOC -> YOLO converted)
YOLO_DATASET_DIR = DATASETS_DIR / "pothole_yolo"


def ensure_dirs() -> None:
    """Create the data, cache, checkpoint and run folders if they don't exist."""
    for d in [DATASETS_DIR, CHECKPOINTS_DIR, RUNS_DIR, CACHE_DIR, KAGGLEHUB_CACHE, CONTEXT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
