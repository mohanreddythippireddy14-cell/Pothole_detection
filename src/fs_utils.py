"""Small filesystem helpers shared by the dataset-building scripts."""

import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    """Symlink dst -> src, or copy the file where symlinks aren't allowed.

    Windows only permits symlinks with Developer Mode or admin rights, so the k-fold
    and final-split dataset builders fall back to a plain copy there.
    """
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)
