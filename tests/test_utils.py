"""Filesystem helper (src/fs_utils.py), config import, and plot grid (scripts/05)."""

import importlib
import sys
from unittest import mock

import pytest

from conftest import load_script


def test_link_or_copy_creates_readable_target(tmp_path):
    from fs_utils import link_or_copy

    src = tmp_path / "a.txt"
    src.write_text("hello")
    dst = tmp_path / "b.txt"
    link_or_copy(src, dst)
    assert dst.read_text() == "hello"


def test_link_or_copy_falls_back_to_copy_when_symlinks_fail(tmp_path):
    from pathlib import Path

    from fs_utils import link_or_copy

    src = tmp_path / "a.txt"
    src.write_text("hello")
    dst = tmp_path / "b.txt"
    with mock.patch.object(Path, "symlink_to", side_effect=OSError("not permitted")):
        link_or_copy(src, dst)
    assert dst.read_text() == "hello"
    assert not dst.is_symlink()


def test_config_import_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.setenv("POTHOLE_DATA_ROOT", str(tmp_path / "data"))
    sys.modules.pop("config", None)
    config = importlib.import_module("config")

    assert not (tmp_path / "data").exists()  # importing must not create folders
    config.ensure_dirs()
    assert config.DATASETS_DIR.is_dir() and config.KAGGLEHUB_CACHE.is_dir()
    sys.modules.pop("config", None)  # don't leak the tmp DATA_ROOT into other tests


@pytest.mark.parametrize("n, expected_panels", [(1, 3), (6, 6), (7, 9), (10, 12)])
def test_make_grid_has_room_for_n_panels(n, expected_panels):
    make_grid = load_script("05_visualize_samples.py").make_grid
    _, axes = make_grid(n)
    assert len(axes) == expected_panels
