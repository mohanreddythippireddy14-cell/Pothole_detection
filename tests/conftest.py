"""Shared test helpers.

The scripts import heavy or network-bound packages at module level (torch,
ultralytics, kagglehub, transformers). The unit tests only exercise pure helper
functions, so when those packages aren't installed they're replaced with empty stub
modules -- CI then needs only numpy/opencv/pandas/Pillow/PyYAML/matplotlib.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless plotting for scripts/05

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))


def _stub_if_missing(name: str, **attrs) -> None:
    try:
        importlib.import_module(name)
    except ImportError:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_if_missing("torch", cuda=types.SimpleNamespace(is_available=lambda: False))
_stub_if_missing("ultralytics", YOLO=object)
_stub_if_missing("kagglehub")
_stub_if_missing("transformers")


def load_script(filename: str):
    """Import a numbered script (e.g. '10_full_pipeline.py') as a module."""
    path = SCRIPTS_DIR / filename
    name = "script_" + path.stem
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
