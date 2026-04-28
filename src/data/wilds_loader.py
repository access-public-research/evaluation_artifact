import importlib
import sys
from pathlib import Path


def _purge_wilds_modules():
    for k in list(sys.modules.keys()):
        if k == "wilds" or k.startswith("wilds."):
            del sys.modules[k]


def _find_vendored_wilds_root(start: Path):
    for parent in [start] + list(start.parents):
        cand = parent / "wilds" / "wilds" / "__init__.py"
        if cand.is_file():
            return parent / "wilds"
    return None


def _resolve_wilds_get_dataset():
    # 1) Try normal import path first (pip install wilds).
    try:
        _purge_wilds_modules()
        wilds_mod = importlib.import_module("wilds")
        gd_mod = importlib.import_module("wilds.get_dataset")
        if hasattr(wilds_mod, "supported_datasets") and hasattr(gd_mod, "get_dataset"):
            return gd_mod.get_dataset
    except Exception:
        pass

    # 2) Try vendored checkout (../wilds or higher).
    here = Path(__file__).resolve()
    vendored_root = _find_vendored_wilds_root(here)
    if vendored_root is not None:
        if str(vendored_root) not in sys.path:
            sys.path.insert(0, str(vendored_root))
        _purge_wilds_modules()
        wilds_mod = importlib.import_module("wilds")
        gd_mod = importlib.import_module("wilds.get_dataset")
        if hasattr(wilds_mod, "supported_datasets") and hasattr(gd_mod, "get_dataset"):
            return gd_mod.get_dataset

    raise RuntimeError(
        "Could not import WILDS `get_dataset`. Install with `python -m pip install wilds` "
        "or ensure a vendored checkout exists at ../wilds/wilds/."
    )


def load_wilds_dataset(name, data_dir, download=False):
    get_dataset = _resolve_wilds_get_dataset()
    return get_dataset(dataset=name, root_dir=data_dir, download=bool(download))


def get_metadata_fields(dataset):
    fields = getattr(dataset, "metadata_fields", None)
    if fields is None:
        fields = getattr(dataset, "_metadata_fields", None)
    return list(fields) if fields is not None else None
