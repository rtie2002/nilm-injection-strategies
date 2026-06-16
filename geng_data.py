"""
DEPRECATED — use nilm_main_pytorch.data (NILM-main PyTorch).

  from nilm_main_pytorch.data import build_train_val_loaders, train_csv_path
"""

import sys
from pathlib import Path

_PYTORCH_ROOT = (
    Path(__file__).resolve().parent
    / "DiffusionModel_NILM"
    / "NILM-main"
    / "NILM-main_pytorch"
)
if str(_PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTORCH_ROOT))

from nilm_main_pytorch.data.datasets import build_train_val_loaders, load_geng_csv
from nilm_main_pytorch.data.paths import test_csv_path, train_csv_path, validation_csv_path


def resolve_geng_paths(data_root, appliance, *, origin=True, train_percent="20", dataset_name="UK_DALE"):
    root = Path(data_root)
    return (
        train_csv_path(root, appliance, origin=origin, train_percent=train_percent, dataset_name=dataset_name),
        validation_csv_path(root, appliance),
    )


build_geng_loaders = build_train_val_loaders

__all__ = [
    "build_geng_loaders",
    "build_train_val_loaders",
    "load_geng_csv",
    "resolve_geng_paths",
    "test_csv_path",
    "train_csv_path",
    "validation_csv_path",
]
