"""Data loaders for Geng UK-DALE 2-column z-score CSVs (NILM-main format)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model.geng_params import FCN_OUTPUT_LENGTH, FCN_TARGET_LENGTH, PARAMS_APPLIANCE


def load_geng_csv(path: Path, crop: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read headerless CSV: col0 aggregate, col1 appliance (already z-scored)."""
    df = pd.read_csv(path, header=None, nrows=crop)
    if df.shape[1] < 2:
        raise ValueError(f"Expected 2 columns in {path}, got {df.shape[1]}")
    arr = df.to_numpy(dtype=np.float32)
    return arr[:, 0], arr[:, 1]


class Seq2SeqWindowDataset(Dataset):
    """EasyS2S: sliding windows, predict full appliance sequence."""

    def __init__(self, aggregate: np.ndarray, appliance: np.ndarray, window_length: int, stride: int = 1):
        self.window_length = window_length
        self.aggregate = aggregate
        self.appliance = appliance
        max_start = len(aggregate) - window_length
        if max_start < 0:
            raise ValueError(f"Series length {len(aggregate)} < window {window_length}")
        self.indices = np.arange(0, max_start + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.indices[idx])
        end = start + self.window_length
        x = torch.from_numpy(self.aggregate[start:end])
        y = torch.from_numpy(self.appliance[start:end])
        return x, y


class Seq2PointWindowDataset(Dataset):
    """S2P / AugLPN: centered window -> midpoint appliance value."""

    def __init__(self, aggregate: np.ndarray, appliance: np.ndarray, window_length: int, stride: int = 1):
        self.window_length = window_length
        self.aggregate = aggregate
        self.appliance = appliance
        self.offset = window_length // 2
        max_start = len(aggregate) - window_length
        if max_start < 0:
            raise ValueError(f"Series length {len(aggregate)} < window {window_length}")
        self.indices = np.arange(0, max_start + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.indices[idx])
        end = start + self.window_length
        mid = start + self.offset
        x = torch.from_numpy(self.aggregate[start:end])
        y = torch.tensor(self.appliance[mid], dtype=torch.float32).unsqueeze(0)
        return x, y


class FCNDataset(Dataset):
    """FCN: padded long input, target center crop to output_length."""

    def __init__(
        self,
        aggregate: np.ndarray,
        appliance: np.ndarray,
        target_length: int = FCN_TARGET_LENGTH,
        output_length: int = FCN_OUTPUT_LENGTH,
    ):
        self.aggregate = aggregate
        self.appliance = appliance
        self.target_length = target_length
        self.output_length = output_length
        self.input_length = target_length + (target_length // 2) * 2
        self.target_offset = (target_length - output_length) // 2

        offset = target_length // 2
        num_windows = int(np.ceil((len(aggregate) - 2 * offset) / target_length))
        pad = num_windows * target_length + 2 * offset - len(aggregate)
        if pad > 0:
            self.aggregate = np.concatenate([aggregate, np.zeros(pad, dtype=np.float32)])
            self.appliance = np.concatenate([appliance, np.zeros(pad, dtype=np.float32)])

        self.targets = self.appliance[offset:-offset]
        self.num_windows = num_windows

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = self.target_length // 2
        inp_start = idx * self.target_length
        x = torch.from_numpy(self.aggregate[inp_start : inp_start + self.input_length])
        t_start = idx * self.target_length
        target = self.targets[t_start : t_start + self.target_length]
        y = torch.from_numpy(
            target[self.target_offset : self.target_offset + self.output_length]
        )
        return x, y


def resolve_geng_paths(
    data_root: Path,
    appliance: str,
    *,
    origin: bool = True,
    train_percent: str = "20",
    dataset_name: str = "UK_DALE",
) -> tuple[Path, Path]:
    app_dir = data_root / appliance
    if origin:
        train_name = f"{appliance}_{train_percent}training_.csv"
    else:
        train_name = f"{dataset_name}Combined{appliance}_file{train_percent}.csv"

    train_path = app_dir / train_name
    val_path = None
    for name in sorted(app_dir.glob("*.csv")):
        if "validation" in name.name.lower():
            val_path = name
            break
    if val_path is None:
        val_path = app_dir / f"{appliance}_validation_.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation CSV not found in {app_dir}")
    return train_path, val_path


def build_geng_loaders(
    *,
    model_name: str,
    appliance: str,
    train_csv: Path,
    val_csv: Path,
    batch_size: int,
    num_workers: int = 0,
    stride: int = 1,
) -> tuple[DataLoader, DataLoader]:
    params = PARAMS_APPLIANCE[appliance]
    window_length = params["window_length"]

    train_agg, train_app = load_geng_csv(train_csv)
    val_agg, val_app = load_geng_csv(val_csv)

    model = model_name.lower()
    if model == "easy_s2s":
        train_ds: Dataset = Seq2SeqWindowDataset(train_agg, train_app, window_length, stride)
        val_ds = Seq2SeqWindowDataset(val_agg, val_app, window_length, stride)
    elif model in ("s2p", "auglpn"):
        train_ds = Seq2PointWindowDataset(train_agg, train_app, window_length, stride)
        val_ds = Seq2PointWindowDataset(val_agg, val_app, window_length, stride)
    elif model == "fcn":
        train_ds = FCNDataset(train_agg, train_app)
        val_ds = FCNDataset(val_agg, val_app)
    else:
        raise ValueError(f"Unknown Geng model: {model_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
