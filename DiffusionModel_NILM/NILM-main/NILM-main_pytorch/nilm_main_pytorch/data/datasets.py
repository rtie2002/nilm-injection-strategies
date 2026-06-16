"""PyTorch datasets and DataLoaders for Geng 2-column z-score CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from nilm_main_pytorch.models.params import FCN_OUTPUT_LENGTH, FCN_TARGET_LENGTH, PARAMS_APPLIANCE


def load_geng_csv(path: Path, crop: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None, nrows=crop)
    if df.shape[1] < 2:
        raise ValueError(f"Expected 2 columns in {path}, got {df.shape[1]}")
    arr = df.to_numpy(dtype=np.float32)
    return arr[:, 0], arr[:, 1]


class Seq2SeqWindowDataset(Dataset):
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
        return (
            torch.from_numpy(self.aggregate[start:end]),
            torch.from_numpy(self.appliance[start:end]),
        )


class Seq2PointWindowDataset(Dataset):
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
        return (
            torch.from_numpy(self.aggregate[start:end]),
            torch.tensor(self.appliance[mid], dtype=torch.float32).unsqueeze(0),
        )


class FCNDataset(Dataset):
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
        y = torch.from_numpy(target[self.target_offset : self.target_offset + self.output_length])
        return x, y


def _make_dataset(
    model_name: str,
    aggregate: np.ndarray,
    appliance: np.ndarray,
    window_length: int,
    stride: int,
) -> Dataset:
    model = model_name.lower()
    if model == "easy_s2s":
        return Seq2SeqWindowDataset(aggregate, appliance, window_length, stride)
    if model in ("s2p", "auglpn"):
        return Seq2PointWindowDataset(aggregate, appliance, window_length, stride)
    if model == "fcn":
        return FCNDataset(aggregate, appliance)
    raise ValueError(f"Unknown model: {model_name}")


def build_loader(
    *,
    model_name: str,
    appliance: str,
    csv_path: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    stride: int = 1,
    crop_rows: int | None = None,
) -> DataLoader:
    window_length = PARAMS_APPLIANCE[appliance]["window_length"]
    agg, app = load_geng_csv(csv_path, crop=crop_rows)
    ds = _make_dataset(model_name, agg, app, window_length, stride)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_train_val_loaders(
    *,
    model_name: str,
    appliance: str,
    train_csv: Path,
    val_csv: Path,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    stride: int = 1,
    val_crop_rows: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_loader = build_loader(
        model_name=model_name,
        appliance=appliance,
        csv_path=train_csv,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        stride=stride,
    )
    val_loader = build_loader(
        model_name=model_name,
        appliance=appliance,
        csv_path=val_csv,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        stride=stride,
        crop_rows=val_crop_rows,
    )
    return train_loader, val_loader
