"""
Geng full-sequence test inference — ports NILM-main/NetFlowExt.py + DataProvider.py.

EasyS2S: custompredictS2SX (sliding windows, overlap average)
S2P / AugLPN: custompredictX (one point per window)
FCN: custompredict_fcn (non-overlapping blocks, center-cropped outputs)
"""

from __future__ import annotations

import numpy as np
import torch

from nilm_main_pytorch.models.params import FCN_OUTPUT_LENGTH, FCN_TARGET_LENGTH


def _batched_forward(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            out = model(batch).cpu().numpy()
            parts.append(out)
    return np.vstack(parts)


def predict_easy_s2s_geng(
    model: torch.nn.Module,
    aggregate: np.ndarray,
    window_length: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Port of NetFlowExt.custompredictS2SX + DoubleSourceProvider4."""
    inputs = np.asarray(aggregate, dtype=np.float32).flatten()
    max_nofw = len(inputs) - window_length + 1
    if max_nofw <= 0:
        raise ValueError(f"Series length {len(inputs)} < window {window_length}")

    indices = np.arange(max_nofw, dtype=np.int64)
    windows = np.stack([inputs[i : i + window_length] for i in indices], axis=0)
    output_container = _batched_forward(model, windows, device, batch_size)

    length = window_length
    n = len(output_container) + length - 1
    sum_arr = np.zeros(n, dtype=np.float64)
    counts_arr = np.zeros(n, dtype=np.float64)
    for i in range(len(output_container)):
        sum_arr[i : i + length] += output_container[i].reshape(-1)
        counts_arr[i : i + length] += 1.0
    return (sum_arr / counts_arr).astype(np.float32)


def predict_s2p_geng(
    model: torch.nn.Module,
    aggregate: np.ndarray,
    window_length: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Port of NetFlowExt.custompredictX + DoubleSourceProvider3."""
    inputs = np.asarray(aggregate, dtype=np.float32).flatten()
    offset = window_length // 2
    max_nofw = len(inputs) - 2 * offset
    if max_nofw <= 0:
        raise ValueError(f"Series length {len(inputs)} too short for window {window_length}")

    indices = np.arange(max_nofw, dtype=np.int64)
    windows = np.stack([inputs[i : i + 2 * offset] for i in indices], axis=0)
    output_container = _batched_forward(model, windows, device, batch_size)
    return output_container.reshape(-1).astype(np.float32)


def _fcn_prepare_aggregate(aggregate: np.ndarray, target_length: int) -> tuple[np.ndarray, int]:
    offset = target_length // 2
    inputs = np.asarray(aggregate, dtype=np.float32).flatten()
    num_windows = int(np.ceil((len(inputs) - 2 * offset) / target_length))
    pad_size = num_windows * target_length + 2 * offset - len(inputs)
    if pad_size > 0:
        inputs = np.concatenate([inputs, np.zeros(pad_size, dtype=np.float32)])
    return inputs, offset


def predict_fcn_geng(
    model: torch.nn.Module,
    aggregate: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    target_length: int = FCN_TARGET_LENGTH,
    output_length: int = FCN_OUTPUT_LENGTH,
) -> np.ndarray:
    """Port of NetFlowExt.custompredict_fcn + DoubleSourceProvider_fcn."""
    inputs, offset = _fcn_prepare_aggregate(aggregate, target_length)
    input_length = 2 * offset + target_length
    num_windows = int(np.ceil((len(inputs) - 2 * offset) / target_length))

    indices = np.arange(num_windows, dtype=np.int64)
    windows = np.stack(
        [inputs[i * target_length : i * target_length + input_length] for i in indices],
        axis=0,
    )
    output_container = _batched_forward(model, windows, device, batch_size)
    return output_container.reshape(-1).astype(np.float32)


def ground_truth_easy_s2s(appliance: np.ndarray) -> np.ndarray:
    """EasyS2S_test.load_dataset — full appliance column."""
    return np.asarray(appliance, dtype=np.float32).flatten()


def ground_truth_s2p(appliance: np.ndarray, window_length: int) -> np.ndarray:
    """S2P_baseline_test.load_dataset — trim both ends by offset."""
    offset = window_length // 2
    series = np.asarray(appliance, dtype=np.float32).flatten()
    return series[offset:-offset]


def ground_truth_fcn(
    appliance: np.ndarray,
    *,
    target_length: int = FCN_TARGET_LENGTH,
    output_length: int = FCN_OUTPUT_LENGTH,
) -> np.ndarray:
    """Per-window center-cropped targets (same length as predict_fcn_geng output)."""
    series = np.asarray(appliance, dtype=np.float32).flatten()
    padded, offset = _fcn_prepare_aggregate(series, target_length)
    interior = padded[offset:-offset]
    target_offset = (target_length - output_length) // 2
    num_windows = int(np.ceil((len(series) - 2 * offset) / target_length))
    chunks: list[np.ndarray] = []
    for idx in range(num_windows):
        start = idx * target_length
        block = interior[start : start + target_length]
        chunks.append(block[target_offset : target_offset + output_length])
    return np.concatenate(chunks).astype(np.float32)


def predict_geng_test(
    model: torch.nn.Module,
    model_name: str,
    aggregate: np.ndarray,
    appliance: np.ndarray,
    window_length: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Geng-aligned inference; returns denorm-ready z-score series (pred, true)."""
    model_key = model_name.lower()
    if model_key == "easy_s2s":
        pred = predict_easy_s2s_geng(model, aggregate, window_length, device, batch_size)
        true = ground_truth_easy_s2s(appliance)
    elif model_key in ("s2p", "auglpn"):
        pred = predict_s2p_geng(model, aggregate, window_length, device, batch_size)
        true = ground_truth_s2p(appliance, window_length)
    elif model_key == "fcn":
        pred = predict_fcn_geng(model, aggregate, device, batch_size)
        true = ground_truth_fcn(appliance)
    else:
        raise ValueError(f"Unknown model for Geng inference: {model_name}")

    if len(pred) != len(true):
        min_len = min(len(pred), len(true))
        pred = pred[:min_len]
        true = true[:min_len]
    return pred, true
