r"""
Add ON/OFF labels directly into source CSV files.

This script edits the original CSV files in place under:
    ../real_data
    ../synthetic_data

It does NOT create a new dataset folder.
It adds or replaces one column:
    on_off

Output columns become:
    aggregate,<appliance>,minute_sin,...,month_cos,on_off

For synthetic CSVs:
    <appliance>,minute_sin,...,month_cos,on_off

______________________________________________________
Step (1)
Go to this folder
______________________________________________________

PowerShell command:
    cd "C:\Users\Raymond Tie\Desktop\PhD\Paper\Injection Strategy Conference\code\injection_strategy"

______________________________________________________
Step (2)
Add ON/OFF labels directly to real_data and synthetic_data
______________________________________________________

PowerShell command:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\add_on_off_to_source_csv.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

APPLIANCES = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]

APPLIANCE_THRESHOLDS_WATTS = {
    "dishwasher": 10.0,
    "fridge": 50.0,
    "kettle": 500.0,
    "microwave": 150.0,
    "washingmachine": 20.0,
}

# Minimum ON duration in 1-minute samples.
# Dishwasher can have small single-point spikes above 10 W, so it needs stricter filtering.
APPLIANCE_MIN_ON_STEPS = {
    "dishwasher": 5,
    "fridge": 1,
    "kettle": 1,
    "microwave": 1,
    "washingmachine": 2,
}

# Spike removal is useful for dishwasher noise, but it can wrongly remove
# short kettle or microwave activations. Use it only where it is helpful.
APPLIANCE_REMOVE_SPIKES = {
    "dishwasher": True,
    "fridge": False,
    "kettle": False,
    "microwave": False,
    "washingmachine": True,
}


def remove_isolated_spikes(power_sequence: np.ndarray, window_size: int = 5, spike_threshold: float = 3.0, background_threshold: float = 50.0) -> np.ndarray:
    # ______________________________________________________
    # Step (A)
    # Remove isolated spikes before ON/OFF detection.
    # This follows the idea used in ukdale_processing.py.
    # ______________________________________________________
    sequence = power_sequence.astype(np.float32).copy()
    half_window = window_size // 2
    padded = np.pad(sequence, half_window, mode="edge")

    for i in range(len(sequence)):
        current_value = sequence[i]
        if current_value < background_threshold:
            continue

        window = padded[i:i + window_size]
        surrounding = np.concatenate([window[:half_window], window[half_window + 1:]])
        median_surrounding = np.median(surrounding)
        low_count = np.sum(surrounding < background_threshold)
        is_background_low = low_count >= len(surrounding) * 0.6

        if is_background_low and current_value > spike_threshold * median_surrounding:
            if current_value > background_threshold * 2:
                sequence[i] = 0.0

    return sequence


def close_short_off_gaps(is_on: np.ndarray, min_off_duration: int) -> np.ndarray:
    # ______________________________________________________
    # Step (B)
    # Fill very short OFF gaps between ON samples.
    # For 1-minute data, default min_off_duration = 1 step.
    # ______________________________________________________
    label = is_on.copy()
    if min_off_duration <= 0 or not np.any(label):
        return label

    on_indices = np.where(label == 1)[0]
    for i in range(len(on_indices) - 1):
        gap = on_indices[i + 1] - on_indices[i]
        if 1 < gap <= min_off_duration + 1:
            label[on_indices[i]:on_indices[i + 1] + 1] = 1
    return label


def remove_short_on_events(is_on: np.ndarray, min_on_duration: int) -> np.ndarray:
    # ______________________________________________________
    # Step (C)
    # Remove ON events that are too short.
    # For 1-minute data, default min_on_duration = 1 step.
    # ______________________________________________________
    label = is_on.copy()
    if min_on_duration <= 1 or not np.any(label):
        return label

    diff = np.diff(np.concatenate([[0], label, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for start, end in zip(starts, ends):
        if (end - start) < min_on_duration:
            label[start:end] = 0
    return label


def expand_event_window(is_on: np.ndarray, expansion_steps: int) -> np.ndarray:
    # ______________________________________________________
    # Step (D)
    # Optionally expand around ON events.
    # Default is 0, meaning exact ON/OFF only.
    # ______________________________________________________
    if expansion_steps <= 0 or not np.any(is_on):
        return is_on.astype(np.int8)

    expanded = np.zeros_like(is_on, dtype=np.int8)
    for t in np.where(is_on == 1)[0]:
        start = max(0, t - expansion_steps)
        end = min(len(is_on), t + expansion_steps + 1)
        expanded[start:end] = 1
    return expanded


def detect_on_off(power_sequence: np.ndarray, threshold: float, min_on_duration: int, remove_spikes: bool, args: argparse.Namespace) -> np.ndarray:
    # ______________________________________________________
    # Step (E)
    # Convert appliance power into 0/1 labels.
    # 1 = appliance ON, 0 = appliance OFF.
    # ______________________________________________________
    sequence = power_sequence.astype(np.float32).copy()

    if remove_spikes:
        sequence = remove_isolated_spikes(sequence)

    sequence[sequence < args.x_noise] = 0.0
    is_on = (sequence >= threshold).astype(np.int8)
    is_on = close_short_off_gaps(is_on, args.min_off_duration)
    is_on = remove_short_on_events(is_on, min_on_duration)
    is_on = expand_event_window(is_on, args.expansion_steps)
    return is_on.astype(np.int8)


def detect_appliance_from_columns(columns: Iterable[str]) -> str | None:
    for appliance in APPLIANCES:
        if appliance in columns:
            return appliance
    return None


def label_one_csv(csv_path: Path, args: argparse.Namespace) -> None:
    # ______________________________________________________
    # Step (F)
    # Read one original CSV, add or replace on_off, then save it back.
    # ______________________________________________________
    df = pd.read_csv(csv_path)
    appliance = detect_appliance_from_columns(df.columns)
    if appliance is None:
        print(f"skip: {csv_path} has no known appliance column")
        return

    threshold = args.thresholds[appliance]
    min_on_duration = args.min_on_steps.get(appliance, APPLIANCE_MIN_ON_STEPS[appliance])
    remove_spikes = False if args.no_spike_removal else APPLIANCE_REMOVE_SPIKES[appliance]
    df["on_off"] = detect_on_off(df[appliance].to_numpy(dtype=np.float32), threshold, min_on_duration, remove_spikes, args)
    df.to_csv(csv_path, index=False)
    print(f"{csv_path.name}: appliance={appliance}, threshold={threshold} W, min_on={min_on_duration} step, spike_removal={remove_spikes}, ON rate={df['on_off'].mean():.4f}")


def parse_key_values(items: Iterable[str], defaults: Dict[str, float | int], value_type: type) -> Dict[str, float | int]:
    values = dict(defaults)
    for item in items:
        name, value = item.split("=", 1)
        values[name.strip()] = value_type(value)
    return values


def main() -> None:
    # ______________________________________________________
    # Step (0)
    # Read settings.
    # This script edits CSV files in place.
    # ______________________________________________________
    script_dir = Path(__file__).resolve().parent
    code_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Add on_off column directly to source real/synthetic CSV files.")
    parser.add_argument("--real-dir", type=Path, default=code_dir / "real_data")
    parser.add_argument("--synthetic-dir", type=Path, default=code_dir / "synthetic_data")
    parser.add_argument("--threshold", action="append", default=[], help="Override threshold, e.g. kettle=2000")
    parser.add_argument("--min-on", action="append", default=[], help="Override min ON steps, e.g. dishwasher=5")
    parser.add_argument("--x-noise", type=float, default=0.0)
    parser.add_argument("--min-off-duration", type=int, default=1, help="Steps. 1 step = 1 minute here.")
    parser.add_argument("--min-on-duration", type=int, default=None, help="Fallback min ON steps. By default, use appliance-specific values.")
    parser.add_argument("--expansion-steps", type=int, default=0)
    parser.add_argument("--no-spike-removal", action="store_true")
    args = parser.parse_args()
    args.thresholds = parse_key_values(args.threshold, APPLIANCE_THRESHOLDS_WATTS, float)
    args.min_on_steps = parse_key_values(args.min_on, APPLIANCE_MIN_ON_STEPS, int)
    if args.min_on_duration is not None:
        args.min_on_steps = {appliance: args.min_on_duration for appliance in APPLIANCES}

    # ______________________________________________________
    # Step (1)
    # Process every CSV under real_data and synthetic_data.
    # ______________________________________________________
    csv_files = sorted(args.real_dir.rglob("*.csv")) + sorted(args.synthetic_dir.glob("*.csv"))
    for csv_path in csv_files:
        label_one_csv(csv_path, args)


if __name__ == "__main__":
    main()



