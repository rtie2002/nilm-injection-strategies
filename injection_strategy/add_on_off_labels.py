r"""
Add ON/OFF labels to all injection-strategy datasets.

This script reads files from:
    ./datasets/<appliance>/

It writes labelled copies to:
    ./datasets_labeled/<appliance>/

CSV output:
    aggregate,<appliance>,minute_sin,...,month_cos,on_off

NPZ output:
    X = aggregate + time features
    y = appliance power
    state = ON/OFF label, same shape as y

______________________________________________________
Step (1)
Go to the injection strategy folder
______________________________________________________

PowerShell command:
    cd "C:\Users\Raymond Tie\Desktop\PhD\Paper\Injection Strategy Conference\code\injection_strategy"

______________________________________________________
Step (2)
Run ON/OFF labelling for all appliances
______________________________________________________

PowerShell command:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\add_on_off_labels.py

______________________________________________________
Step (3)
Run only one appliance if needed
______________________________________________________

Example:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\add_on_off_labels.py --appliance kettle

______________________________________________________
Step (4)
Change ON threshold if needed
______________________________________________________

Example:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\add_on_off_labels.py --threshold kettle=2000 --threshold microwave=200
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
    "kettle": 2000.0,
    "microwave": 200.0,
    "washingmachine": 20.0,
}


def remove_isolated_spikes(
    power_sequence: np.ndarray,
    window_size: int = 5,
    spike_threshold: float = 3.0,
    background_threshold: float = 50.0,
) -> np.ndarray:
    # ______________________________________________________
    # Step (A)
    # Remove isolated high spikes before ON/OFF detection.
    # This follows the idea in ukdale_processing.py.
    # ______________________________________________________
    sequence = power_sequence.astype(np.float32).copy()
    n = len(sequence)
    half_window = window_size // 2
    padded = np.pad(sequence, half_window, mode="edge")

    for i in range(n):
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
    # If an appliance is ON, briefly OFF, then ON again,
    # fill the short OFF gap as ON.
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
    # Remove very short ON activations.
    # With 1-minute data, default min_on_duration = 1 step.
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
    # Expand around detected ON points.
    # Use 0 for exact ON only. Use 1 or 2 to include nearby context.
    # ______________________________________________________
    if expansion_steps <= 0 or not np.any(is_on):
        return is_on.astype(np.int8)

    expanded = np.zeros_like(is_on, dtype=np.int8)
    for t in np.where(is_on == 1)[0]:
        start = max(0, t - expansion_steps)
        end = min(len(is_on), t + expansion_steps + 1)
        expanded[start:end] = 1
    return expanded


def detect_on_off(
    power_sequence: np.ndarray,
    threshold: float,
    x_noise: float,
    remove_spikes: bool,
    min_off_duration: int,
    min_on_duration: int,
    expansion_steps: int,
) -> np.ndarray:
    # ______________________________________________________
    # Step (E)
    # Convert appliance power into 0/1 ON/OFF labels.
    # 1 = appliance ON, 0 = appliance OFF.
    # ______________________________________________________
    sequence = power_sequence.astype(np.float32).copy()

    if remove_spikes:
        sequence = remove_isolated_spikes(sequence)

    sequence[sequence < x_noise] = 0.0
    is_on = (sequence >= threshold).astype(np.int8)
    is_on = close_short_off_gaps(is_on, min_off_duration)
    is_on = remove_short_on_events(is_on, min_on_duration)
    is_on = expand_event_window(is_on, expansion_steps)
    return is_on.astype(np.int8)


def label_csv(csv_path: Path, output_path: Path, appliance: str, args: argparse.Namespace) -> None:
    # ______________________________________________________
    # Step (F)
    # Process one CSV dataset.
    # Add one new column named on_off.
    # ______________________________________________________
    df = pd.read_csv(csv_path)
    if appliance not in df.columns:
        print(f"  skip CSV without appliance column: {csv_path.name}")
        return

    threshold = args.thresholds[appliance]
    df["on_off"] = detect_on_off(
        df[appliance].to_numpy(dtype=np.float32),
        threshold=threshold,
        x_noise=args.x_noise,
        remove_spikes=not args.no_spike_removal,
        min_off_duration=args.min_off_duration,
        min_on_duration=args.min_on_duration,
        expansion_steps=args.expansion_steps,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  CSV  {csv_path.name} -> {output_path.name}, ON rate={df['on_off'].mean():.3f}")


def label_npz(npz_path: Path, output_path: Path, appliance: str, args: argparse.Namespace) -> None:
    # ______________________________________________________
    # Step (G)
    # Process one NPZ dataset.
    # Save state as a 0/1 array with the same shape as y.
    # ______________________________________________________
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]

    threshold = args.thresholds[appliance]
    flat_state = detect_on_off(
        y.reshape(-1),
        threshold=threshold,
        x_noise=args.x_noise,
        remove_spikes=not args.no_spike_removal,
        min_off_duration=args.min_off_duration,
        min_on_duration=args.min_on_duration,
        expansion_steps=args.expansion_steps,
    )
    state = flat_state.reshape(y.shape)

    meta = {key: data[key] for key in data.files if key not in {"X", "y"}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, state=state, **meta)
    print(f"  NPZ  {npz_path.name} -> {output_path.name}, ON rate={state.mean():.3f}")


def process_appliance(args: argparse.Namespace, appliance: str) -> None:
    # ______________________________________________________
    # Step (1)
    # Process one appliance folder inside ./datasets.
    # ______________________________________________________
    input_dir = args.input_dir / appliance
    output_dir = args.output_dir / appliance

    if not input_dir.exists():
        print(f"skip missing folder: {input_dir}")
        return

    print(f"\n{appliance}: threshold={args.thresholds[appliance]} W")

    # ______________________________________________________
    # Step (2)
    # Add on_off label to every CSV file.
    # Output format remains similar to original data, plus on_off.
    # ______________________________________________________
    for csv_path in sorted(input_dir.glob("*.csv")):
        label_csv(csv_path, output_dir / csv_path.name, appliance, args)

    # ______________________________________________________
    # Step (3)
    # Add state array to every NPZ file.
    # This lets model code use labels later for F1-score.
    # ______________________________________________________
    for npz_path in sorted(input_dir.glob("*.npz")):
        label_npz(npz_path, output_dir / npz_path.name, appliance, args)


def parse_thresholds(items: Iterable[str]) -> Dict[str, float]:
    thresholds = dict(APPLIANCE_THRESHOLDS_WATTS)
    for item in items:
        name, value = item.split("=", 1)
        thresholds[name.strip()] = float(value)
    return thresholds


def main() -> None:
    # ______________________________________________________
    # Step (0)
    # Read command-line settings.
    # Default: process all appliances in ./datasets.
    # ______________________________________________________
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Add ON/OFF labels to injection-strategy datasets.")
    parser.add_argument("--input-dir", type=Path, default=script_dir / "datasets")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "datasets_labeled")
    parser.add_argument("--appliance", choices=["all"] + APPLIANCES, default="all")
    parser.add_argument("--threshold", action="append", default=[], help="Override threshold, e.g. kettle=2000")
    parser.add_argument("--x-noise", type=float, default=0.0)
    parser.add_argument("--min-off-duration", type=int, default=1, help="Steps. 1 step = 1 minute here.")
    parser.add_argument("--min-on-duration", type=int, default=1, help="Steps. 1 step = 1 minute here.")
    parser.add_argument("--expansion-steps", type=int, default=0, help="Steps around ON events. 0 = no expansion.")
    parser.add_argument("--no-spike-removal", action="store_true")
    args = parser.parse_args()
    args.thresholds = parse_thresholds(args.threshold)

    # ______________________________________________________
    # Step (4)
    # Run appliance by appliance.
    # ______________________________________________________
    appliances = APPLIANCES if args.appliance == "all" else [args.appliance]
    for appliance in appliances:
        process_appliance(args, appliance)


if __name__ == "__main__":
    main()
