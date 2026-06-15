r"""
Build NILM injection-strategy datasets.

This script implements the current D1/D2/D3 experiment design.

Common aggregate construction:
    X_s = X_r - Y_r + Y_s

Methods:
    D0: Real only
    D1: Full-window append
    D2: ON-event insertion
    D3: Balanced event insertion

______________________________________________________
Step (1)
Go to the injection strategy folder
______________________________________________________

PowerShell command:
    cd "C:\Users\Raymond Tie\Desktop\PhD\Paper\Injection Strategy Conference\code\injection_strategy"

______________________________________________________
Step (2)
Build all datasets
______________________________________________________

PowerShell command:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\build_injection_datasets.py

______________________________________________________
Step (3)
Build one appliance
______________________________________________________

Example:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\build_injection_datasets.py --appliance kettle

Output folder:
    ./datasets/<appliance>/

Each NPZ contains:
    X     = aggregate + 8 time features, shape (windows, timesteps, 9)
    y     = appliance target, shape (windows, timesteps)
    state = ON/OFF label, shape (windows, timesteps)

Each CSV uses the readable format:
    aggregate,<appliance>,minute_sin,...,month_cos,on_off
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

TIME_COLUMNS = [
    "minute_sin", "minute_cos",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]

APPLIANCES = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]


# These thresholds are only fallback labels if a CSV has no on_off column.
APPLIANCE_THRESHOLDS_WATTS = {
    "dishwasher": 10.0,
    "fridge": 50.0,
    "kettle": 500.0,
    "microwave": 150.0,
    "washingmachine": 20.0,
}


# ______________________________________________________
# Step (A)
# Cut continuous rows into fixed NILM windows.
# ______________________________________________________
def sliding_windows(values: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    if len(values) < window_len:
        raise ValueError(f"Need at least {window_len} rows, got {len(values)}")
    starts = np.arange(0, len(values) - window_len + 1, stride)
    return np.stack([values[s:s + window_len] for s in starts], axis=0)


# ______________________________________________________
# Step (B)
# Load real data.
# Real data contains aggregate X_r, appliance Y_r, time features, and on_off.
# ______________________________________________________
def load_real_windows(
    csv_path: Path,
    appliance: str,
    window_len: int,
    stride: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = ["aggregate", appliance] + TIME_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    aggregate = df["aggregate"].to_numpy(dtype=np.float32)
    appliance_power = df[appliance].to_numpy(dtype=np.float32)
    time_features = df[TIME_COLUMNS].to_numpy(dtype=np.float32)

    if "on_off" in df.columns:
        on_off = df["on_off"].to_numpy(dtype=np.int8)
    else:
        on_off = (appliance_power >= threshold).astype(np.int8)

    x_rows = np.column_stack([aggregate, time_features])
    pair_rows = np.column_stack([aggregate, appliance_power, time_features])

    X_real = sliding_windows(x_rows, window_len, stride)
    y_real = sliding_windows(appliance_power[:, None], window_len, stride)[:, :, 0]
    real_pair = sliding_windows(pair_rows, window_len, stride)
    state_real = sliding_windows(on_off[:, None], window_len, stride)[:, :, 0].astype(np.int8)
    return X_real, y_real, real_pair, state_real


# ______________________________________________________
# Step (C)
# Load synthetic appliance data.
# Synthetic data contains only Y_s, time features, and on_off.
# ______________________________________________________
def load_synthetic_sequence(csv_path: Path, appliance: str, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = [appliance] + TIME_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    y = df[appliance].to_numpy(dtype=np.float32)
    time_features = df[TIME_COLUMNS].to_numpy(dtype=np.float32)
    if "on_off" in df.columns:
        state = df["on_off"].to_numpy(dtype=np.int8)
    else:
        state = (y >= threshold).astype(np.int8)
    return y, time_features, state


def load_synthetic_windows(
    csv_path: Path,
    appliance: str,
    window_len: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, time_features, state = load_synthetic_sequence(csv_path, appliance, threshold)
    if len(y) % window_len != 0:
        raise ValueError(f"{csv_path.name} rows {len(y)} not divisible by window length {window_len}")
    n = len(y) // window_len
    return (
        y.reshape(n, window_len),
        time_features.reshape(n, window_len, len(TIME_COLUMNS)),
        state.reshape(n, window_len),
    )


# ______________________________________________________
# Step (D)
# Utility samplers.
# ______________________________________________________
def sample_indices(rng: np.random.Generator, pool_size: int, n: int) -> np.ndarray:
    if pool_size <= 0:
        raise ValueError("Cannot sample from an empty pool")
    return rng.choice(pool_size, size=n, replace=n > pool_size)


def on_window_rate(state: np.ndarray) -> float:
    if len(state) == 0:
        return 0.0
    return float(state.any(axis=1).mean())


def extract_on_events(y: np.ndarray, state: np.ndarray, min_len: int = 1) -> list[dict]:
    # ______________________________________________________
    # Step (E)
    # Extract contiguous synthetic ON events from the labelled synthetic signal.
    # Each record keeps the original event location before insertion.
    # ______________________________________________________
    mask = state.astype(np.int8)
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    events = []
    for event_id, (start, end) in enumerate(zip(starts, ends)):
        if (end - start) >= min_len:
            events.append({
                "event_id": event_id,
                "source_start": int(start),
                "source_end": int(end),
                "values": y[start:end].astype(np.float32),
            })
    if not events:
        raise ValueError("No synthetic ON events found. Check synthetic on_off labels.")
    return events


def find_off_runs(state_window: np.ndarray, length: int) -> np.ndarray:
    # Return valid start positions where the real target appliance is OFF for length samples.
    if length > len(state_window):
        return np.array([], dtype=int)
    valid = []
    for start in range(0, len(state_window) - length + 1):
        if state_window[start:start + length].sum() == 0:
            valid.append(start)
    return np.array(valid, dtype=int)


def full_off_window_indices(state_windows: np.ndarray) -> np.ndarray:
    # Windows where the target appliance is OFF for the whole window.
    return np.where(state_windows.sum(axis=1) == 0)[0]


# ______________________________________________________
# Step (F)
# D1: Full-window append.
# Select full synthetic windows, then construct X_s = X_r - Y_r + Y_s.
# ______________________________________________________
def make_full_window_samples(
    rng: np.random.Generator,
    real_pair_windows: np.ndarray,
    y_synth_windows: np.ndarray,
    state_synth_windows: np.ndarray,
    n_syn: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_syn == 0:
        t = real_pair_windows.shape[1]
        return np.empty((0, t, 9), dtype=np.float32), np.empty((0, t), dtype=np.float32), np.empty((0, t), dtype=np.int8)

    syn_idx = sample_indices(rng, len(y_synth_windows), n_syn)
    bg_idx = sample_indices(rng, len(real_pair_windows), n_syn)

    real_bg = real_pair_windows[bg_idx]
    x_r = real_bg[:, :, 0]
    y_r = real_bg[:, :, 1]
    time_features = real_bg[:, :, 2:]

    y_s = y_synth_windows[syn_idx].astype(np.float32)
    state_s = state_synth_windows[syn_idx].astype(np.int8)
    x_s = x_r - y_r + y_s
    X_s = np.concatenate([x_s[:, :, None], time_features], axis=2).astype(np.float32)
    return X_s, y_s, state_s


# ______________________________________________________
# Step (G)
# D2: ON-event insertion.
# Cut synthetic ON events and insert them into real OFF background windows.
# ______________________________________________________
def make_event_inserted_samples(
    rng: np.random.Generator,
    real_pair_windows: np.ndarray,
    state_real_windows: np.ndarray,
    events: list[dict],
    n_syn: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    window_len = real_pair_windows.shape[1]
    if n_syn == 0:
        empty_meta = pd.DataFrame(columns=[
            "synthetic_sample_index", "source_event_id", "source_start", "source_end",
            "source_length", "used_start_in_source", "used_length",
            "real_background_window", "background_rule", "insert_at", "insert_end",
            "original_background_energy", "inserted_event_energy", "final_aggregate_energy",
        ])
        return (
            np.empty((0, window_len, 9), dtype=np.float32),
            np.empty((0, window_len), dtype=np.float32),
            np.empty((0, window_len), dtype=np.int8),
            empty_meta,
        )

    X_out = []
    y_out = []
    state_out = []
    meta_rows = []
    full_off_candidates = full_off_window_indices(state_real_windows)

    # Prefer full target-OFF windows. This avoids removing real target events outside
    # the inserted synthetic event. If none exist, fall back to interval-OFF insertion.
    for sample_i in range(n_syn):
        event_record = events[int(rng.integers(0, len(events)))]
        event = event_record["values"]
        used_start_in_source = 0
        if len(event) > window_len:
            used_start_in_source = int(rng.integers(0, len(event) - window_len + 1))
            event = event[used_start_in_source:used_start_in_source + window_len]

        event_len = len(event)
        chosen = None
        background_rule = "full_window_off"
        if len(full_off_candidates) > 0:
            bg_idx = int(full_off_candidates[int(rng.integers(0, len(full_off_candidates)))])
            insert_at = int(rng.integers(0, window_len - event_len + 1))
            chosen = (bg_idx, insert_at)
        else:
            background_rule = "insert_interval_off"
            for _try in range(200):
                bg_idx = int(rng.integers(0, len(real_pair_windows)))
                starts = find_off_runs(state_real_windows[bg_idx], event_len)
                if len(starts) > 0:
                    insert_at = int(starts[int(rng.integers(0, len(starts)))])
                    chosen = (bg_idx, insert_at)
                    break

        # Fallback: if no OFF gap is found, use an all-OFF position from the least active real window.
        if chosen is None:
            background_rule = "least_active_fallback"
            off_counts = (state_real_windows == 0).sum(axis=1)
            bg_idx = int(np.argmax(off_counts))
            starts = find_off_runs(state_real_windows[bg_idx], min(event_len, window_len))
            insert_at = int(starts[0]) if len(starts) else 0
            event = event[:window_len - insert_at]
            event_len = len(event)
            chosen = (bg_idx, insert_at)

        bg_idx, insert_at = chosen
        real_bg = real_pair_windows[bg_idx]
        x_r = real_bg[:, 0]
        y_r = real_bg[:, 1]
        time_features = real_bg[:, 2:]

        background = x_r - y_r
        y_insert = np.zeros(window_len, dtype=np.float32)
        state_insert = np.zeros(window_len, dtype=np.int8)
        y_insert[insert_at:insert_at + event_len] = event
        state_insert[insert_at:insert_at + event_len] = 1

        x_s = background + y_insert
        X_s = np.concatenate([x_s[:, None], time_features], axis=1).astype(np.float32)

        meta_rows.append({
            "synthetic_sample_index": sample_i,
            "source_event_id": int(event_record["event_id"]),
            "source_start": int(event_record["source_start"]),
            "source_end": int(event_record["source_end"]),
            "source_length": int(event_record["source_end"] - event_record["source_start"]),
            "used_start_in_source": int(used_start_in_source),
            "used_length": int(event_len),
            "real_background_window": int(bg_idx),
            "background_rule": background_rule,
            "insert_at": int(insert_at),
            "insert_end": int(insert_at + event_len),
            "original_background_energy": float(background.sum()),
            "inserted_event_energy": float(y_insert.sum()),
            "final_aggregate_energy": float(x_s.sum()),
        })

        X_out.append(X_s)
        y_out.append(y_insert)
        state_out.append(state_insert)

    return np.stack(X_out), np.stack(y_out), np.stack(state_out), pd.DataFrame(meta_rows)


# ______________________________________________________
# Step (H)
# Save readable CSV (training code loads CSV only).
# ______________________________________________________
def save_csv_like_original(path: Path, X: np.ndarray, y: np.ndarray, state: np.ndarray, appliance: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = X[:, :, 0].reshape(-1)
    appliance_power = y.reshape(-1)
    time_features = X[:, :, 1:].reshape(-1, len(TIME_COLUMNS))
    df = pd.DataFrame(np.column_stack([aggregate, appliance_power, time_features]), columns=["aggregate", appliance] + TIME_COLUMNS)
    df["on_off"] = state.reshape(-1).astype(np.int8)
    df.to_csv(path, index=False)


def save_dataset(path: Path, X: np.ndarray, y: np.ndarray, state: np.ndarray, appliance: str) -> None:
    csv_path = path if path.suffix == ".csv" else path.with_suffix(".csv")
    save_csv_like_original(csv_path, X, y, state, appliance)


def save_event_metadata(path: Path, event_meta: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_meta.to_csv(path, index=False)


def combine(real_X: np.ndarray, real_y: np.ndarray, real_state: np.ndarray, syn_X: np.ndarray, syn_y: np.ndarray, syn_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([real_X, syn_X], axis=0),
        np.concatenate([real_y, syn_y], axis=0),
        np.concatenate([real_state, syn_state], axis=0),
    )


# ______________________________________________________
# Step (I)
# Build all D0/D1/D2/D3 datasets for one appliance.
# ______________________________________________________
def build_for_appliance(args: argparse.Namespace, appliance: str) -> None:
    rng = np.random.default_rng(args.seed)
    threshold = args.thresholds.get(appliance, APPLIANCE_THRESHOLDS_WATTS[appliance])

    train_csv = args.real_dir / "training" / f"{appliance}_train_100k.csv"
    val_csv = args.real_dir / "validating" / f"{appliance}_val_50k.csv"
    test_csv = args.real_dir / "testing" / "house1" / f"{appliance}_test_50k.csv"
    house2_csv = args.real_dir / "testing" / "house2" / f"{appliance}_house2_test_50k.csv"
    synth_csv = args.synthetic_dir / f"{appliance}_synthetic.csv"

    X_train, y_train, train_pair, state_train = load_real_windows(train_csv, appliance, args.window_len, args.stride, threshold)
    X_val, y_val, _, state_val = load_real_windows(val_csv, appliance, args.window_len, args.stride, threshold)
    X_test_h1, y_test_h1, _, state_test_h1 = load_real_windows(test_csv, appliance, args.window_len, args.stride, threshold)
    X_test_h2, y_test_h2, _, state_test_h2 = load_real_windows(house2_csv, appliance, args.window_len, args.stride, threshold)

    y_synth_windows, _, state_synth_windows = load_synthetic_windows(synth_csv, appliance, args.window_len, threshold)
    y_synth_seq, _, state_synth_seq = load_synthetic_sequence(synth_csv, appliance, threshold)
    events = extract_on_events(y_synth_seq, state_synth_seq, min_len=args.min_event_len)

    n_real = len(X_train)
    split_dir = args.output_dir / appliance

    print(f"\n{appliance}: real windows={n_real}, synthetic windows={len(y_synth_windows)}, events={len(events)}")
    print(f"  real ON-window rate={on_window_rate(state_train):.3f}")
    print(f"  full synthetic ON-window rate={on_window_rate(state_synth_windows):.3f}")

    save_dataset(split_dir / "val_house1.csv", X_val, y_val, state_val, appliance)
    save_dataset(split_dir / "test_house1.csv", X_test_h1, y_test_h1, state_test_h1, appliance)
    save_dataset(split_dir / "test_house2.csv", X_test_h2, y_test_h2, state_test_h2, appliance)
    save_dataset(split_dir / "train_real_only.csv", X_train, y_train, state_train, appliance)

    # Experiment 1: D1/D2/D3 at rho = 100%.
    n_syn = n_real

    X_full, y_full, state_full = make_full_window_samples(rng, train_pair, y_synth_windows, state_synth_windows, n_syn)
    X_aug, y_aug, state_aug = combine(X_train, y_train, state_train, X_full, y_full, state_full)
    save_dataset(split_dir / "train_d1_full_window_append_100.csv", X_aug, y_aug, state_aug, appliance)
    print(f"  D1 full-window append 100: X={X_aug.shape}, synthetic ON-window rate={on_window_rate(state_full):.3f}")

    X_event, y_event, state_event, event_meta = make_event_inserted_samples(rng, train_pair, state_train, events, n_syn)
    X_aug, y_aug, state_aug = combine(X_train, y_train, state_train, X_event, y_event, state_event)
    save_dataset(split_dir / "train_d2_on_event_insertion_100.csv", X_aug, y_aug, state_aug, appliance)
    save_event_metadata(split_dir / "train_d2_on_event_insertion_100_event_metadata.csv", event_meta)
    print(f"  D2 ON-event insertion 100: X={X_aug.shape}, synthetic ON-window rate={on_window_rate(state_event):.3f}")

    X_bal, y_bal, state_bal, event_meta = make_balanced_samples(rng, train_pair, state_train, y_synth_windows, state_synth_windows, events, n_syn)
    X_aug, y_aug, state_aug = combine(X_train, y_train, state_train, X_bal, y_bal, state_bal)
    save_dataset(split_dir / "train_d3_balanced_event_insertion_100.csv", X_aug, y_aug, state_aug, appliance)
    save_event_metadata(split_dir / "train_d3_balanced_event_insertion_100_event_metadata.csv", event_meta)
    print(f"  D3 balanced event insertion 100: X={X_aug.shape}, synthetic ON-window rate={on_window_rate(state_bal):.3f}")

    # Experiment 2: ratio sensitivity for D3 only.
    for ratio in args.ratios:
        if ratio == 0 or abs(ratio - 1.0) < 1e-12:
            continue
        n_syn = int(round(ratio * n_real))
        X_bal, y_bal, state_bal, event_meta = make_balanced_samples(rng, train_pair, state_train, y_synth_windows, state_synth_windows, events, n_syn)
        X_aug, y_aug, state_aug = combine(X_train, y_train, state_train, X_bal, y_bal, state_bal)
        pct = int(round(ratio * 100))
        save_dataset(split_dir / f"train_d3_balanced_event_insertion_{pct}.csv", X_aug, y_aug, state_aug, appliance)
        save_event_metadata(split_dir / f"train_d3_balanced_event_insertion_{pct}_event_metadata.csv", event_meta)
        print(f"  D3 balanced event insertion {pct}: X={X_aug.shape}, synthetic ON-window rate={on_window_rate(state_bal):.3f}")


def make_balanced_samples(
    rng: np.random.Generator,
    real_pair_windows: np.ndarray,
    state_real_windows: np.ndarray,
    y_synth_windows: np.ndarray,
    state_synth_windows: np.ndarray,
    events: list[dict],
    n_syn: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    # D3 = 50% D2 event-inserted samples + 50% D1 full-window samples.
    n_event = n_syn // 2
    n_full = n_syn - n_event
    X_event, y_event, state_event, event_meta = make_event_inserted_samples(rng, real_pair_windows, state_real_windows, events, n_event)
    X_full, y_full, state_full = make_full_window_samples(rng, real_pair_windows, y_synth_windows, state_synth_windows, n_full)
    return (
        np.concatenate([X_event, X_full], axis=0),
        np.concatenate([y_event, y_full], axis=0),
        np.concatenate([state_event, state_full], axis=0),
        event_meta,
    )


def parse_thresholds(items: Iterable[str]) -> Dict[str, float]:
    thresholds = dict(APPLIANCE_THRESHOLDS_WATTS)
    for item in items:
        name, value = item.split("=", 1)
        thresholds[name.strip()] = float(value)
    return thresholds


def main() -> None:
    default_code_dir = Path(__file__).resolve().parents[1]
    default_data_dir = default_code_dir / "data" if (default_code_dir / "data").exists() else default_code_dir

    parser = argparse.ArgumentParser(description="Build D1/D2/D3 injection-strategy NILM datasets.")
    parser.add_argument("--real-dir", type=Path, default=default_data_dir / "real_data")
    parser.add_argument("--synthetic-dir", type=Path, default=default_data_dir / "synthetic_data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "datasets")
    parser.add_argument("--appliance", choices=["all"] + APPLIANCES, default="all")
    parser.add_argument("--window-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.50, 1.00, 2.00])
    parser.add_argument("--min-event-len", type=int, default=1)
    parser.add_argument("--threshold", action="append", default=[], help="Fallback threshold, e.g. --threshold kettle=500")
    args = parser.parse_args()
    args.thresholds = parse_thresholds(args.threshold)

    appliances = APPLIANCES if args.appliance == "all" else [args.appliance]
    for appliance in appliances:
        build_for_appliance(args, appliance)


if __name__ == "__main__":
    main()
