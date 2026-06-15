r"""
Preview NILM augmented waveform files.

This script is only for checking the situation inside .npz files.
It does not train the model and does not change the dataset.

______________________________________________________
Step (1)
Go to the injection strategy folder
______________________________________________________

PowerShell command:
    cd "C:\Users\Raymond Tie\Desktop\PhD\Paper\Injection Strategy Conference\code\injection_strategy"

______________________________________________________
Step (2)
Preview one appliance and one experiment
______________________________________________________

Example:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\plot_augmented_waveforms.py --appliance kettle --dataset train_balanced_100

This creates PNG and CSV files under:
    .\waveform_preview\kettle\train_balanced_100\

______________________________________________________
Step (3)
Preview all training experiments for one appliance
______________________________________________________

Example:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\plot_augmented_waveforms.py --appliance kettle --dataset all

______________________________________________________
Step (4)
Preview more windows
______________________________________________________

Example:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\plot_augmented_waveforms.py --appliance kettle --dataset train_balanced_100 --num-windows 12

Output CSV columns:
    window_id
    timestep
    aggregate
    appliance
    background_estimate = aggregate - appliance
    8 time-feature columns
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TIME_COLUMNS = [
    "minute_sin", "minute_cos",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]

APPLIANCES = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]

TRAIN_DATASETS = [
    "train_real_only",
    "train_on_focused_100",
    "train_full_distribution_100",
    "train_balanced_25",
    "train_balanced_50",
    "train_balanced_100",
    "train_balanced_200",
]


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # ______________________________________________________
    # Step (A)
    # Load one .npz dataset.
    # X contains aggregate + time features.
    # y contains target appliance power.
    # ______________________________________________________
    data = np.load(path)
    return data["X"], data["y"]


def get_real_window_count(dataset_dir: Path) -> int:
    # ______________________________________________________
    # Step (B)
    # Find where synthetic windows start.
    # In the builder script, augmented training data is saved as:
    # first = real windows, second = synthetic windows.
    # ______________________________________________________
    real_path = dataset_dir / "train_real_only.npz"
    if not real_path.exists():
        return 0
    X_real, _ = load_npz(real_path)
    return len(X_real)


def choose_window_ids(total: int, real_count: int, num_windows: int, mode: str) -> list[int]:
    # ______________________________________________________
    # Step (C)
    # Choose which windows to preview.
    # synthetic mode starts after the real-only part.
    # mixed mode shows some real and some synthetic windows.
    # ______________________________________________________
    if total == 0:
        return []

    if mode == "real":
        start = 0
        end = min(real_count, total)
    elif mode == "synthetic":
        start = min(real_count, total - 1)
        end = total
    else:
        start = 0
        end = total

    if end <= start:
        start = 0
        end = total

    count = min(num_windows, end - start)
    if count <= 0:
        return []

    return np.linspace(start, end - 1, count, dtype=int).tolist()


def make_window_table(X: np.ndarray, y: np.ndarray, window_ids: Iterable[int]) -> pd.DataFrame:
    # ______________________________________________________
    # Step (D)
    # Convert selected windows into readable CSV rows.
    # Each row is one timestep inside one window.
    # ______________________________________________________
    rows = []
    for window_id in window_ids:
        aggregate = X[window_id, :, 0]
        appliance = y[window_id]
        background = aggregate - appliance
        time_features = X[window_id, :, 1:]

        for t in range(X.shape[1]):
            row = {
                "window_id": window_id,
                "timestep": t,
                "aggregate": float(aggregate[t]),
                "appliance": float(appliance[t]),
                "background_estimate": float(background[t]),
            }
            for i, col in enumerate(TIME_COLUMNS):
                row[col] = float(time_features[t, i])
            rows.append(row)

    return pd.DataFrame(rows)


def _svg_polyline(values: list[float], color: str, width: int, height: int, min_y: float, max_y: float) -> str:
    if len(values) <= 1:
        return ""
    span = max(max_y - min_y, 1e-9)
    points = []
    for i, value in enumerate(values):
        x = 30 + i * (width - 50) / (len(values) - 1)
        y = 10 + (max_y - value) * (height - 30) / span
        points.append(f"{x:.1f},{y:.1f}")
    return f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.6" />'


def save_html_plots(table: pd.DataFrame, output_dir: Path, appliance: str) -> None:
    # ______________________________________________________
    # Step (E1)
    # Save no-dependency HTML waveform plots.
    # This works even when matplotlib is not installed.
    # Open waveform_preview.html in a browser to inspect the signal.
    # ______________________________________________________
    width = 920
    height = 260
    sections = []
    for window_id, part in table.groupby("window_id"):
        aggregate = part["aggregate"].tolist()
        appliance_y = part["appliance"].tolist()
        background = part["background_estimate"].tolist()
        min_y = min(aggregate + appliance_y + background)
        max_y = max(aggregate + appliance_y + background)
        svg = "\n".join([
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="260" role="img">',
            '<rect x="0" y="0" width="100%" height="100%" fill="white" stroke="#ddd"/>',
            _svg_polyline(aggregate, "#1f77b4", width, height, min_y, max_y),
            _svg_polyline(appliance_y, "#d62728", width, height, min_y, max_y),
            _svg_polyline(background, "#2ca02c", width, height, min_y, max_y),
            f'<text x="30" y="24" font-size="14" font-family="Arial">window {int(window_id)} | min={min_y:.3f}, max={max_y:.3f}</text>',
            '</svg>',
        ])
        sections.append(f"<h3>Window {int(window_id)}</h3>\n{svg}")

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{appliance} waveform preview</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
.legend span {{ display: inline-block; margin-right: 18px; }}
.blue {{ color: #1f77b4; }} .red {{ color: #d62728; }} .green {{ color: #2ca02c; }}
h3 {{ margin-top: 28px; margin-bottom: 8px; }}
</style>
</head>
<body>
<h2>{appliance} waveform preview</h2>
<div class="legend">
<span class="blue">aggregate</span>
<span class="red">appliance</span>
<span class="green">background = aggregate - appliance</span>
</div>
{''.join(sections)}
</body>
</html>
"""
    (output_dir / "waveform_preview.html").write_text(html, encoding="utf-8")

def save_plots(table: pd.DataFrame, output_dir: Path, appliance: str) -> None:
    # ______________________________________________________
    # Step (E)
    # Save one PNG waveform plot for each selected window.
    # If matplotlib is missing, the CSV preview still works.
    # ______________________________________________________
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so PNG plots are skipped. CSV preview is still saved.")
        return

    for window_id, part in table.groupby("window_id"):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(part["timestep"], part["aggregate"], label="aggregate", linewidth=1.4)
        ax.plot(part["timestep"], part["appliance"], label=appliance, linewidth=1.2)
        ax.plot(part["timestep"], part["background_estimate"], label="background", linewidth=1.0, alpha=0.75)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Power")
        ax.set_title(f"{appliance} window {window_id}")
        ax.legend(loc="upper right")
        ax.grid(True, linewidth=0.4, alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_dir / f"window_{int(window_id):04d}.png", dpi=160)
        plt.close(fig)


def preview_dataset(args: argparse.Namespace, dataset_name: str) -> None:
    # ______________________________________________________
    # Step (F)
    # Preview one dataset, such as train_balanced_100.
    # It saves both CSV and PNG files.
    # ______________________________________________________
    dataset_dir = args.dataset_dir / args.appliance
    npz_path = dataset_dir / f"{dataset_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    X, y = load_npz(npz_path)
    real_count = get_real_window_count(dataset_dir)
    window_ids = choose_window_ids(len(X), real_count, args.num_windows, args.mode)

    output_dir = args.output_dir / args.appliance / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    table = make_window_table(X, y, window_ids)
    csv_path = output_dir / "waveform_preview.csv"
    table.to_csv(csv_path, index=False)
    save_plots(table, output_dir, args.appliance)

    print(f"{args.appliance}/{dataset_name}: X={X.shape}, y={y.shape}")
    print(f"  real windows before synthetic part: {real_count}")
    print(f"  preview window ids: {window_ids}")
    print(f"  saved CSV: {csv_path}")
    print(f"  saved HTML: {output_dir / 'waveform_preview.html'}")
    print(f"  saved PNG folder: {output_dir}")


def main() -> None:
    # ______________________________________________________
    # Step (0)
    # Command that starts this script:
    # cd "C:\Users\Raymond Tie\Desktop\PhD\Paper\Injection Strategy Conference\code\injection_strategy"
    # & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\plot_augmented_waveforms.py --appliance kettle --dataset train_balanced_100
    # ______________________________________________________
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Preview augmented NILM waveform datasets.")
    parser.add_argument("--dataset-dir", type=Path, default=script_dir / "datasets")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "waveform_preview")
    parser.add_argument("--appliance", choices=APPLIANCES, required=True)
    parser.add_argument("--dataset", default="train_balanced_100", help="Dataset name without .npz, or all")
    parser.add_argument("--num-windows", type=int, default=6)
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real", "mixed"],
        default="synthetic",
        help="Which part to preview. For augmented train files, synthetic starts after train_real_only length.",
    )
    args = parser.parse_args()

    # ______________________________________________________
    # Step (1)
    # Decide whether to preview one dataset or all training datasets.
    # ______________________________________________________
    dataset_names = TRAIN_DATASETS if args.dataset == "all" else [args.dataset]
    for dataset_name in dataset_names:
        preview_dataset(args, dataset_name)


if __name__ == "__main__":
    main()


