r"""
Visualize Geng mix CSV files with real vs synthetic background colors.

Expects labeled CSV from build_geng_mix.py:
    aggregate,<appliance>,source
    source: 0 = real (green background), 1 = synthetic (light red background)

Examples:
    python data/geng_mix_visualize.py DiffusionModel_NILM/NILM-main/dataset_preprocess/created_data/UK_DALE/kettle/UK_DALECombinedkettle_file20_labeled.csv

    python data/geng_mix_visualize.py kettle/UK_DALECombinedkettle_file20_labeled.csv --span 2048

    python data/geng_mix_visualize.py --path UK_DALECombinedfridge_file10_labeled.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, Slider

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DIFFUSION_DIR = BASE_DIR / "DiffusionModel_NILM"
DEFAULT_MIX_ROOT = (
    DIFFUSION_DIR / "NILM-main" / "dataset_preprocess" / "created_data" / "UK_DALE"
)

APPLIANCES = ("dishwasher", "fridge", "kettle", "microwave", "washingmachine")

REAL_COLOR = "#c8f7c5"      # light green
SYN_COLOR = "#ffd6d6"       # light red


def resolve_csv_path(path_str: str) -> Path:
    raw = path_str.strip().strip('"').strip("'")
    p = Path(raw)
    if p.is_file():
        return p.resolve()

    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                Path.cwd() / p,
                BASE_DIR / p,
                SCRIPT_DIR / p,
                DIFFUSION_DIR / p,
                DEFAULT_MIX_ROOT / p,
            ]
        )
        # e.g. kettle/UK_DALECombinedkettle_file20_labeled.csv
        if len(p.parts) == 1:
            for app in APPLIANCES:
                candidates.append(DEFAULT_MIX_ROOT / app / p.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"CSV not found: {raw}\nTried:\n  {tried}")


def detect_appliance_column(df: pd.DataFrame, file_path: Path) -> str | None:
    for appliance in APPLIANCES:
        if appliance in df.columns:
            return appliance
    name = file_path.name.lower()
    for appliance in APPLIANCES:
        if appliance in name:
            return appliance
    return None


def find_source_segments(source: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (start, end, source_value) for contiguous runs."""
    src = np.asarray(source).astype(int)
    if len(src) == 0:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(src)):
        if src[i] != src[i - 1]:
            segments.append((start, i, int(src[i - 1])))
            start = i
    segments.append((start, len(src), int(src[-1])))
    return segments


def interactive_viewer(file_path: Path, initial_span: int = 2048) -> None:
    print(f"\nLoading: {file_path}")
    df = pd.read_csv(file_path)
    total_points = len(df)
    if total_points == 0:
        print("CSV is empty.")
        return

    if "source" not in df.columns:
        raise ValueError(
            "CSV must have a 'source' column (0=real, 1=synthetic). "
            "Run build_geng_mix.py to generate *_labeled.csv files."
        )

    appliance = detect_appliance_column(df, file_path)
    plot_columns: list[str] = []
    if "aggregate" in df.columns:
        plot_columns.append("aggregate")
    if appliance and appliance in df.columns:
        plot_columns.append(appliance)
    if not plot_columns:
        plot_columns = [
            c
            for c in df.columns
            if c != "source" and pd.api.types.is_numeric_dtype(df[c])
        ]

    sequences = {col: df[col].to_numpy(dtype=float) for col in plot_columns}
    source = df["source"].to_numpy(dtype=int)
    segments = find_source_segments(source)

    n_real = int((source == 0).sum())
    n_syn = int((source == 1).sum())

    print(f"Total points: {total_points:,}")
    print(f"Appliance: {appliance or 'unknown'}")
    print(f"Real rows (source=0, first block): {n_real:,}")
    print(f"Synthetic rows (source=1, second block): {n_syn:,}")
    print("Row order: real block then syn block (Geng concat; NILM shuffles windows at train time)")
    print(f"Plotted: {', '.join(plot_columns)}")

    state = {
        "start_idx": 0,
        "view_span": min(initial_span, total_points),
        "show_source": True,
        "patches": [],
        "labels": [],
        "selection_rect": None,
        "sel_start": None,
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.30, left=0.08, right=0.95, top=0.90)
    legend_text = fig.text(
        0.5,
        0.02,
        "Green = real (source 0)  |  Light red = synthetic (source 1)",
        ha="center",
        fontsize=10,
        color="#333333",
    )

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    lines: dict[str, object] = {}
    label_to_legline: dict[str, object] = {}

    end_idx = min(state["view_span"], total_points)
    x_range = np.arange(0, end_idx)
    for i, (label, values) in enumerate(sequences.items()):
        line, = ax.plot(
            x_range,
            values[:end_idx],
            label=label,
            color=colors[i % len(colors)],
            alpha=0.9,
            linewidth=1.2,
            picker=5,
        )
        lines[label] = line

    legend = ax.legend(loc="upper right")
    for legline in legend.get_lines():
        legline.set_picker(5)
        label_to_legline[legline.get_label()] = legline

    ax.grid(True, alpha=0.25, zorder=2)
    ax.set_xlabel("Row index (6 s timestep)")
    ax.set_ylabel("Power (W)")

    max_start = max(0, total_points - 1)
    ax_pos = plt.axes([0.1, 0.12, 0.5, 0.03])
    pos_slider = Slider(ax_pos, "Start", 0, max_start, valinit=0, valstep=1, valfmt="%d")

    ax_span = plt.axes([0.1, 0.07, 0.5, 0.03])
    span_slider = Slider(
        ax_span,
        "Span",
        10,
        max(10, min(total_points, 100_000)),
        valinit=state["view_span"],
        valstep=10,
        valfmt="%d",
    )

    ax_prev = plt.axes([0.65, 0.09, 0.08, 0.04])
    ax_next = plt.axes([0.74, 0.09, 0.08, 0.04])
    ax_fit = plt.axes([0.83, 0.09, 0.12, 0.04])
    ax_toggle = plt.axes([0.65, 0.04, 0.30, 0.04])

    btn_prev = Button(ax_prev, "Back")
    btn_next = Button(ax_next, "Forward")
    btn_fit = Button(ax_fit, "Fit Y")
    btn_toggle = Button(ax_toggle, "Source BG ON")

    def clear_patches() -> None:
        for patch in state["patches"]:
            patch.remove()
        for label in state["labels"]:
            label.remove()
        state["patches"] = []
        state["labels"] = []

    def redraw_view(_=None) -> None:
        state["start_idx"] = int(pos_slider.val)
        state["view_span"] = int(span_slider.val)
        end = min(state["start_idx"] + state["view_span"], total_points)
        x_new = np.arange(state["start_idx"], end)

        for label, line in lines.items():
            line.set_xdata(x_new)
            line.set_ydata(sequences[label][state["start_idx"]:end])

        ax.set_xlim(state["start_idx"], max(state["start_idx"] + 1, end))
        ax.set_title(
            f"{file_path.name} | rows {state['start_idx']:,}–{end:,} | "
            f"span {state['view_span']:,} | concat order",
            fontsize=11,
            fontweight="bold",
        )

        clear_patches()
        if state["show_source"]:
            visible_real = 0
            visible_syn = 0
            for s, e, src in segments:
                if e <= state["start_idx"] or s >= end:
                    continue
                color = REAL_COLOR if src == 0 else SYN_COLOR
                patch = ax.axvspan(s, e, color=color, alpha=0.55, zorder=0)
                state["patches"].append(patch)
                if src == 0:
                    visible_real += min(e, end) - max(s, state["start_idx"])
                else:
                    visible_syn += min(e, end) - max(s, state["start_idx"])

            legend_text.set_text(
                f"Green=real ({visible_real:,} visible) | "
                f"Light red=synthetic ({visible_syn:,} visible)"
            )
        else:
            legend_text.set_text("Source backgrounds hidden")

        fig.canvas.draw_idle()

    def auto_fit(_=None) -> None:
        y_min, y_max = float("inf"), float("-inf")
        found = False
        for line in lines.values():
            if line.get_visible():
                y = line.get_ydata()
                if len(y):
                    y_min = min(y_min, float(np.min(y)))
                    y_max = max(y_max, float(np.max(y)))
                    found = True
        if found:
            span = (y_max - y_min) or 1.0
            ax.set_ylim(y_min - span * 0.1, y_max + span * 0.1)
            redraw_view()

    def toggle_source(_=None) -> None:
        state["show_source"] = not state["show_source"]
        btn_toggle.label.set_text(
            f"Source BG {'ON' if state['show_source'] else 'OFF'}"
        )
        redraw_view()

    def on_pick(event) -> None:
        target = event.artist
        line = target if target in lines.values() else lines.get(target.get_label())
        if line:
            visible = not line.get_visible()
            line.set_visible(visible)
            if line.get_label() in label_to_legline:
                label_to_legline[line.get_label()].set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw_idle()

    pos_slider.on_changed(redraw_view)
    span_slider.on_changed(redraw_view)
    btn_prev.on_clicked(
        lambda _: pos_slider.set_val(max(0, state["start_idx"] - state["view_span"] // 2))
    )
    btn_next.on_clicked(
        lambda _: pos_slider.set_val(min(max_start, state["start_idx"] + state["view_span"] // 2))
    )
    btn_fit.on_clicked(auto_fit)
    btn_toggle.on_clicked(toggle_source)
    fig.canvas.mpl_connect("pick_event", on_pick)

    auto_fit()
    print("\nControls: Start/Span sliders, Back/Forward, Fit Y, toggle source backgrounds")
    print("Legend: click to show/hide aggregate or appliance trace")
    plt.show()


def list_labeled_csvs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("*_labeled*.csv"))


def choose_file() -> Path | None:
    files = list_labeled_csvs(DEFAULT_MIX_ROOT)
    if not files:
        print(f"No *_labeled*.csv under {DEFAULT_MIX_ROOT}")
        return None

    print("\nLabeled Geng mix CSV files:")
    for i, f in enumerate(files):
        print(f" [{i}] {f.relative_to(BASE_DIR)}")

    user_input = input("\nEnter index or path: ").strip().strip('"')
    if not user_input:
        return None
    if user_input.isdigit() and int(user_input) < len(files):
        return files[int(user_input)]
    try:
        return resolve_csv_path(user_input)
    except FileNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize Geng mix CSV with real (green) vs synthetic (red) backgrounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/geng_mix_visualize.py DiffusionModel_NILM/.../UK_DALECombinedkettle_file20_labeled.csv\n"
            "  python data/geng_mix_visualize.py kettle/UK_DALECombinedkettle_file20_labeled.csv\n"
        ),
    )
    parser.add_argument("csv_file", nargs="?", help="Labeled mix CSV path")
    parser.add_argument("--path", "-p", type=str, default=None, help="Alias for csv_file")
    parser.add_argument("--span", type=int, default=2048, help="Initial visible row count")
    args = parser.parse_args()

    path_arg = args.path or args.csv_file
    if not path_arg:
        file_path = choose_file()
    else:
        try:
            file_path = resolve_csv_path(path_arg)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return 1

    if file_path is None:
        return 1

    try:
        interactive_viewer(file_path, initial_span=args.span)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
