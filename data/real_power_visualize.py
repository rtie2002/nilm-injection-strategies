r"""
Visualize real-power NILM CSV files.

This script is compatible with the current data folder:
    ./real_data
    ./synthetic_data

Expected real CSV columns:
    aggregate,<appliance>,minute_sin,...,month_cos,on_off

Expected synthetic CSV columns:
    <appliance>,minute_sin,...,month_cos,on_off

______________________________________________________
Step (1)
Go to the data folder
______________________________________________________

PowerShell command:
    cd "C:\Users\Raymond Tie\Desktop\PhD\Paper\Injection Strategy Conference\code\data"

______________________________________________________
Step (2)
Open the file picker
______________________________________________________

PowerShell command:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\real_power_visualize.py

______________________________________________________
Step (3)
Open one CSV directly
______________________________________________________

Example:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\real_power_visualize.py --path .\real_data\training\kettle_train_100k.csv

______________________________________________________
Step (4)
Show fewer or more points at first view
______________________________________________________

Example:
    & "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\real_power_visualize.py --path .\synthetic_data\kettle_synthetic.csv --span 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for this visualizer. "
        "Install it in the Python environment or use the HTML preview script instead."
    ) from exc

APPLIANCES = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]
TIME_COLUMNS = {
    "minute_sin", "minute_cos",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
}


def detect_appliance_column(df: pd.DataFrame, file_path: Path) -> str | None:
    # ______________________________________________________
    # Step (A)
    # Find the appliance column.
    # It can be detected from CSV columns or from file name.
    # ______________________________________________________
    for appliance in APPLIANCES:
        if appliance in df.columns:
            return appliance

    name = file_path.name.lower()
    for appliance in APPLIANCES:
        if appliance in name:
            return appliance

    return None


def get_plot_columns(df: pd.DataFrame, appliance: str | None) -> list[str]:
    # ______________________________________________________
    # Step (B)
    # Decide which signals to plot.
    # Plot aggregate if available, then the appliance power.
    # Do not plot time features or on_off as normal lines.
    # ______________________________________________________
    columns = []
    if "aggregate" in df.columns:
        columns.append("aggregate")
    if appliance and appliance in df.columns:
        columns.append(appliance)

    if not columns:
        ignored = TIME_COLUMNS | {"on_off"}
        columns = [c for c in df.columns if c not in ignored and pd.api.types.is_numeric_dtype(df[c])]

    return columns


def find_on_segments(on_off: np.ndarray) -> list[tuple[int, int, int]]:
    # ______________________________________________________
    # Step (C)
    # Convert 0/1 on_off sequence into ON segments.
    # Each segment is start index, end index, segment number.
    # ______________________________________________________
    mask = np.asarray(on_off).astype(int)
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e), i + 1) for i, (s, e) in enumerate(zip(starts, ends))]


def interactive_viewer(file_path: Path, initial_span: int = 1024) -> None:
    # ______________________________________________________
    # Step (1)
    # Load one CSV file from real_data or synthetic_data.
    # ______________________________________________________
    print(f"\nLoading data: {file_path}")
    df = pd.read_csv(file_path)
    total_points = len(df)
    if total_points == 0:
        print("CSV is empty.")
        return

    appliance = detect_appliance_column(df, file_path)
    plot_columns = get_plot_columns(df, appliance)
    if not plot_columns:
        print("No plottable power column found.")
        return

    # ______________________________________________________
    # Step (2)
    # Prepare power sequences and ON/OFF highlights.
    # ______________________________________________________
    sequences = {col: df[col].to_numpy(dtype=float) for col in plot_columns}
    on_segments = find_on_segments(df["on_off"].to_numpy()) if "on_off" in df.columns else []

    print(f"Total points: {total_points:,}")
    print(f"Appliance: {appliance or 'unknown'}")
    print(f"Plotted columns: {', '.join(plot_columns)}")
    if "on_off" in df.columns:
        print(f"ON samples: {int(df['on_off'].sum()):,} ({df['on_off'].mean():.4f})")
        print(f"ON periods: {len(on_segments):,}")
    else:
        print("No on_off column found; highlights disabled.")

    # ______________________________________________________
    # Step (3)
    # Create interactive plot with sliders and buttons.
    # ______________________________________________________
    state = {
        "start_idx": 0,
        "view_span": min(initial_span, total_points),
        "show_on_off": True,
        "patches": [],
        "labels": [],
        "selection_rect": None,
        "sel_start": None,
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.30, left=0.08, right=0.95, top=0.92)
    fig_text = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=10, color="darkgreen", fontweight="bold")

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    lines = {}
    label_to_legline = {}

    end_idx = min(state["view_span"], total_points)
    x_range = np.arange(0, end_idx)
    for i, (label, values) in enumerate(sequences.items()):
        line, = ax.plot(x_range, values[:end_idx], label=label, color=colors[i % len(colors)], alpha=0.85, picker=5)
        lines[label] = line

    legend = ax.legend(loc="upper right")
    for legline in legend.get_lines():
        legline.set_picker(5)
        label_to_legline[legline.get_label()] = legline

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power")

    max_start = max(0, total_points - 1)
    ax_pos = plt.axes([0.1, 0.12, 0.5, 0.03])
    pos_slider = Slider(ax_pos, "Start", 0, max_start, valinit=0, valstep=1, valfmt="%d")

    ax_span = plt.axes([0.1, 0.07, 0.5, 0.03])
    span_slider = Slider(ax_span, "Span", 10, max(10, min(total_points, 50000)), valinit=state["view_span"], valstep=10, valfmt="%d")

    ax_prev = plt.axes([0.65, 0.09, 0.08, 0.04])
    ax_next = plt.axes([0.74, 0.09, 0.08, 0.04])
    ax_fit = plt.axes([0.83, 0.09, 0.12, 0.04])
    ax_toggle = plt.axes([0.65, 0.04, 0.17, 0.04])

    btn_prev = Button(ax_prev, "Back")
    btn_next = Button(ax_next, "Forward")
    btn_fit = Button(ax_fit, "Fit Y")
    btn_toggle = Button(ax_toggle, "Highlights ON")

    def clear_highlights() -> None:
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
        ax.set_title(f"{file_path.name} | {state['start_idx']:,} to {end:,} | span {state['view_span']:,}")

        clear_highlights()
        if on_segments and state["show_on_off"]:
            y_min, y_max = ax.get_ylim()
            label_y = y_max - (y_max - y_min) * 0.05
            visible = [(s, e, n) for s, e, n in on_segments if e > state["start_idx"] and s < end]
            for s, e, n in visible:
                patch = ax.axvspan(s, e, color="lightgreen", alpha=0.30, zorder=-1)
                state["patches"].append(patch)
                label_x = (max(s, state["start_idx"]) + min(e, end)) / 2
                text = ax.text(label_x, label_y, str(n), ha="center", va="top", fontsize=8, color="darkgreen", fontweight="bold")
                state["labels"].append(text)
            fig_text.set_text(f"ON periods: {len(on_segments)} | visible: {len(visible)}")
        else:
            fig_text.set_text("")

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

    def toggle_highlights(_=None) -> None:
        state["show_on_off"] = not state["show_on_off"]
        btn_toggle.label.set_text(f"Highlights {'ON' if state['show_on_off'] else 'OFF'}")
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

    def on_mouse_press(event) -> None:
        if event.inaxes != ax:
            return
        if event.button == 1:
            state["sel_start"] = (event.xdata, event.ydata)
            if state["selection_rect"]:
                state["selection_rect"].remove()
            rect = plt.Rectangle((event.xdata, event.ydata), 0, 0, fill=False, color="red", linestyle="--")
            state["selection_rect"] = rect
            ax.add_patch(rect)
        elif event.button == 3:
            if state["selection_rect"]:
                state["selection_rect"].remove()
                state["selection_rect"] = None
            fig.canvas.draw_idle()

    def on_mouse_move(event) -> None:
        if state["sel_start"] and event.inaxes == ax:
            state["selection_rect"].set_width(event.xdata - state["sel_start"][0])
            state["selection_rect"].set_height(event.ydata - state["sel_start"][1])
            fig.canvas.draw_idle()

    def on_mouse_release(event) -> None:
        if state["sel_start"] and event.button == 1:
            x0, x1 = sorted([state["sel_start"][0], event.xdata])
            state["sel_start"] = None
            print("\n" + "=" * 40)
            print("PERIOD STATISTICS")
            for label, line in lines.items():
                if not line.get_visible():
                    continue
                x = line.get_xdata()
                y = line.get_ydata()
                mask = (x >= x0) & (x <= x1)
                selected = y[mask]
                if len(selected):
                    print(f"{label:18} mean={selected.mean():9.3f} max={selected.max():9.3f} points={len(selected)}")
            print("=" * 40)

    pos_slider.on_changed(redraw_view)
    span_slider.on_changed(redraw_view)
    btn_prev.on_clicked(lambda _: pos_slider.set_val(max(0, state["start_idx"] - state["view_span"] // 2)))
    btn_next.on_clicked(lambda _: pos_slider.set_val(min(max_start, state["start_idx"] + state["view_span"] // 2)))
    btn_fit.on_clicked(auto_fit)
    btn_toggle.on_clicked(toggle_highlights)

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)

    auto_fit()
    plt.show()


def list_csv_files(data_dir: Path) -> list[Path]:
    # ______________________________________________________
    # Step (4)
    # Find all CSV files under code/data/real_data and synthetic_data.
    # ______________________________________________________
    roots = [data_dir / "real_data", data_dir / "synthetic_data"]
    files = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.csv")))
    return files


def choose_file(data_dir: Path) -> Path | None:
    files = list_csv_files(data_dir)
    if not files:
        print(f"No CSV files found under {data_dir}")
        return None

    print(f"\nScanning: {data_dir}")
    for i, file in enumerate(files):
        print(f" [{i}] {file.relative_to(data_dir)}")

    user_input = input("\nEnter index, appliance name, or full path: ").strip().strip('"')
    if not user_input:
        return None

    input_path = Path(user_input)
    if input_path.exists():
        return input_path

    if user_input.isdigit() and int(user_input) < len(files):
        return files[int(user_input)]

    matches = [file for file in files if user_input.lower() in file.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print("Multiple matches:")
        for i, file in enumerate(matches):
            print(f" [{i}] {file.relative_to(data_dir)}")
        choice = input("Choose match index: ").strip()
        if choice.isdigit() and int(choice) < len(matches):
            return matches[int(choice)]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize current real_power NILM CSV files.")
    parser.add_argument("--path", type=Path, help="CSV file path to visualize")
    parser.add_argument("--span", type=int, default=1024, help="Initial number of samples shown")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent
    file_path = args.path
    if file_path is None:
        file_path = choose_file(data_dir)
    elif not file_path.is_absolute():
        file_path = (Path.cwd() / file_path).resolve()

    if file_path and file_path.exists():
        interactive_viewer(file_path, initial_span=args.span)
    else:
        print(f"Error: file not found: {file_path}")


if __name__ == "__main__":
    main()

