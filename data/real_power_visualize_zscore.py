r"""
Visualize Geng NILM CSV files in z-score space (what EasyS2S trains on).

Supports:
  - Standard Geng CSVs (2 columns, no header): aggregate_z, appliance_z
    e.g. kettle_validation_.csv, UK_DALECombinedkettle_file20.csv
  - Labeled mix CSVs (watts + source): converted to z-score for display
    e.g. UK_DALECombinedkettle_file20_labeled.csv

Examples:
    python data/real_power_visualize_zscore.py --path DiffusionModel_NILM/NILM-main/dataset_preprocess/created_data/UK_DALE/kettle/kettle_validation_.csv

    python data/real_power_visualize_zscore.py kettle/UK_DALECombinedkettle_file20.csv --span 2048

    python data/real_power_visualize_zscore.py --browse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, Slider

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_UK_DALE = (
    BASE_DIR
    / "DiffusionModel_NILM"
    / "NILM-main"
    / "dataset_preprocess"
    / "created_data"
    / "UK_DALE"
)

AGG_MEAN = 522.0
AGG_STD = 814.0

APPLIANCES = ("kettle", "microwave", "fridge", "dishwasher", "washingmachine")

# Geng / NILM-main_pytorch normalization + ON thresholds (watts)
APPLIANCE_PARAMS: dict[str, dict[str, float]] = {
    "kettle": {"mean": 700.0, "std": 1000.0, "on_power_threshold": 200.0},
    "microwave": {"mean": 500.0, "std": 800.0, "on_power_threshold": 200.0},
    "fridge": {"mean": 200.0, "std": 400.0, "on_power_threshold": 50.0},
    "dishwasher": {"mean": 700.0, "std": 1000.0, "on_power_threshold": 10.0},
    "washingmachine": {"mean": 400.0, "std": 700.0, "on_power_threshold": 20.0},
}

REAL_COLOR = "#c8f7c5"
SYN_COLOR = "#ffd6d6"


def norm_on_threshold_z(appliance: str) -> float:
    p = APPLIANCE_PARAMS[appliance]
    return (p["on_power_threshold"] - p["mean"]) / p["std"]


def watts_to_zscore(df_w: pd.DataFrame, appliance: str) -> pd.DataFrame:
    p = APPLIANCE_PARAMS[appliance]
    out = pd.DataFrame(
        {
            "aggregate": (df_w["aggregate"].to_numpy(dtype=np.float64) - AGG_MEAN) / AGG_STD,
            appliance: (df_w[appliance].to_numpy(dtype=np.float64) - p["mean"]) / p["std"],
        }
    )
    if "source" in df_w.columns:
        out["source"] = df_w["source"].to_numpy(dtype=np.int32)
    return out


def detect_appliance(file_path: Path, columns: list[str]) -> str | None:
    for app in APPLIANCES:
        if app in columns:
            return app
    name = file_path.name.lower()
    for app in APPLIANCES:
        if app in name:
            return app
    return None


def resolve_csv_path(path_str: str) -> Path:
    raw = path_str.strip().strip('"').strip("'")
    p = Path(raw)
    if p.is_file():
        return p.resolve()

    candidates: list[Path] = []
    if not p.is_absolute():
        candidates.extend(
            [
                Path.cwd() / p,
                BASE_DIR / p,
                SCRIPT_DIR / p,
                DEFAULT_UK_DALE / p,
            ]
        )
        if len(p.parts) == 1:
            for app in APPLIANCES:
                candidates.append(DEFAULT_UK_DALE / app / p.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"CSV not found: {raw}\nTried:\n  {tried}")


def load_zscore_frame(file_path: Path) -> tuple[pd.DataFrame, str | None, str]:
    """Return (dataframe in z-score, appliance name, load mode label)."""
    # Try headerless Geng NILM CSV first (2 z-score columns).
    try:
        raw = pd.read_csv(file_path, header=None, nrows=5)
        if raw.shape[1] >= 2 and raw.shape[1] <= 3:
            df = pd.read_csv(file_path, header=None)
            if df.shape[1] == 2:
                appliance = detect_appliance(file_path, [])
                if appliance is None:
                    raise ValueError("Could not infer appliance from filename.")
                df = df.iloc[:, :2].copy()
                df.columns = ["aggregate", appliance]
                return df, appliance, "z-score (2-col, no header)"
    except (pd.errors.ParserError, ValueError):
        pass

    # Labeled or other CSV with headers (often watts).
    df = pd.read_csv(file_path)
    appliance = detect_appliance(file_path, list(df.columns))
    if appliance is None:
        raise ValueError(f"Could not detect appliance column in {file_path.name}")

    if "aggregate" not in df.columns or appliance not in df.columns:
        raise ValueError(f"Expected columns aggregate and {appliance} in {file_path.name}")

    # Heuristic: Geng z-score aggregate rarely exceeds ~3; watts mains are hundreds+.
    agg_sample = df["aggregate"].iloc[: min(5000, len(df))].to_numpy(dtype=np.float64)
    if np.nanmax(np.abs(agg_sample)) > 20.0:
        return watts_to_zscore(df, appliance), appliance, "watts -> z-score (labeled/header CSV)"

    out = df[["aggregate", appliance]].copy()
    if "source" in df.columns:
        out["source"] = df["source"].astype(int)
    return out, appliance, "z-score (header CSV)"


def find_on_segments_z(appliance_z: np.ndarray, threshold_z: float) -> list[tuple[int, int, int]]:
    mask = (np.asarray(appliance_z, dtype=np.float64) > threshold_z).astype(int)
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e), i + 1) for i, (s, e) in enumerate(zip(starts, ends))]


def find_source_segments(source: np.ndarray, value: int) -> list[tuple[int, int]]:
    mask = (np.asarray(source).astype(int) == value).astype(int)
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def interactive_viewer(file_path: Path, initial_span: int = 1024) -> None:
    print(f"\nLoading: {file_path}")
    df, appliance, mode = load_zscore_frame(file_path)
    total_points = len(df)
    if total_points == 0:
        print("CSV is empty.")
        return

    assert appliance is not None
    threshold_z = norm_on_threshold_z(appliance)
    on_segments = find_on_segments_z(df[appliance].to_numpy(), threshold_z)

    has_source = "source" in df.columns
    real_segments = find_source_segments(df["source"].to_numpy(), 0) if has_source else []
    syn_segments = find_source_segments(df["source"].to_numpy(), 1) if has_source else []

    sequences = {
        "aggregate (z)": df["aggregate"].to_numpy(dtype=np.float64),
        f"{appliance} (z)": df[appliance].to_numpy(dtype=np.float64),
    }

    print(f"Mode: {mode}")
    print(f"Total points: {total_points:,}")
    print(f"Appliance: {appliance}")
    print(f"ON threshold (z): {threshold_z:.4f}  (= {APPLIANCE_PARAMS[appliance]['on_power_threshold']:.0f} W)")
    print(f"ON segments (z > thr): {len(on_segments):,}")
    if has_source:
        n_real = int((df["source"] == 0).sum())
        n_syn = int((df["source"] == 1).sum())
        print(f"source=0 (real block): {n_real:,} rows | source=1 (syn block): {n_syn:,} rows")

    state = {
        "start_idx": 0,
        "view_span": min(initial_span, total_points),
        "show_on": True,
        "show_source": has_source,
        "patches": [],
        "labels": [],
        "selection_rect": None,
        "sel_start": None,
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.30, left=0.08, right=0.95, top=0.92)
    fig_text = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=10, color="darkgreen", fontweight="bold")

    colors = ["#1f77b4", "#d62728"]
    lines: dict[str, plt.Line2D] = {}
    label_to_legline: dict[str, plt.Line2D] = {}

    end_idx = min(state["view_span"], total_points)
    x_range = np.arange(0, end_idx)
    for i, (label, values) in enumerate(sequences.items()):
        line, = ax.plot(
            x_range,
            values[:end_idx],
            label=label,
            color=colors[i % len(colors)],
            alpha=0.9,
            picker=5,
        )
        lines[label] = line

    ax.axhline(threshold_z, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.7, label=f"ON thr z={threshold_z:.3f}")
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=0.8, alpha=0.6)

    legend = ax.legend(loc="upper right", fontsize=9)
    for legline in legend.get_lines():
        legline.set_picker(5)
        label_to_legline[legline.get_label()] = legline

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Sample index (6 s per row)")
    ax.set_ylabel("Z-score")

    max_start = max(0, total_points - 1)
    pos_slider = Slider(plt.axes([0.1, 0.12, 0.5, 0.03]), "Start", 0, max_start, valinit=0, valstep=1, valfmt="%d")
    span_slider = Slider(
        plt.axes([0.1, 0.07, 0.5, 0.03]),
        "Span",
        10,
        max(10, min(total_points, 50000)),
        valinit=state["view_span"],
        valstep=10,
        valfmt="%d",
    )

    btn_prev = Button(plt.axes([0.65, 0.09, 0.08, 0.04]), "Back")
    btn_next = Button(plt.axes([0.74, 0.09, 0.08, 0.04]), "Forward")
    btn_fit = Button(plt.axes([0.83, 0.09, 0.12, 0.04]), "Fit Y")
    btn_on = Button(plt.axes([0.65, 0.04, 0.12, 0.04]), "ON marks")
    btn_src = Button(plt.axes([0.78, 0.04, 0.17, 0.04]), "Source bg")

    def clear_decorations() -> None:
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
            f"{file_path.name} | z-score | {state['start_idx']:,}–{end:,} | span {state['view_span']:,}",
            fontsize=11,
        )

        clear_decorations()

        if has_source and state["show_source"]:
            for s, e in real_segments:
                if e > state["start_idx"] and s < end:
                    patch = ax.axvspan(s, e, color=REAL_COLOR, alpha=0.35, zorder=-2)
                    state["patches"].append(patch)
            for s, e in syn_segments:
                if e > state["start_idx"] and s < end:
                    patch = ax.axvspan(s, e, color=SYN_COLOR, alpha=0.35, zorder=-2)
                    state["patches"].append(patch)

        if state["show_on"]:
            y_min, y_max = ax.get_ylim()
            label_y = y_max - (y_max - y_min) * 0.05
            visible = [(s, e, n) for s, e, n in on_segments if e > state["start_idx"] and s < end]
            for s, e, n in visible:
                patch = ax.axvspan(s, e, color="lightgreen", alpha=0.22, zorder=-1)
                state["patches"].append(patch)
                label_x = (max(s, state["start_idx"]) + min(e, end)) / 2
                text = ax.text(
                    label_x,
                    label_y,
                    str(n),
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="darkgreen",
                    fontweight="bold",
                )
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

    def toggle_on(_=None) -> None:
        state["show_on"] = not state["show_on"]
        btn_on.label.set_text(f"ON {'ON' if state['show_on'] else 'OFF'}")
        redraw_view()

    def toggle_source(_=None) -> None:
        if not has_source:
            return
        state["show_source"] = not state["show_source"]
        btn_src.label.set_text(f"Src bg {'ON' if state['show_source'] else 'OFF'}")
        redraw_view()

    def on_pick(event) -> None:
        target = event.artist
        line = target if target in lines.values() else lines.get(getattr(target, "get_label", lambda: "")())
        if line:
            visible = not line.get_visible()
            line.set_visible(visible)
            lbl = line.get_label()
            if lbl in label_to_legline:
                label_to_legline[lbl].set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw_idle()

    pos_slider.on_changed(redraw_view)
    span_slider.on_changed(redraw_view)
    btn_prev.on_clicked(lambda _: pos_slider.set_val(max(0, state["start_idx"] - state["view_span"] // 2)))
    btn_next.on_clicked(lambda _: pos_slider.set_val(min(max_start, state["start_idx"] + state["view_span"] // 2)))
    btn_fit.on_clicked(auto_fit)
    btn_on.on_clicked(toggle_on)
    btn_src.on_clicked(toggle_source)
    fig.canvas.mpl_connect("pick_event", on_pick)

    if not has_source:
        btn_src.label.set_text("No source col")
        btn_src.active = False

    auto_fit()
    plt.show()


def list_ukdale_csvs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.csv"))


def choose_file(root: Path) -> Path | None:
    files = list_ukdale_csvs(root)
    if not files:
        print(f"No CSV under {root}")
        return None

    print(f"\nUK_DALE CSVs under {root}:")
    for i, f in enumerate(files):
        try:
            rel = f.relative_to(root)
        except ValueError:
            rel = f.name
        print(f" [{i}] {rel}")

    user_input = input("\nIndex, appliance name, or path: ").strip().strip('"')
    if not user_input:
        return None

    p = Path(user_input)
    if p.is_file():
        return p.resolve()
    if user_input.isdigit() and int(user_input) < len(files):
        return files[int(user_input)]

    matches = [f for f in files if user_input.lower() in f.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        for i, f in enumerate(matches):
            print(f" [{i}] {f.name}")
        choice = input("Match index: ").strip()
        if choice.isdigit() and int(choice) < len(matches):
            return matches[int(choice)]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive viewer for Geng NILM CSVs in z-score space (model input scale)."
    )
    parser.add_argument("--path", type=str, default=None, help="CSV path (2-col z-score or labeled watts)")
    parser.add_argument("--span", type=int, default=1024, help="Initial window length in samples")
    parser.add_argument(
        "--browse",
        action="store_true",
        help=f"Pick file under --data-root (default: UK_DALE created_data)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_UK_DALE,
        help="Root for --browse file picker",
    )
    args = parser.parse_args()

    file_path: Path | None = None
    if args.path:
        file_path = resolve_csv_path(args.path)
    else:
        file_path = choose_file(args.data_root.resolve())

    if file_path and file_path.is_file():
        interactive_viewer(file_path, initial_span=args.span)
    else:
        print("No file selected or file not found.")
        print("Example:")
        print(
            "  python data/real_power_visualize_zscore.py "
            "--path DiffusionModel_NILM/NILM-main/dataset_preprocess/created_data/UK_DALE/kettle/kettle_validation_.csv"
        )


if __name__ == "__main__":
    main()
