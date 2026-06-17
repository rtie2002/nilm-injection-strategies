"""
Train Geng NILM models â€” PyTorch version.

  python -m nilm_main_pytorch.train --model easy_s2s --appliance kettle
  python -m nilm_main_pytorch.train --model s2p --appliance microwave --augmented
"""

from __future__ import annotations

import sys
from pathlib import Path

_PYTORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTORCH_ROOT))

import argparse
import json
import time

import numpy as np
import torch

from nilm_main_pytorch.data.datasets import build_train_val_loaders
from nilm_main_pytorch.data.paths import require_csv, train_csv_path, validation_csv_path
from nilm_main_pytorch.losses import build_loss_fn
from nilm_main_pytorch.metrics import evaluate_loader
from nilm_main_pytorch.models import build_model
from nilm_main_pytorch.models.params import PARAMS_APPLIANCE
from nilm_main_pytorch.utils import (
    add_device_cli_args,
    checkpoint_path,
    data_root_path,
    device_options_from_args,
    get_device,
    gpu_stats,
    load_config,
    log_device,
    merge_cli_config,
    model_training_config,
    norm_stats,
    portable_path_str,
    relativize_config,
    results_dir_path,
    set_seed,
)




def _setup_plot_style() -> None:
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.8,
            "axes.linewidth": 0.8,
        }
    )


def _figure_dir(cfg: dict, model_name: str, appliance: str, augmented: bool) -> Path:
    from nilm_main_pytorch.utils import experiment_run_suffix

    suffix = experiment_run_suffix(cfg, augmented=augmented)
    return results_dir_path(cfg) / "figures" / f"{model_name.lower()}_{appliance}_{suffix}"


def _graph_paths(cfg: dict, model_name: str, appliance: str, augmented: bool) -> dict[str, str]:
    out_dir = _figure_dir(cfg, model_name, appliance, augmented)
    return {
        "figure_dir": portable_path_str(out_dir),
        "loss_curve_png": portable_path_str(out_dir / "loss_curve.png"),
        "loss_curve_pdf": portable_path_str(out_dir / "loss_curve.pdf"),
        "metric_summary_png": portable_path_str(out_dir / "metric_summary.png"),
        "metric_summary_pdf": portable_path_str(out_dir / "metric_summary.pdf"),
        "on_samples_png": portable_path_str(out_dir / "on_period_samples.png"),
        "on_samples_pdf": portable_path_str(out_dir / "on_period_samples.pdf"),
    }


def _save_loss_curve(train_losses: list[float], val_losses: list[float], out_dir: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not train_losses or not val_losses:
        return
    _setup_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
    markevery = max(1, len(epochs) // 8)
    ax.plot(epochs, train_losses, marker="o", markersize=3.5, markevery=markevery, label="Train loss")
    ax.plot(epochs, val_losses, marker="s", markersize=3.5, markevery=markevery, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=min(8, max(3, len(epochs) // 5)), integer=True, min_n_ticks=3))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=True, loc="best")
    fig.savefig(out_dir / "loss_curve.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / "loss_curve.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_metric_summary(metrics: dict[str, float], out_dir: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    _setup_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["MAE (W)", "SAE", "F1"]
    values = [float(metrics.get("mae", 0.0)), float(metrics.get("sae", 0.0)), float(metrics.get("f1", 0.0))]
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_dir / "metric_summary.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / "metric_summary.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _denorm(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * std + mean


def _on_slice_bounds(on_mask: np.ndarray, window_len: int, pad: int = 48) -> slice:
    on_idx = np.flatnonzero(on_mask)
    if on_idx.size == 0:
        return slice(0, window_len)
    start = max(0, int(on_idx[0]) - pad)
    end = min(window_len, int(on_idx[-1]) + pad + 1)
    return slice(start, end)


def _aligned_prediction_slice(pred_w: np.ndarray, window_len: int, sl: slice) -> tuple[np.ndarray, np.ndarray, bool]:
    pred_arr = np.ravel(np.asarray(pred_w, dtype=np.float64))
    if pred_arr.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=np.float64), False
    if pred_arr.size == 1:
        mid = window_len // 2
        return np.asarray([mid], dtype=int), pred_arr, True
    if pred_arr.size == window_len:
        t = np.arange(sl.start, sl.stop)
        return t, pred_arr[sl], False

    start = max(0, (window_len - pred_arr.size) // 2)
    end = start + pred_arr.size
    left = max(sl.start, start)
    right = min(sl.stop, end)
    if right <= left:
        return np.asarray([], dtype=int), np.asarray([], dtype=np.float64), False
    t = np.arange(left, right)
    return t, pred_arr[(left - start):(right - start)], False


def _dataset_window_energy(ds, app_w: np.ndarray, threshold_w: float) -> np.ndarray:
    on_power = np.where(app_w >= threshold_w, app_w, 0.0)
    if hasattr(ds, "indices") and hasattr(ds, "window_length"):
        window_len = int(ds.window_length)
        starts = np.asarray(ds.indices, dtype=np.int64)
        sums = np.concatenate([[0.0], np.cumsum(on_power, dtype=np.float64)])
        return sums[starts + window_len] - sums[starts]
    if hasattr(ds, "target_length") and hasattr(ds, "num_windows"):
        target_len = int(ds.target_length)
        offset = target_len // 2
        target_offset = int(getattr(ds, "target_offset", 0))
        output_len = int(getattr(ds, "output_length", target_len))
        energies = np.zeros(len(ds), dtype=np.float64)
        for idx in range(len(ds)):
            start = idx * target_len + offset + target_offset
            end = start + output_len
            energies[idx] = on_power[start:end].sum()
        return energies
    return np.zeros(len(ds), dtype=np.float64)


def _dataset_plot_window(ds, idx: int) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(ds, "indices") and hasattr(ds, "window_length"):
        start = int(ds.indices[idx])
        end = start + int(ds.window_length)
        return np.asarray(ds.aggregate[start:end]), np.asarray(ds.appliance[start:end])
    if hasattr(ds, "target_length") and hasattr(ds, "input_length"):
        start = idx * int(ds.target_length)
        end = start + int(ds.input_length)
        return np.asarray(ds.aggregate[start:end]), np.asarray(ds.appliance[start:end])
    raise TypeError("Unsupported dataset type for ON-period plotting")


def _appliance_plot_ylim(true_w: np.ndarray, pred_y: np.ndarray, threshold_w: float) -> float:
    peak = float(threshold_w)
    if true_w.size:
        peak = max(peak, float(np.max(true_w)))
    if pred_y.size:
        peak = max(peak, float(np.max(pred_y)))
    return max(peak * 1.15, threshold_w * 2.0, 10.0)


def _save_on_period_samples(
    model,
    val_loader,
    device: torch.device,
    stats: dict,
    threshold_w: float,
    out_dir: Path,
    title: str,
    n_samples: int = 3,
) -> None:
    import matplotlib.pyplot as plt

    ds = val_loader.dataset
    if not hasattr(ds, "aggregate") or not hasattr(ds, "appliance"):
        print("validation dataset has no aggregate/appliance arrays; skip ON-period graph", flush=True)
        return

    agg_series = np.asarray(ds.aggregate)
    app_series = np.asarray(ds.appliance)
    if agg_series.ndim != 1 or app_series.ndim != 1:
        print("validation dataset arrays are not 1D series; skip ON-period graph", flush=True)
        return

    app_w_series = _denorm(app_series, stats["appliance_mean"], stats["appliance_std"])
    on_energy = _dataset_window_energy(ds, app_w_series, threshold_w)
    candidates = np.flatnonzero(on_energy > 0)
    if len(candidates) == 0:
        print("no ON validation windows found; skip ON-period graph", flush=True)
        return

    _setup_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = candidates[np.argsort(on_energy[candidates])[-n_samples:][::-1]]
    n_rows = 2 * len(picks)
    fig, axes = plt.subplots(n_rows, 1, figsize=(3.5, 2.4 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    model.eval()
    for row, idx in enumerate(picks):
        ax_agg = axes[2 * row]
        ax_app = axes[2 * row + 1]

        x_np, y_np = _dataset_plot_window(ds, int(idx))
        x = torch.from_numpy(x_np[None, :].astype(np.float32)).to(device)
        with torch.no_grad():
            pred = model(x).detach().cpu().numpy()[0]

        agg_w = _denorm(x_np, stats["aggregate_mean"], stats["aggregate_std"])
        true_w = _denorm(y_np, stats["appliance_mean"], stats["appliance_std"])
        pred_w = np.clip(_denorm(pred, stats["appliance_mean"], stats["appliance_std"]), 0.0, None)

        window_len = len(true_w)
        on_mask = true_w >= threshold_w
        sl = _on_slice_bounds(on_mask, window_len, pad=48)
        t = np.arange(sl.start, sl.stop)
        mark_step = max(1, len(t) // 12)

        ax_agg.plot(t, agg_w[sl], color="#7f7f7f", linewidth=1.4, linestyle="-", label="Aggregate", zorder=1)
        ax_agg.set_ylabel("Agg (W)")
        ax_agg.set_title(f"val window {idx} (ON energy {on_energy[idx]:.0f} W)", pad=4)
        ax_agg.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax_agg.legend(frameon=True, loc="upper right", fontsize=7)

        true_slice = true_w[sl]
        ax_app.plot(
            t,
            true_slice,
            color="#d62728",
            linewidth=2.0,
            linestyle="-",
            label="Ground truth",
            zorder=3,
        )

        pred_t, pred_y, is_point = _aligned_prediction_slice(pred_w, window_len, sl)
        if pred_t.size:
            if is_point:
                ax_app.scatter(pred_t, pred_y, color="#1f77b4", s=20, label="Prediction", zorder=4)
            else:
                ax_app.plot(
                    pred_t,
                    pred_y,
                    color="#1f77b4",
                    linewidth=1.8,
                    linestyle="--",
                    marker="o",
                    markersize=3.5,
                    markevery=mark_step,
                    label="Prediction",
                    zorder=4,
                )

        ax_app.axhline(
            threshold_w,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.0,
            label=f"ON threshold ({threshold_w:.0f} W)",
            zorder=2,
        )
        ax_app.set_ylim(0.0, _appliance_plot_ylim(true_slice, pred_y, threshold_w))
        ax_app.set_ylabel("App (W)")
        ax_app.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax_app.legend(frameon=True, loc="upper right", fontsize=7)
        ax_app.set_xlabel("Timestep in window")

    fig.suptitle(title, fontsize=10, y=1.01)
    fig.savefig(out_dir / "on_period_samples.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / "on_period_samples.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved ON-period graph -> {out_dir / 'on_period_samples.png'}", flush=True)
def _run_epoch_batches(
    model,
    loader,
    device: torch.device,
    loss_fn,
    *,
    train: bool,
    optimizer=None,
    epoch: int = 0,
    total_epochs: int = 0,
    progress: bool = False,
    label: str = "train",
) -> float:
    n_batches = len(loader)
    progress_every = max(1, n_batches // 10)
    total_loss = 0.0
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for bi, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")

            if train and bi == 0 and epoch == 1 and progress:
                pdev = next(model.parameters()).device
                print(
                    f"  [check] batch tensors on {x.device} | model weights on {pdev}",
                    flush=True,
                )

            if train:
                optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                loss.backward()
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            if progress and (bi % progress_every == 0 or bi == n_batches - 1):
                print(
                    f"  [{label}] epoch {epoch}/{total_epochs} "
                    f"batch {bi + 1}/{n_batches} loss {loss.item():.4f} | {gpu_stats(device)}",
                    flush=True,
                )

    return total_loss / max(n_samples, 1)


def train_one(
    cfg: dict,
    *,
    verbose: bool = True,
    show_device: bool = True,
) -> dict:
    appliance = cfg["data"]["appliance"]
    model_name = cfg["model"]["name"]
    augmented = bool(cfg["data"]["augmented"])
    train_cfg = cfg["training"]
    dl_cfg = cfg.get("dataloader", {})

    set_seed(int(train_cfg.get("seed", 2024)))
    device = log_device(cfg) if show_device else get_device(cfg)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data_root = data_root_path(cfg)
    if verbose:
        print(f"data_root: {portable_path_str(data_root)}", flush=True)

    injection_rho = cfg["data"].get("injection_rho")
    train_csv = require_csv(
        train_csv_path(
            data_root,
            appliance,
            origin=not augmented if injection_rho is None else int(injection_rho) == 0,
            train_percent=str(cfg["data"]["train_percent"]),
            dataset_name=cfg["data"].get("dataset_name", "UK_DALE"),
            injection_rho=int(injection_rho) if injection_rho is not None else None,
        ),
        "Training",
    )
    val_csv = require_csv(validation_csv_path(data_root, appliance), "Validation")

    train_params = model_training_config(cfg, model_name)
    batch_size = int(train_params["batch_size"])
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", True))

    val_crop = cfg.get("evaluation", {}).get("val_max_rows", None)
    if val_crop is not None:
        val_crop = int(val_crop)

    train_loader, val_loader = build_train_val_loaders(
        model_name=model_name,
        appliance=appliance,
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        val_crop_rows=val_crop,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    if verbose:
        print(
            f"dataset: {n_train:,} train windows, {len(train_loader)} batches | "
            f"{n_val:,} val windows | batch_size={batch_size} | workers={num_workers}",
            flush=True,
        )

    model = build_model(model_name, appliance).to(device)
    loss_fn, on_thr_w = build_loss_fn(
        model_name,
        appliance,
        augmented=augmented,
        huber_delta=float(train_cfg.get("huber_delta", 0.5)),
        regression_alpha=float(train_cfg.get("regression_alpha", 1.0)),
        switch_beta=float(train_cfg.get("switch_beta", 0.1)),
    )
    stats = norm_stats(appliance)
    sample_second = float(cfg.get("evaluation", {}).get("sample_second", 6.0))
    metrics_every = int(cfg.get("evaluation", {}).get("metrics_every_epochs", 0))
    # TensorFlow customfit uses zero-based min_epoch. EasyS2S_train.py passes min_epoch=1,
    # so best-checkpoint evaluation starts after epoch 2 in one-based logs.
    min_epoch = int(train_cfg.get("min_epoch", 1))
    eval_train_loss = bool(train_cfg.get("eval_train_loss_each_epoch", True))
    progress_batches = bool(train_cfg.get("progress_batches", True)) and verbose

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_params["lr"]))
    epochs = int(train_params["epochs"])
    patience = int(train_params["patience"])
    early_stopping = bool(train_params.get("early_stopping", True))
    save_path = checkpoint_path(cfg, model_name, appliance, augmented)

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    since = time.time()
    last_metrics: dict[str, float] | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        train_step_loss = _run_epoch_batches(
            model,
            train_loader,
            device,
            loss_fn,
            train=True,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=epochs,
            progress=progress_batches,
            label="train",
        )
        t_train = time.time() - t0

        model.eval()
        train_loss = train_step_loss
        t_train_eval = 0.0
        if eval_train_loss:
            # Match TensorFlow NetFlowExt.customfit: after each epoch it runs the
            # network again on the training provider with dropout/noise disabled.
            t_train_eval0 = time.time()
            train_loss = _run_epoch_batches(
                model,
                train_loader,
                device,
                loss_fn,
                train=False,
                epoch=epoch,
                total_epochs=epochs,
                progress=False,
                label="train_eval",
            )
            t_train_eval = time.time() - t_train_eval0

        t1 = time.time()
        if verbose:
            print(f"  [val] epoch {epoch}/{epochs} running...", flush=True)
        val_loss = _run_epoch_batches(
            model,
            val_loader,
            device,
            loss_fn,
            train=False,
            epoch=epoch,
            total_epochs=epochs,
            progress=False,
            label="val",
        )
        t_val = time.time() - t1
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        # Metrics requires another full pass over the validation loader; avoid doing it every epoch unless requested.
        run_metrics = metrics_every > 0 and ((epoch % metrics_every == 0) or (epoch == epochs))
        t_metrics = 0.0
        if run_metrics:
            t2 = time.time()
            last_metrics = evaluate_loader(
                model,
                val_loader,
                device,
                stats["appliance_mean"],
                stats["appliance_std"],
                on_thr_w,
                sample_second,
            )
            t_metrics = time.time() - t2

        if verbose:
            parts = [
                f"epoch {epoch}/{epochs}",
                f"train {train_loss:.4f} ({t_train:.1f}s)",
                f"val {val_loss:.4f} ({t_val:.1f}s)",
            ]
            if eval_train_loss and t_train_eval > 0:
                parts.append(f"train_eval {t_train_eval:.1f}s")
            if last_metrics and run_metrics:
                parts.append(f"MAE {last_metrics['mae']:.1f}W")
                parts.append(f"F1 {last_metrics['f1']:.3f}")
                if t_metrics > 0:
                    parts.append(f"metrics {t_metrics:.1f}s")
            parts.append(gpu_stats(device))
            marker = " *best*" if val_loss < best_val else ""
            print(" | ".join(parts) + marker, flush=True)

        # Match TensorFlow NetFlowExt.customfit early stopping:
        #   if epoch_zero_based >= min_epoch: compare val loss and save best.
        #   stop only when best_valid_epoch + patience < current_epoch.
        epoch_zero_based = epoch - 1
        if epoch_zero_based >= min_epoch:
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                stale = 0
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "cfg": relativize_config(cfg),
                        "best_val_loss": best_val,
                        "epoch": epoch,
                        "appliance": appliance,
                        "augmented": augmented,
                        "model_name": model_name,
                    },
                    save_path,
                )
                if verbose:
                    print(f"  -> saved {portable_path_str(save_path)}", flush=True)
            else:
                stale += 1
                if early_stopping and stale > patience:
                    if verbose:
                        print(f"early stop at epoch {epoch} (patience {patience})", flush=True)
                    break

    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": relativize_config(cfg),
                "best_val_loss": val_loss,
                "epoch": epoch,
                "appliance": appliance,
                "augmented": augmented,
                "model_name": model_name,
            },
            save_path,
        )
        best_val = val_loss
        best_epoch = epoch

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_metrics = evaluate_loader(
        model,
        val_loader,
        device,
        stats["appliance_mean"],
        stats["appliance_std"],
        on_thr_w,
        sample_second,
    )
    figure_dir = _figure_dir(cfg, model_name, appliance, augmented)
    graph_paths = _graph_paths(cfg, model_name, appliance, augmented)
    if bool(cfg.get("evaluation", {}).get("save_plots", True)):
        plot_title = (
            f"{model_name.upper()} {appliance} rho={injection_rho}%"
            if injection_rho is not None
            else f"{model_name.upper()} {appliance} {'aug' if augmented else 'origin'} {cfg['data']['train_percent']}"
        )
        _save_loss_curve(train_losses, val_losses, figure_dir, plot_title)
        _save_metric_summary(val_metrics, figure_dir, f"Validation metrics ({appliance})")
        _save_on_period_samples(model, val_loader, device, stats, on_thr_w, figure_dir, f"ON-period samples ({appliance})")
        if verbose:
            print(f"figures saved -> {graph_paths['figure_dir']}", flush=True)

    elapsed = time.time() - since
    if verbose:
        print(
            f"done {appliance} {'aug' if augmented else 'origin'} pct={cfg['data']['train_percent']} | "
            f"best epoch {best_epoch} | {elapsed / 60:.1f} min total | {gpu_stats(device)}",
            flush=True,
        )

    return {
        "model": model_name.lower(),
        "appliance": appliance,
        "augmented": augmented,
        "injection_rho": injection_rho,
        "train_csv": portable_path_str(train_csv),
        "val_csv": portable_path_str(val_csv),
        "val_mae": val_metrics["mae"],
        "val_sae": val_metrics["sae"],
        "val_f1": val_metrics["f1"],
        "best_epoch": best_epoch,
        "checkpoint": portable_path_str(save_path),
        "elapsed_s": elapsed,
        "status": "trained",
        **graph_paths,
    }


def parse_args() -> argparse.Namespace:
    from nilm_main_pytorch.models.params import ALL_APPLIANCES

    p = argparse.ArgumentParser(description="Train Geng NILM models (PyTorch)")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", choices=["easy_s2s", "s2p", "fcn", "auglpn"], required=True)
    p.add_argument("--appliance", choices=list(ALL_APPLIANCES), required=True)
    p.add_argument("--augmented", action="store_true")
    p.add_argument("--train-percent", default="20")
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    add_device_cli_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dev, gpu_id, require_cuda = device_options_from_args(args)
    cfg = merge_cli_config(
        load_config(args.config),
        model=args.model,
        appliance=args.appliance,
        augmented=args.augmented,
        train_percent=args.train_percent,
        data_root=args.data_root,
        epochs=args.epochs,
        device=dev,
        gpu_id=gpu_id,
        require_cuda=require_cuda,
    )
    result = train_one(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


