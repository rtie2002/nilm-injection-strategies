from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from model.cnn import CNN
from metrics import evaluate_loader

CODE_DIR = Path(__file__).resolve().parent


class NILMDataset(Dataset):
    def __init__(self, aggregate, appliance, on_off=None):
        self.aggregate = torch.tensor(aggregate, dtype=torch.float32)
        self.appliance = torch.tensor(appliance, dtype=torch.float32)
        self.on_off = None if on_off is None else torch.tensor(on_off, dtype=torch.int8)

    def __len__(self):
        return len(self.aggregate)

    def __getitem__(self, idx):
        if self.on_off is None:
            return self.aggregate[idx], self.appliance[idx]
        return self.aggregate[idx], self.appliance[idx], self.on_off[idx]


def load_arrays(appliance, dataset_name, data_dir, window_len):
    path_npz = data_dir / appliance / f"{dataset_name}.npz"
    path_csv = data_dir / appliance / f"{dataset_name}.csv"
    path = path_npz if path_npz.exists() else path_csv
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path_npz} or {path_csv}")

    if path.suffix == ".npz":
        data = np.load(path)
        aggregate = data["X"][:, :, 0]
        appliance_power = data["y"]
        on_off = data["state"] if "state" in data else None
    else:
        df = pd.read_csv(path)
        n = len(df) // window_len
        aggregate = df["aggregate"].to_numpy(dtype=np.float32).reshape(n, window_len)
        appliance_power = df[appliance].to_numpy(dtype=np.float32).reshape(n, window_len)
        on_off = df["on_off"].to_numpy(dtype=np.int8).reshape(n, window_len) if "on_off" in df.columns else None

    return aggregate, appliance_power, on_off


def get_norm_stats(norm_cfg, appliance):
    method = norm_cfg["method"].lower()

    if method == "none":
        return {"method": "none"}

    if method == "zscore":
        if appliance not in norm_cfg:
            raise KeyError(f"No normalization stats for appliance: {appliance}")

        app_stats = norm_cfg[appliance]
        return {
            "method": "zscore",
            "aggregate_mean": float(norm_cfg["aggregate_mean"]),
            "aggregate_std": float(norm_cfg["aggregate_std"]),
            "appliance_mean": float(app_stats["mean"]),
            "appliance_std": float(app_stats["std"]),
        }

    raise ValueError(f"Unknown normalization.method: {method}")


def normalize(aggregate, appliance, stats):
    if stats["method"] == "none":
        return aggregate, appliance

    aggregate = (aggregate - stats["aggregate_mean"]) / stats["aggregate_std"]
    appliance = (appliance - stats["appliance_mean"]) / stats["appliance_std"]
    return aggregate, appliance


def denormalize_appliance(appliance, stats):
    if stats["method"] == "none":
        return appliance

    if torch.is_tensor(appliance):
        return appliance * stats["appliance_std"] + stats["appliance_mean"]

    return appliance * stats["appliance_std"] + stats["appliance_mean"]


def denormalize_appliance_watts(appliance, stats):
    watts = denormalize_appliance(appliance, stats)
    if torch.is_tensor(watts):
        return torch.clamp(watts, min=0.0)
    return np.clip(watts, 0.0, None)


def to_watts_tensor(tensor, stats):
    return denormalize_appliance_watts(tensor, stats).detach().cpu().numpy()


def get_on_threshold(cfg, appliance):
    return float(cfg["metrics"]["on_thresholds"][appliance])


def get_norm_on_threshold(on_threshold_w: float, stats: dict) -> float:
    """ON/OFF boundary in normalized space: (threshold_w - mean) / std."""
    if stats["method"] == "none":
        return on_threshold_w
    return (on_threshold_w - stats["appliance_mean"]) / stats["appliance_std"]


def huber_loss_elementwise(y_true: torch.Tensor, y_pred: torch.Tensor, delta: float) -> torch.Tensor:
    residual = torch.abs(y_true - y_pred)
    small = residual.pow(2)
    large = delta * residual - 0.5 * delta**2
    return torch.where(residual < delta, small, large)


def huber_loss_batch(y_true: torch.Tensor, y_pred: torch.Tensor, delta: float) -> torch.Tensor:
    return huber_loss_elementwise(y_true, y_pred, delta).mean(dim=-1).mean()


def switch_state_penalty(y_true: torch.Tensor, y_pred: torch.Tensor, norm_threshold: float) -> torch.Tensor:
    true_state = (y_true > norm_threshold).float()
    pred_state = (y_pred > norm_threshold).float()
    return (true_state - pred_state).pow(2).mean(dim=-1)


def build_training_loss(cfg, norm_stats):
    train_cfg = cfg["training"]
    loss_name = train_cfg.get("loss", "l1").lower()
    delta = float(train_cfg.get("huber_delta", 0.5))

    if loss_name == "l1":
        regression_loss = lambda pred, y: nn.functional.l1_loss(pred, y)
        return regression_loss, regression_loss, None

    if loss_name == "combined":
        alpha = float(train_cfg.get("regression_alpha", 1.0))
        beta = float(train_cfg.get("switch_beta", 0.1))
        on_threshold_w = get_on_threshold(cfg, cfg["data"]["appliance"])
        norm_threshold = get_norm_on_threshold(on_threshold_w, norm_stats)

        def regression_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return huber_loss_batch(y, pred, delta)

        def combined_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            switch = switch_state_penalty(y, pred, norm_threshold).mean()
            return alpha * regression_loss(pred, y) + beta * switch

        return combined_loss, regression_loss, norm_threshold

    raise ValueError(f"Unknown training.loss: {loss_name}")


def load_data(cfg):
    data = cfg["data"]
    loader = cfg["dataloader"]
    data_dir = CODE_DIR / data["data_dir"]
    window_len = data["window_len"]
    appliance = data["appliance"]

    train_agg, train_app, train_on = load_arrays(appliance, data["train_dataset"], data_dir, window_len)
    val_agg, val_app, val_on = load_arrays(appliance, data["val_dataset"], data_dir, window_len)
    test_agg, test_app, test_on = load_arrays(appliance, data["test_dataset"], data_dir, window_len)

    norm_stats = get_norm_stats(cfg["normalization"], appliance)
    train_agg, train_app = normalize(train_agg, train_app, norm_stats)
    val_agg, val_app = normalize(val_agg, val_app, norm_stats)
    test_agg, test_app = normalize(test_agg, test_app, norm_stats)

    train_loader = DataLoader(
        NILMDataset(train_agg, train_app),
        batch_size=loader["batch_size"],
        shuffle=loader["shuffle_train"],
    )
    val_loader = DataLoader(
        NILMDataset(val_agg, val_app, val_on),
        batch_size=loader["batch_size"],
        shuffle=loader["shuffle_val"],
    )
    test_loader = DataLoader(
        NILMDataset(test_agg, test_app, test_on),
        batch_size=loader["batch_size"],
        shuffle=loader["shuffle_test"],
    )

    return train_loader, val_loader, test_loader, norm_stats


def resolve_device(cfg) -> torch.device:
    """Pick training device and print GPU info at startup."""
    train_cfg = cfg.get("training", {})
    device_name = str(train_cfg.get("device", "cuda")).lower()
    require_gpu = bool(train_cfg.get("require_gpu", device_name == "cuda"))
    cuda_available = torch.cuda.is_available()

    print("=" * 10)
    print("Device check")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")
    else:
        print("  No GPU detected — training will use CPU unless you install CUDA PyTorch.")

    if device_name in ("cuda", "gpu"):
        if not cuda_available:
            if require_gpu:
                raise RuntimeError(
                    "training.device is 'cuda' but no GPU was found. "
                    "Install a CUDA-enabled PyTorch build or set training.device: cpu."
                )
            print("  WARNING: GPU requested but not available — falling back to CPU.")
            device = torch.device("cpu")
        else:
            gpu_id = int(train_cfg.get("gpu_id", 0))
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(
                    f"training.gpu_id={gpu_id} is invalid; only {torch.cuda.device_count()} GPU(s) found."
                )
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
    elif device_name == "cpu":
        if require_gpu:
            raise RuntimeError("training.require_gpu is true but training.device is 'cpu'.")
        device = torch.device("cpu")
    elif device_name == "auto":
        device = torch.device("cuda" if cuda_available else "cpu")
    else:
        raise ValueError(f"Unknown training.device: {device_name} (use cuda, cpu, or auto)")

    print(f"  Using device: {device}")
    if device.type == "cuda":
        print(f"  Active GPU: {torch.cuda.get_device_name(device)}")
    print("=" * 10)
    return device


def get_checkpoint_path(cfg) -> Path:
    appliance = cfg["data"]["appliance"]
    train_cfg = cfg["training"]
    save_dir = CODE_DIR / train_cfg.get("save_dir", "checkpoints")
    name_template = train_cfg.get("checkpoint_name", "{appliance}_best_epoch.pt")
    return save_dir / name_template.format(appliance=appliance)


def build_model(cfg, device):
    model_cfg = cfg["model"]
    name = model_cfg["name"].lower()

    if name == "cnn":
        return CNN(
            input_channels=model_cfg["input_channels"],
            hidden_channels=model_cfg["hidden_channels"],
            output_channels=model_cfg["output_channels"],
        ).to(device)

    if name == "transformer":
        raise NotImplementedError("Transformer model not added yet. Set model.name: cnn")

    raise ValueError(f"Unknown model.name: {name}")


def _setup_paper_style() -> None:
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
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


def plot_epoch_curve(values, cfg, save_stem: Path, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    _setup_paper_style()

    series = np.asarray(values, dtype=float)
    epochs = np.arange(1, len(series) + 1)
    n_epochs = len(series)
    markevery = max(1, n_epochs // 8)

    fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
    ax.plot(epochs, series, marker="o", markersize=3.5, markevery=markevery, color="#1f77b4", label="Validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(1, n_epochs)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=min(8, max(3, n_epochs // 5)), integer=True, min_n_ticks=3))

    y_min = float(series.min())
    y_max = float(series.max())
    y_pad = max((y_max - y_min) * 0.12, y_max * 0.05, 1e-4)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=True, loc="best")

    save_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(save_stem.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(train_losses, val_losses, cfg, save_stem: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    _setup_paper_style()

    train_arr = np.asarray(train_losses, dtype=float)
    val_arr = np.asarray(val_losses, dtype=float)
    epochs = np.arange(1, len(train_arr) + 1)
    n_epochs = len(train_arr)
    appliance = cfg["data"]["appliance"]
    model_name = cfg["model"]["name"].upper()
    markevery = max(1, n_epochs // 8)

    fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
    ax.plot(epochs, train_arr, marker="o", markersize=3.5, markevery=markevery, label="Train loss")
    ax.plot(epochs, val_arr, marker="s", markersize=3.5, markevery=markevery, label="Validation loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss (normalized)")
    ax.set_title(f"{model_name} training curve ({appliance})")
    ax.set_xlim(1, n_epochs)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=min(8, max(3, n_epochs // 5)), integer=True, min_n_ticks=3))

    y_ref = np.concatenate([train_arr[1:] if n_epochs > 1 else train_arr, val_arr])
    y_min = float(y_ref.min())
    y_max = float(y_ref.max())
    y_pad = max((y_max - y_min) * 0.12, y_max * 0.05, 1e-4)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=True, loc="best")

    save_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(save_stem.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _on_slice_bounds(on_mask: np.ndarray, window_len: int, pad: int = 48) -> slice:
    on_idx = np.flatnonzero(on_mask)
    if on_idx.size == 0:
        return slice(0, window_len)
    start = max(0, int(on_idx[0]) - pad)
    end = min(window_len, int(on_idx[-1]) + pad + 1)
    return slice(start, end)


def plot_on_period_samples(model, cfg, norm_stats, device, save_stem: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_paper_style()

    train_cfg = cfg["training"]
    data = cfg["data"]
    appliance = data["appliance"]
    window_len = data["window_len"]
    split = train_cfg.get("on_sample_split", "val").lower()
    n_samples = int(train_cfg.get("n_on_samples", 3))
    on_threshold = get_on_threshold(cfg, appliance)

    dataset_name = data[f"{split}_dataset"]
    data_dir = CODE_DIR / data["data_dir"]
    agg_w, app_w, on_off = load_arrays(appliance, dataset_name, data_dir, window_len)

    if on_off is None:
        on_off = (app_w >= on_threshold).astype(np.int8)

    on_energy = (app_w * on_off).sum(axis=1)
    candidate_idx = np.flatnonzero(on_energy > 0)
    if candidate_idx.size == 0:
        print(f"no ON windows found in {split} set — skipping on-period plot")
        return

    pick = candidate_idx[np.argsort(on_energy[candidate_idx])[-n_samples:][::-1]]
    agg_n, _ = normalize(agg_w, app_w, norm_stats)

    model.eval()
    fig, axes = plt.subplots(n_samples, 1, figsize=(3.5, 3.5 * n_samples), constrained_layout=True)
    if n_samples == 1:
        axes = [axes]

    for ax, win_idx in zip(axes, pick):
        x = torch.tensor(agg_n[win_idx : win_idx + 1], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_w = to_watts_tensor(model(x)[0], norm_stats)

        true_w = app_w[win_idx]
        mains_w = agg_w[win_idx]
        sl = _on_slice_bounds(on_off[win_idx], window_len)
        t = np.arange(sl.start, sl.stop)
        mark_step = max(1, len(t) // 12)

        ax.plot(t, mains_w[sl], color="#7f7f7f", linewidth=1.4, linestyle="-", label="Aggregate", zorder=1)
        ax.plot(
            t,
            true_w[sl],
            color="#d62728",
            linewidth=2.0,
            linestyle="-",
            label="Ground truth",
            zorder=3,
        )
        ax.plot(
            t,
            pred_w[sl],
            color="#1f77b4",
            linewidth=1.8,
            linestyle="--",
            marker="o",
            markersize=3.5,
            markevery=mark_step,
            label="Prediction",
            zorder=4,
        )
        ax.axhline(
            on_threshold,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.0,
            label=f"ON threshold ({on_threshold:.0f} W)",
            zorder=2,
        )
        ax.set_ylabel("Power (W)")
        ax.set_title(f"{split} window {win_idx} (ON energy {on_energy[win_idx]:.0f} W)", pad=4)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(frameon=True, loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Timestep in window")
    fig.suptitle(f"{cfg['model']['name'].upper()} ON-period samples ({appliance})", fontsize=10, y=1.01)

    save_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(save_stem.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_checkpoint_model(model, save_path: Path, device) -> dict | None:
    if not save_path.exists():
        return None
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


if __name__ == "__main__":
    with open(CODE_DIR / "hyperparameter.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg)

    train_loader, val_loader, test_loader, norm_stats = load_data(cfg)

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        batch = next(iter(loader))
        x, y = batch[0], batch[1]
        y_watts = denormalize_appliance(y.numpy(), norm_stats)
        print(
            f"{name}: {len(loader.dataset)} windows, "
            f"x={tuple(x.shape)}, y={tuple(y.shape)}, "
            f"y_watts min={y_watts.min():.1f}, max={y_watts.max():.1f}"
        )

    model = build_model(cfg, device)
    print(f"model: {cfg['model']['name']}")
    print(f"normalization: {norm_stats}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    train_losses = []
    val_losses = []
    val_maes = []
    val_f1s = []

    torch.manual_seed(cfg["training"]["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    loss_fn, regression_loss_fn, norm_on_threshold = build_training_loss(cfg, norm_stats)
    save_path = get_checkpoint_path(cfg)
    on_threshold = get_on_threshold(cfg, cfg["data"]["appliance"])
    to_watts = lambda tensor: to_watts_tensor(tensor, norm_stats)

    print(f"loss: {cfg['training'].get('loss', 'l1')} (curve plots Huber/L1 regression only)")
    if norm_on_threshold is not None:
        print(f"F1/train switch threshold: {on_threshold} W ({norm_on_threshold:.4f} normalized)")

    since = time.time()

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"Epoch {epoch}/{cfg['training']['epochs']}")
        print("-" * 10)

        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += regression_loss_fn(pred, y).item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0], batch[1]
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_loss += regression_loss_fn(pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_metrics = evaluate_loader(model, val_loader, device, to_watts, on_threshold)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_metrics["mae"])
        val_f1s.append(val_metrics["f1"])

        print(f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
        print(
            f"val MAE: {val_metrics['mae']:.2f} W | "
            f"val SAE: {val_metrics['sae']:.2f}% | "
            f"val F1: {val_metrics['f1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_metrics
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "norm_stats": norm_stats,
                    "best_val_loss": best_val_loss,
                    "best_val_metrics": best_val_metrics,
                    "epoch": epoch,
                    "appliance": cfg["data"]["appliance"],
                    "train_dataset": cfg["data"]["train_dataset"],
                    "cfg": cfg,
                },
                save_path,
            )
            print(f"saved best model -> {save_path.name} | epoch {epoch} | val loss: {best_val_loss:.4f}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            test_loss += regression_loss_fn(pred, y).item() * x.size(0)
    test_loss /= len(test_loader.dataset)
    test_metrics = evaluate_loader(model, test_loader, device, to_watts, on_threshold)

    elapsed = time.time() - since
    print("-" * 10)
    print(f"test loss: {test_loss:.4f}")
    print(
        f"test MAE: {test_metrics['mae']:.2f} W | "
        f"test SAE: {test_metrics['sae']:.2f}% | "
        f"test F1: {test_metrics['f1']:.4f}"
    )
    print(f"best val loss: {best_val_loss:.4f}")
    print(f"training time: {elapsed:.1f}s")
    print("=" * 10)
    print("Best checkpoint summary")
    print(f"  file: {save_path}")
    print(f"  appliance: {cfg['data']['appliance']}")
    print(f"  train dataset: {cfg['data']['train_dataset']}")
    print(f"  best epoch: {best_epoch} (lowest val Huber loss)")
    if best_val_metrics:
        print(f"  val MAE: {best_val_metrics['mae']:.2f} W")
        print(f"  val SAE: {best_val_metrics['sae']:.2f}%")
        print(f"  val F1: {best_val_metrics['f1']:.4f}")
    print("=" * 10)

    appliance = cfg["data"]["appliance"]
    model_name = cfg["model"]["name"].upper()
    figures_dir = CODE_DIR / "figures"

    plot_loss_curve(train_losses, val_losses, cfg, CODE_DIR / cfg["training"]["loss_plot_path"])
    plot_epoch_curve(val_maes, cfg, CODE_DIR / cfg["training"]["mae_plot_path"], "MAE (W)", f"{model_name} val MAE ({appliance})")
    plot_epoch_curve(val_f1s, cfg, CODE_DIR / cfg["training"]["f1_plot_path"], "F1 score", f"{model_name} val F1 ({appliance})")

    ckpt = load_checkpoint_model(model, save_path, device)
    if ckpt is not None:
        print(f"loaded best checkpoint (epoch {ckpt.get('epoch')}) for ON-period plots")
    plot_on_period_samples(model, cfg, norm_stats, device, CODE_DIR / cfg["training"]["on_sample_plot_path"])

    print(f"loss plot saved: {(figures_dir / 'loss_curve.png')}")
    print(f"mae plot saved: {(figures_dir / 'mae_curve.png')}")
    print(f"f1 plot saved: {(figures_dir / 'f1_curve.png')}")
    print(f"on-period plot saved: {(figures_dir / 'on_period_samples.png')}")