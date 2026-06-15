import numpy as np


def power_to_on_off(power_w: np.ndarray, threshold: float) -> np.ndarray:
    return (power_w >= threshold).astype(np.int8)


def compute_mae(pred_w: np.ndarray, true_w: np.ndarray) -> float:
    return float(np.mean(np.abs(pred_w - true_w)))


def compute_sae(pred_w: np.ndarray, true_w: np.ndarray) -> float:
    true_energy = float(true_w.sum())
    if true_energy == 0.0:
        return 0.0 if float(pred_w.sum()) == 0.0 else 100.0
    return float(abs(true_energy - float(pred_w.sum())) / true_energy * 100.0)


def compute_f1(pred_w: np.ndarray, true_w: np.ndarray, threshold: float) -> float:
    """F1 from regression outputs: binarize both pred and true power at threshold (W)."""
    pred_on = power_to_on_off(pred_w, threshold)
    true_on = power_to_on_off(true_w, threshold)

    tp = int(np.sum((pred_on == 1) & (true_on == 1)))
    fp = int(np.sum((pred_on == 1) & (true_on == 0)))
    fn = int(np.sum((pred_on == 0) & (true_on == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_metrics(pred_w: np.ndarray, true_w: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "mae": compute_mae(pred_w, true_w),
        "sae": compute_sae(pred_w, true_w),
        "f1": compute_f1(pred_w, true_w, threshold),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return f"MAE: {metrics['mae']:.2f} W | SAE: {metrics['sae']:.2f}% | F1: {metrics['f1']:.4f}"


def evaluate_loader(model, loader, device, to_watts, threshold: float) -> dict[str, float]:
    model.eval()
    pred_chunks = []
    true_chunks = []

    import torch

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            pred = model(x)
            pred_chunks.append(np.asarray(to_watts(pred.cpu())).reshape(-1))
            true_chunks.append(np.asarray(to_watts(y)).reshape(-1))

    pred_w = np.concatenate(pred_chunks)
    true_w = np.concatenate(true_chunks)
    return compute_metrics(pred_w, true_w, threshold)
