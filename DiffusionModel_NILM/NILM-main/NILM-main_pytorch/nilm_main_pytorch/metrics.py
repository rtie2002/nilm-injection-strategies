"""
Geng NILM metrics — exact port of NILM-main/nilm_metric.py + EasyS2S_test.py usage.

UK-DALE sampling: 6 s per row (sample_second=6.0).

Test scripts use:
  MAE  -> get_abs_error(...)[0]
  SAE  -> get_sae(..., sample_second=6.0)   # ratio, not percent
  F1   -> recall_precision_accuracy_f1(...)[3]
"""

from __future__ import annotations

import numpy as np
import torch

# UK-DALE Geng pipeline: 6 s resolution (EasyS2S_test.py sample_second = 6.0)
DEFAULT_SAMPLE_SECOND = 6.0


def get_TP(target: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    assert target.shape == prediction.shape
    target = 1 - np.clip(target, threshold, 0) / threshold
    prediction = 1 - np.clip(prediction, threshold, 0) / threshold
    return float(np.sum(np.logical_and(target, prediction) * 1.0))


def get_FP(target: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    assert target.shape == prediction.shape
    target = np.clip(target, threshold, 0) / threshold
    prediction = 1 - np.clip(prediction, threshold, 0) / threshold
    return float(np.sum(np.logical_and(target, prediction) * 1.0))


def get_FN(target: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    assert target.shape == prediction.shape
    target = 1 - np.clip(target, threshold, 0) / threshold
    prediction = np.clip(prediction, threshold, 0) / threshold
    return float(np.sum(np.logical_and(target, prediction) * 1.0))


def get_TN(target: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    assert target.shape == prediction.shape
    target = np.clip(target, threshold, 0) / threshold
    prediction = np.clip(prediction, threshold, 0) / threshold
    return float(np.sum(np.logical_and(target, prediction) * 1.0))


def get_recall(target: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    tp = get_TP(target, prediction, threshold)
    fn = get_FN(target, prediction, threshold)
    if tp + fn <= 0.0:
        return tp / (tp + fn + 1e-9)
    return tp / (tp + fn)


def get_precision(target: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    tp = get_TP(target, prediction, threshold)
    fp = get_FP(target, prediction, threshold)
    if tp + fp <= 0.0:
        return tp / (tp + fp + 1e-9)
    return tp / (tp + fp)


def get_F1(target: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    recall = get_recall(target, prediction, threshold)
    precision = get_precision(target, prediction, threshold)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def get_abs_error(target: np.ndarray, prediction: np.ndarray) -> tuple[float, ...]:
    assert target.shape == prediction.shape
    data = np.abs(target - prediction)
    mean = float(np.mean(data))
    std = float(np.std(data))
    sorted_data = np.sort(data)
    min_v = float(sorted_data[0])
    max_v = float(sorted_data[-1])
    quartile1 = float(np.percentile(data, 25))
    median = float(np.percentile(data, 50))
    quartile2 = float(np.percentile(data, 75))
    return mean, std, min_v, max_v, quartile1, median, quartile2


def get_sae(target: np.ndarray, prediction: np.ndarray, sample_second: float) -> float:
    """Signal aggregate error: |E_pred - E_true| / |E_true| (kWh via sample_second)."""
    r = np.sum(target * sample_second * 1.0 / 3600.0)
    rhat = np.sum(prediction * sample_second * 1.0 / 3600.0)
    return float(np.abs(r - rhat) / np.abs(r))


def tp_tn_fp_fn(states_pred: np.ndarray, states_ground: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum(np.logical_and(states_pred == 1, states_ground == 1)))
    fp = int(np.sum(np.logical_and(states_pred == 1, states_ground == 0)))
    fn = int(np.sum(np.logical_and(states_pred == 0, states_ground == 1)))
    tn = int(np.sum(np.logical_and(states_pred == 0, states_ground == 0)))
    return tp, tn, fp, fn


def _recall(tp: float, fn: float) -> float:
    return tp / float(tp + fn)


def _precision(tp: float, fp: float) -> float:
    return tp / float(tp + fp)


def _f1(prec: float, rec: float) -> float:
    return 2 * (prec * rec) / float(prec + rec)


def _accuracy(tp: float, tn: float, p: float, n: float) -> float:
    return (tp + tn) / float(p + n)


def recall_precision_accuracy_f1(
    pred: np.ndarray,
    ground: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, float] | None:
    """Exact copy of nilm_metric.recall_precision_accuracy_f1 (used in *_test.py)."""
    pred = np.asarray(pred).flatten()
    ground = np.asarray(ground).flatten()
    if len(pred) == 0:
        return None

    pr = np.array([0 if p < threshold else 1 for p in pred])
    gr = np.array([0 if p < threshold else 1 for p in ground])

    tp, tn, fp, fn = tp_tn_fp_fn(pr, gr)
    p = float(np.sum(pr))
    n = float(len(pr) - p)

    res_recall = _recall(tp, fn)
    res_precision = _precision(tp, fp)
    res_f1 = _f1(res_precision, res_recall)
    res_accuracy = _accuracy(tp, tn, p, n)
    return res_recall, res_precision, res_accuracy, res_f1


def compute_mae(pred_w: np.ndarray, true_w: np.ndarray) -> float:
    return get_abs_error(true_w.flatten(), pred_w.flatten())[0]


def compute_sae(
    pred_w: np.ndarray,
    true_w: np.ndarray,
    sample_second: float = DEFAULT_SAMPLE_SECOND,
) -> float:
    return get_sae(true_w.flatten(), pred_w.flatten(), sample_second)


def compute_f1(pred_w: np.ndarray, true_w: np.ndarray, threshold: float) -> float:
    on_off = recall_precision_accuracy_f1(pred_w.flatten(), true_w.flatten(), threshold)
    if on_off is None:
        return 0.0
    return float(on_off[3])


def compute_metrics(
    pred_w: np.ndarray,
    true_w: np.ndarray,
    threshold: float,
    sample_second: float = DEFAULT_SAMPLE_SECOND,
) -> dict[str, float]:
    pred_f = np.asarray(pred_w).flatten()
    true_f = np.asarray(true_w).flatten()
    on_off = recall_precision_accuracy_f1(pred_f, true_f, threshold)
    out: dict[str, float] = {
        "mae": compute_mae(pred_f, true_f),
        "sae": compute_sae(pred_f, true_f, sample_second),
        "f1": float(on_off[3]) if on_off is not None else 0.0,
    }
    if on_off is not None:
        out["recall"] = float(on_off[0])
        out["precision"] = float(on_off[1])
        out["accuracy"] = float(on_off[2])
    return out


def denorm_appliance(tensor: torch.Tensor, mean: float, std: float) -> np.ndarray:
    return tensor.detach().cpu().numpy() * std + mean


def prepare_predictions_geng(pred_w: np.ndarray) -> np.ndarray:
    """EasyS2S_test.py: clip negatives to zero before metrics."""
    out = np.asarray(pred_w, dtype=np.float64).flatten().copy()
    out[out <= 0.0] = 0.0
    return out


def evaluate_loader(
    model,
    loader,
    device: torch.device,
    appliance_mean: float,
    appliance_std: float,
    threshold_w: float,
    sample_second: float = DEFAULT_SAMPLE_SECOND,
) -> dict[str, float]:
    model.eval()
    pred_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)
            pred_chunks.append(denorm_appliance(pred.cpu(), appliance_mean, appliance_std).reshape(-1))
            true_chunks.append(denorm_appliance(y, appliance_mean, appliance_std).reshape(-1))

    pred_w = prepare_predictions_geng(np.concatenate(pred_chunks))
    true_w = prepare_predictions_geng(np.concatenate(true_chunks))
    return compute_metrics(pred_w, true_w, threshold_w, sample_second)


def apply_postprocess(pred_w: np.ndarray, *, augmented: bool, threshold_w: float, ewma_alpha: float) -> np.ndarray:
    pred_w = prepare_predictions_geng(pred_w)
    if augmented:
        from nilm_main_pytorch.ewma import conditional_ema

        return conditional_ema(pred_w, threshold=threshold_w, alpha=ewma_alpha)
    return pred_w
