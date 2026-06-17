# Pipeline A — Geng Reproduction Checklist

**Goal:** Match Geng et al. (Energy 2025) before changing injection strategy (Pipeline B is separate).

## What is aligned

| Component | Geng (TF) | PyTorch port |
|-----------|-----------|--------------|
| Data z-score 2-col CSV | `prepare_all_ukdale.py` | Same files |
| Mix `D_mix = concat(real, syn)` | `ChunkS2S_Slider(shuffle=True)` at train | `build_geng_mix.py` concat only |
| EasyS2S window | 600 | 600 |
| Loss origin / augmented | MSE / combined (α=1, β=0.1, Huber δ=0.5) | `losses.py` |
| Metrics MAE / SAE / F1 | `nilm_metric.py` | `metrics.py` (exact port) |
| Test inference EasyS2S | `custompredictS2SX` overlap average | `inference.py` (default in `test.py`) |
| Test inference S2P / AugLPN | `custompredictX` | `inference.py` |
| Hyperparams | Table 3 / `*_train.py` | `config/default.yaml` `per_model` |

## Training CSV map (Tables 5–7, EasyS2S)

| Scenario | `--train-percent` | Training CSV |
|----------|-------------------|--------------|
| Origin 100k | `10` (no `--augmented`) | `{app}_10training_.csv` |
| Origin 200k | `20` (no `--augmented`) | `{app}_20training_.csv` |
| 100k + 100k | `10` `--augmented` | `UK_DALECombined{app}_file10.csv` |
| 200k + 200k | `20` `--augmented` | `UK_DALECombined{app}_file20.csv` |
| 100k + 200k | `10_20` `--augmented` | `UK_DALECombined{app}_file10_20.csv` |
| 200k + 100k | `20_10` `--augmented` | `UK_DALECombined{app}_file20_10.csv` |

Build mixes (all 4 augmented scenarios, including asymmetric 100k+200k / 200k+100k):

```bash
cd DiffusionModel_NILM && python build_geng_mix.py
```

If you already have `file10` / `file20` and only need the asymmetric mixes:

```bash
python build_geng_mix.py --scenario missing
# or explicitly:
python build_geng_mix.py --scenario 10_20
python build_geng_mix.py --scenario 20_10
```

## Reproduction commands

```bash
# 1. UK-DALE CSVs
python NILM-main/dataset_preprocess/prepare_all_ukdale.py

# 2. Diffusion sample (≥200k syn timesteps per appliance)
python run_diffusion_all.py --no-train

# 3. Geng mixes (all 4 augmented scenarios)
python build_geng_mix.py

# 4. Train + test (example: EasyS2S kettle, 200k+200k)
cd NILM-main/NILM-main_pytorch
python -m nilm_main_pytorch.train --model easy_s2s --appliance kettle --augmented --train-percent 20
python -m nilm_main_pytorch.test --model easy_s2s --appliance kettle --augmented --train-percent 20 --test-house 2 --ewma
```

## Validate against TensorFlow

Pick one appliance (e.g. kettle), one condition (origin 200k + augmented 200k+200k):

1. Train TF: `EasyS2S_train.py` / `EasyS2S_Abtrain.py`
2. Train PyTorch with same CSV
3. Compare test MAE / SAE / F1 on house 2

Use `--window-eval` on PyTorch test only for debugging (old window-batched eval, not paper protocol).

## Pipeline B (thesis injection) — NOT Pipeline A

- `injection_strategy/build_injection_datasets.py` (D0–D4)
- Root `model_train.py`, `model/cnn.py`, `hyperparameter.yaml`
- Do **not** mix with Geng EasyS2S until A reproduces; then adapt injection on top of A.
