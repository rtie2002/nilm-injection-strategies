# NILM-main — PyTorch Version

**PyTorch port of `NILM-main/` (TensorFlow).**  
Original TF scripts remain in the parent folder `../` (EasyS2S_train.py, S2P_baseline_train.py, etc.).

## Location

```text
DiffusionModel_NILM/NILM-main/
  EasyS2S_*.py, DataProvider.py, ...     ← TensorFlow (original)
  dataset_preprocess/created_data/        ← shared CSV data
  NILM-main_pytorch/                    ← this folder (PyTorch)
    nilm_main_pytorch/                  ← Python package
    config/default.yaml
    checkpoints/
    results/
```

## Quick start

```bash
cd DiffusionModel_NILM/NILM-main/NILM-main_pytorch

python -m nilm_main_pytorch.train --model easy_s2s --appliance kettle
python -m nilm_main_pytorch.train --model s2p --appliance kettle --augmented
python -m nilm_main_pytorch.test --model easy_s2s --appliance kettle --augmented --test-house 2 --ewma
python -m nilm_main_pytorch.run_experiments --suite table8 --phase train
python -m nilm_main_pytorch.run_all_scenarios --phase train_test
```

## Models

| Model | Paper tables |
|-------|----------------|
| easy_s2s | Tables 5–7 |
| s2p, fcn, auglpn | Tables 8–9 |

Config: `config/default.yaml`  
Reproduction checklist: `REPRODUCTION_A.md`
