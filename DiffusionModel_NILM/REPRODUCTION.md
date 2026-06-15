# Geng et al. (Energy 2025) — Full Reproduction Master Plan

Paper: *A diffusion model-based framework to enhance the robustness of non-intrusive load disaggregation*  
PDF: `A diffusion model-based framework to enhance the robustness of non-intrusive load disaggregation.pdf`

This document explains **Algorithm 1 (zeros + ON windows)**, **how aggregate power X is built**, **how mixing works**, what is **missing in the repo**, which **scripts we must add**, and the **full experiment grid** to reproduce Tables 5–9 — including the **TensorFlow → PyTorch** port.

---

## Part A — Core concepts (paper + code)

### A1. Algorithm 1 — what “putting 0” actually means

**Common misunderstanding:** Algorithm 1 does **not** mean “keep only ON timesteps.”  
**Correct:** It (1) **forces noise to zero**, (2) finds **ON events**, (3) keeps a **window around each ON**, including OFF minutes before/after.

#### Paper pseudocode (Sec 3.2; numeric value in **Table 2**)

**Table 2 lists three different “window” concepts — do not confuse them:**

| Table 2 row | Symbol | Value | Meaning |
|-------------|--------|-------|---------|
| **Algorithm1-window** | `l_window` | **100** | Timesteps kept **before and after** each ON index in Algorithm 1 |
| Window length | diffusion train window | **512** | Sliding window inside the diffusion model (`microwave.yaml` `window: 512`) |
| *(Table 3)* Window length | NILM input window | **600** | Seq2seq / S2P training window @ 6 s |

At UK-DALE **6 s** sampling: `l_window = 100` → **±10 minutes** of context per ON timestep (100 × 6 s = 600 s each side).

The paper **does not** justify why 100 was chosen; it is only given as a fixed hyperparameter in Table 2.

```text
Input:  power sequence x[1..T], threshold x_threshold, noise floor x_noise, window l_window
Step 2: x[x < x_noise] = 0          ← EXPLICITLY SET LOW POWER TO ZERO
Step 3: find indices where x >= x_threshold  (ON starts)
Step 4-7: for each ON index:
            keep timesteps [index - l_window, index + l_window]
Step 10: output only kept rows (still contain 0 W and high W)
Step 11-12: MinMax normalize selected excerpt → diffusion training CSV
```

#### Worked example (microwave, threshold = 200 W, l_window = 3 for demo)

```text
Original watts:  [0, 0, 0, 50, 800, 850, 820, 40, 0, 0, 0, 0, 1200, 1180, 0]
After step 2:    [0, 0, 0, 50, 800, 850, 820, 40,  0, 0, 0, 0, 1200, 1180, 0]
ON indices:      50W may be <200; ON starts at index 4 and 12
Keep windows:    merge [1..7] and [9..15] → most indices selected
Selected rows:   include many 0 W + ON burst + decay — NOT only 800 W rows
```

**Evidence in paper:** Sec 3.2 prose: *“window length **before** appliance turns on and **after** it turns off.”*

**Evidence in repo:** Not shipped. Closest implementation: your PhD file  
`PhD/Code/multi-domain NILM/.../algorithm1_v2_multivariate.py` (lines 160–200).

**Scope:** Algorithm 1 is **only for diffusion training data** (paper Sec 4.1).  
Real NILM train/test CSVs from `ukdale_processing.py` use the **full timeline** (no Algorithm 1).

---

### A2. Diffusion training vs diffusion output

| Stage | Data | Contains 0 W? |
|-------|------|----------------|
| Diffusion **training** (after Alg. 1) | ON-focused excerpts, MinMax scaled | **Yes** — many rows are 0 or low |
| Diffusion **generation** (`main.py --sample`) | Long sequences (seq len 10 000, window 512) | **Yes** — model learned OFF→ON→OFF shapes |
| Post-filter (Sec 4.1) | Discard invalid; enforce power > 0 where needed | Can **keep** zeros in timeline |

Diffusion trains on **per-appliance** single-channel CSV (`Data/datasets/{app}.csv`).  
It does **not** train on aggregate mains.

---

### A3. How aggregate power X is reconstructed (synthetic NILM rows)

**Paper (Sec 3.2):**  
> *“The synthesized aggregate power can be expressed as the **sum of the synthesized electrical appliance powers**.”*

For each synthetic timestep `t` and each target appliance model:

```text
X_syn(t) = Y_kettle_syn(t) + Y_microwave_syn(t) + Y_fridge_syn(t)
         + Y_dishwasher_syn(t) + Y_washingmachine_syn(t)

Y_target(t) = Y_<appliance>_syn(t)    # column trained by NILM
```

**NOT used for synthetic rows:**

```text
❌ X_syn = X_real_background - Y_real_target + Y_syn   (your D1/D2 injection rule)
❌ X_syn = real mains from house 2
```

**Requirements:**

1. Generate **five independent** synthetic watt traces (one diffusion model per appliance).
2. Align by **row index** `t` (same length; ideally generated with matched length).
3. Sum in **watts** (after inverse MinMax — see `InverseTransform.py`).
4. Z-score `X_syn` and `Y_target` with **the same mean/std** as real house-2 training data (paper Sec 4.1).

**Real rows** in the mixed CSV still use **real mains** from `ukdale_processing.py`:

```text
X_real(t) = house-2 aggregate meter (channel 1)
Y_real(t) = house-2 appliance submeter
```

---

### A4. How mixing works (exact procedure)

Paper notation: `D_mix = D_real ∪ D_syn`, sizes in `{10⁵, 2×10⁵}`, **shuffle in DataLoader**.

#### Step-by-step (e.g. Table 8 `*` = 200k+200k)

```text
1. REAL BLOCK
   - Take first 200,000 timesteps from house-2 train stream (6 s sampling)
   - Each row: (X_real, Y_appliance) from ukdale_processing
   - Includes long OFF periods

2. SYNTHETIC BLOCK
   - Take first 200,000 timesteps from synthetic timeline
   - Each row: (X_syn = sum of 5 apps, Y_appliance_syn)
   - Also includes OFF and ON minutes

3. CONCATENATE (union at row level)
   D_mix = [real_rows (200k); syn_rows (200k)]  → 400k rows
   NOT window append. NOT event insertion.

4. Z-SCORE entire D_mix (paper: after mixing, before NILM training)
   X_norm = (X - mean_X) / std_X
   Y_norm = (Y - mean_Y) / std_Y
   Stats computed on the 400k mixed training set (or paper may use real stats — verify; Sec 4.1 says mixed set normalized)

5. SAVE CSV for NILM
   Two columns: [aggregate, appliance]  (no header or header=0 per their loaders)

6. AT TRAIN TIME (DataProvider ChunkS2S_Slider)
   - Load full CSV into memory
   - Build sliding windows of length 600 (Table 3)
   - Shuffle WINDOW START INDICES each epoch (shuffle=True)
   - Each window: 600 consecutive rows from shuffled-index sampling
```

**What mixing is NOT (your injection strategies):**

| Method | Geng mix |
|--------|----------|
| D1 full-window append | Row-level concat |
| D2 ON-event insert into OFF | Full synthetic timelines |
| D3 balanced | N/A |

**Missing in GitHub repo:** Steps 2–5 script. Training code assumes pre-built files:

```text
{appliance}_20training_.csv                 # baseline real
UK_DALECombined{appliance}_file20.csv       # mixed augmented
```

(`EasyS2S_train.py` lines 148–158; `TrainPercent='20'` encodes experiment id, not 20%.)

---

## Part B — What the repo has vs what is missing

### B1. Present

| Item | Path | Stack |
|------|------|--------|
| Diffusion model | `main.py`, `Models/diffusion/` | PyTorch |
| NILM EasyS2S | `NILM-main/EasyS2S_*.py` | TensorFlow/Keras |
| NILM S2P / FCN / AugLPN | `S2P_*`, `fcn_*`, `AugLPNNILM_*` | TensorFlow |
| Real preprocessor | `NILM-main/dataset_preprocess/ukdale_processing.py` | pandas |
| Metrics | `NILM-main/nilm_metric.py` | numpy |
| 2 microwave `.h5` weights | `NILM-main/models/EasyS2S/UK_DALE/` | TF |

### B2. Missing (must build or obtain)

| # | Item | Why needed |
|---|------|------------|
| 1 | UK-DALE raw `.dat` (houses 1, 2) | All steps |
| 2 | Algorithm 1 exporter → `Data/datasets/{app}.csv` | Diffusion train |
| 3 | Config YAML ×4 (kettle, fridge, DW, WM) | Diffusion train |
| 4 | Fix `main.py` `args.opts` bug | Diffusion run |
| 5 | Diffusion checkpoints `.pt` ×5 | Skip 20k-epoch train |
| 6 | Batch sample + inverse MinMax script | Synthetic watts |
| 7 | **Sum-of-5 aggregate builder** | Synthetic X |
| 8 | **Mix script** (crop + concat + z-score + save) | NILM augmented train |
| 9 | Real NILM train/val CSVs ×5 | Baseline |
| 10 | Mixed NILM CSVs ×5 × mix ratios | Tables 5–9 grid |
| 11 | `*genvalidation*` CSVs | `EasyS2S_Abtrain` val |
| 12 | Pretrained `.h5` all models × conditions | Or train 30+ runs |
| 13 | `requirements.txt` / Docker | Repro env |
| 14 | End-to-end runner + results collector | Automate tables |
| 15 | PyTorch NILM models | Replace TensorFlow |

---

## Part C — Scripts we need to implement

Place under `DiffusionModel_NILM/scripts/` (to be created).

### C1. `algorithm1_export.py`

**Purpose:** Paper Algorithm 1 → `Data/datasets/{appliance}.csv` (single `power` column, MinMax).

**How:**

1. Read UK-DALE house-2 appliance @ 6 s (or reuse `ukdale_processing` alignment).
2. Run Algorithm 1 (`l_window=100`, thresholds from paper Table 1).
3. MinMax scale to [0,1] with fixed `max_power` per appliance.
4. Save CSV for `CustomDataset` in `microwave.yaml`.

**Input:** `--ukdale-dir`, `--appliance`, `--output Data/datasets/{app}.csv`  
**Borrow from:** `algorithm1_v2_multivariate.py` in PhD repo.

---

### C2. `train_diffusion_all.sh` / `train_diffusion_all.py`

**Purpose:** Train diffusion for all 5 appliances.

```bash
for app in kettle microwave fridge dishwasher washingmachine; do
  python main.py --name $app --config Config/${app}.yaml --gpu 0 --train
done
```

**Also:** fix `main.py`:

```python
parser.add_argument('opts', nargs=argparse.REMAINDER, default=None)
```

**Create:** `Config/{kettle,fridge,dishwasher,washingmachine}.yaml` (copy microwave, change paths).

**Paper params (Table 2):** 20 000 epochs, batch 128, window 512, seq 10 000.

---

### C3. `sample_diffusion_all.py`

**Purpose:** Sample synthetic watts for all appliances.

**How:**

1. Load checkpoint per appliance (`--milestone`).
2. `trainer.sample(...)` → `ddpm_fake_{app}.npy`.
3. Inverse MinMax using scaler fit on **original house-2 appliance watts** (extend `InverseTransform.py`).
4. Post-filter: `power < threshold → 0`, drop invalid segments (Sec 4.1).
5. Save: `generatedData/{app}_watts.csv` with columns `[power]` or `[{app}]`.

**Output length:** ≥ 200 000 timesteps per appliance for Table 8/9 `200k+200k`.

---

### C4. `build_synthetic_aggregate.py`

**Purpose:** Build synthetic NILM rows with **X = sum of 5 appliances**.

```python
for t in range(n_syn):
    X_syn[t] = sum(Y_app[t] for app in APPLIANCES)
    Y_target[t] = Y_target_app[t]
```

**Input:** `generatedData/{app}_watts.csv` for all 5 (same `n_syn` rows).  
**Output:** `mixed/synthetic_{target_appliance}_{n_syn}.csv` with `[aggregate, appliance]`.

---

### C5. `build_geng_mix.py`

**Purpose:** Implement `D_mix = D_real ∪ D_syn`.

**CLI example:**

```bash
python scripts/build_geng_mix.py \
  --appliance microwave \
  --n-real 200000 --n-syn 200000 \
  --real-csv NILM-main/dataset_preprocess/created_data/UK_DALE/microwave/microwave_20training_.csv \
  --syn-csv mixed/synthetic_microwave_200000.csv \
  --zscore \
  --shuffle-seed 2024 \
  --output NILM-main/dataset_preprocess/created_data/UK_DALE/microwave/UK_DALECombinedmicrowave_file20.csv
```

**Steps inside:**

1. Load `n_real` real rows, `n_syn` synthetic rows.
2. `pd.concat([real, syn], ignore_index=True)`.
3. Optional: shuffle rows before save (paper also shuffles at DataLoader).
4. Compute z-score on combined columns; save normalized CSV **or** save raw + stats JSON (match their loader).

---

### C6. `prepare_real_nilm.py`

**Purpose:** Wrapper around `ukdale_processing.py` for all 5 appliances.

**Output per appliance:**

```text
{appliance}_20training_.csv      # crop 200k for baseline
{appliance}_validation_.csv
{appliance}_test_.csv              # house 2
# + house 1 test for cross-house (Table 6, 9)
```

---

### C7. `run_nilm_experiments.py`

**Purpose:** Train + test grid; write `results/geng_tables.csv`.

**Modes:**

- `--model {easy_s2s,s2p,fcn,auglpn}`
- `--appliance {kettle,...}`
- `--condition {origin_200k,100k+100k,...,200k+200k}`
- `--test-house {1,2}`

**Augmented runs:** enable `combined_loss` + EWMA (`EasyS2S_Abtrain.py` pattern).

---

### C8. `collect_tables.py`

**Purpose:** Aggregate MAE/SAE/F1 → LaTeX rows matching Tables 5–9.

---

## Part D — Full experiment matrix (repeat to reproduce paper)

### D1. Data mix conditions (all tables)

| Label | n_real | n_syn | Used in |
|-------|--------|-------|---------|
| Origin(200k) | 200 000 | 0 | Baseline |
| 100k+100k | 100 000 | 100 000 | Tables 5–7 |
| 100k+200k | 100 000 | 200 000 | Tables 5–7 |
| 200k+100k | 200 000 | 100 000 | Tables 5–7 |
| 200k+200k | 200 000 | 200 000 | Tables 5–7; **Tables 8–9 `*`** |

### D2. Table 5 — EasyS2S, test house **2**

| Runs | 5 appliances × 5 conditions = **25 trains** |
| Model | `EasyS2S_Abtrain.py` |
| Augmented loss | `combined_loss` (Huber + switch, α=1, β=0.1) |
| Test | `EasyS2S_test.py`, `originHome=True`, EWMA for augmented |

### D3. Table 6 — EasyS2S, test house **1**

Same 25 trains; test on house 1 (`originHome=False` or house-1 test CSV).

### D4. Table 7 — EasyS2S averages

Average MAE/SAE/F1 over 5 appliances per condition (post-process from D2/D3 logs).

### D5. Table 8 — S2P, FCN, AugLPN, test house **2**

| Model | Baseline | Augmented (`*`) |
|-------|----------|-----------------|
| S2P | Origin(200k) | 200k+200k + framework loss |
| FCN | same | same |
| AugLPN | same | same |

**Runs:** 3 models × 5 appliances × 2 conditions = **30 trains** (test once per train).

### D6. Table 9 — S2P, FCN, AugLPN, test house **1**

Same 30 trains; evaluate on house 1 test set.

### D7. NILM hyperparameters (Table 3 — must match)

| Parameter | Value |
|-----------|--------|
| Window length | **600** timesteps @ 6 s |
| Batch size | 1024 |
| Epochs | 100 |
| Patience | 20 |
| Optimizer | Adam, lr 1e-3 |
| Loss (augmented) | Huber δ=0.5 + ON/OFF term α=0.1 |
| Post-process | EWMA β ∈ {0.5, 0.9} |

### D8. Metrics (must match `nilm_metric.py`)

- **MAE** — mean |pred − truth| (watts, denormalized)
- **SAE** — |Σ pred − Σ truth| / Σ truth
- **F1** — ON/OFF with thresholds from paper Table 1

---

## Part E — TensorFlow → PyTorch port (required for unified repo)

| Component | Current | Action |
|-----------|---------|--------|
| Diffusion | PyTorch | Keep |
| EasyS2S CNN | TF `EasyS2S_Model.py` | Port to `torch.nn` |
| S2P | TF | Port |
| FCN | TF | Port |
| AugLPN | TF | Port |
| `DataProvider` sliding window | TF feed | `Dataset` + `DataLoader` |
| `combined_loss` | TF graph | PyTorch loss module |
| EWMA | numpy in test | Keep |
| Training loop | `NetFlowExt.py` | Standard PyTorch loop |

**Port order:** DataLoader → EasyS2S → combined loss → S2P → FCN → AugLPN → experiment runner.

**Note:** `../model/cnn.py` in `nilm-injection-strategies` is a **different** architecture (your thesis CNN), not EasyS2S.

---

## Part F — Master checklist (do in this order)

### Phase 0 — Repo fixes (1 day)

- [ ] Fix `main.py` `args.opts` bug
- [ ] Add `scripts/` directory
- [ ] Add `requirements.txt` (torch, tensorflow==2.3, keras==2.4, pandas, sklearn, pypdf)
- [ ] Copy/create `Config/*.yaml` for 5 appliances

### Phase 1 — Data download (1 day)

- [ ] Download UK-DALE houses 1 & 2
- [ ] Place in `NILM-main/dataset_preprocess/UK_DALE/`
- [ ] Verify channel mapping matches `ukdale_processing.py` `params_appliance`

### Phase 2 — Algorithm 1 + diffusion train data (2–3 days)

- [ ] Implement `scripts/algorithm1_export.py`
- [ ] Export `Data/datasets/{app}.csv` for all 5 appliances
- [ ] Visual check: selected excerpts have 0 W before/after ON (plot like paper Fig. 5)

### Phase 3 — Diffusion train + sample (days–weeks GPU)

- [ ] `scripts/train_diffusion_all.py` — train 5 models (20k epochs each)
- [ ] `scripts/sample_diffusion_all.py` — ≥200k watts per appliance
- [ ] Inverse MinMax → `generatedData/{app}_watts.csv`
- [ ] Verify: ~90%+ rows below ON threshold for sparse appliances (OFF-heavy)

### Phase 4 — Aggregate + mix (2 days)

- [ ] `scripts/build_synthetic_aggregate.py` — X = sum(5 apps)
- [ ] `scripts/prepare_real_nilm.py` — real CSVs via ukdale_processing
- [ ] `scripts/build_geng_mix.py` — all mix ratios (100k/200k grid)
- [ ] Output mixed CSVs for each appliance × condition
- [ ] Sanity check: synthetic row `aggregate` ≈ sum of five appliance files at same index

### Phase 5 — NILM training TensorFlow (1–2 weeks)

- [ ] Baseline Origin(200k) for EasyS2S, S2P, FCN, AugLPN × 5 apps
- [ ] Augmented 200k+200k with `combined_loss` × same
- [ ] Optional: full 5×5 mix grid for Tables 5–7
- [ ] Save all `.h5` checkpoints with clear naming

### Phase 6 — Evaluation (2–3 days)

- [ ] Test house 2 → Table 5, 8
- [ ] Test house 1 → Table 6, 9
- [ ] Enable EWMA for augmented models
- [ ] `scripts/collect_tables.py` → compare to paper PDF

### Phase 7 — PyTorch port (2–4 weeks, parallel track)

- [ ] Port models + loss + DataLoader
- [ ] Re-run one appliance (microwave) to validate against TF
- [ ] Full grid in PyTorch

---

## Part G — Mapping to our thesis repo (`nilm-injection-strategies`)

We use **1-minute** data, not 6 s UK-DALE. Direct numeric match to Geng tables is unlikely.

| Geng | Our repo |
|------|----------|
| 200k @ 6 s | 100k @ 1 min (`*_train_100k.csv`) |
| Window 600 (~60 min) | Window 512 (~512 min) |
| Sum-of-5 + row mix | `train_d4_geng_timestep_mix_100.csv` (D4) |
| D1/D2/D3 injection | Our contribution |
| TF EasyS2S/S2P/FCN/AugLPN | PyTorch CNN (`model/cnn.py`) |

**Recommendation:** Use this Geng pipeline for a **reference baseline** (D4); keep D1–D3 as injection-strategy science.

---

## Part H — Quick reference

| Question | Answer |
|----------|--------|
| Where are zeros set? | Algorithm 1 step 2: `x[x < x_noise] = 0`; plus OFF minutes kept in windows |
| How is synthetic X built? | **Sum of 5 synthetic appliance powers** (not D1 background swap) |
| How is data mixed? | **Row concat** real+syn → z-score → shuffle windows at train time |
| Where is mix in code? | **Not published** — must implement `build_geng_mix.py` |
| Table 8 test house? | **2** |
| Table 9 test house? | **1** |
| Table 8/9 augmented data? | **200k+200k** for `*` models |
| Diffusion stack? | **PyTorch** (`main.py`) |
| NILM stack? | **TensorFlow** (must port to PyTorch) |

---

## Part I — Contacts

- Upstream: https://github.com/linfengYang/DiffusionModel_NILM  
- Paper DOI: 10.1016/j.energy.2025.135423  
- Author email (README): miles_gzy@163.com (for checkpoints / mix scripts)
