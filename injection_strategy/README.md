# Injection Strategy Dataset Builder

This folder builds datasets for the ICSIMA injection-strategy experiment.

The script loads both:

- real data: aggregate smart meter signal, `X_r`, and real appliance signal, `Y_r`
- synthetic data: synthetic appliance signal, `Y_s`

The fixed construction is:

```text
X_s = X_r - Y_r + Y_s
```

This means:

```text
synthetic aggregate = real aggregate - real appliance + synthetic appliance
```

The research variable is how `Y_s` is selected.

______________________________________________________
Step (1)
Go to the injection strategy folder
______________________________________________________

```powershell
cd "C:\Users\Raymond Tie\Desktop\PhD\Paper\Injection Strategy Conference\code\injection_strategy"
```

______________________________________________________
Step (2)
Build all datasets using the default setting
______________________________________________________

Default setting:

```text
window length = 512
stride = 512
appliances = all
ratios = 25%, 50%, 100%, 200%
```

Command:

```powershell
& "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\build_injection_datasets.py
```

This creates datasets in:

```text
.\datasets\<appliance>\
```

______________________________________________________
Step (3)
Build only one appliance if needed
______________________________________________________

Example for kettle:

```powershell
& "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\build_injection_datasets.py --appliance kettle
```

Available appliances:

```text
dishwasher
fridge
kettle
microwave
washingmachine
```

______________________________________________________
Step (4)
Change window length or stride
______________________________________________________

Example using overlap:

```powershell
& "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\build_injection_datasets.py --window-len 512 --stride 256
```

Smaller stride gives more training windows.

______________________________________________________
Step (5)
Override ON thresholds if needed
______________________________________________________

Example:

```powershell
& "C:\Users\Raymond Tie\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\build_injection_datasets.py --threshold kettle=2000 --threshold microwave=200
```

Current default thresholds:

```text
dishwasher = 10 W
fridge = 50 W
kettle = 2000 W
microwave = 200 W
washingmachine = 20 W
```

______________________________________________________
Step (6)
What files are created
______________________________________________________

For each appliance, the script creates:

```text
train_real_only.npz
train_on_focused_100.npz
train_full_distribution_100.npz
train_balanced_25.npz
train_balanced_50.npz
train_balanced_100.npz
train_balanced_200.npz
val_house1.npz
test_house1.npz
test_house2.npz
```

Each `.npz` contains:

```text
X = aggregate + 8 time features, shape (windows, timesteps, 9)
y = appliance target, shape (windows, timesteps)
```

For `X`:

```text
channel 0 = aggregate power
channel 1-8 = time features
```

______________________________________________________
Step (7)
What each training dataset means
______________________________________________________

```text
train_real_only
```
Only real training windows.

```text
train_on_focused_100
```
Real windows plus synthetic windows sampled from ON periods only.

```text
train_full_distribution_100
```
Real windows plus synthetic windows sampled from the full synthetic sequence.

```text
train_balanced_25 / 50 / 100 / 200
```
Real windows plus balanced synthetic windows.
Balanced means 50% from ON-focused pool and 50% from full-distribution pool.
The number is the injection ratio.

______________________________________________________
Step (8)
Experiment idea
______________________________________________________

Experiment 1:

```text
real_only vs on_focused_100 vs full_distribution_100 vs balanced_100
```

This tests appliance-state sampling.

Experiment 2:

```text
balanced_25 vs balanced_50 vs balanced_100 vs balanced_200
```

This tests injection ratio.

Validation:

```text
val_house1
```

Testing:

```text
test_house1
test_house2
```
