# Injection Dataset Statistics Summary

This summary explains the generated D0/D1/D2/D3 datasets.

Window length: `512` timesteps.

Definitions:

- `full_windows`: total windows in the training dataset, including real and synthetic windows.
- `synthetic_windows`: added synthetic windows only.
- `full_timesteps`: `full_windows x 512`.
- `synthetic_timesteps`: `synthetic_windows x 512`.
- `full_on_window_rate`: fraction of all training windows that contain at least one ON timestep.
- `synthetic_on_window_rate`: fraction of synthetic windows that contain at least one ON timestep.
- `full_on_sample_rate`: fraction of all timesteps labelled ON.
- `synthetic_on_sample_rate`: fraction of synthetic timesteps labelled ON.
- `energy_per_window`: sum of appliance power within one window, averaged over windows.

Important note: D2 has `synthetic_on_window_rate = 100%` by design because every D2 synthetic window contains an inserted ON event. This does not mean all timesteps are ON.

## dishwasher

| Method | Ratio | Full windows | Synthetic windows | Full timesteps | Synthetic timesteps | Full ON-window | Synthetic ON-window | Full ON-sample | Synthetic ON-sample | Full energy/window | Synthetic energy/window |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 Real only | 0% | 195 | 0 | 99,840 | 0 | 16.92% | 0.00% | 2.56% | 0.00% | 9,944.54 | 0.00 |
| D1 Full-window append | 100% | 390 | 195 | 199,680 | 99,840 | 20.00% | 23.08% | 3.05% | 3.55% | 12,566.49 | 15,188.43 |
| D2 ON-event insertion | 100% | 390 | 195 | 199,680 | 99,840 | 58.46% | 100.00% | 10.62% | 18.68% | 40,811.70 | 71,678.84 |
| D3 Balanced event insertion | 25% | 244 | 49 | 124,928 | 25,088 | 25.00% | 57.14% | 4.14% | 10.44% | 16,185.66 | 41,022.75 |
| D3 Balanced event insertion | 50% | 293 | 98 | 150,016 | 50,176 | 31.74% | 61.22% | 5.45% | 11.21% | 21,234.53 | 43,699.29 |
| D3 Balanced event insertion | 100% | 390 | 195 | 199,680 | 99,840 | 39.49% | 62.05% | 6.96% | 11.36% | 27,310.70 | 44,676.84 |
| D3 Balanced event insertion | 200% | 585 | 390 | 299,520 | 199,680 | 46.15% | 60.77% | 8.18% | 11.00% | 32,079.20 | 43,146.54 |

Main reading at 100% ratio:

- D1 synthetic ON-window rate: 23.08%.
- D2 synthetic ON-window rate: 100.00%.
- D3 synthetic ON-window rate: 62.05%.
- D3 sits between D1 and D2 for ON-window exposure in most sparse appliances.

## fridge

| Method | Ratio | Full windows | Synthetic windows | Full timesteps | Synthetic timesteps | Full ON-window | Synthetic ON-window | Full ON-sample | Synthetic ON-sample | Full energy/window | Synthetic energy/window |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 Real only | 0% | 195 | 0 | 99,840 | 0 | 100.00% | 0.00% | 38.24% | 0.00% | 18,170.14 | 0.00 |
| D1 Full-window append | 100% | 390 | 195 | 199,680 | 99,840 | 100.00% | 100.00% | 38.28% | 38.32% | 18,302.69 | 18,435.23 |
| D2 ON-event insertion | 100% | 390 | 195 | 199,680 | 99,840 | 100.00% | 100.00% | 21.24% | 4.25% | 10,080.00 | 1,989.87 |
| D3 Balanced event insertion | 25% | 244 | 49 | 124,928 | 25,088 | 100.00% | 100.00% | 35.20% | 23.10% | 16,739.40 | 11,045.64 |
| D3 Balanced event insertion | 50% | 293 | 98 | 150,016 | 50,176 | 100.00% | 100.00% | 32.73% | 21.78% | 15,581.28 | 10,429.98 |
| D3 Balanced event insertion | 100% | 390 | 195 | 199,680 | 99,840 | 100.00% | 100.00% | 29.74% | 21.23% | 14,147.91 | 10,125.68 |
| D3 Balanced event insertion | 200% | 585 | 390 | 299,520 | 199,680 | 100.00% | 100.00% | 27.01% | 21.39% | 12,882.32 | 10,238.40 |

Main reading at 100% ratio:

- D1 synthetic ON-window rate: 100.00%.
- D2 synthetic ON-window rate: 100.00%.
- D3 synthetic ON-window rate: 100.00%.
- D3 sits between D1 and D2 for ON-window exposure in most sparse appliances.

## kettle

| Method | Ratio | Full windows | Synthetic windows | Full timesteps | Synthetic timesteps | Full ON-window | Synthetic ON-window | Full ON-sample | Synthetic ON-sample | Full energy/window | Synthetic energy/window |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 Real only | 0% | 195 | 0 | 99,840 | 0 | 69.23% | 0.00% | 0.86% | 0.00% | 8,338.08 | 0.00 |
| D1 Full-window append | 100% | 390 | 195 | 199,680 | 99,840 | 69.49% | 69.74% | 0.72% | 0.58% | 8,307.89 | 8,277.69 |
| D2 ON-event insertion | 100% | 390 | 195 | 199,680 | 99,840 | 84.62% | 100.00% | 0.67% | 0.48% | 6,443.11 | 4,548.14 |
| D3 Balanced event insertion | 25% | 244 | 49 | 124,928 | 25,088 | 72.13% | 83.67% | 0.80% | 0.53% | 7,991.71 | 6,613.27 |
| D3 Balanced event insertion | 50% | 293 | 98 | 150,016 | 50,176 | 74.06% | 83.67% | 0.74% | 0.50% | 7,626.77 | 6,211.41 |
| D3 Balanced event insertion | 100% | 390 | 195 | 199,680 | 99,840 | 76.41% | 83.59% | 0.69% | 0.52% | 7,329.34 | 6,320.59 |
| D3 Balanced event insertion | 200% | 585 | 390 | 299,520 | 199,680 | 78.29% | 82.82% | 0.62% | 0.50% | 6,903.43 | 6,186.10 |

Main reading at 100% ratio:

- D1 synthetic ON-window rate: 69.74%.
- D2 synthetic ON-window rate: 100.00%.
- D3 synthetic ON-window rate: 83.59%.
- D3 sits between D1 and D2 for ON-window exposure in most sparse appliances.

## microwave

| Method | Ratio | Full windows | Synthetic windows | Full timesteps | Synthetic timesteps | Full ON-window | Synthetic ON-window | Full ON-sample | Synthetic ON-sample | Full energy/window | Synthetic energy/window |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 Real only | 0% | 195 | 0 | 99,840 | 0 | 80.00% | 0.00% | 0.80% | 0.00% | 3,177.16 | 0.00 |
| D1 Full-window append | 100% | 390 | 195 | 199,680 | 99,840 | 70.77% | 61.54% | 0.57% | 0.35% | 2,819.21 | 2,461.25 |
| D2 ON-event insertion | 100% | 390 | 195 | 199,680 | 99,840 | 90.00% | 100.00% | 0.56% | 0.32% | 2,181.85 | 1,186.53 |
| D3 Balanced event insertion | 25% | 244 | 49 | 124,928 | 25,088 | 81.15% | 85.71% | 0.71% | 0.34% | 2,907.61 | 1,834.91 |
| D3 Balanced event insertion | 50% | 293 | 98 | 150,016 | 50,176 | 80.20% | 80.61% | 0.63% | 0.28% | 2,649.90 | 1,600.76 |
| D3 Balanced event insertion | 100% | 390 | 195 | 199,680 | 99,840 | 80.77% | 81.54% | 0.57% | 0.34% | 2,550.82 | 1,924.48 |
| D3 Balanced event insertion | 200% | 585 | 390 | 299,520 | 199,680 | 81.03% | 81.54% | 0.47% | 0.31% | 2,237.12 | 1,767.09 |

Main reading at 100% ratio:

- D1 synthetic ON-window rate: 61.54%.
- D2 synthetic ON-window rate: 100.00%.
- D3 synthetic ON-window rate: 81.54%.
- D3 sits between D1 and D2 for ON-window exposure in most sparse appliances.

## washingmachine

| Method | Ratio | Full windows | Synthetic windows | Full timesteps | Synthetic timesteps | Full ON-window | Synthetic ON-window | Full ON-sample | Synthetic ON-sample | Full energy/window | Synthetic energy/window |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 Real only | 0% | 195 | 0 | 99,840 | 0 | 26.15% | 0.00% | 4.52% | 0.00% | 12,557.05 | 0.00 |
| D1 Full-window append | 100% | 390 | 195 | 199,680 | 99,840 | 33.33% | 40.51% | 6.05% | 7.59% | 18,146.22 | 23,735.39 |
| D2 ON-event insertion | 100% | 390 | 195 | 199,680 | 99,840 | 63.08% | 100.00% | 10.27% | 16.01% | 28,198.28 | 43,839.51 |
| D3 Balanced event insertion | 25% | 244 | 49 | 124,928 | 25,088 | 33.61% | 63.27% | 5.74% | 10.60% | 16,269.08 | 31,041.44 |
| D3 Balanced event insertion | 50% | 293 | 98 | 150,016 | 50,176 | 40.27% | 68.37% | 6.62% | 10.81% | 18,860.53 | 31,403.17 |
| D3 Balanced event insertion | 100% | 390 | 195 | 199,680 | 99,840 | 48.97% | 71.79% | 8.23% | 11.94% | 23,582.49 | 34,607.94 |
| D3 Balanced event insertion | 200% | 585 | 390 | 299,520 | 199,680 | 55.90% | 70.77% | 9.72% | 12.32% | 28,353.34 | 36,251.48 |

Main reading at 100% ratio:

- D1 synthetic ON-window rate: 40.51%.
- D2 synthetic ON-window rate: 100.00%.
- D3 synthetic ON-window rate: 71.79%.
- D3 sits between D1 and D2 for ON-window exposure in most sparse appliances.
