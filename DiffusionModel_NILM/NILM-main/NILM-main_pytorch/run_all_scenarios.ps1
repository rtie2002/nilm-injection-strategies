# One-click: all EasyS2S scenarios x 5 appliances (train + test)
# Usage: right-click -> Run with PowerShell, or from this folder:
#   .\run_all_scenarios.ps1

Set-Location $PSScriptRoot
python -m nilm_main_pytorch.run_all_scenarios --phase train_test --skip-errors @args
