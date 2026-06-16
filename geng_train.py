"""
DEPRECATED — use NILM-main PyTorch package.

  cd DiffusionModel_NILM/NILM-main/NILM-main_pytorch
  python -m nilm_main_pytorch.train --model easy_s2s --appliance kettle
"""

import sys
from pathlib import Path

_PYTORCH_ROOT = (
    Path(__file__).resolve().parent
    / "DiffusionModel_NILM"
    / "NILM-main"
    / "NILM-main_pytorch"
)
if str(_PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTORCH_ROOT))

from nilm_main_pytorch.train import main

if __name__ == "__main__":
    main()
