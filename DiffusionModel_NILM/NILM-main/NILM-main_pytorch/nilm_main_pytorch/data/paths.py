"""UK-DALE CSV path resolution for Geng PyTorch pipeline."""

from __future__ import annotations

from pathlib import Path

AGGREGATE_MEAN = 522.0
AGGREGATE_STD = 814.0


def appliance_dir(data_root: Path, appliance: str) -> Path:
    return data_root / appliance


def geng_rho_train_csv_path(
    data_root: Path,
    appliance: str,
    rho_pct: int,
    dataset_name: str = "UK_DALE",
) -> Path:
    """Geng injection-ratio CSV from build_geng_rho_datasets.py."""
    return appliance_dir(data_root, appliance) / f"{dataset_name}Combined{appliance}_rho{int(rho_pct)}.csv"


def train_csv_path(
    data_root: Path,
    appliance: str,
    *,
    origin: bool,
    train_percent: str = "20",
    dataset_name: str = "UK_DALE",
    injection_rho: int | None = None,
) -> Path:
    app_dir = appliance_dir(data_root, appliance)
    if injection_rho is not None:
        return geng_rho_train_csv_path(data_root, appliance, injection_rho, dataset_name)
    if origin:
        name = f"{appliance}_{train_percent}training_.csv"
    else:
        name = f"{dataset_name}Combined{appliance}_file{train_percent}.csv"
    return app_dir / name


def validation_csv_path(data_root: Path, appliance: str) -> Path:
    app_dir = appliance_dir(data_root, appliance)
    for path in sorted(app_dir.glob("*.csv")):
        if "validation" in path.name.lower():
            return path
    return app_dir / f"{appliance}_validation_.csv"


def test_csv_path(data_root: Path, appliance: str, test_house: int) -> Path:
    app_dir = appliance_dir(data_root, appliance)
    if test_house == 2:
        return app_dir / f"{appliance}_test_.csv"
    if test_house == 1:
        return app_dir / f"{appliance}_test_home1Small_.csv"
    raise ValueError("test_house must be 1 or 2")


def require_csv(path: Path, label: str) -> Path:
    if not path.is_file():
        from nilm_main_pytorch.utils import portable_path_str

        raise FileNotFoundError(f"{label} CSV not found: {portable_path_str(path)}")
    return path
