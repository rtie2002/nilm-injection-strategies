"""Per-appliance hyperparameters — Geng PyTorch package (from NILM-main/Arguments.py)."""

PARAMS_APPLIANCE = {
    "kettle": {
        "window_length": 600,
        "on_power_threshold": 200,
        "mean": 700,
        "std": 1000,
    },
    "microwave": {
        "window_length": 600,
        "on_power_threshold": 200,
        "mean": 500,
        "std": 800,
    },
    "fridge": {
        "window_length": 600,
        "on_power_threshold": 50,
        "mean": 200,
        "std": 400,
    },
    "dishwasher": {
        "window_length": 600,
        "on_power_threshold": 10,
        "mean": 700,
        "std": 1000,
    },
    "washingmachine": {
        "window_length": 600,
        "on_power_threshold": 20,
        "mean": 400,
        "std": 700,
    },
}

# FCN uses a longer input window (see fcn_train.py).
FCN_TARGET_LENGTH = 2053
FCN_OUTPUT_LENGTH = 1053

ALL_APPLIANCES = tuple(PARAMS_APPLIANCE.keys())


def norm_on_threshold(appliance: str) -> float:
    p = PARAMS_APPLIANCE[appliance]
    return (p["on_power_threshold"] - p["mean"]) / p["std"]
