import torch
from torch import nn
from torchsummary import summary


class CNN(nn.Module):
    """
    NILM CNN: aggregate window -> appliance window (same length).

    (B, 512) aggregate
        -> (B, 1, 512)
        -> input_block    (B, h,   512)
        -> expand_block   (B, h*2, 512)
        -> bottleneck     (B, h*4, 512)
        -> contract1      (B, h*2, 512)
        -> contract2      (B, h,   512)
        -> output_block   (B, 1,   512)
        -> (B, 512) normalized appliance power (linear output; no ReLU)
    """

    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int):
        super().__init__()
        h = hidden_channels

        self.input_block = nn.Sequential(
            nn.Conv1d(input_channels, h, kernel_size=7, padding=3),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
        )

        self.expand_block = nn.Sequential(
            nn.Conv1d(h, h * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(h * 2),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(h * 2, h * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(h * 4),
            nn.ReLU(inplace=True),
        )

        self.contract1 = nn.Sequential(
            nn.Conv1d(h * 4, h * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(h * 2),
            nn.ReLU(inplace=True),
        )

        self.contract2 = nn.Sequential(
            nn.Conv1d(h * 2, h, kernel_size=5, padding=2),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
        )

        self.output_block = nn.Sequential(
            nn.Conv1d(h, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_block(x)
        x = self.expand_block(x)
        x = self.bottleneck(x)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.output_block(x)

        return x.squeeze(1)


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    config_path = Path(__file__).resolve().parent.parent / "hyperparameter.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_cfg["name"].lower() != "cnn":
        raise ValueError("cnn.py demo only supports model.name: cnn")

    model = CNN(
        input_channels=model_cfg["input_channels"],
        hidden_channels=model_cfg["hidden_channels"],
        output_channels=model_cfg["output_channels"],
    ).to(device)

    window_len = cfg["data"]["window_len"]
    x = torch.randn(2, window_len, device=device)
    y = model(x)
    print(f"input: {tuple(x.shape)} -> output: {tuple(y.shape)}")
    print(summary(model, (window_len,), device=str(device)))
