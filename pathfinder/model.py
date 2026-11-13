"""Small 1D ResNet that reads 1 s of IMU and predicts displacement + its own uncertainty.

Same idea as TLIO (Liu et al. 2020, BSD): input is gravity-aligned IMU (rotated into the
world frame with the filter's current orientation), output is the 3D displacement over the
window and a log standard deviation per axis. Ours is ~10x smaller, so it can run on-device.
"""

import torch
import torch.nn as nn

WINDOW = 200          # 1 s at 200 Hz


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 5, stride, 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 5, 1, 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + self.skip(x))


class DispNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv1d(6, w, 7, 2, 3, bias=False), nn.BatchNorm1d(w), nn.ReLU())
        self.body = nn.Sequential(Block(w, w, 1), Block(w, 2 * w, 2), Block(2 * w, 4 * w, 2),
                                  Block(4 * w, 4 * w, 2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(4 * w, 128), nn.ReLU(), nn.Linear(128, 6))

    def forward(self, x):                     # x: (B, 6, WINDOW) [gyro_w, acc_w]
        z = self.pool(self.body(self.stem(x))).flatten(1)
        out = self.head(z)
        disp, logstd = out[:, :3], out[:, 3:].clamp(-7, 3)
        return disp, logstd


def nll(pred, logstd, target):
    return (((pred - target) ** 2) / (2 * torch.exp(2 * logstd)) + logstd).mean()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def beta_nll(pred, logstd, target, beta: float = 0.5):
    """Gaussian NLL weighted by var^beta (stop-gradient), Seitzer et al., ICLR 2022."""
    var = torch.exp(2 * logstd)
    loss = ((pred - target) ** 2) / (2 * var) + logstd
    return (loss * var.detach() ** beta).mean()
