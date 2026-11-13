"""Wraps a trained DispNet as the filter's learned-displacement source."""

from pathlib import Path

import numpy as np
import torch

from .model import WINDOW, DispNet


class Predictor:
    window = WINDOW
    every = 20                      # one learned update every 0.1 s

    def __init__(self, path: str | Path, device: str = "cpu", min_std: float = 0.01):
        ck = torch.load(path, map_location=device, weights_only=True)
        self.net = DispNet(ck["width"]).to(device).eval()
        self.net.load_state_dict(ck["state"])
        self.device = device
        self.min_std = min_std
        self.name = Path(path).parent.name

    @torch.no_grad()
    def __call__(self, win_world: np.ndarray):
        x = torch.from_numpy(win_world.T[None].astype(np.float32)).to(self.device)
        d, ls = self.net(x)
        std = np.maximum(torch.exp(ls)[0].cpu().numpy(), self.min_std)
        return d[0].cpu().numpy().astype(float), np.diag(std ** 2)
