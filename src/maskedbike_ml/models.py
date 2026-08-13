from __future__ import annotations

import math
import torch
from torch import nn


class CSCAE(nn.Module):
    """Convolutional side-channel autoencoder from thesis Table 1."""

    def __init__(self, trace_length: int, latent_ratio: float = 0.10, channels: int = 1):
        super().__init__()
        half = (trace_length + 1) // 2
        flat = channels * half
        latent = max(2, int(flat * latent_ratio))
        self.trace_length = trace_length
        self.half = half
        self.channels = channels
        self.latent_size = latent
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
        )
        self.encoder_dense = nn.Linear(flat, latent)
        self.decoder_dense = nn.Linear(latent, flat)
        self.decoder = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_conv(x[:, None, :])
        return self.encoder_dense(x.flatten(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.encode(x)
        decoded = self.decoder_dense(code).reshape(-1, self.channels, self.half)
        reconstructed = self.decoder(decoded)[:, 0, : self.trace_length]
        return code, reconstructed


class PaperClassifier(nn.Module):
    """Four Dense + BatchNorm + ReLU blocks from thesis Table 2."""

    def __init__(self, latent_size: int, widths: list[int] | None = None):
        super().__init__()
        if widths is None:
            base_exp = int(math.log2(latent_size))
            widths = [2 ** max(0, base_exp - i) for i in range(4)]
        if len(widths) != 4 or any(int(width) < 1 for width in widths):
            raise ValueError("classifier requires exactly four positive hidden widths")
        self.widths = [int(width) for width in widths]
        layers: list[nn.Module] = []
        current = latent_size
        for width in self.widths:
            layers += [nn.Linear(current, width), nn.BatchNorm1d(width), nn.ReLU()]
            current = width
        layers.append(nn.Linear(current, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
