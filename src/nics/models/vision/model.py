"""NICS Vision -- compact convolutional classifier, trained from random init.

A CNN was chosen over a Vision Transformer for the first version because
this project trains on CPU only (no discrete GPU in the current hardware
audit); a small ViT trains far slower per step than a comparably-sized CNN
on CPU. No pretrained weights are loaded anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class NicsVisionConfig:
    num_classes: int
    in_channels: int = 3
    image_size: int = 32
    base_channels: int = 16
    dropout: float = 0.1


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class NicsVisionModel(nn.Module):
    """Small residual-free CNN classifier, random weight initialisation."""

    def __init__(self, cfg: NicsVisionConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels
        self.stem = ConvBlock(cfg.in_channels, c)
        self.block2 = ConvBlock(c, c * 2)
        self.block3 = ConvBlock(c * 2, c * 4)
        reduced = cfg.image_size // 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(c * 4 * reduced * reduced, cfg.num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)
