"""GeoNet: our own small convolutional network for rock and mineral photos.

Written from scratch for this project and initialised from random weights: no pretrained
checkpoint, no torchvision model, no external architecture code.

Design for a CPU-only laptop:
  * inverted-residual blocks (1x1 expand -> 3x3 depthwise -> squeeze-excitation -> 1x1
    project, with a skip when shapes match): few multiply-adds per pixel, which is what
    a CPU without a GPU can afford;
  * squeeze-excitation lets the network weigh colour channels globally: colour and
    lustre matter for minerals (malachite green, azurite blue, pyrite brass);
  * two heads: the class (quartz, pyrite, granite...) and its coarse group (sulfide,
    igneous rock...). The group head is a cheap extra signal for classes with few photos.

The default `micro` width has about 1.1 M parameters at 160x160 input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeoNetConfig:
    num_classes: int
    num_groups: int
    stem: int = 24
    # (expand ratio, out channels, blocks, first stride)
    stages: list = field(default_factory=lambda: [[1, 24, 1, 1], [4, 40, 2, 2], [4, 64, 3, 2], [4, 112, 3, 2], [4, 160, 2, 2]])
    head: int = 448
    dropout: float = 0.2
    se_ratio: float = 0.25


class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, ratio: float):
        super().__init__()
        mid = max(8, int(ch * ratio))
        self.fc1 = nn.Conv2d(ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(s))))


class InvertedResidual(nn.Module):
    def __init__(self, cin: int, cout: int, expand: int, stride: int, se_ratio: float):
        super().__init__()
        mid = cin * expand
        self.skip = stride == 1 and cin == cout
        layers: list[nn.Module] = []
        if expand != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU()]
        layers += [nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.SiLU(),
                   SqueezeExcite(mid, se_ratio / expand),  # squeeze to cin * se_ratio channels
                   nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        y = self.body(x)
        return x + y if self.skip else y


class GeoNet(nn.Module):
    def __init__(self, cfg: GeoNetConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(nn.Conv2d(3, cfg.stem, 3, 2, 1, bias=False), nn.BatchNorm2d(cfg.stem), nn.SiLU())
        blocks, cin = [], cfg.stem
        for expand, cout, n, stride in cfg.stages:
            for i in range(n):
                blocks.append(InvertedResidual(cin, cout, expand, stride if i == 0 else 1, cfg.se_ratio))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.neck = nn.Sequential(nn.Conv2d(cin, cfg.head, 1, bias=False), nn.BatchNorm2d(cfg.head), nn.SiLU())
        self.drop = nn.Dropout(cfg.dropout)
        self.cls = nn.Linear(cfg.head, cfg.num_classes)
        self.grp = nn.Linear(cfg.head, cfg.num_groups)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        # zero the last BN of every residual block: each block starts as identity, trains more stably
        for b in self.blocks:
            if isinstance(b, InvertedResidual) and b.skip:
                nn.init.zeros_(b.body[-1].weight)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.neck(self.blocks(self.stem(x)))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.drop(self.features(x))
        return self.cls(f), self.grp(f)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ImageNet-free normalisation: per-channel mean/std measured on our own training set at
# build time are used when available; these defaults are just mid-grey.
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.25, 0.25, 0.25)
