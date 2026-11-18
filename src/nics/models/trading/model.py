"""NICS Trading -- market-sequence Transformer, trained from random initialisation.

Input: a window of candles already turned into scale-free features
(returns and ranges measured in ATR units, see nics.data.market), plus
the geometry of a scanner signal when one exists.

Two heads, trained together:
  - horizon head: does price end the next N bars down / flat / up?
    (self-supervised from candles alone, so every bar is a training sample)
  - signal head: probability that a scanner signal reaches TP1 before SL
    (supervised by the real outcomes stored with our signals)

No pretrained weights are loaded. The Transformer blocks are the same
from-scratch blocks as NICS Language (nics.models.text.model.Block).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nics.models.text.model import Block, NicsTextConfig, RMSNorm

N_HORIZON_CLASSES = 3  # 0 = down, 1 = flat, 2 = up


@dataclass
class NicsTradingConfig:
    n_features: int
    n_signal_features: int
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    max_seq_len: int = 96
    dropout: float = 0.1

    def block_config(self) -> NicsTextConfig:
        # Block only reads d_model / n_heads / dropout from this config.
        return NicsTextConfig(
            vocab_size=1,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
        )


class NicsTradingModel(nn.Module):
    def __init__(self, cfg: NicsTradingConfig):
        super().__init__()
        self.cfg = cfg
        bcfg = cfg.block_config()
        self.in_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(bcfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.horizon_head = nn.Linear(cfg.d_model, N_HORIZON_CLASSES)
        self.signal_proj = nn.Linear(cfg.n_signal_features, cfg.d_model)
        self.signal_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, n_features) -> summary of the window at its last bar."""
        b, t, _ = x.shape
        assert t <= self.cfg.max_seq_len, f"window {t} exceeds max_seq_len {self.cfg.max_seq_len}"
        pos = torch.arange(t, device=x.device)
        h = self.drop(self.in_proj(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        return self.norm_f(h)[:, -1, :]

    def forward_horizon(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.horizon_head(self.encode(x))
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    def forward_signal(
        self, x: torch.Tensor, signal_features: torch.Tensor, targets: torch.Tensor | None = None
    ):
        h = self.encode(x)
        s = F.silu(self.signal_proj(signal_features))
        logit = self.signal_head(torch.cat([h, s], dim=-1)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit, targets.float()) if targets is not None else None
        return logit, loss
