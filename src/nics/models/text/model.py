"""NICS Text -- decoder-only Transformer, trained from random initialisation.

No pretrained weights are loaded anywhere in this module. Every tensor
originates from the initialisation routines below, seeded explicitly by
the caller (see nics.common.seed.set_seed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NicsTextConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 256
    dropout: float = 0.0
    # "learned": a trained position table (nics-text-nano, kept for old checkpoints).
    # "rope": rotary positions applied to queries and keys, no position parameters.
    pos_encoding: str = "learned"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def rope_tables(head_dim: int, max_seq_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for rotary position encoding, shape (max_seq_len, head_dim // 2)."""
    assert head_dim % 2 == 0, "rotary positions need an even head dimension"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels of x (b, heads, t, head_dim) by position-dependent angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: NicsTextConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: dict | None = None,
    ) -> torch.Tensor:
        """Self-attention.

        rope:  (cos, sin) already sliced to the positions of x, or None.
        cache: per-layer dict holding the keys/values of earlier positions. When
               given, x holds only the NEW positions and the dict is updated.
        """
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if rope is not None:
            q = apply_rope(q, *rope)
            k = apply_rope(k, *rope)
        dropout_p = self.dropout if self.training else 0.0
        past = cache.get("k") if cache is not None else None
        if past is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        else:
            k = torch.cat([past, k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
            mask = None
            if t > 1:
                # new position i may see every cached position and new positions <= i
                p = past.shape[2]
                mask = torch.ones(t, p + t, dtype=torch.bool, device=x.device)
                mask[:, p:] = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        if cache is not None:
            cache["k"], cache["v"] = k, v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 4.0):
        super().__init__()
        hidden = int(dim * hidden_mult * 2 / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, cfg: NicsTextConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model)

    def forward(self, x: torch.Tensor, rope=None, cache: dict | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, cache)
        x = x + self.ffn(self.norm2(x))
        return x


class NicsTextModel(nn.Module):
    """Decoder-only Transformer language model, random weight initialisation."""

    def __init__(self, cfg: NicsTextConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        elif cfg.pos_encoding == "rope":
            cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len)
            # Fixed tables, not parameters: kept out of checkpoints and of the parameter count.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            raise ValueError(f"unknown pos_encoding: {cfg.pos_encoding!r}")
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: the input embedding and output projection share
        # parameters, which is standard practice for small LMs and is not
        # a form of loading pretrained weights -- both start random.
        self.lm_head.weight = self.tok_emb.weight
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

    def new_cache(self) -> list[dict]:
        """Empty key/value cache for incremental generation (one dict per layer)."""
        return [{} for _ in self.blocks]

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: list[dict] | None = None,
        start_pos: int = 0,
    ):
        """Logits (and loss when targets are given).

        With a cache, idx holds only the tokens not yet seen and start_pos is the
        number of tokens already in the cache.
        """
        b, t = idx.shape
        assert start_pos + t <= self.cfg.max_seq_len, (
            f"sequence length {start_pos + t} exceeds max_seq_len {self.cfg.max_seq_len}"
        )
        x = self.tok_emb(idx) * math.sqrt(self.cfg.d_model)
        rope = None
        if self.cfg.pos_encoding == "learned":
            x = x + self.pos_emb(torch.arange(start_pos, start_pos + t, device=idx.device))
        else:
            rope = (self.rope_cos[start_pos : start_pos + t], self.rope_sin[start_pos : start_pos + t])
        for i, block in enumerate(self.blocks):
            x = block(x, rope, cache[i] if cache is not None else None)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Continue idx by up to max_new_tokens tokens.

        temperature 0 picks the most likely token (deterministic). Generation stops
        at eos_id (batch size 1) or when the context is full. With use_cache the
        keys/values of earlier tokens are reused, so each step costs one token of
        compute instead of the whole sequence; without it the context slides.
        """
        self.eval()
        if use_cache:
            max_new_tokens = min(max_new_tokens, self.cfg.max_seq_len - idx.shape[1])
            cache = self.new_cache()
            step_input, pos = idx, 0
        for _ in range(max(0, max_new_tokens)):
            if use_cache:
                logits, _ = self(step_input, cache=cache, start_pos=pos)
                pos += step_input.shape[1]
            else:
                logits, _ = self(idx[:, -self.cfg.max_seq_len :])
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k < logits.size(-1):
                    kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            step_input = next_id
            if eos_id is not None and idx.shape[0] == 1 and next_id.item() == eos_id:
                break
        return idx
