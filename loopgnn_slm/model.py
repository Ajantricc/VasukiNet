from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoopGNNConfig:
    vocab_size: int
    block_size: int = 128
    d_model: int = 192
    n_heads: int = 6
    dropout: float = 0.1
    max_loops: int = 8
    gnn_steps: int = 2
    gnn_edge_radius: int = 2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LoopGNNConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(y))


class LocalCausalGraphConv(nn.Module):
    """Token graph message passing.

    Nodes are token positions. Directed causal edges connect each token to prior nearby
    tokens within `gnn_edge_radius`, plus a self edge. This gives the model an explicit
    graph-based local reasoning channel alongside attention.
    """

    def __init__(self, cfg: LoopGNNConfig):
        super().__init__()
        self.radius = cfg.gnn_edge_radius
        self.msg = nn.Linear(cfg.d_model, cfg.d_model)
        self.gate = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        agg = x
        count = torch.ones(t, device=x.device, dtype=x.dtype).view(1, t, 1)
        for r in range(1, self.radius + 1):
            shifted = F.pad(x[:, :-r, :], (0, 0, r, 0))
            agg = agg + shifted
            count = count + (torch.arange(t, device=x.device) >= r).to(x.dtype).view(1, t, 1)
        agg = agg / count.clamp_min(1.0)
        m = self.msg(agg)
        z = torch.sigmoid(self.gate(torch.cat([x, m], dim=-1)))
        return self.norm(x + self.dropout(z * m))


class LoopBlock(nn.Module):
    def __init__(self, cfg: LoopGNNConfig):
        super().__init__()
        self.time_proj = nn.Sequential(nn.Linear(2, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model))
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.gnn = LocalCausalGraphConv(cfg)
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.shortcut_gate = nn.Linear(2, cfg.d_model)

    def forward(self, x: torch.Tensor, loop_idx: int, total_loops: int) -> torch.Tensor:
        tau = float(loop_idx + 1) / float(max(total_loops, 1))
        h = 1.0 / float(max(total_loops, 1))
        cond = torch.tensor([tau, h], dtype=x.dtype, device=x.device).view(1, 1, 2)
        cond_vec = self.time_proj(cond)
        gate = torch.sigmoid(self.shortcut_gate(cond))

        y = x + cond_vec
        y = y + self.attn(self.ln1(y))
        for _ in range(1):
            y = self.gnn(self.ln2(y))
        y = y + self.mlp(self.ln3(y))
        return (1.0 - gate) * x + gate * y


class LoopGNNSmallLM(nn.Module):
    def __init__(self, cfg: LoopGNNConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.loop = LoopBlock(cfg)
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loops: int | None = None,
                return_hidden: bool = False):
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.cfg.block_size}")
        loops = int(loops or self.cfg.max_loops)
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for i in range(loops):
            x = self.loop(x, i, loops)
        h = self.final_norm(x)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_hidden:
            return logits, loss, h
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.9,
                 top_k: int = 50, loops: int | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx, loops=loops)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
