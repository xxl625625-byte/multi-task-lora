"""小型因果基座。q_proj、v_proj、o_proj 的名字和正式模型一致，方便换基座。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LoRAFactor(nn.Module):
    def __init__(self, rank: int, in_features: int, out_features: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, rank))


class TrainableLoRA(nn.Module):
    def __init__(self, specs: dict[str, tuple[int, int]], rank: int, alpha: int):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.factors = nn.ModuleDict(
            {name: LoRAFactor(rank, in_features, out_features) for name, (out_features, in_features) in specs.items()}
        )

    def export(self) -> dict[str, dict[str, torch.Tensor]]:
        exported = {}
        for name, factor in self.factors.items():
            exported[name] = {
                "A": factor.A.detach().cpu().clone(),
                "B": factor.B.detach().cpu().clone(),
            }
        return exported


def _attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int) -> torch.Tensor:
    length, width = query.shape
    head_dim = width // n_heads
    query = query.view(length, n_heads, head_dim).transpose(0, 1)
    key = key.view(length, n_heads, head_dim).transpose(0, 1)
    value = value.view(length, n_heads, head_dim).transpose(0, 1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=query.device), diagonal=1)
    scores = scores.masked_fill(mask, -1e9)
    weights = torch.softmax(scores, dim=-1)
    mixed = torch.matmul(weights, value)
    return mixed.transpose(0, 1).contiguous().view(length, width)


class Block(nn.Module):
    def __init__(self, width: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, hidden: torch.Tensor, apply) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normed = self.ln1(hidden)
        query = apply("q_proj", self.q_proj, normed)
        key = self.k_proj(normed)
        value = apply("v_proj", self.v_proj, normed)
        attended = _attend(query, key, value, self.n_heads)
        output = apply("o_proj", self.o_proj, attended)
        hidden = hidden + output
        hidden = hidden + self.mlp(self.ln2(hidden))
        return hidden, {"q_proj": normed, "v_proj": normed, "o_proj": attended}


class TinyCausal(nn.Module):
    def __init__(self, vocab_size: int, width: int = 32, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        if width % n_heads != 0:
            raise ValueError("隐藏宽度必须能被头数整除")
        self.width = width
        self.embed = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList(Block(width, n_heads) for _ in range(n_layers))
        self.ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def target_specs(self, target_modules: list[str]) -> dict[str, tuple[int, int]]:
        specs = {}
        for layer, _block in enumerate(self.blocks):
            for name in target_modules:
                specs[f"L{layer}_{name}"] = (self.width, self.width)
        return specs

    def forward(
        self,
        token_ids: torch.Tensor,
        span: torch.Tensor | None = None,
        factors: TrainableLoRA | None = None,
        dense: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embed(token_ids)
        captured: dict[str, torch.Tensor] = {}

        def apply(short_name: str, layer: nn.Linear, states: torch.Tensor, layer_index: int) -> torch.Tensor:
            key = f"L{layer_index}_{short_name}"
            result = layer(states)
            if factors is not None and key in factors.factors:
                factor = factors.factors[key]
                result = result + factors.scale * (states @ factor.A.T @ factor.B.T)
            if dense is not None and key in dense:
                result = result + states @ dense[key].T
            return result

        for index, block in enumerate(self.blocks):
            def bound_apply(short_name, layer, states, layer_index=index):
                return apply(short_name, layer, states, layer_index)

            hidden, inputs = block(hidden, bound_apply)
            if span is not None and span.numel() > 0:
                for short_name, states in inputs.items():
                    captured[f"L{index}_{short_name}"] = _pool(states, span)
        logits = self.head(self.ln(hidden))
        return logits, captured


def _pool(states: torch.Tensor, span: torch.Tensor) -> torch.Tensor:
    chosen = states.index_select(0, span)
    norms = chosen.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    chosen = chosen / norms
    mean = chosen.mean(dim=0)
    return mean / mean.norm().clamp_min(1e-8)


def restricted_cross_entropy(logits_last: torch.Tensor, candidate_ids: list[int], gold_id: int) -> torch.Tensor:
    if gold_id not in candidate_ids:
        raise ValueError("正确答案不在候选项里")
    gathered = logits_last[candidate_ids]
    gold_index = candidate_ids.index(gold_id)
    return F.cross_entropy(gathered.unsqueeze(0), torch.tensor([gold_index], device=logits_last.device))
