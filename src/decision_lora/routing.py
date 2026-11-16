"""整题路由。分数高只表示更贴近某个专家的强方向。"""

from __future__ import annotations

import math


def layer_gap(sigma1: float, sigma2: float) -> float:
    if sigma1 <= 0.0:
        return 0.0
    if sigma2 <= 0.0:
        return math.inf
    return sigma1 / sigma2


def expert_score(parts: list[dict], gap_min: float = 1.5) -> float | None:
    """parts 里每一项是一层一个模块的 sigma1、sigma2 和已经取过绝对值的余弦。"""
    if not parts:
        return None
    usable = []
    for part in parts:
        if layer_gap(part["sigma1"], part["sigma2"]) < gap_min:
            continue
        usable.append(part["sigma1"] * part["cosine_abs"])
    if not usable:
        return None
    return sum(usable) / len(usable)


def route(
    scores: dict[str, float],
    margin: float = 0.1,
    allow_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, float]:
    """返回每个专家的权重。全 0 表示只走基座。"""
    allow_pairs = allow_pairs or set()
    weights = {name: 0.0 for name in scores}
    if not scores:
        return weights
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_score = ranked[0]
    if top_score == 0.0:
        return weights
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = (top_score - second_score) / top_score
    if len(ranked) > 1 and gap < margin:
        pair = tuple(sorted((ranked[0][0], ranked[1][0])))
        if pair in allow_pairs:
            first = math.exp(ranked[0][1])
            second = math.exp(ranked[1][1])
            total = first + second
            weights[ranked[0][0]] = first / total
            weights[ranked[1][0]] = second / total
            return weights
        return weights
    weights[top_name] = 1.0
    return weights


def filter_by_modality(
    scores: dict[str, float],
    sample_modality: str,
    expert_modality: dict[str, str],
) -> dict[str, float]:
    kept = {}
    for name, score in scores.items():
        kind = expert_modality[name]
        if sample_modality == "dialogue" and kind != "dialogue":
            continue
        if sample_modality != "dialogue" and kind == "dialogue":
            continue
        kept[name] = score
    return kept
