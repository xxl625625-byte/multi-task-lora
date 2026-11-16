"""开发集门槛。留出任务不参与这些判断。"""

from __future__ import annotations


def passes_expert_gate(
    base_f1: float,
    shared_f1: float,
    group_f1: float,
    base_cost: float,
    group_cost: float,
    margin: float = 0.01,
) -> bool:
    return (
        group_f1 >= base_f1 + margin
        and group_f1 >= shared_f1 + margin
        and group_cost <= base_cost
    )


def shared_caught_up(group_f1: float, long_shared_f1: float, margin: float = 0.01) -> bool:
    return group_f1 < long_shared_f1 + margin


def rollback_triggered(
    candidate_f1: float,
    reference_f1: float,
    max_norm_ratio: float | None,
    candidate_cosine: float | None,
    reference_cosine: float | None,
    margin: float = 0.01,
    norm_limit: float = 1.5,
) -> bool:
    if candidate_f1 < reference_f1 - margin:
        return True
    if max_norm_ratio is not None and max_norm_ratio > norm_limit:
        return True
    if (
        candidate_cosine is not None
        and reference_cosine is not None
        and candidate_cosine > reference_cosine
    ):
        return True
    return False


def choose_fallback(base_f1: float, shared_f1: float, base_cost: float, shared_cost: float) -> str:
    """门槛没过时，在基座和共享 LoRA 里选主结果。"""
    shared_ok = shared_cost <= base_cost
    if shared_ok and shared_f1 > base_f1:
        return "shared"
    return "base"
