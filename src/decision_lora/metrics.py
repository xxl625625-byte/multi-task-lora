"""按任务计算宏平均 F1 和高代价错误率。"""

from __future__ import annotations

from collections import defaultdict


def _f1_one(truth: list[str], pred: list[str], label: str) -> float:
    tp = fp = fn = 0
    for gold, guess in zip(truth, pred):
        if guess == label and gold == label:
            tp += 1
        elif guess == label and gold != label:
            fp += 1
        elif guess != label and gold == label:
            fn += 1
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(truth: list[str], pred: list[str], labels: list[str]) -> float:
    if not truth:
        return 0.0
    scores = [_f1_one(truth, pred, label) for label in labels]
    return sum(scores) / len(scores)


def high_cost_rate(truth: list[str], pred: list[str], high_cost: str | None) -> float | None:
    if high_cost is None or not truth:
        return None
    errors = 0
    eligible = 0
    for gold, guess in zip(truth, pred):
        if gold == high_cost:
            continue
        eligible += 1
        if guess == high_cost:
            errors += 1
    if eligible == 0:
        return 0.0
    return errors / eligible


def average_tasks(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def group_rows(rows: list[dict], key: str = "task") -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)
