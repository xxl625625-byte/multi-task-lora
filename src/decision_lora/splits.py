"""按案例主键切分。派生样本跟着原始案例走。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from decision_lora.schema import TaskCard
from decision_lora.simulate import Sample


def assign_splits(
    samples: list[Sample],
    cards: dict[str, TaskCard],
    dev_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, list[Sample]]:
    by_task: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        by_task[sample.task][sample.case_id].append(sample)

    rng = np.random.default_rng(seed)
    grouped = {"train": [], "dev": [], "holdout_near": [], "holdout_far": []}
    for task, cases in by_task.items():
        card = cards[task]
        case_ids = sorted(cases)
        if card.holdout == "near":
            bucket = "holdout_near"
        elif card.holdout == "far":
            bucket = "holdout_far"
        else:
            bucket = None
        if bucket is not None:
            for case_id in case_ids:
                grouped[bucket].extend(cases[case_id])
            continue
        order = case_ids.copy()
        rng.shuffle(order)
        if len(order) < 2:
            train_ids = set(order)
            dev_ids: set[str] = set()
        else:
            dev_count = max(1, int(round(len(order) * dev_ratio)))
            dev_count = min(dev_count, len(order) - 1)
            dev_ids = set(order[:dev_count])
            train_ids = set(order[dev_count:])
        for case_id in order:
            target = "dev" if case_id in dev_ids else "train"
            if case_id not in train_ids and case_id not in dev_ids:
                target = "train"
            grouped[target].extend(cases[case_id])
    return grouped
