"""四组 8+3 切分。分组文件与 baseline 的 groups.json 相同，不再重抽。"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from decision_lora.schema import task_order
from decision_lora.simulate import Sample


def load_groups(path: Path) -> list[dict]:
    groups = json.loads(path.read_text(encoding="utf-8"))
    names = set(task_order())
    numbers = []
    for group in groups:
        train = list(group["train_tasks"])
        ood = list(group["ood_tasks"])
        number = group["group"]
        if len(train) != 8 or len(ood) != 3:
            raise ValueError(f"第 {number} 组不是 8 个域内加 3 个域外")
        if set(train) & set(ood):
            raise ValueError(f"第 {number} 组域内和域外有重叠")
        if set(train) | set(ood) != names:
            raise ValueError(f"第 {number} 组没有覆盖全部 11 个任务")
        numbers.append(number)
    if numbers != [1, 2, 3, 4]:
        raise ValueError("分组编号必须是 1 到 4，且按这个顺序")
    return groups


def split_fold(
    samples: list[Sample],
    group: dict,
    dev_ratio: float = 0.2,
) -> dict[str, list[Sample]]:
    """域外任务整段留出。域内按案例切，同一 case_id 的派生样本不拆开。"""
    by_task: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_task[sample.task].append(sample)
    missing = [name for name in group["train_tasks"] + group["ood_tasks"] if name not in by_task]
    if missing:
        raise ValueError("这些任务没有样本：" + "、".join(missing))
    train: list[Sample] = []
    dev: list[Sample] = []
    ood: list[Sample] = []
    for name in group["ood_tasks"]:
        ood.extend(by_task[name])
    for name in group["train_tasks"]:
        rng = random.Random(f"{group['seed']}:{group['group']}:{name}")
        task_train, task_dev = _split_cases(by_task[name], dev_ratio, rng)
        train.extend(task_train)
        dev.extend(task_dev)
    train_ids = {(row.task, row.case_id) for row in train}
    leaked = [row for row in dev if (row.task, row.case_id) in train_ids]
    if leaked:
        raise ValueError("训练集和验证集出现了同一任务的同一案例")
    ood_names = set(group["ood_tasks"])
    if any(row.task in ood_names for row in train + dev):
        raise ValueError("域外任务进入了训练或验证")
    return {"train": train, "dev": dev, "ood": ood}


def _split_cases(rows: list[Sample], valid_ratio: float, rng: random.Random):
    cases: dict[str, list[Sample]] = defaultdict(list)
    for row in rows:
        cases[row.case_id].append(row)
    by_answer: dict[str, list[str]] = defaultdict(list)
    for case_id, items in cases.items():
        by_answer[items[0].answer].append(case_id)
    train_ids: list[str] = []
    valid_ids: list[str] = []
    for items in by_answer.values():
        rng.shuffle(items)
        if len(items) == 1:
            train_ids.append(items[0])
            continue
        n_valid = int(round(len(items) * valid_ratio))
        n_valid = min(max(n_valid, 1), len(items) - 1)
        valid_ids.extend(items[:n_valid])
        train_ids.extend(items[n_valid:])
    if not valid_ids and train_ids:
        valid_ids.append(train_ids.pop())
    if not train_ids:
        raise ValueError("切分后训练集为空")
    train = [row for case_id in train_ids for row in cases[case_id]]
    valid = [row for case_id in valid_ids for row in cases[case_id]]
    return train, valid
