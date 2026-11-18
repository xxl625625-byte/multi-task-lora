"""读取原始 jsonl，并按 groups.json 切成和 baseline 相同的 8+3。"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from decision_lora.finetune.tasks import TASK_ORDER, expand_row

SEED = 20261117
VALID_RATIO = 0.2


def log(message):
    print(message, file=sys.stderr, flush=True)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def material_key(row):
    return re.sub(r"\s+", "", row["材料"])


def iter_jsonl(path):
    with open(path, encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON：{exc}") from exc


def load_groups(path):
    groups = read_json(path)
    if not isinstance(groups, list) or len(groups) != 4:
        raise ValueError("groups.json 必须是 4 组")
    seen = []
    for group in groups:
        train_tasks = group.get("train_tasks")
        ood_tasks = group.get("ood_tasks")
        if len(train_tasks) != 8 or len(ood_tasks) != 3:
            raise ValueError(f"第 {group.get('group')} 组必须是 8 个训练任务加 3 个域外任务")
        if set(train_tasks) & set(ood_tasks):
            raise ValueError(f"第 {group['group']} 组的训练任务和域外任务有重叠")
        unknown = [name for name in train_tasks + ood_tasks if name not in TASK_ORDER]
        if unknown:
            raise ValueError("未知任务：" + "、".join(unknown))
        seen.extend(train_tasks + ood_tasks)
    missing = [name for name in TASK_ORDER if name not in seen]
    if missing:
        raise ValueError("分组没有覆盖：" + "、".join(missing))
    return groups


def load_raw(raw_dir):
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise ValueError(f"找不到原始数据目录：{raw_dir}")
    task_files = [raw_dir / f"{name}.jsonl" for name in TASK_ORDER]
    combined = raw_dir / "all.jsonl"
    present = [path for path in task_files if path.is_file()]
    if present and combined.is_file():
        raise ValueError("data/raw 里不要同时放 all.jsonl 和按任务拆开的 jsonl")
    rows = {name: [] for name in TASK_ORDER}
    if combined.is_file():
        sources = [(combined, None)]
    elif len(present) == len(task_files):
        sources = [(path, path.stem) for path in task_files]
    else:
        missing = [path.name for path in task_files if not path.is_file()]
        raise ValueError("原始数据还没放齐。请放 11 个 jsonl，或一个 all.jsonl。缺少：" + "、".join(missing))
    for path, task_name in sources:
        for line_number, obj in iter_jsonl(path):
            try:
                row = expand_row(obj, task_name)
            except ValueError as exc:
                raise ValueError(f"{path} 第 {line_number} 行：{exc}") from exc
            rows[row["任务"]].append(row)
    empty = [name for name, items in rows.items() if not items]
    if empty:
        raise ValueError("这些任务没有样本：" + "、".join(empty))
    for name, items in rows.items():
        seen = {}
        for index, row in enumerate(items, 1):
            key = material_key(row)
            if key in seen:
                raise ValueError(f"{name} 第 {index} 条材料和第 {seen[key]} 条重复")
            seen[key] = index
    return rows


def split_task_rows(rows, valid_ratio, rng):
    keyed = [row["分组键"] for row in rows]
    if any(keyed) and not all(keyed):
        raise ValueError("同一任务里，分组键要么都有，要么都没有")
    if all(keyed):
        buckets = defaultdict(list)
        for row in rows:
            buckets[row["分组键"]].append(row)
        keys = list(buckets)
        if len(keys) < 2:
            raise ValueError("分组键只有 1 个，无法把同一实体拆开")
        rng.shuffle(keys)
        n_valid = int(round(len(keys) * valid_ratio))
        n_valid = min(max(n_valid, 1), len(keys) - 1)
        valid_keys = set(keys[:n_valid])
        train, valid = [], []
        for key, items in buckets.items():
            (valid if key in valid_keys else train).extend(items)
        return train, valid
    by_answer = defaultdict(list)
    for row in rows:
        by_answer[row["答案"]].append(row)
    train, valid = [], []
    for items in by_answer.values():
        rng.shuffle(items)
        if len(items) == 1:
            train.append(items[0])
            continue
        n_valid = int(round(len(items) * valid_ratio))
        n_valid = min(max(n_valid, 1), len(items) - 1)
        valid.extend(items[:n_valid])
        train.extend(items[n_valid:])
    if not valid:
        rng.shuffle(train)
        valid.append(train.pop())
    if not train:
        raise ValueError("切分后训练集为空")
    return train, valid


def prepare_group(group, rows_by_task, valid_ratio):
    ood_rows = [row for name in group["ood_tasks"] for row in rows_by_task[name]]
    ood_keys = {row["隔离键"] for row in ood_rows if row.get("隔离键")}
    train_rows, valid_rows = [], []
    dropped = {"分组键": 0}
    counts = {}
    for name in group["ood_tasks"]:
        counts[name] = {"train": 0, "valid": 0, "ood": len(rows_by_task[name])}
    for name in group["train_tasks"]:
        kept = []
        for row in rows_by_task[name]:
            if row.get("隔离键") and row["隔离键"] in ood_keys:
                dropped["分组键"] += 1
                continue
            kept.append(row)
        if len(kept) < 2:
            raise ValueError(f"第 {group['group']} 组的「{name}」去掉与域外重叠的样本后不足 2 条")
        rng = random.Random(f"{group['seed']}:{group['group']}:{name}")
        task_train, task_valid = split_task_rows(kept, valid_ratio, rng)
        train_rows.extend(task_train)
        valid_rows.extend(task_valid)
        counts[name] = {"train": len(task_train), "valid": len(task_valid), "ood": 0}
    train_keys = {material_key(row) + "\n" + row["任务"] for row in train_rows}
    leaked = [row for row in valid_rows if material_key(row) + "\n" + row["任务"] in train_keys]
    if leaked:
        raise ValueError("训练集和验证集出现了同一任务的同一材料")
    return {
        "train": train_rows,
        "valid": valid_rows,
        "ood": ood_rows,
        "meta": {
            "group": group["group"],
            "seed": group["seed"],
            "valid_ratio": valid_ratio,
            "train_tasks": group["train_tasks"],
            "ood_tasks": group["ood_tasks"],
            "dropped_overlap": dropped,
            "counts": counts,
        },
    }


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_all(raw_dir, groups_path, out_dir, valid_ratio):
    groups = load_groups(groups_path)
    rows = load_raw(raw_dir)
    out_dir = Path(out_dir)
    for group in groups:
        packed = prepare_group(group, rows, valid_ratio)
        folder = out_dir / f"group_{group['group']:02d}"
        write_jsonl(folder / "train.jsonl", packed["train"])
        write_jsonl(folder / "valid.jsonl", packed["valid"])
        write_jsonl(folder / "ood.jsonl", packed["ood"])
        write_json(folder / "split_meta.json", packed["meta"])
        log(
            f"第 {group['group']} 组  train={len(packed['train'])}  "
            f"valid={len(packed['valid'])}  ood={len(packed['ood'])}"
        )
    log(f"已写入 {out_dir.resolve()}")


def load_prepared(path):
    rows = []
    required = ("任务", "题型", "判断目标", "判断规则", "材料", "选项", "答案")
    for line_number, obj in iter_jsonl(path):
        if not isinstance(obj, dict) or any(key not in obj for key in required):
            raise ValueError(f"{path} 第 {line_number} 行缺少训练字段")
        if obj["任务"] not in TASK_ORDER:
            raise ValueError(f"{path} 第 {line_number} 行任务未登记：{obj['任务']}")
        if obj["答案"] not in [item["代号"] for item in obj["选项"]]:
            raise ValueError(f"{path} 第 {line_number} 行答案不在选项里")
        rows.append(obj)
    if not rows:
        raise ValueError(f"数据为空：{path}")
    return rows
