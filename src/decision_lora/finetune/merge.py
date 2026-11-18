"""探针之间的硬约束和贪心合并。余弦只排序，验证集退步才拒收。"""

import math
import random
from collections import defaultdict

import numpy as np

DROP_LIMIT = 0.01

MERGE_TRAIT = {
    "大宗商品判别": ("exclusive", "neutral"),
    "是否烟草许可": ("exclusive", "neutral"),
    "是否道路运输许可": ("exclusive", "neutral"),
    "是否食品许可": ("exclusive", "neutral"),
    "是否建筑市场资质": ("exclusive", "neutral"),
    "是否商务特许许可": ("exclusive", "neutral"),
    "是否有排污许可": ("exclusive", "neutral"),
    "批发商身份识别": ("exclusive", "strict"),
    "地址核验": ("graded", "lenient"),
    "店企核验": ("exclusive", "lenient"),
    "品牌名和店铺名一致性判别": ("exclusive", "lenient"),
}


def cannot_merge(left, right):
    left_form, left_tolerance = MERGE_TRAIT[left]
    right_form, right_tolerance = MERGE_TRAIT[right]
    if left_form != right_form:
        return True
    if {left_tolerance, right_tolerance} == {"lenient", "strict"}:
        return True
    return False


def blocked_pairs(names):
    blocked = set()
    ordered = list(names)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if cannot_merge(left, right):
                key = (left, right) if left < right else (right, left)
                blocked.add(key)
    return blocked


def merge_tasks(tasks, cosine, blocked, drop_of, drop_limit):
    clusters = {task: {task} for task in tasks}

    def pair_key(left, right):
        return (left, right) if left < right else (right, left)

    def cluster_score(left, right):
        scores = []
        for one in left:
            for other in right:
                if pair_key(one, other) in blocked:
                    return None
                value = cosine.get(pair_key(one, other))
                if value is None or value < 0:
                    return None
                scores.append(value)
        if not scores:
            return None
        return sum(scores) / len(scores)

    while True:
        names = list(clusters)
        best = None
        best_score = -1.0
        for index, left_name in enumerate(names):
            for right_name in names[index + 1:]:
                score = cluster_score(clusters[left_name], clusters[right_name])
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best = (left_name, right_name)
        if best is None:
            break
        left_name, right_name = best
        drops = drop_of(set(clusters[left_name]), set(clusters[right_name]))
        if any(value > drop_limit for value in drops.values()):
            break
        clusters[left_name].update(clusters[right_name])
        del clusters[right_name]
    return [sorted(members) for members in clusters.values()]


def factor_inner(scale_left, matrix_b_left, matrix_a_left, scale_right, matrix_b_right, matrix_a_right):
    gram_b = matrix_b_left.T @ matrix_b_right
    gram_a = matrix_a_left @ matrix_a_right.T
    return float(scale_left * scale_right * np.sum(gram_b * gram_a))


def bundle_cosine(left, right):
    numerator = 0.0
    denominator = 0.0
    for name, one in left.items():
        other = right.get(name)
        if other is None:
            continue
        left_norm = factor_inner(one["scale"], one["B"], one["A"], one["scale"], one["B"], one["A"])
        right_norm = factor_inner(other["scale"], other["B"], other["A"], other["scale"], other["B"], other["A"])
        if left_norm <= 0.0 or right_norm <= 0.0:
            continue
        inner = factor_inner(one["scale"], one["B"], one["A"], other["scale"], other["B"], other["A"])
        cosine = inner / math.sqrt(left_norm * right_norm)
        weight = 0.5 * (left_norm + right_norm)
        numerator += weight * cosine
        denominator += weight
    if denominator == 0.0:
        return None
    return numerator / denominator


def rows_of(rows, names):
    chosen = set(names)
    return [row for row in rows if row["任务"] in chosen]


def equal_rows(rows, rng, count):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["任务"]].append(row)
    tasks = sorted(by_task)
    bags = {}
    cursors = {}
    for task, items in by_task.items():
        bag = list(items)
        rng.shuffle(bag)
        bags[task] = bag
        cursors[task] = 0
    stream = []
    for _ in range(count):
        task = tasks[rng.randrange(len(tasks))]
        if cursors[task] >= len(bags[task]):
            rng.shuffle(bags[task])
            cursors[task] = 0
        stream.append(bags[task][cursors[task]])
        cursors[task] += 1
    return stream


def update_budget(row_count, args):
    group_size = args.batch_size * args.accumulation
    per_epoch = max(1, math.ceil(row_count / group_size))
    full = per_epoch * args.epochs
    probe = max(1, min(500, full // 5))
    return full, probe, per_epoch


def seeded(seed):
    return random.Random(seed)
