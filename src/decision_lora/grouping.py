"""探针余弦与贪心合并。余弦只排序，开发集退步才决定拒收。"""

from __future__ import annotations

from collections.abc import Callable, Iterable


DropFn = Callable[[frozenset[str], frozenset[str]], dict[str, float]]


def merge_tasks(
    tasks: list[str],
    cosine: dict[tuple[str, str], float | None],
    blocked: set[tuple[str, str]],
    drop_of: DropFn,
    drop_limit: float = 0.01,
) -> list[list[str]]:
    """从单任务开始合并。

    余弦为负或硬约束挡住的对不尝试。按平均余弦从高到低试，
    第一次有成员下降超过 drop_limit，整个合并停止。
    """
    clusters: dict[str, set[str]] = {task: {task} for task in tasks}

    def pair_key(left: str, right: str) -> tuple[str, str]:
        return (left, right) if left < right else (right, left)

    def cluster_score(left: set[str], right: set[str]) -> float | None:
        scores: list[float] = []
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
        best: tuple[str, str] | None = None
        best_score = -1.0
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                score = cluster_score(clusters[left_name], clusters[right_name])
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best = (left_name, right_name)
        if best is None:
            break
        left_name, right_name = best
        drops = drop_of(frozenset(clusters[left_name]), frozenset(clusters[right_name]))
        if any(value > drop_limit for value in drops.values()):
            break
        clusters[left_name].update(clusters[right_name])
        del clusters[right_name]

    return [sorted(members) for members in clusters.values()]


def blocked_pairs(names: Iterable[str], cannot) -> set[tuple[str, str]]:
    blocked: set[tuple[str, str]] = set()
    ordered = list(names)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if cannot(left, right):
                key = (left, right) if left < right else (right, left)
                blocked.add(key)
    return blocked
