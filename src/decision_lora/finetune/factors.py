"""把专家 LoRA 减去通用语料 LoRA。系数按模块分别算，不拆开 A、B 各自相减。"""

import math
from collections import defaultdict

import numpy as np

from decision_lora.finetune.merge import factor_inner

NORM_LIMIT = 1.5
GAP_MIN = 1.5
ROUTE_MARGIN = 0.1


def low_rank_usv(matrix_b, matrix_a):
    orthogonal, triangle = np.linalg.qr(matrix_a.T, mode="reduced")
    core = matrix_b @ triangle.T
    left, values, vt = np.linalg.svd(core, full_matrices=False)
    right = orthogonal @ vt.T
    return left, values, right


def lambda_star(expert, general, name):
    one = expert[name]
    other = general[name]
    denominator = factor_inner(other["scale"], other["B"], other["A"], other["scale"], other["B"], other["A"])
    if denominator == 0.0:
        return 0.0
    inner = factor_inner(one["scale"], one["B"], one["A"], other["scale"], other["B"], other["A"])
    return float(np.clip(inner / denominator, 0.0, 1.0))


def residual_factors(expert, general, mode):
    packed = {}
    lambdas = {}
    ratios = []
    for name, one in expert.items():
        other = general[name]
        scale = one["scale"]
        if mode == "L0":
            matrix_b = scale * one["B"]
            matrix_a = one["A"]
            coefficient = 0.0
            ratio = 1.0
        elif mode == "L1":
            coefficient = lambda_star(expert, general, name)
            matrix_b = np.concatenate([scale * one["B"], -coefficient * other["scale"] * other["B"]], axis=1)
            matrix_a = np.concatenate([one["A"], other["A"]], axis=0)
            left_norm = factor_inner(scale, one["B"], one["A"], scale, one["B"], one["A"])
            gen_norm = factor_inner(other["scale"], other["B"], other["A"], other["scale"], other["B"], other["A"])
            inner = factor_inner(scale, one["B"], one["A"], other["scale"], other["B"], other["A"])
            residual_norm = left_norm + coefficient * coefficient * gen_norm - 2.0 * coefficient * inner
            ratio = math.sqrt(max(residual_norm, 0.0) / left_norm) if left_norm > 0.0 else None
        else:
            left, values, right = low_rank_usv(other["scale"] * other["B"], other["A"])
            sigma = float(values[0]) if values.size else 0.0
            if sigma == 0.0:
                matrix_b = scale * one["B"]
                matrix_a = one["A"]
                coefficient = 0.0
                ratio = 1.0
            else:
                direction_u = left[:, 0]
                direction_v = right[:, 0]
                projected = (scale * one["B"]) @ (one["A"] @ direction_v)
                coefficient = float(np.clip(np.dot(direction_u, projected) / sigma, 0.0, 1.0))
                matrix_b = np.concatenate([scale * one["B"], (-coefficient * sigma * direction_u).reshape(-1, 1)], axis=1)
                matrix_a = np.concatenate([one["A"], direction_v.reshape(1, -1)], axis=0)
                dense_i = (scale * one["B"]) @ one["A"]
                dense_r = matrix_b @ matrix_a
                base = float(np.linalg.norm(dense_i))
                ratio = float(np.linalg.norm(dense_r) / base) if base > 0.0 else None
        packed[name] = (
            np.ascontiguousarray(matrix_a, dtype=np.float32),
            np.ascontiguousarray(matrix_b, dtype=np.float32),
        )
        lambdas[name] = coefficient
        if ratio is not None:
            ratios.append(ratio)
    return packed, lambdas, (max(ratios) if ratios else None)


def prototype_of(matrix_a, matrix_b):
    _left, values, right = low_rank_usv(matrix_b, matrix_a)
    if values.size == 0 or float(values[0]) < 1e-8:
        return None
    sigma2 = float(values[1]) if values.size > 1 else 0.0
    return {"v": right[:, 0].astype(np.float32), "sigma1": float(values[0]), "sigma2": sigma2}


def prototypes_of(experts):
    prototypes = {}
    for name, modules in experts.items():
        prototypes[name] = {module: prototype_of(matrix_a, matrix_b) for module, (matrix_a, matrix_b) in modules.items()}
    return prototypes


def mean_pairwise_cosine(vectors):
    if len(vectors) < 2:
        return None
    total = 0.0
    count = 0
    for index, left in enumerate(vectors):
        for right in vectors[index + 1:]:
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denom == 0.0:
                continue
            total += float(np.dot(left, right) / denom)
            count += 1
    if count == 0:
        return None
    return total / count


def cosine_of_prototypes(prototypes):
    by_module = defaultdict(list)
    for modules in prototypes.values():
        for module, proto in modules.items():
            if proto is not None:
                by_module[module].append(proto["v"])
    values = [mean_pairwise_cosine(items) for items in by_module.values()]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def rollback_triggered(candidate, reference, norm_ratio, candidate_cosine, reference_cosine, drop_limit):
    if candidate < reference - drop_limit:
        return True
    if norm_ratio is not None and norm_ratio > NORM_LIMIT:
        return True
    if candidate_cosine is not None and reference_cosine is not None and candidate_cosine > reference_cosine:
        return True
    return False


def save_factors(path, experts, prototypes):
    payload = {}
    for name, modules in experts.items():
        for module, (matrix_a, matrix_b) in modules.items():
            payload[f"expert::{name}::{module}::A"] = matrix_a
            payload[f"expert::{name}::{module}::B"] = matrix_b
    for name, modules in prototypes.items():
        for module, proto in modules.items():
            if proto is None:
                continue
            payload[f"proto::{name}::{module}::v"] = proto["v"]
            payload[f"proto::{name}::{module}::sigma"] = np.array([proto["sigma1"], proto["sigma2"]], dtype=np.float32)
    np.savez(path, **payload)


def load_factors(path):
    data = np.load(path)
    experts = defaultdict(dict)
    prototypes = defaultdict(dict)
    for key in data.files:
        kind, name, module, part = key.split("::")
        if kind == "expert" and part == "A":
            experts[name][module] = (data[key], data[f"expert::{name}::{module}::B"])
        elif kind == "proto" and part == "v":
            sigma = data[f"proto::{name}::{module}::sigma"]
            prototypes[name][module] = {"v": data[key], "sigma1": float(sigma[0]), "sigma2": float(sigma[1])}
    return dict(experts), dict(prototypes)
