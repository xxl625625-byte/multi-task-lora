"""通用语料投影相减。系数只去掉和专用增量重叠的部分。"""

from __future__ import annotations

import numpy as np


def factor_inner(
    scale_i: float,
    matrix_b_i: np.ndarray,
    matrix_a_i: np.ndarray,
    scale_g: float,
    matrix_b_g: np.ndarray,
    matrix_a_g: np.ndarray,
) -> float:
    """<s_i B_i A_i, s_g B_g A_g>_F，不展开完整矩阵。"""
    gram_b = matrix_b_i.T @ matrix_b_g
    gram_a = matrix_a_i @ matrix_a_g.T
    return float(scale_i * scale_g * np.sum(gram_b * gram_a))


def delta_from_factors(scale: float, matrix_b: np.ndarray, matrix_a: np.ndarray) -> np.ndarray:
    return scale * (matrix_b @ matrix_a)


def lambda_star(delta_i: np.ndarray, delta_gen: np.ndarray) -> float:
    denominator = float(np.sum(delta_gen * delta_gen))
    if denominator == 0.0:
        return 0.0
    value = float(np.sum(delta_i * delta_gen) / denominator)
    return float(np.clip(value, 0.0, 1.0))


def residual_matrix(delta_i: np.ndarray, delta_gen: np.ndarray, coefficient: float) -> np.ndarray:
    return delta_i - coefficient * delta_gen


def concat_factors(
    scale_i: float,
    matrix_b_i: np.ndarray,
    matrix_a_i: np.ndarray,
    scale_g: float,
    matrix_b_g: np.ndarray,
    matrix_a_g: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    """缩放写进 B，前向不再乘一次 s。"""
    stacked_b = np.concatenate(
        [scale_i * matrix_b_i, -coefficient * scale_g * matrix_b_g],
        axis=1,
    )
    stacked_a = np.concatenate([matrix_a_i, matrix_a_g], axis=0)
    return stacked_b @ stacked_a


def separate_ab_product(
    matrix_b_i: np.ndarray,
    matrix_a_i: np.ndarray,
    matrix_b_g: np.ndarray,
    matrix_a_g: np.ndarray,
) -> np.ndarray:
    return (matrix_b_i - matrix_b_g) @ (matrix_a_i - matrix_a_g)


def rank1_residual(delta_i: np.ndarray, delta_gen: np.ndarray) -> tuple[np.ndarray, float]:
    """只减去通用增量的最大奇异成分。σ1 为 0 时不减。"""
    if float(np.linalg.norm(delta_gen)) == 0.0:
        return delta_i.copy(), 0.0
    left, values, right = np.linalg.svd(delta_gen, full_matrices=False)
    sigma = float(values[0])
    if sigma == 0.0:
        return delta_i.copy(), 0.0
    direction_u = left[:, 0]
    direction_v = right[0]
    coefficient = float(np.dot(direction_u, delta_i @ direction_v) / sigma)
    coefficient = float(np.clip(coefficient, 0.0, 1.0))
    component = sigma * np.outer(direction_u, direction_v)
    return delta_i - coefficient * component, coefficient


def prototype(matrix: np.ndarray) -> dict | None:
    """第一右奇异向量。代码里取 Vh 的第 0 行，对应数学记号的第一列。"""
    if float(np.linalg.norm(matrix)) < 1e-12:
        return None
    _left, values, right = np.linalg.svd(matrix, full_matrices=False)
    sigma2 = float(values[1]) if values.size > 1 else 0.0
    return {"v": right[0].copy(), "sigma1": float(values[0]), "sigma2": sigma2}


def norm_ratio(residual: np.ndarray, delta_i: np.ndarray) -> float | None:
    base = float(np.linalg.norm(delta_i))
    if base == 0.0:
        return None
    return float(np.linalg.norm(residual) / base)


def mean_pairwise_cosine(vectors: list[np.ndarray]) -> float | None:
    if len(vectors) < 2:
        return None
    total = 0.0
    count = 0
    for index, left in enumerate(vectors):
        for right in vectors[index + 1 :]:
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denom == 0.0:
                continue
            total += float(np.dot(left, right) / denom)
            count += 1
    if count == 0:
        return None
    return total / count
