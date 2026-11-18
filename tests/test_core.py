import unittest

import numpy as np

from decision_lora.gates import (
    choose_fallback,
    passes_expert_gate,
    rollback_triggered,
    shared_caught_up,
)
from decision_lora.grouping import merge_tasks
from decision_lora.residual import (
    concat_factors,
    delta_from_factors,
    factor_inner,
    lambda_star,
    rank1_residual,
    residual_matrix,
    separate_ab_product,
)
from decision_lora.routing import expert_score, filter_by_modality, route
from decision_lora.folds import load_groups, split_fold
from decision_lora.schema import cannot_merge, task_cards, task_order
from decision_lora.simulate import build_samples
from decision_lora.splits import assign_splits
from pathlib import Path


class MergeTests(unittest.TestCase):
    def test_negative_cosine_is_skipped_and_first_failure_stops(self):
        tasks = ["地址", "品牌", "中介"]
        cosine = {
            ("中介", "地址"): 0.2,
            ("品牌", "地址"): 0.9,
            ("中介", "品牌"): -0.4,
        }
        tried = []

        def drop_of(left, right):
            tried.append(tuple(sorted(left | right)))
            return {name: 0.0 for name in left | right}

        groups = merge_tasks(tasks, cosine, set(), drop_of, 0.01)
        self.assertEqual(tried, [("品牌", "地址")])
        flat = {tuple(group) for group in groups}
        self.assertIn(("品牌", "地址"), flat)
        self.assertIn(("中介",), flat)

    def test_joint_drop_stops_the_rest(self):
        tasks = ["甲", "乙", "丙"]
        cosine = {("乙", "甲"): 0.9, ("丙", "乙"): 0.4, ("丙", "甲"): 0.3}
        tried = []

        def drop_of(left, right):
            tried.append(set(left | right))
            if left == {"甲"} and right == {"乙"} or left == {"乙"} and right == {"甲"}:
                return {name: 0.0 for name in left | right}
            return {name: 0.05 for name in left | right}

        groups = merge_tasks(tasks, cosine, set(), drop_of, 0.01)
        self.assertEqual(len(tried), 2)
        sizes = sorted(len(group) for group in groups)
        self.assertEqual(sizes, [1, 2])

    def test_hard_block_prevents_attempt(self):
        tasks = ["甲", "乙"]
        groups = merge_tasks(tasks, {("甲", "乙"): 0.8}, {("甲", "乙")}, lambda *_args: {}, 0.01)
        self.assertEqual(sorted(len(group) for group in groups), [1, 1])


class ResidualTests(unittest.TestCase):
    def test_projection_removes_only_parallel_part(self):
        gen = np.array([[1.0, 0.0], [0.0, 5.0]])
        delta = np.array([[1.0, 0.0], [0.0, 0.0]])
        coefficient = lambda_star(delta, gen)
        self.assertLess(coefficient, 0.1)
        kept = residual_matrix(delta, gen, coefficient)
        full = residual_matrix(delta, gen, 1.0)
        self.assertLess(np.linalg.norm(kept), np.linalg.norm(full))

    def test_orthogonal_delta_is_not_subtracted(self):
        gen = np.array([[0.0, 2.0], [0.0, 0.0]])
        delta = np.array([[3.0, 0.0], [0.0, 0.0]])
        self.assertEqual(lambda_star(delta, gen), 0.0)

    def test_factor_inner_and_concat_match_dense_residual(self):
        rng = np.random.default_rng(0)
        b_i = rng.normal(size=(4, 2))
        a_i = rng.normal(size=(2, 5))
        b_g = rng.normal(size=(4, 2))
        a_g = rng.normal(size=(2, 5))
        scale = 1.0
        delta_i = delta_from_factors(scale, b_i, a_i)
        delta_g = delta_from_factors(scale, b_g, a_g)
        self.assertAlmostEqual(factor_inner(scale, b_i, a_i, scale, b_g, a_g), float(np.sum(delta_i * delta_g)))
        coefficient = lambda_star(delta_i, delta_g)
        dense = residual_matrix(delta_i, delta_g, coefficient)
        stacked = concat_factors(scale, b_i, a_i, scale, b_g, a_g, coefficient)
        np.testing.assert_allclose(stacked, dense, atol=1e-8)
        separate = separate_ab_product(b_i, a_i, b_g, a_g)
        self.assertGreater(np.linalg.norm(separate - (delta_i - delta_g)), 1e-6)

    def test_rank1_keeps_sigma_clip(self):
        gen = np.array([[4.0, 0.0], [0.0, 0.0]])
        delta = np.array([[1.0, 0.0], [0.0, 9.0]])
        residual, coefficient = rank1_residual(delta, gen)
        self.assertAlmostEqual(coefficient, 0.25)
        np.testing.assert_allclose(residual[1, 1], 9.0)


class RouteTests(unittest.TestCase):
    def test_close_scores_abstain(self):
        weights = route({"E0": 1.0, "E1": 0.95}, margin=0.1)
        self.assertEqual(weights, {"E0": 0.0, "E1": 0.0})

    def test_clear_winner_is_top1(self):
        weights = route({"E0": 2.0, "E1": 0.2}, margin=0.1)
        self.assertEqual(weights["E0"], 1.0)
        self.assertEqual(weights["E1"], 0.0)

    def test_unstable_layers_drop_out(self):
        score = expert_score(
            [{"sigma1": 1.0, "sigma2": 0.9, "cosine_abs": 1.0}, {"sigma1": 2.0, "sigma2": 0.1, "cosine_abs": 0.5}],
            gap_min=1.5,
        )
        self.assertAlmostEqual(score, 1.0)

    def test_dialogue_does_not_enter_structured_pool(self):
        kept = filter_by_modality(
            {"talk": 0.4, "form": 0.8},
            "structured",
            {"talk": "dialogue", "form": "structured"},
        )
        self.assertEqual(list(kept), ["form"])


class GateTests(unittest.TestCase):
    def test_group_must_beat_base_and_shared(self):
        self.assertFalse(passes_expert_gate(0.5, 0.55, 0.555, 0.1, 0.1))
        self.assertTrue(passes_expert_gate(0.5, 0.55, 0.57, 0.1, 0.1))
        self.assertFalse(passes_expert_gate(0.5, 0.55, 0.7, 0.1, 0.2))

    def test_fallback_prefers_base_when_shared_cost_is_higher(self):
        self.assertEqual(choose_fallback(0.4, 0.6, 0.1, 0.2), "base")
        self.assertEqual(choose_fallback(0.4, 0.6, 0.2, 0.2), "shared")

    def test_rollback_on_norm_or_f1(self):
        self.assertTrue(rollback_triggered(0.50, 0.52, 1.0, 0.1, 0.2))
        self.assertTrue(rollback_triggered(0.52, 0.52, 1.6, 0.1, 0.2))
        self.assertFalse(rollback_triggered(0.52, 0.52, 1.1, 0.1, 0.2))
        self.assertTrue(shared_caught_up(0.60, 0.60))


class DataTests(unittest.TestCase):
    def test_cases_stay_together_inside_a_fold(self):
        cards = task_cards()
        self.assertEqual(len(task_order()), 11)
        samples = build_samples(cards)
        splits = assign_splits(samples, cards, seed=42)
        train_cases = {(row.task, row.case_id) for row in splits["train"]}
        dev_cases = {(row.task, row.case_id) for row in splits["dev"]}
        self.assertFalse(train_cases & dev_cases)
        self.assertEqual(splits["holdout_near"], [])
        groups = load_groups(Path(__file__).resolve().parents[1] / "configs" / "groups.json")
        for group in groups:
            fold = split_fold(samples, group)
            self.assertEqual(len(set(row.task for row in fold["ood"])), 3)
            self.assertFalse({row.task for row in fold["train"]} & set(group["ood_tasks"]))
            train_ids = {(row.task, row.case_id) for row in fold["train"]}
            dev_ids = {(row.task, row.case_id) for row in fold["dev"]}
            self.assertFalse(train_ids & dev_ids)

    def test_lenient_and_strict_cannot_share_a_lora(self):
        cards = task_cards()
        self.assertEqual(cannot_merge(cards["店企核验"], cards["批发商身份识别"]), "容差方向相反")
        self.assertIsNone(cannot_merge(cards["大宗商品判别"], cards["是否烟草许可"]))
        self.assertEqual(cannot_merge(cards["地址核验"], cards["批发商身份识别"]), "输出形式不同")


if __name__ == "__main__":
    unittest.main()
