"""按方案顺序执行。门槛没过就停，不用留出任务改决定。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from decision_lora.gates import (
    choose_fallback,
    passes_expert_gate,
    rollback_triggered,
    shared_caught_up,
)
from decision_lora.grouping import blocked_pairs, merge_tasks
from decision_lora.metrics import average_tasks, high_cost_rate, macro_f1
from decision_lora.model import TinyCausal
from decision_lora.residual import (
    lambda_star,
    mean_pairwise_cosine,
    norm_ratio,
    prototype,
    rank1_residual,
    residual_matrix,
)
from decision_lora.routing import expert_score, filter_by_modality, route
from decision_lora.schema import cannot_merge, task_cards
from decision_lora.simulate import GENERAL_CORPUS, Sample, build_samples, write_jsonl
from decision_lora.splits import assign_splits
from decision_lora.tokenizer import CharTokenizer
from decision_lora.train import Trainer, probe_steps


def _dense(factors: dict, scale: float) -> dict[str, torch.Tensor]:
    return {name: scale * (pair["B"] @ pair["A"]) for name, pair in factors.items()}


def _numpy_delta(factors: dict, name: str, scale: float) -> np.ndarray:
    pair = factors[name]
    return (scale * (pair["B"] @ pair["A"])).numpy()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _score_rows(rows: list[Sample], predictions: dict[str, str], cards) -> tuple[dict, dict]:
    by_task = defaultdict(lambda: {"truth": [], "pred": []})
    for row in rows:
        by_task[row.task]["truth"].append(row.answer)
        by_task[row.task]["pred"].append(predictions[id(row)])
    f1_scores = {}
    cost_scores = {}
    for task, pair in by_task.items():
        card = cards[task]
        labels = [code for code, _meaning in card.options]
        f1_scores[task] = macro_f1(pair["truth"], pair["pred"], labels)
        cost = high_cost_rate(pair["truth"], pair["pred"], card.high_cost)
        if cost is not None:
            cost_scores[task] = cost
    return f1_scores, cost_scores


class Pipeline:
    def __init__(self, config: dict, root: Path):
        self.config = config
        self.root = root
        self.cards = task_cards()
        self.scale = config["alpha"] / config["rank"]
        self.margin = config["f1_margin"]

    def run(self) -> dict:
        torch.manual_seed(self.config["seed"])
        samples = build_samples(self.cards)
        data_path = self.root / "data" / "simulated" / "samples.jsonl"
        write_jsonl(data_path, samples)
        splits = assign_splits(samples, self.cards, seed=self.config["seed"])
        texts = []
        for sample in samples:
            from decision_lora.prompt import render_prompt

            text, _start, _end = render_prompt(self.cards[sample.task], sample.material)
            texts.append(text)
        texts.extend(GENERAL_CORPUS)
        tokenizer = CharTokenizer(texts)
        for card in self.cards.values():
            tokenizer.check_codes([code for code, _meaning in card.options])

        model = TinyCausal(
            tokenizer.vocab_size,
            width=self.config["d_model"],
            n_layers=self.config["n_layers"],
            n_heads=self.config["n_heads"],
        )
        trainer = self._trainer(model, tokenizer, self.config["target_modules"])
        visible = [name for name, card in self.cards.items() if card.holdout is None]
        dev_counts = defaultdict(int)
        for row in splits["dev"]:
            dev_counts[row.task] += 1

        print("训练探针")
        probes = {}
        probe_f1 = {}
        steps = probe_steps(self.config["steps_T"])
        for task in visible:
            train_rows = [row for row in splits["train"] if row.task == task][: self.config.get("probe_sample_cap", 2000)]
            probes[task] = trainer.train_decision(train_rows, steps)
            probe_f1[task] = self._f1_for(trainer, splits["dev"], task, probes[task])

        eligible = [
            task
            for task in visible
            if dev_counts[task] >= self.config["min_dev_for_merge"]
        ]
        cosine = self._cosine_matrix(probes, eligible)
        blocked = blocked_pairs(
            eligible,
            lambda left, right: cannot_merge(self.cards[left], self.cards[right]) is not None,
        )

        def drop_of(left: frozenset[str], right: frozenset[str]) -> dict[str, float]:
            members = sorted(left | right)
            rows = [row for row in splits["train"] if row.task in members]
            joint = trainer.train_decision(rows, steps)
            drops = {}
            for task in members:
                joint_f1 = self._f1_for(trainer, splits["dev"], task, joint)
                drops[task] = probe_f1[task] - joint_f1
            return drops

        print("按探针合并")
        merged = merge_tasks(eligible, cosine, blocked, drop_of, self.margin) if len(eligible) >= 2 else [[task] for task in eligible]
        held_out_of_merge = [task for task in visible if task not in eligible]
        groups = merged + [[task] for task in held_out_of_merge]

        report = {
            "base_name": self.config["base_name"],
            "dev_floor_relaxed": self.config.get("dev_floor_relaxed", False),
            "formal_min_dev_for_merge": self.config["formal_min_dev_for_merge"],
            "samples": {key: len(value) for key, value in splits.items()},
            "data_path": str(data_path),
            "groups_before_recheck": groups,
        }
        modules = list(self.config["target_modules"])
        adopted = self._train_and_gate(trainer, splits, groups, probes, probe_f1, modules, report, allow_capacity_retry=True)
        out = self.root / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "report.json"
        report_path.write_text(json.dumps(adopted, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"主结果": adopted["adopted"], "报告": str(report_path)}, ensure_ascii=False))
        return adopted

    def _trainer(self, model, tokenizer, modules) -> Trainer:
        return Trainer(
            model,
            tokenizer,
            self.cards,
            self.config["rank"],
            self.config["alpha"],
            self.config["lr"],
            self.config["seed"],
            self.config["max_length"],
            modules,
        )

    def _train_and_gate(self, trainer, splits, groups, probes, probe_f1, modules, report, allow_capacity_retry: bool):
        print("训练组 LoRA 与共享 LoRA")
        group_models, group_names = self._fit_groups(
            trainer, splits, groups, probe_f1, recheck=allow_capacity_retry
        )
        self._group_tasks = {name: list(payload["tasks"]) for name, payload in group_models.items()}
        report["experts"] = self._group_tasks
        shared = trainer.train_decision(splits["train"], self.config["steps_T"])
        base_f1, base_cost = self._bundle_metrics(trainer, splits["dev"], None)
        shared_f1, shared_cost = self._bundle_metrics(trainer, splits["dev"], shared)
        group_f1, group_cost = self._oracle_metrics(trainer, splits["dev"], group_models, group_names)
        report.setdefault("gates", []).append(
            {
                "base_f1": base_f1,
                "shared_f1": shared_f1,
                "group_f1": group_f1,
                "base_cost": base_cost,
                "shared_cost": shared_cost,
                "group_cost": group_cost,
                "target_modules": modules,
            }
        )
        passed = passes_expert_gate(base_f1, shared_f1, group_f1, base_cost, group_cost, self.margin)
        if not passed and allow_capacity_retry and "o_proj" not in modules:
            print("组 LoRA 未过门槛，目标模块加上 o_proj 后重比一次")
            trainer.target_modules = modules + ["o_proj"]
            return self._train_and_gate(
                trainer,
                splits,
                groups,
                probes,
                probe_f1,
                modules + ["o_proj"],
                report,
                allow_capacity_retry=False,
            )
        if not passed:
            choice = choose_fallback(base_f1, shared_f1, base_cost, shared_cost)
            report["adopted"] = choice
            report["stopped_at"] = "expert_gate"
            return report

        long_steps = max(1, len(group_names) * self.config["steps_T"])
        print("加长共享 LoRA 后复测")
        long_shared = trainer.train_decision(splits["train"], long_steps)
        long_f1, _long_cost = self._bundle_metrics(trainer, splits["dev"], long_shared)
        report["long_shared_f1"] = long_f1
        report["long_shared_steps"] = long_steps
        if shared_caught_up(group_f1, long_f1, self.margin):
            report["adopted"] = "shared_long"
            report["stopped_at"] = "shared_catchup"
            return report

        print("训练通用语料并选择相减档位")
        gen_factors, used_steps = trainer.train_language_model(
            GENERAL_CORPUS[:-2] or GENERAL_CORPUS,
            self.config["steps_T"],
            GENERAL_CORPUS[-2:],
        )
        report["general_steps"] = used_steps
        rung, residuals, rung_report = self._choose_rung(trainer, splits, group_models, gen_factors)
        report["rung"] = rung
        report["rung_detail"] = rung_report
        prototypes = {
            name: {module: prototype(matrix.numpy()) for module, matrix in matrices.items()}
            for name, matrices in residuals.items()
        }
        route_f1, _route_cost = self._routed_metrics(trainer, splits["dev"], residuals, prototypes)
        report["route_f1"] = route_f1
        report["oracle_f1_for_route"] = group_f1
        if route_f1 < group_f1 - self.margin:
            report["adopted"] = "base"
            report["stopped_at"] = "router"
            report["router_note"] = "路由相对理想组别下降超过 1 个百分点，业务推理退回冻结基座"
            self._evaluate_holdout(trainer, splits, None, None, report, use_route=False)
            return report

        report["adopted"] = f"residual_{rung}"
        report["stopped_at"] = "completed"
        self._evaluate_holdout(trainer, splits, residuals, prototypes, report, use_route=True)
        return report

    def _fit_groups(self, trainer, splits, groups, probe_f1, recheck: bool):
        fitted = {}
        names = []
        for index, members in enumerate(groups):
            name = "E" + str(index)
            names.append(name)
            rows = [row for row in splits["train"] if row.task in members]
            factors = trainer.train_decision(rows, self.config["steps_T"])
            kept = list(members)
            if recheck:
                for task in members:
                    task_rows = [row for row in splits["dev"] if row.task == task]
                    if not task_rows:
                        continue
                    f1_now = self._f1_for(trainer, task_rows, task, factors)
                    if probe_f1[task] - f1_now > self.margin:
                        kept.remove(task)
                        single_rows = [row for row in splits["train"] if row.task == task]
                        single_name = name + "_" + task
                        fitted[single_name] = {
                            "tasks": [task],
                            "factors": trainer.train_decision(single_rows, self.config["steps_T"]),
                        }
            if kept:
                if kept != list(members):
                    rows = [row for row in splits["train"] if row.task in kept]
                    factors = trainer.train_decision(rows, self.config["steps_T"])
                fitted[name] = {"tasks": kept, "factors": factors}
        order = list(fitted)
        return fitted, order

    def _choose_rung(self, trainer, splits, group_models, gen_factors):
        measured = {}

        def measure(mode: str):
            residuals = {}
            ratios = []
            vectors = defaultdict(list)
            lambdas = {}
            for name, payload in group_models.items():
                matrices = {}
                lambdas[name] = {}
                for module in payload["factors"]:
                    delta_i = _numpy_delta(payload["factors"], module, self.scale)
                    delta_g = _numpy_delta(gen_factors, module, self.scale)
                    if mode == "L0":
                        matrix = delta_i
                        coefficient = 0.0
                    elif mode == "L1":
                        coefficient = lambda_star(delta_i, delta_g)
                        matrix = residual_matrix(delta_i, delta_g, coefficient)
                    else:
                        matrix, coefficient = rank1_residual(delta_i, delta_g)
                    matrices[module] = torch.tensor(matrix, dtype=torch.float32)
                    lambdas[name][module] = coefficient
                    ratio = norm_ratio(matrix, delta_i)
                    if ratio is not None:
                        ratios.append(ratio)
                    proto = prototype(matrix)
                    if proto is not None:
                        vectors[module].append(proto["v"])
                residuals[name] = matrices
            f1, _cost = self._oracle_residual_metrics(trainer, splits["dev"], group_models, residuals)
            cosines = [mean_pairwise_cosine(items) for items in vectors.values()]
            cosines = [value for value in cosines if value is not None]
            measured[mode] = {
                "f1": f1,
                "max_norm_ratio": max(ratios) if ratios else None,
                "prototype_cosine": sum(cosines) / len(cosines) if cosines else None,
                "lambdas": lambdas,
            }
            return residuals

        reference = measure("L0")
        candidate = measure("L1")
        if not self._triggered("L1", "L0", measured):
            return "L1", candidate, measured
        fallback = measure("L2")
        if not self._triggered("L2", "L0", measured):
            return "L2", fallback, measured
        return "L0", reference, measured

    def _triggered(self, mode: str, reference: str, measured: dict) -> bool:
        return rollback_triggered(
            measured[mode]["f1"],
            measured[reference]["f1"],
            measured[mode]["max_norm_ratio"],
            measured[mode]["prototype_cosine"],
            measured[reference]["prototype_cosine"],
            self.margin,
            self.config["norm_ratio_limit"],
        )

    def _evaluate_holdout(self, trainer, splits, residuals, prototypes, report, use_route: bool):
        holdout = {}
        for key in ("holdout_near", "holdout_far"):
            if use_route:
                predictions, zero_rate = self._predict_routed(trainer, splits[key], residuals, prototypes)
            else:
                predictions = self._predict_base(trainer, splits[key])
                zero_rate = None
            f1, cost = _score_rows(splits[key], predictions, self.cards)
            holdout[key] = {
                "f1": f1,
                "high_cost": cost,
                "macro_f1": average_tasks(f1),
                "abstain_rate": zero_rate,
            }
        report["holdout"] = holdout

    def _f1_for(self, trainer, rows, task, factors) -> float:
        subset = [row for row in rows if row.task == task]
        if not subset:
            return 0.0
        predictions = {}
        for row in subset:
            code, _captured = trainer.predict(row, factors)
            predictions[id(row)] = code
        f1, _cost = _score_rows(subset, predictions, self.cards)
        return f1.get(task, 0.0)

    def _bundle_metrics(self, trainer, rows, factors):
        predictions = {}
        for row in rows:
            if factors is None:
                code, _captured = trainer.predict_with_matrices(row, None)
            else:
                code, _captured = trainer.predict(row, factors)
            predictions[id(row)] = code
        f1, cost = _score_rows(rows, predictions, self.cards)
        return average_tasks(f1), average_tasks(cost)

    def _oracle_metrics(self, trainer, rows, group_models, group_names):
        task_to_name = {}
        for name in group_names:
            for task in group_models[name]["tasks"]:
                task_to_name[task] = name
        predictions = {}
        for row in rows:
            factors = group_models[task_to_name[row.task]]["factors"]
            code, _captured = trainer.predict(row, factors)
            predictions[id(row)] = code
        f1, cost = _score_rows(rows, predictions, self.cards)
        return average_tasks(f1), average_tasks(cost)

    def _oracle_residual_metrics(self, trainer, rows, group_models, residuals):
        task_to_name = {}
        for name, payload in group_models.items():
            for task in payload["tasks"]:
                task_to_name[task] = name
        predictions = {}
        for row in rows:
            code, _captured = trainer.predict_with_matrices(row, residuals[task_to_name[row.task]])
            predictions[id(row)] = code
        f1, cost = _score_rows(rows, predictions, self.cards)
        return average_tasks(f1), average_tasks(cost)

    def _routed_metrics(self, trainer, rows, residuals, prototypes):
        predictions, _zero = self._predict_routed(trainer, rows, residuals, prototypes)
        f1, cost = _score_rows(rows, predictions, self.cards)
        return average_tasks(f1), average_tasks(cost)

    def _predict_base(self, trainer, rows):
        predictions = {}
        for row in rows:
            code, _captured = trainer.predict_with_matrices(row, None)
            predictions[id(row)] = code
        return predictions

    def _predict_routed(self, trainer, rows, residuals, prototypes):
        task_modality = {name: card.modality for name, card in self.cards.items()}
        group_tasks = self._group_tasks
        predictions = {}
        zeros = 0
        for row in rows:
            _base_code, captured = trainer.predict_with_matrices(row, None)
            scores = {}
            for expert, modules in prototypes.items():
                parts = []
                for module, proto in modules.items():
                    if proto is None or module not in captured:
                        continue
                    hidden = captured[module].numpy()
                    cosine = abs(float(np.dot(proto["v"], hidden)))
                    parts.append(
                        {
                            "sigma1": proto["sigma1"],
                            "sigma2": proto["sigma2"],
                            "cosine_abs": cosine,
                        }
                    )
                score = expert_score(parts, self.config["spectral_gap_min"])
                if score is not None:
                    scores[expert] = score
            modalities = {}
            for expert in scores:
                tasks = group_tasks.get(expert, [])
                modalities[expert] = task_modality[tasks[0]] if tasks else "structured"
            scores = filter_by_modality(scores, self.cards[row.task].modality, modalities)
            weights = route(scores, self.config["route_margin"], set())
            if not weights or all(value == 0.0 for value in weights.values()):
                zeros += 1
                predictions[id(row)] = _base_code
                continue
            mixed = None
            for expert, weight in weights.items():
                if weight == 0.0:
                    continue
                mixed = _mix(mixed, residuals[expert], weight)
            code, _captured = trainer.predict_with_matrices(row, mixed)
            predictions[id(row)] = code
        zero_rate = zeros / len(rows) if rows else None
        return predictions, zero_rate

    def _cosine_matrix(self, probes, tasks):
        matrix = {}
        for index, left in enumerate(tasks):
            for right in tasks[index + 1 :]:
                key = (left, right) if left < right else (right, left)
                matrix[key] = _bundle_cosine(probes[left], probes[right], self.scale)
        return matrix


def _mix(accumulator, matrices, weight: float):
    mixed = {} if accumulator is None else accumulator
    for name, matrix in matrices.items():
        added = weight * matrix
        mixed[name] = added if name not in mixed else mixed[name] + added
    return mixed


def _bundle_cosine(left: dict, right: dict, scale: float) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for name in left:
        if name not in right:
            continue
        delta_left = _numpy_delta(left, name, scale)
        delta_right = _numpy_delta(right, name, scale)
        norm_left = float(np.linalg.norm(delta_left))
        norm_right = float(np.linalg.norm(delta_right))
        if norm_left == 0.0 or norm_right == 0.0:
            continue
        cosine = float(np.sum(delta_left * delta_right) / (norm_left * norm_right))
        weight = 0.5 * (norm_left ** 2 + norm_right ** 2)
        numerator += weight * cosine
        denominator += weight
    if denominator == 0.0:
        return None
    return numerator / denominator
