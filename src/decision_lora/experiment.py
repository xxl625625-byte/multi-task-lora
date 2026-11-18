"""四组对照。域外只出现在最终报告里，不参与合并、相减档位和模型选择。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from decision_lora.folds import load_groups, split_fold
from decision_lora.gates import rollback_triggered
from decision_lora.grouping import blocked_pairs, merge_tasks
from decision_lora.schema import cannot_merge, task_cards, task_order
from decision_lora.metrics import accuracy, positive_recall
from decision_lora.model import TinyCausal
from decision_lora.pipeline import _bundle_cosine, _mix, _numpy_delta, load_config
from decision_lora.prompt import render_prompt
from decision_lora.residual import lambda_star, mean_pairwise_cosine, norm_ratio, prototype, rank1_residual, residual_matrix
from decision_lora.routing import expert_score, filter_by_modality, route
from decision_lora.simulate import GENERAL_CORPUS, Sample, build_samples, write_jsonl
from decision_lora.tokenizer import CharTokenizer
from decision_lora.train import Trainer, probe_steps


def run_compare(config: dict, root: Path) -> dict:
    torch.manual_seed(config["seed"])
    cards = task_cards()
    samples = build_samples(cards)
    write_jsonl(root / "data" / "simulated" / "samples.jsonl", samples)
    prompts = []
    for sample in samples:
        text, _start, _end = render_prompt(cards[sample.task], sample.material)
        prompts.append(text)
    prompts.extend(GENERAL_CORPUS)
    tokenizer = CharTokenizer(prompts)
    for card in cards.values():
        tokenizer.check_codes([code for code, _meaning in card.options])
    model = TinyCausal(
        tokenizer.vocab_size,
        width=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
    )
    trainer = Trainer(
        model,
        tokenizer,
        cards,
        config["rank"],
        config["alpha"],
        config["lr"],
        config["seed"],
        config["max_length"],
        list(config["target_modules"]),
    )
    groups = load_groups(root / "configs" / "groups.json")
    folds = [run_fold(trainer, samples, group, config) for group in groups]
    report = {
        "protocol": "11 任务、4 组、每组 8 个域内 3 个域外",
        "selection": "合并、相减档位只看该组域内验证集。域外不参与选择。",
        "baseline": "一个混合 LoRA，只在该组 8 个域内任务上训练",
        "method": "8 个域内任务做探针合并，再减去通用语料投影，整题路由",
        "data": "模拟样本。这组数字不能当成业务 baseline。",
        "folds": folds,
    }
    path = root / "outputs" / "compare.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(report)
    return report


def run_fold(trainer: Trainer, samples: list[Sample], group: dict, config: dict) -> dict:
    number = group["group"]
    print(f"第 {number} 组：域内 {len(group['train_tasks'])}，域外 {len(group['ood_tasks'])}")
    split = split_fold(samples, group)
    cards = trainer.cards
    scale = config["alpha"] / config["rank"]
    margin = config["f1_margin"]
    visible = list(group["train_tasks"])
    probes = {}
    probe_acc = {}
    for task in visible:
        rows = [row for row in split["train"] if row.task == task]
        probes[task] = trainer.train_decision(rows, probe_steps(config["steps_T"]))
        task_dev = [row for row in split["dev"] if row.task == task]
        probe_acc[task] = _accuracy(trainer, task_dev, probes[task])

    cosine = {}
    for index, left in enumerate(visible):
        for right in visible[index + 1 :]:
            key = (left, right) if left < right else (right, left)
            cosine[key] = _bundle_cosine(probes[left], probes[right], scale)
    blocked = blocked_pairs(
        visible,
        lambda left, right: cannot_merge(cards[left], cards[right]) is not None,
    )

    def drop_of(left, right):
        members = list(left | right)
        rows = [row for row in split["train"] if row.task in members]
        factors = trainer.train_decision(rows, probe_steps(config["steps_T"]))
        drops = {}
        for task in members:
            task_dev = [row for row in split["dev"] if row.task == task]
            drops[task] = probe_acc[task] - _accuracy(trainer, task_dev, factors)
        return drops

    clusters = merge_tasks(visible, cosine, blocked, drop_of, margin)
    fitted = _fit_clusters(trainer, split, clusters, probe_acc, config)
    shared = trainer.train_decision(split["train"], config["steps_T"])
    gen_factors, used_steps = trainer.train_language_model(
        GENERAL_CORPUS[:-2] or GENERAL_CORPUS,
        config["steps_T"],
        GENERAL_CORPUS[-2:],
    )
    rung, residuals, detail = _choose_rung(trainer, split, fitted, gen_factors, config)
    prototypes = {
        name: {module: prototype(matrix.numpy()) for module, matrix in matrices.items()}
        for name, matrices in residuals.items()
    }
    group_tasks = {name: payload["tasks"] for name, payload in fitted.items()}
    return {
        "group": number,
        "train_tasks": group["train_tasks"],
        "ood_tasks": group["ood_tasks"],
        "clusters": [payload["tasks"] for payload in fitted.values()],
        "rung": rung,
        "general_steps": used_steps,
        "rung_detail": _jsonable(detail),
        "in_domain": {
            "base": _pack(trainer, split["dev"], None, cards),
            "shared": _pack(trainer, split["dev"], shared, cards),
            "method": _pack_routed(trainer, split["dev"], residuals, prototypes, group_tasks, config),
        },
        "ood": {
            "base": _pack(trainer, split["ood"], None, cards),
            "shared": _pack(trainer, split["ood"], shared, cards),
            "method": _pack_routed(trainer, split["ood"], residuals, prototypes, group_tasks, config),
        },
    }


def _fit_clusters(trainer, split, clusters, probe_acc, config):
    fitted = {}
    margin = config["f1_margin"]
    for index, members in enumerate(clusters):
        name = "E" + str(index)
        rows = [row for row in split["train"] if row.task in members]
        factors = trainer.train_decision(rows, config["steps_T"])
        kept = list(members)
        for task in members:
            task_dev = [row for row in split["dev"] if row.task == task]
            if probe_acc[task] - _accuracy(trainer, task_dev, factors) > margin:
                kept.remove(task)
                single = [row for row in split["train"] if row.task == task]
                fitted[name + "_" + task] = {
                    "tasks": [task],
                    "factors": trainer.train_decision(single, config["steps_T"]),
                }
        if not kept:
            continue
        if kept != list(members):
            rows = [row for row in split["train"] if row.task in kept]
            factors = trainer.train_decision(rows, config["steps_T"])
        fitted[name] = {"tasks": kept, "factors": factors}
    return fitted


def _choose_rung(trainer, split, fitted, gen_factors, config):
    scale = config["alpha"] / config["rank"]
    measured = {}

    def measure(mode: str):
        residuals = {}
        ratios = []
        vectors = defaultdict(list)
        lambdas = {}
        for name, payload in fitted.items():
            matrices = {}
            lambdas[name] = {}
            for module in payload["factors"]:
                delta_i = _numpy_delta(payload["factors"], module, scale)
                delta_g = _numpy_delta(gen_factors, module, scale)
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
        score = _macro_accuracy(trainer, split["dev"], fitted, residuals)
        cosines = [mean_pairwise_cosine(items) for items in vectors.values()]
        cosines = [value for value in cosines if value is not None]
        measured[mode] = {
            "accuracy": score,
            "max_norm_ratio": max(ratios) if ratios else None,
            "prototype_cosine": sum(cosines) / len(cosines) if cosines else None,
            "lambdas": lambdas,
        }
        return residuals

    reference = measure("L0")
    candidate = measure("L1")
    if not _triggered("L1", measured, config):
        return "L1", candidate, measured
    fallback = measure("L2")
    if not _triggered("L2", measured, config):
        return "L2", fallback, measured
    return "L0", reference, measured


def _triggered(mode: str, measured: dict, config: dict) -> bool:
    return rollback_triggered(
        measured[mode]["accuracy"],
        measured["L0"]["accuracy"],
        measured[mode]["max_norm_ratio"],
        measured[mode]["prototype_cosine"],
        measured["L0"]["prototype_cosine"],
        config["f1_margin"],
        config["norm_ratio_limit"],
    )


def _accuracy(trainer, rows, factors) -> float:
    if not rows:
        return 0.0
    truth = [row.answer for row in rows]
    pred = [trainer.predict(row, factors)[0] for row in rows]
    return accuracy(truth, pred)


def _macro_accuracy(trainer, rows, fitted, residuals) -> float:
    task_to_name = {}
    for name, payload in fitted.items():
        for task in payload["tasks"]:
            task_to_name[task] = name
    by_task = defaultdict(lambda: {"truth": [], "pred": []})
    for row in rows:
        code, _captured = trainer.predict_with_matrices(row, residuals[task_to_name[row.task]])
        by_task[row.task]["truth"].append(row.answer)
        by_task[row.task]["pred"].append(code)
    scores = [accuracy(pair["truth"], pair["pred"]) for pair in by_task.values()]
    return sum(scores) / len(scores) if scores else 0.0


def _pack(trainer, rows, factors, cards) -> dict:
    predictions = {}
    for row in rows:
        if factors is None:
            code, _captured = trainer.predict_with_matrices(row, None)
        else:
            code, _captured = trainer.predict(row, factors)
        predictions[id(row)] = code
    return _summarize(rows, predictions, cards, None)


def _pack_routed(trainer, rows, residuals, prototypes, group_tasks, config) -> dict:
    predictions = {}
    zeros = 0
    cards = trainer.cards
    task_modality = {name: card.modality for name, card in cards.items()}
    for row in rows:
        base_code, captured = trainer.predict_with_matrices(row, None)
        scores = {}
        for expert, modules in prototypes.items():
            parts = []
            for module, proto in modules.items():
                if proto is None or module not in captured:
                    continue
                hidden = captured[module].numpy()
                parts.append(
                    {
                        "sigma1": proto["sigma1"],
                        "sigma2": proto["sigma2"],
                        "cosine_abs": abs(float(np.dot(proto["v"], hidden))),
                    }
                )
            score = expert_score(parts, config["spectral_gap_min"])
            if score is not None:
                scores[expert] = score
        modalities = {}
        for expert in scores:
            tasks = group_tasks.get(expert, [])
            modalities[expert] = task_modality[tasks[0]] if tasks else "structured"
        scores = filter_by_modality(scores, cards[row.task].modality, modalities)
        weights = route(scores, config["route_margin"], set())
        if not weights or all(value == 0.0 for value in weights.values()):
            zeros += 1
            predictions[id(row)] = base_code
            continue
        mixed = None
        for expert, weight in weights.items():
            if weight == 0.0:
                continue
            mixed = _mix(mixed, residuals[expert], weight)
        code, _captured = trainer.predict_with_matrices(row, mixed)
        predictions[id(row)] = code
    abstain = zeros / len(rows) if rows else None
    return _summarize(rows, predictions, cards, abstain)


def _summarize(rows, predictions, cards, abstain) -> dict:
    by_task = defaultdict(lambda: {"truth": [], "pred": []})
    for row in rows:
        by_task[row.task]["truth"].append(row.answer)
        by_task[row.task]["pred"].append(predictions[id(row)])
    per_task = {}
    for name in task_order():
        if name not in by_task:
            continue
        pair = by_task[name]
        item = {
            "n": len(pair["truth"]),
            "accuracy": accuracy(pair["truth"], pair["pred"]),
        }
        if cards[name].output_form != "graded":
            item["recall"] = positive_recall(pair["truth"], pair["pred"], "A")
        per_task[name] = item
    ordered = list(per_task.values())
    return {
        "n": len(rows),
        "macro_accuracy": sum(item["accuracy"] for item in ordered) / len(ordered) if ordered else 0.0,
        "abstain_rate": abstain,
        "per_task": per_task,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if value is None:
        return None
    return value


def _print(report: dict) -> None:
    for fold in report["folds"]:
        print(f"第 {fold['group']} 组  相减档位 {fold['rung']}  域外：{'、'.join(fold['ood_tasks'])}")
        for split_name, title in (("in_domain", "域内"), ("ood", "域外")):
            block = fold[split_name]
            print(
                f"  {title}  基座 {block['base']['macro_accuracy']:.3f}"
                f"  混合LoRA {block['shared']['macro_accuracy']:.3f}"
                f"  本方法 {block['method']['macro_accuracy']:.3f}"
            )


def load_and_compare(config_path: Path, root: Path) -> dict:
    return run_compare(load_config(config_path), root)
