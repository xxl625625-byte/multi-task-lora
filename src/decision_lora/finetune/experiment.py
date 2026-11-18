"""一组的完整流程：探针、合并、全量专家、通用语料相减、路由评估。"""

from collections import defaultdict
from pathlib import Path

from decision_lora.finetune.batch import encode_rows
from decision_lora.finetune.corpus import load_general_corpus
from decision_lora.finetune.data import load_prepared, log, prepare_all, read_json, write_json
from decision_lora.finetune.factors import (
    cosine_of_prototypes,
    load_factors,
    prototypes_of,
    residual_factors,
    rollback_triggered,
    save_factors,
)
from decision_lora.finetune.lora_train import export_lora, fit_decision, fit_language, load_model
from decision_lora.finetune.merge import DROP_LIMIT, blocked_pairs, bundle_cosine, merge_tasks, rows_of, update_budget
from decision_lora.finetune.route import predict_base, predict_experts, route_rows
from decision_lora.finetune.score import pack_report, print_report

TASK_NAME = "eleven_task_method_lora"
FORMAT_VERSION = 1


def task_accuracy(rows, predictions):
    report = pack_report(rows, predictions)
    return {name: item["accuracy"] for name, item in report["per_task"].items()}


def macro_of(rows, predictions):
    if not rows:
        return None
    return pack_report(rows, predictions)["macro_accuracy"]


def output_dir(root, group_id):
    return Path(root) / f"group_{group_id:02d}"


def train_cluster(model, tokenizer, args, torch, device, dtype, precision, group_id, train_rows, valid_rows, members, probe_accuracy, index):
    name = "E" + str(index)
    member_train = rows_of(train_rows, members)
    member_valid = rows_of(valid_rows, members)
    full, _probe, per_epoch = update_budget(len(member_train), args)
    log(f"第 {group_id} 组训练 {name}：{'、'.join(members)}，{full} 步")
    torch.manual_seed(args.seed + 100 + index)
    factors = fit_decision(
        model, tokenizer, member_train, member_valid, args, torch, device, dtype, precision,
        full, per_epoch, f"第 {group_id} 组 {name}",
    )
    if len(members) == 1:
        return {name: {"tasks": list(members), "factors": factors}}
    applied = {name: residual_factors(factors, factors, "L0")[0]}
    predictions = predict_experts(
        model, tokenizer, member_valid, applied, {task: name for task in members}, args, torch, device, dtype,
    )
    accuracy = task_accuracy(member_valid, predictions)
    dropouts = [task for task in members if task in probe_accuracy and probe_accuracy[task] - accuracy[task] > DROP_LIMIT]
    fitted = {}
    kept = [task for task in members if task not in dropouts]
    for offset, task in enumerate(dropouts):
        single_name = name + "_" + str(offset)
        single_train = rows_of(train_rows, [task])
        single_valid = rows_of(valid_rows, [task])
        single_full, _probe, single_every = update_budget(len(single_train), args)
        log(f"第 {group_id} 组 {task} 合并后退步，拆成单任务")
        torch.manual_seed(args.seed + 200 + index * 10 + offset)
        fitted[single_name] = {
            "tasks": [task],
            "factors": fit_decision(
                model, tokenizer, single_train, single_valid, args, torch, device, dtype, precision,
                single_full, single_every, f"第 {group_id} 组 {single_name}",
            ),
        }
    if not kept:
        return fitted
    if dropouts:
        kept_train = rows_of(train_rows, kept)
        kept_valid = rows_of(valid_rows, kept)
        kept_full, _probe, kept_every = update_budget(len(kept_train), args)
        log(f"第 {group_id} 组重训去掉退步任务后的 {name}")
        torch.manual_seed(args.seed + 300 + index)
        factors = fit_decision(
            model, tokenizer, kept_train, kept_valid, args, torch, device, dtype, precision,
            kept_full, kept_every, f"第 {group_id} 组 {name} 重训",
        )
    fitted[name] = {"tasks": kept, "factors": factors}
    return fitted


def evaluate_saved(model, tokenizer, args, torch, device, dtype, valid_rows, ood_rows, experts, prototypes, manifest):
    base_valid = predict_base(model, tokenizer, valid_rows, args, torch, device, dtype)
    base_ood = predict_base(model, tokenizer, ood_rows, args, torch, device, dtype)
    routed_valid, valid_abstain = route_rows(model, tokenizer, valid_rows, experts, prototypes, args, torch, device, dtype)
    routed_ood, ood_abstain = route_rows(model, tokenizer, ood_rows, experts, prototypes, args, torch, device, dtype)
    valid_report = pack_report(valid_rows, routed_valid)
    oracle_accuracy = manifest["rung_detail"][manifest["rung"]]["accuracy"]
    adopted = "residual_" + manifest["rung"]
    if valid_report["macro_accuracy"] < oracle_accuracy - DROP_LIMIT:
        adopted = "base"
        valid_report = pack_report(valid_rows, base_valid)
        ood_report = pack_report(ood_rows, base_ood)
    else:
        ood_report = pack_report(ood_rows, routed_ood)
    return {
        "group": manifest["group"],
        "adopted": adopted,
        "rung": manifest["rung"],
        "train_tasks": manifest["train_tasks"],
        "ood_tasks": manifest["ood_tasks"],
        "clusters": manifest["clusters"],
        "in_domain": valid_report,
        "ood": ood_report,
        "abstain_rate": {"in_domain": valid_abstain, "ood": ood_abstain},
        "base": {"in_domain": pack_report(valid_rows, base_valid), "ood": pack_report(ood_rows, base_ood)},
        "routed": {"in_domain": pack_report(valid_rows, routed_valid), "ood": pack_report(ood_rows, routed_ood)},
    }


def train_group(model, tokenizer, args, torch, device, dtype, precision, group_id):
    folder = Path(args.prepared) / f"group_{group_id:02d}"
    train_rows = load_prepared(folder / "train.jsonl")
    valid_rows = load_prepared(folder / "valid.jsonl")
    ood_rows = load_prepared(folder / "ood.jsonl")
    encode_rows(tokenizer, train_rows, args.max_length)
    encode_rows(tokenizer, valid_rows, args.max_length)
    encode_rows(tokenizer, ood_rows, args.max_length)
    meta = read_json(folder / "split_meta.json")
    destination = output_dir(args.output, group_id)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"输出目录非空：{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    train_tasks = list(meta["train_tasks"])
    valid_counts = defaultdict(int)
    for row in valid_rows:
        valid_counts[row["任务"]] += 1
    eligible = [task for task in train_tasks if valid_counts[task] >= args.min_dev_for_merge]
    log(f"第 {group_id} 组可参与合并的域内任务 {len(eligible)} 个，门槛 {args.min_dev_for_merge} 条验证样本")

    probes = {}
    probe_accuracy = {}
    for index, task in enumerate(eligible, 1):
        task_train = rows_of(train_rows, [task])
        task_valid = rows_of(valid_rows, [task])
        full, probe, _per_epoch = update_budget(len(task_train), args)
        log(f"第 {group_id} 组探针 {index}/{len(eligible)} {task}  全量 {full} 步，探针 {probe} 步")
        torch.manual_seed(args.seed + index)
        factors = fit_decision(
            model, tokenizer, task_train, None, args, torch, device, dtype, precision,
            probe, None, f"第 {group_id} 组探针 {task}",
        )
        probes[task] = factors
        predictions = predict_experts(
            model, tokenizer, task_valid, {"probe": residual_factors(factors, factors, "L0")[0]},
            {task: "probe"}, args, torch, device, dtype,
        )
        probe_accuracy[task] = task_accuracy(task_valid, predictions)[task]

    cosine = {}
    ordered = list(eligible)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            key = (left, right) if left < right else (right, left)
            cosine[key] = bundle_cosine(probes[left], probes[right])

    def drop_of(left, right):
        members = sorted(left | right)
        member_train = rows_of(train_rows, members)
        _full, probe, _per_epoch = update_budget(len(member_train), args)
        log(f"第 {group_id} 组试合并：{'、'.join(members)}")
        fit_decision(
            model, tokenizer, member_train, None, args, torch, device, dtype, precision,
            probe, None, f"第 {group_id} 组试合并",
        )
        factors = export_lora(model)
        applied = {"trial": residual_factors(factors, factors, "L0")[0]}
        drops = {}
        for task in members:
            task_valid = rows_of(valid_rows, [task])
            predictions = predict_experts(
                model, tokenizer, task_valid, applied, {task: "trial" for task in members}, args, torch, device, dtype,
            )
            drops[task] = probe_accuracy[task] - task_accuracy(task_valid, predictions)[task]
        return drops

    clusters = merge_tasks(eligible, cosine, blocked_pairs(eligible), drop_of, DROP_LIMIT) if eligible else []
    clusters.extend([[task] for task in train_tasks if task not in eligible])
    log("第 %s 组合并结果：%s" % (group_id, " | ".join("、".join(cluster) for cluster in clusters)))
    fitted = {}
    for index, members in enumerate(clusters):
        fitted.update(train_cluster(
            model, tokenizer, args, torch, device, dtype, precision, group_id,
            train_rows, valid_rows, members, probe_accuracy, index,
        ))
    covered = {task for payload in fitted.values() for task in payload["tasks"]}
    if covered != set(train_tasks):
        raise RuntimeError("有域内任务没有分到专家")

    train_texts, valid_texts = load_general_corpus(args.general_corpus)
    log(f"第 {group_id} 组通用语料 {args.general_corpus}：训练 {len(train_texts)} 条，早停 {len(valid_texts)} 条")
    torch.manual_seed(args.seed + 1)
    general, general_steps = fit_language(
        model, tokenizer, args, torch, device, dtype, precision, args.general_updates, train_texts, valid_texts,
    )
    measured = {}
    built = {}
    for mode in ("L0", "L1", "L2"):
        experts = {}
        ratios = []
        lambdas = {}
        for name, payload in fitted.items():
            packed, module_lambdas, ratio = residual_factors(payload["factors"], general, mode)
            experts[name] = packed
            lambdas[name] = module_lambdas
            if ratio is not None:
                ratios.append(ratio)
        task_to_expert = {task: name for name, payload in fitted.items() for task in payload["tasks"]}
        predictions = predict_experts(model, tokenizer, valid_rows, experts, task_to_expert, args, torch, device, dtype)
        prototypes = prototypes_of(experts)
        measured[mode] = {
            "accuracy": macro_of(valid_rows, predictions),
            "max_norm_ratio": max(ratios) if ratios else None,
            "prototype_cosine": cosine_of_prototypes(prototypes),
            "lambdas": {name: {module: float(value) for module, value in items.items()} for name, items in lambdas.items()},
        }
        built[mode] = (experts, prototypes)
        log(f"第 {group_id} 组 {mode} 域内宏平均准确率 {measured[mode]['accuracy']:.4f}")

    rung = "L1"
    if rollback_triggered(
        measured["L1"]["accuracy"], measured["L0"]["accuracy"], measured["L1"]["max_norm_ratio"],
        measured["L1"]["prototype_cosine"], measured["L0"]["prototype_cosine"], DROP_LIMIT,
    ):
        rung = "L2"
        if rollback_triggered(
            measured["L2"]["accuracy"], measured["L0"]["accuracy"], measured["L2"]["max_norm_ratio"],
            measured["L2"]["prototype_cosine"], measured["L0"]["prototype_cosine"], DROP_LIMIT,
        ):
            rung = "L0"
    experts, prototypes = built[rung]
    log(f"第 {group_id} 组采用相减档位 {rung}")
    manifest = {
        "task": TASK_NAME,
        "format_version": FORMAT_VERSION,
        "group": group_id,
        "train_tasks": train_tasks,
        "ood_tasks": list(meta["ood_tasks"]),
        "clusters": {name: payload["tasks"] for name, payload in fitted.items()},
        "rung": rung,
        "general_steps": general_steps,
        "general_corpus": str(Path(args.general_corpus).resolve()),
        "general_train_size": len(train_texts),
        "general_valid_size": len(valid_texts),
        "rung_detail": measured,
        "max_length": args.max_length,
        "seed": args.seed,
        "base_model": str(Path(args.model).resolve()),
    }
    write_json(destination / "method_config.json", manifest)
    save_factors(destination / "factors.npz", experts, prototypes)
    report = evaluate_saved(model, tokenizer, args, torch, device, dtype, valid_rows, ood_rows, experts, prototypes, manifest)
    write_json(destination / "eval.json", report)
    write_json(folder / "method_eval.json", report)
    print_report(f"第 {group_id} 组 域内验证（{report['adopted']}）", report["in_domain"])
    print_report(f"第 {group_id} 组 域外测试（{report['adopted']}）", report["ood"])
    return report


def selected_groups(raw_group):
    if raw_group is None:
        return [1, 2, 3, 4]
    if raw_group not in (1, 2, 3, 4):
        raise ValueError("group 只能是 1、2、3、4")
    return [raw_group]


def summary_of(reports):
    return [{
        "group": item["group"],
        "adopted": item["adopted"],
        "rung": item["rung"],
        "ood_tasks": item["ood_tasks"],
        "in_domain_macro_accuracy": item["in_domain"]["macro_accuracy"],
        "ood_macro_accuracy": item["ood"]["macro_accuracy"],
    } for item in reports]


def run_train(args):
    import json

    prepare_all(args.raw, args.groups, args.prepared, args.valid_ratio)
    model, tokenizer, torch, device, dtype, precision = load_model(args)
    reports = []
    for group_id in selected_groups(args.group):
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        reports.append(train_group(model, tokenizer, args, torch, device, dtype, precision, group_id))
    payload = summary_of(reports)
    write_json(Path(args.output) / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_evaluate(args):
    import json

    model, tokenizer, torch, device, dtype, _precision = load_model(args)
    reports = []
    for group_id in selected_groups(args.group):
        destination = output_dir(args.output, group_id)
        manifest = read_json(destination / "method_config.json")
        if manifest.get("task") != TASK_NAME:
            raise ValueError(f"{destination} 不是本脚本的结果")
        experts, prototypes = load_factors(destination / "factors.npz")
        folder = Path(args.prepared) / f"group_{group_id:02d}"
        report = evaluate_saved(
            model, tokenizer, args, torch, device, dtype,
            load_prepared(folder / "valid.jsonl"), load_prepared(folder / "ood.jsonl"),
            experts, prototypes, manifest,
        )
        write_json(destination / "eval.json", report)
        write_json(folder / "method_eval.json", report)
        print_report(f"第 {group_id} 组 域内验证（{report['adopted']}）", report["in_domain"])
        print_report(f"第 {group_id} 组 域外测试（{report['adopted']}）", report["ood"])
        reports.append(report)
    payload = summary_of(reports)
    write_json(Path(args.output) / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
