"""和 baseline 相同的准确率、召回率、AUC。"""

import math
from collections import defaultdict

from decision_lora.finetune.tasks import TASK_ORDER


def roc_auc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average = (index + 1 + end + 1) / 2
        for cursor in range(index, end + 1):
            if pairs[cursor][1] == 1:
                rank_sum += average
        index = end + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def binary_metrics(golds, preds, scores):
    tp = sum(gold == "A" and pred == "A" for gold, pred in zip(golds, preds))
    fp = sum(gold != "A" and pred == "A" for gold, pred in zip(golds, preds))
    fn = sum(gold == "A" and pred != "A" for gold, pred in zip(golds, preds))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "accuracy": sum(gold == pred for gold, pred in zip(golds, preds)) / len(golds),
        "precision": precision,
        "recall": recall,
        "auc": roc_auc(scores, [int(gold == "A") for gold in golds]),
        "positive": "A",
    }


def graded_metrics(golds, preds, probabilities, codes):
    per_class = {}
    aucs = []
    for code in codes:
        support = sum(gold == code for gold in golds)
        tp = sum(gold == code and pred == code for gold, pred in zip(golds, preds))
        fp = sum(gold != code and pred == code for gold, pred in zip(golds, preds))
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else (0.0 if support else None)
        recall = tp / (tp + fn) if support else None
        per_class[code] = {"support": support, "precision": precision, "recall": recall}
        if support and support < len(golds):
            auc = roc_auc([probs[code] for probs in probabilities], [int(gold == code) for gold in golds])
            if auc is not None:
                aucs.append(auc)
    present = [item for item in per_class.values() if item["support"]]
    return {
        "accuracy": sum(gold == pred for gold, pred in zip(golds, preds)) / len(golds),
        "macro_precision": sum(item["precision"] for item in present) / len(present),
        "macro_recall": sum(item["recall"] for item in present) / len(present),
        "macro_ovr_auc": sum(aucs) / len(aucs) if aucs else None,
        "per_class": per_class,
    }


def score_task(rows, predictions):
    golds = [row["答案"] for row in rows]
    preds = [item["代号"] for item in predictions]
    if rows[0]["题型"] == "是非题":
        metrics = binary_metrics(golds, preds, [item["概率"]["A"] for item in predictions])
    else:
        codes = [item["代号"] for item in rows[0]["选项"]]
        metrics = graded_metrics(golds, preds, [item["概率"] for item in predictions], codes)
    metrics["n"] = len(rows)
    metrics["题型"] = rows[0]["题型"]
    return metrics


def validation_nll(rows, predictions):
    total = 0.0
    for row, prediction in zip(rows, predictions):
        probability = min(max(prediction["概率"][row["答案"]], 1e-12), 1.0)
        total += -math.log(probability)
    return total / len(rows)


def pack_report(rows, predictions):
    by_task = defaultdict(lambda: {"rows": [], "predictions": []})
    for row, prediction in zip(rows, predictions):
        bucket = by_task[row["任务"]]
        bucket["rows"].append(row)
        bucket["predictions"].append(prediction)
    per_task = {name: score_task(bucket["rows"], bucket["predictions"]) for name, bucket in by_task.items()}
    ordered = [per_task[name] for name in TASK_ORDER if name in per_task]
    return {
        "n": len(rows),
        "micro_accuracy": sum(row["答案"] == prediction["代号"] for row, prediction in zip(rows, predictions)) / len(rows),
        "macro_accuracy": sum(item["accuracy"] for item in ordered) / len(ordered),
        "per_task": per_task,
    }


def fmt_metric(value):
    if value is None:
        return "—"
    return f"{value:.4f}"


def print_report(title, report):
    print(
        f"{title}  n={report['n']}  微平均准确率={fmt_metric(report['micro_accuracy'])}  "
        f"宏平均准确率={fmt_metric(report['macro_accuracy'])}"
    )
    for name in TASK_ORDER:
        item = report["per_task"].get(name)
        if item is None:
            continue
        if item["题型"] == "是非题":
            detail = f"P={fmt_metric(item['precision'])} R={fmt_metric(item['recall'])} AUC={fmt_metric(item['auc'])}"
        else:
            detail = (
                f"宏P={fmt_metric(item['macro_precision'])} 宏R={fmt_metric(item['macro_recall'])} "
                f"宏AUC={fmt_metric(item['macro_ovr_auc'])}"
            )
        print(f"  {name}  n={item['n']}  acc={fmt_metric(item['accuracy'])}  {detail}")
