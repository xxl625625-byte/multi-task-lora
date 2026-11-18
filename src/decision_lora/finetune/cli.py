"""参数解析。真正的训练入口是仓库根目录的 train.py。"""

import argparse
import math
from pathlib import Path

import numpy as np

from decision_lora.finetune.corpus import load_general_corpus
from decision_lora.finetune.data import SEED, VALID_RATIO, load_groups, log
from decision_lora.finetune.experiment import run_evaluate, run_train
from decision_lora.finetune.factors import low_rank_usv, residual_factors
from decision_lora.finetune.merge import DROP_LIMIT, cannot_merge, merge_tasks
from decision_lora.finetune.route import route_weights

ROOT = Path(__file__).resolve().parents[3]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def selftest():
    groups = load_groups(ROOT / "configs" / "groups.json")
    if [group["group"] for group in groups] != [1, 2, 3, 4]:
        raise AssertionError("分组不是 1 到 4")
    if any(len(group["train_tasks"]) != 8 or len(group["ood_tasks"]) != 3 for group in groups):
        raise AssertionError("分组不是 8+3")
    if not cannot_merge("地址核验", "批发商身份识别"):
        raise AssertionError("地址和批发商不应合并")
    if not cannot_merge("店企核验", "批发商身份识别"):
        raise AssertionError("宽口径和严口径不应合并")
    if cannot_merge("是否烟草许可", "是否食品许可"):
        raise AssertionError("两类许可应该允许尝试合并")
    tried = []

    def drop_of(left, right):
        tried.append(tuple(sorted(left | right)))
        if "甲" in left | right and "乙" in left | right and len(left | right) == 2:
            return {name: 0.0 for name in left | right}
        return {name: 0.05 for name in left | right}

    merged = merge_tasks(
        ["甲", "乙", "丙"],
        {("乙", "甲"): 0.9, ("丙", "乙"): 0.4, ("丙", "甲"): 0.3},
        set(),
        drop_of,
        DROP_LIMIT,
    )
    if len(tried) != 2:
        raise AssertionError("第一次合并失败后不应再试")
    if sorted(len(group) for group in merged) != [1, 2]:
        raise AssertionError("合并结果不对")
    orthogonal_expert = {"m": {"A": np.array([[1.0, 0.0]]), "B": np.array([[3.0], [0.0]]), "scale": 1.0}}
    orthogonal_general = {"m": {"A": np.array([[0.0, 1.0]]), "B": np.array([[0.0], [2.0]]), "scale": 1.0}}
    packed, _lambdas, _ratio = residual_factors(orthogonal_expert, orthogonal_general, "L1")
    if not np.allclose(packed["m"][1] @ packed["m"][0], np.array([[3.0, 0.0], [0.0, 0.0]])):
        raise AssertionError("相减后的低秩因子和稠密残差不一致")
    matrix_b = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    matrix_a = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    left, values, right = low_rank_usv(matrix_b, matrix_a)
    dense = matrix_b @ matrix_a
    _u, expected, _vt = np.linalg.svd(dense, full_matrices=False)
    if not np.allclose(values, expected[:values.size]):
        raise AssertionError("低秩奇异值和稠密奇异值不一致")
    if not np.allclose(dense @ right[:, 0], values[0] * left[:, 0]):
        raise AssertionError("右奇异向量不正确")
    if route_weights({"E0": 1.0, "E1": 0.95}):
        raise AssertionError("分差不够时应弃权")
    if route_weights({"E0": 2.0, "E1": 0.2}) != {"E0": 1.0}:
        raise AssertionError("分差够时应只选第一名")
    train_texts, valid_texts = load_general_corpus(ROOT / "data" / "general" / "corpus.txt")
    if len(valid_texts) < 2 or len(train_texts) < 2:
        raise AssertionError("通用语料切分不正确")
    log("selftest 通过")


def main(argv=None):
    parser = argparse.ArgumentParser(description="十一任务决策 LoRA 微调入口")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("selftest")
    for name in ("run", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", required=True)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
        sub.add_argument("--attention", choices=("auto", "sdpa", "eager"), default="auto")
        sub.add_argument("--threads", type=positive_int, default=4)
        sub.add_argument("--batch-size", type=positive_int, default=1)
        sub.add_argument("--prepared", default=str(ROOT / "data" / "groups"))
        sub.add_argument("--output", default=str(ROOT / "outputs_method"))
        sub.add_argument("--group", type=int)
        sub.add_argument("--max-length", type=positive_int, default=2048)
        sub.add_argument("--no-checkpointing", action="store_true")
        if name == "run":
            sub.add_argument("--raw", default=str(ROOT / "data" / "raw"))
            sub.add_argument("--groups", default=str(ROOT / "configs" / "groups.json"))
            sub.add_argument("--valid-ratio", type=float, default=VALID_RATIO)
            sub.add_argument("--epochs", type=positive_int, default=3)
            sub.add_argument("--accumulation", type=positive_int, default=8)
            sub.add_argument("--rank", type=positive_int, default=8)
            sub.add_argument("--lr", type=float, default=1e-4)
            sub.add_argument("--seed", type=int, default=SEED)
            sub.add_argument("--log-every", type=positive_int, default=10)
            sub.add_argument("--min-dev-for-merge", type=positive_int, default=200)
            sub.add_argument("--general-updates", type=positive_int, default=200)
            sub.add_argument("--general-corpus", default=str(ROOT / "data" / "general" / "corpus.txt"))
    args = parser.parse_args(argv)
    if args.command == "selftest":
        selftest()
        return
    if args.command == "run":
        if not math.isfinite(args.lr) or args.lr <= 0:
            parser.error("lr 必须是有限正数")
        run_train(args)
        return
    run_evaluate(args)
