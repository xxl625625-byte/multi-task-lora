"""整题路由。一道题只选一个专家，分差不够就走冻结基座。"""

import math
import re
from collections import defaultdict

import numpy as np

from decision_lora.finetune.batch import autocast, encode_row, make_batch, predict_rows, prompt_text
from decision_lora.finetune.factors import GAP_MIN, ROUTE_MARGIN
from decision_lora.finetune.lora_train import clear_active, set_active
from decision_lora.finetune.tasks import render_prompt

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def expert_score(parts):
    usable = []
    for part in parts:
        sigma1 = part["sigma1"]
        sigma2 = part["sigma2"]
        if sigma1 <= 0.0:
            gap = 0.0
        elif sigma2 <= 0.0:
            gap = math.inf
        else:
            gap = sigma1 / sigma2
        if gap < GAP_MIN:
            continue
        usable.append(sigma1 * part["cosine_abs"])
    if not usable:
        return None
    return sum(usable) / len(usable)


def route_weights(scores):
    if not scores:
        return {}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_score = ranked[0]
    if top_score == 0.0:
        return {}
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if len(ranked) > 1 and (top_score - second) / top_score < ROUTE_MARGIN:
        return {}
    return {top_name: 1.0}


def predict_experts(model, tokenizer, rows, experts, task_to_expert, args, torch, device, dtype):
    clear_active()
    grouped = defaultdict(list)
    for row in rows:
        grouped[task_to_expert[row["任务"]]].append(row)
    predictions = {}
    with model.disable_adapter():
        for name, part in grouped.items():
            set_active(torch, device, experts[name])
            batch = predict_rows(model, tokenizer, part, args.batch_size, torch, device, dtype, args.max_length)
            for row, prediction in zip(part, batch):
                predictions[id(row)] = prediction
    clear_active()
    return [predictions[id(row)] for row in rows]


def predict_base(model, tokenizer, rows, args, torch, device, dtype):
    clear_active()
    with model.disable_adapter():
        return predict_rows(model, tokenizer, rows, args.batch_size, torch, device, dtype, args.max_length)


def span_indexes(tokenizer, row, input_ids):
    prompt = prompt_text(tokenizer, render_prompt(row))
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("路由需要 fast tokenizer 的 offset mapping")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(input_ids):
        raise ValueError(f"{row['任务']} 的路由切分和训练切分不一致")
    char_start = prompt.index("【判断规则】")
    char_end = prompt.index("【选项】")
    indexes = []
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if start == end:
            continue
        if start < char_end and end > char_start:
            indexes.append(index)
    return indexes


def layer_of(name):
    found = _LAYER_RE.search(name)
    if found is None:
        raise ValueError(f"模块名里没有层号：{name}")
    return int(found.group(1))


def pool_hidden(hidden, indexes):
    picked = hidden[indexes].float()
    picked = picked / picked.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    mean = picked.mean(0)
    return (mean / mean.norm().clamp_min(1e-8)).detach().cpu().numpy()


def route_rows(model, tokenizer, rows, experts, prototypes, args, torch, device, dtype):
    clear_active()
    causal = model.get_base_model()
    chosen = []
    abstain = []
    with model.disable_adapter(), torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            part = rows[start:start + args.batch_size]
            samples = [encode_row(tokenizer, row, args.max_length) for row in part]
            spans = [span_indexes(tokenizer, row, sample["input_ids"]) for row, sample in zip(part, samples)]
            batch = make_batch(torch, samples, tokenizer.pad_token_id, device)
            with autocast(torch, device, dtype):
                output = causal.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
            for row_index, (row, span) in enumerate(zip(part, spans)):
                if not span:
                    abstain.append(row)
                    continue
                pooled = {}
                for name in prototypes[next(iter(prototypes))]:
                    layer = layer_of(name)
                    pooled[layer] = pool_hidden(output.hidden_states[layer][row_index], span)
                scores = {}
                for expert, modules in prototypes.items():
                    parts = []
                    for module, proto in modules.items():
                        if proto is None:
                            continue
                        hidden = pooled[layer_of(module)]
                        parts.append({
                            "sigma1": proto["sigma1"],
                            "sigma2": proto["sigma2"],
                            "cosine_abs": abs(float(np.dot(proto["v"], hidden))),
                        })
                    score = expert_score(parts)
                    if score is not None:
                        scores[expert] = score
                weights = route_weights(scores)
                winner = next((name for name, weight in weights.items() if weight == 1.0), None)
                if winner is None:
                    abstain.append(row)
                else:
                    chosen.append((row, winner))
    predictions = {}
    if abstain:
        for row, prediction in zip(abstain, predict_base(model, tokenizer, abstain, args, torch, device, dtype)):
            predictions[id(row)] = prediction
    grouped = defaultdict(list)
    for row, expert in chosen:
        grouped[expert].append(row)
    with model.disable_adapter():
        for expert, part in grouped.items():
            set_active(torch, device, experts[expert])
            batch = predict_rows(model, tokenizer, part, args.batch_size, torch, device, dtype, args.max_length)
            for row, prediction in zip(part, batch):
                predictions[id(row)] = prediction
    clear_active()
    return [predictions[id(row)] for row in rows], (len(abstain) / len(rows) if rows else None)
