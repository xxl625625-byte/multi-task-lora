"""把判断题编成选项代号上的交叉熵，和 baseline 的前向一致。"""

import contextlib

from decision_lora.finetune.tasks import render_prompt


def runtime(device, precision):
    import torch

    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError("设备只支持 cpu 或 cuda[:编号]")
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请安装对应显卡的 PyTorch，或使用 --device cpu")
        torch.cuda.set_device(dev)
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if dev.type == "cuda" else "fp32"
    if dev.type == "cpu" and precision != "fp32":
        raise ValueError("CPU 请使用 fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("这块 GPU 不支持 BF16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    return torch, dev, dtype, precision


def autocast(torch, device, dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def load_tokenizer(path):
    from pathlib import Path

    from transformers import AutoTokenizer

    if not Path(path).is_dir():
        raise ValueError(f"目录不存在：{path}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer 没有 pad 或 eos")
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal(model_path, attention):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation=attention,
    )
    model.config.use_cache = False
    return model


def prompt_text(tokenizer, user_text):
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError("需要 transformers>=4.51，才能关闭 Qwen3 的思考模式") from exc


def continuation_id(tokenizer, prompt, code):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + code, add_special_tokens=False)
    if full[:len(prefix)] != prefix:
        raise ValueError(f"代号 {code!r} 和前文粘在了一起")
    extra = full[len(prefix):]
    if len(extra) != 1:
        shown = tokenizer.convert_ids_to_tokens(extra)
        raise ValueError(f"代号 {code!r} 被分成 {len(extra)} 个 token：{shown}")
    return prefix, extra[0]


def encode_row(tokenizer, row, max_length):
    prompt = prompt_text(tokenizer, render_prompt(row))
    prefix, token_ids = None, []
    for item in row["选项"]:
        ids, token_id = continuation_id(tokenizer, prompt, item["代号"])
        if prefix is None:
            prefix = ids
        elif ids != prefix:
            raise ValueError("同一题的选项切分不一致")
        token_ids.append(token_id)
    if len(prefix) > max_length:
        raise ValueError(f"输入含 {len(prefix)} tokens，超过 max_length={max_length}；不会截断")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("不同代号落到了同一个 token")
    codes = [item["代号"] for item in row["选项"]]
    return {
        "input_ids": prefix,
        "code_token_ids": token_ids,
        "target": [1.0 if code == row["答案"] else 0.0 for code in codes],
        "answer_index": codes.index(row["答案"]),
        "codes": codes,
        "names": [item["名称"] for item in row["选项"]],
        "任务": row["任务"],
    }


def encode_rows(tokenizer, rows, max_length):
    samples = []
    for number, row in enumerate(rows, 1):
        try:
            samples.append(encode_row(tokenizer, row, max_length))
        except ValueError as exc:
            raise ValueError(f"第 {number} 条（{row['任务']}）：{exc}") from exc
    return samples


def make_batch(torch, samples, pad_id, device):
    width = max(len(sample["input_ids"]) for sample in samples)
    choices = max(len(sample["code_token_ids"]) for sample in samples)
    batch = len(samples)
    input_ids = torch.full((batch, width), pad_id, dtype=torch.long)
    attention = torch.zeros((batch, width), dtype=torch.long)
    code_ids = torch.zeros((batch, choices), dtype=torch.long)
    code_mask = torch.zeros((batch, choices), dtype=torch.float32)
    target = torch.zeros((batch, choices), dtype=torch.float32)
    for index, sample in enumerate(samples):
        length = len(sample["input_ids"])
        input_ids[index, :length] = torch.tensor(sample["input_ids"])
        attention[index, :length] = 1
        count = len(sample["code_token_ids"])
        code_ids[index, :count] = torch.tensor(sample["code_token_ids"])
        code_mask[index, :count] = 1
        target[index, :count] = torch.tensor(sample["target"], dtype=torch.float32)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "code_token_ids": code_ids.to(device),
        "code_mask": code_mask.to(device),
        "target": target.to(device),
    }


def last_token_logits(torch, model, input_ids, attention_mask):
    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = causal.model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
    ).last_hidden_state
    last = attention_mask.sum(1) - 1
    picked = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
    return causal.lm_head(picked)


def option_log_probs(torch, token_logits, code_token_ids, code_mask):
    scores = token_logits.gather(1, code_token_ids).float()
    scores = scores.masked_fill(code_mask == 0, torch.finfo(scores.dtype).min)
    return torch.log_softmax(scores, dim=-1)


def decision_loss(torch, model, batch):
    token_logits = last_token_logits(torch, model, batch["input_ids"], batch["attention_mask"])
    log_probs = option_log_probs(torch, token_logits, batch["code_token_ids"], batch["code_mask"])
    if not torch.isfinite(log_probs).all():
        raise RuntimeError("选项概率出现非有限数值")
    return -(batch["target"] * log_probs).sum(-1).mean()


def predict_rows(model, tokenizer, rows, batch_size, torch, device, dtype, max_length):
    import math

    samples = encode_rows(tokenizer, rows, max_length)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            part = samples[start:start + batch_size]
            batch = make_batch(torch, part, tokenizer.pad_token_id, device)
            with autocast(torch, device, dtype):
                token_logits = last_token_logits(torch, model, batch["input_ids"], batch["attention_mask"])
            log_probs = option_log_probs(torch, token_logits, batch["code_token_ids"], batch["code_mask"])
            for row_index, sample in enumerate(part):
                count = len(sample["codes"])
                probs = [math.exp(value) for value in log_probs[row_index, :count].tolist()]
                chosen = max(range(count), key=lambda index: probs[index])
                predictions.append({
                    "任务": sample["任务"],
                    "代号": sample["codes"][chosen],
                    "名称": sample["names"][chosen],
                    "概率": {code: prob for code, prob in zip(sample["codes"], probs)},
                })
    return predictions
