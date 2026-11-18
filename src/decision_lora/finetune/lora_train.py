"""在同一份基座上反复重置 LoRA，分别训探针、专家和通用语料。"""

import math
import random

from decision_lora.finetune.batch import autocast, decision_loss, encode_row, load_causal, load_tokenizer, make_batch, predict_rows, runtime
from decision_lora.finetune.data import log
from decision_lora.finetune.merge import equal_rows
from decision_lora.finetune.score import validation_nll

ACTIVE = {}


def reset_lora(model, torch):
    for module in model.modules():
        if not hasattr(module, "lora_A"):
            continue
        for layer in module.lora_A.values():
            torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        for layer in module.lora_B.values():
            torch.nn.init.zeros_(layer.weight)


def export_lora(model):
    factors = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        factors[name] = {
            "A": module.lora_A["default"].weight.detach().float().cpu().numpy().copy(),
            "B": module.lora_B["default"].weight.detach().float().cpu().numpy().copy(),
            "scale": float(module.scaling["default"]),
        }
    if not factors:
        raise RuntimeError("没有导出到 LoRA 参数")
    return factors


def clone_lora(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_lora(model, state):
    import torch

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name].to(parameter.device))


def install_hooks(model, torch):
    for name, module in model.named_modules():
        if not hasattr(module, "base_layer"):
            continue

        def make(module_name):
            def hook(_module, inputs, output):
                pair = ACTIVE.get(module_name)
                if pair is None:
                    return output
                matrix_a, matrix_b = pair
                mixed = torch.nn.functional.linear(torch.nn.functional.linear(inputs[0], matrix_a), matrix_b)
                return output + mixed.to(dtype=output.dtype)

            return hook

        module.base_layer.register_forward_hook(make(name))


def clear_active():
    ACTIVE.clear()


def set_active(torch, device, factors):
    clear_active()
    if not factors:
        return
    for name, (matrix_a, matrix_b) in factors.items():
        ACTIVE[name] = (torch.tensor(matrix_a, device=device), torch.tensor(matrix_b, device=device))


def _optimizer(model, args, torch, max_updates):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    warmup = max(1, int(max_updates * 0.1))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (max_updates - step) / max(1, max_updates - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    return parameters, optimizer, scheduler


def _scaler(torch, device, precision):
    enabled = device.type == "cuda" and precision == "fp16"
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def fit_decision(model, tokenizer, train_rows, valid_rows, args, torch, device, dtype, precision, max_updates, eval_every, tag):
    clear_active()
    reset_lora(model, torch)
    parameters, optimizer, scheduler = _optimizer(model, args, torch, max_updates)
    scaler = _scaler(torch, device, precision)
    stream = equal_rows(train_rows, random.Random(args.seed), max_updates * args.batch_size * args.accumulation)
    cursor = 0
    updates = 0
    best_state = None
    best_nll = float("inf")
    model.train()
    while updates < max_updates:
        group = stream[cursor:cursor + args.batch_size * args.accumulation]
        cursor += len(group)
        if not group:
            break
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(group), args.batch_size):
            part = group[start:start + args.batch_size]
            samples = [encode_row(tokenizer, row, args.max_length) for row in part]
            batch = make_batch(torch, samples, tokenizer.pad_token_id, device)
            with autocast(torch, device, dtype):
                loss = decision_loss(torch, model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("训练 loss 非有限数；可改用 --precision fp32")
            total_loss += float(loss.detach()) * len(part)
            seen += len(part)
            scaler.scale(loss * (len(part) / len(group))).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm) and not scaler.is_enabled():
            raise RuntimeError("梯度非有限数，请改用 --precision fp32")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
            updates += 1
        else:
            log(f"{tag} FP16 梯度溢出，本组已跳过")
            continue
        if updates == max_updates or updates % args.log_every == 0:
            log(f"{tag} updates={updates}/{max_updates} loss={total_loss / seen:.5f}")
        if valid_rows and eval_every and (updates % eval_every == 0 or updates == max_updates):
            predictions = predict_rows(model, tokenizer, valid_rows, args.batch_size, torch, device, dtype, args.max_length)
            nll = validation_nll(valid_rows, predictions)
            log(f"{tag} 验证对数损失 {nll:.5f}")
            if nll < best_nll:
                best_nll = nll
                best_state = clone_lora(model)
            model.train()
    if updates == 0:
        raise RuntimeError(f"{tag} 没有成功的参数更新")
    if best_state is not None:
        restore_lora(model, best_state)
    return export_lora(model)


def language_perplexity(model, tokenizer, texts, torch, device, dtype):
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                continue
            tokens = torch.tensor([ids], dtype=torch.long, device=device)
            with autocast(torch, device, dtype):
                loss = model(tokens, labels=tokens).loss
            total += float(loss) * (len(ids) - 1)
            count += len(ids) - 1
    model.train()
    if count == 0:
        return float("inf")
    return math.exp(total / count)


def fit_language(model, tokenizer, args, torch, device, dtype, precision, max_updates, train_texts, valid_texts):
    clear_active()
    reset_lora(model, torch)
    parameters, optimizer, scheduler = _optimizer(model, args, torch, max_updates)
    scaler = _scaler(torch, device, precision)
    rng = random.Random(args.seed + 1)
    best_state = clone_lora(model)
    previous = float("inf")
    used = 0
    model.train()
    for step in range(max_updates):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.accumulation):
            text = train_texts[rng.randrange(len(train_texts))]
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                continue
            tokens = torch.tensor([ids], dtype=torch.long, device=device)
            with autocast(torch, device, dtype):
                loss = model(tokens, labels=tokens).loss
            if not torch.isfinite(loss):
                raise RuntimeError("通用语料 loss 非有限数")
            losses.append(loss)
            scaler.scale(loss / args.accumulation).backward()
        if not losses:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            log("通用语料 FP16 梯度溢出，本步已跳过")
            continue
        scheduler.step()
        used = step + 1
        if used % 2 == 0:
            perplexity = language_perplexity(model, tokenizer, valid_texts, torch, device, dtype)
            log(f"通用语料 updates={used}/{max_updates} 验证困惑度 {perplexity:.4f}")
            if perplexity > previous:
                restore_lora(model, best_state)
                log("通用语料验证困惑度上升，停在上一档")
                break
            previous = perplexity
            best_state = clone_lora(model)
            model.train()
    if used == 0:
        raise RuntimeError("通用语料没有成功的参数更新")
    if previous == float("inf"):
        best_state = clone_lora(model)
    restore_lora(model, best_state)
    return export_lora(model), used


def load_model(args):
    torch, device, dtype, precision = runtime(args.device, args.precision)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    attention = ("sdpa" if device.type == "cuda" else "eager") if args.attention == "auto" else args.attention
    tokenizer = load_tokenizer(args.model)
    base = load_causal(args.model, attention)
    base.config.pad_token_id = tokenizer.pad_token_id
    if args.max_length > base.config.max_position_embeddings:
        raise ValueError("max_length 超过模型的位置上限")
    from peft import LoraConfig, TaskType, get_peft_model

    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )).to(device)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    install_hooks(model, torch)
    return model, tokenizer, torch, device, dtype, precision
