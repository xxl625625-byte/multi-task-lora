"""训练专用 LoRA 和通用语料 LoRA。采样按任务等概率。"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.optim import AdamW

from decision_lora.model import TinyCausal, TrainableLoRA, restricted_cross_entropy
from decision_lora.prompt import render_prompt
from decision_lora.schema import TaskCard
from decision_lora.simulate import Sample
from decision_lora.tokenizer import CharTokenizer


def probe_steps(steps_t: int) -> int:
    return max(1, min(500, steps_t // 5))


class Trainer:
    def __init__(
        self,
        model: TinyCausal,
        tokenizer: CharTokenizer,
        cards: dict[str, TaskCard],
        rank: int,
        alpha: int,
        lr: float,
        seed: int,
        max_length: int,
        target_modules: list[str],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cards = cards
        self.rank = rank
        self.alpha = alpha
        self.lr = lr
        self.seed = seed
        self.max_length = max_length
        self.target_modules = target_modules

    def _new_lora(self) -> TrainableLoRA:
        return TrainableLoRA(self.model.target_specs(self.target_modules), self.rank, self.alpha)

    def _encode_decision(self, sample: Sample) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
        card = self.cards[sample.task]
        text, start, end = render_prompt(card, sample.material)
        ids = self.tokenizer.encode(text)
        if len(ids) > self.max_length:
            raise ValueError(f"{sample.case_id} 超过长度上限 {self.max_length}")
        if not 0 <= start < end <= len(ids):
            raise ValueError(f"{sample.case_id} 的路由跨度无效")
        candidate_ids = [self.tokenizer.token_id(code) for code, _meaning in card.options]
        gold = self.tokenizer.token_id(sample.answer)
        tokens = torch.tensor(ids, dtype=torch.long)
        span = torch.arange(start, end, dtype=torch.long)
        return tokens, span, candidate_ids, gold

    def train_decision(self, samples: list[Sample], steps: int) -> dict[str, dict[str, torch.Tensor]]:
        if not samples:
            raise ValueError("没有训练样本")
        by_task: dict[str, list[Sample]] = {}
        for sample in samples:
            by_task.setdefault(sample.task, []).append(sample)
        tasks = sorted(by_task)
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        lora = self._new_lora()
        optimizer = AdamW(lora.parameters(), lr=self.lr)
        self.model.eval()
        lora.train()
        for _step in range(steps):
            task = tasks[int(rng.integers(len(tasks)))]
            bucket = by_task[task]
            sample = bucket[int(rng.integers(len(bucket)))]
            tokens, _span, candidate_ids, gold = self._encode_decision(sample)
            logits, _captured = self.model(tokens, factors=lora)
            loss = restricted_cross_entropy(logits[-1], candidate_ids, gold)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return lora.export()

    def train_language_model(self, texts: list[str], steps: int, valid_texts: list[str]) -> tuple[dict, int]:
        rng = np.random.default_rng(self.seed + 1)
        torch.manual_seed(self.seed + 1)
        lora = self._new_lora()
        optimizer = AdamW(lora.parameters(), lr=self.lr)
        previous = math.inf
        used = 0
        for step in range(steps):
            text = texts[int(rng.integers(len(texts)))]
            ids = self.tokenizer.encode(text)
            if len(ids) < 2:
                continue
            tokens = torch.tensor(ids, dtype=torch.long)
            logits, _captured = self.model(tokens[:-1], factors=lora)
            target = tokens[1:]
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            used = step + 1
            if valid_texts and used % 2 == 0:
                perplexity = self._perplexity(lora, valid_texts)
                if perplexity > previous:
                    break
                previous = perplexity
        return lora.export(), used

    def _perplexity(self, lora: TrainableLoRA, texts: list[str]) -> float:
        total = 0.0
        count = 0
        with torch.no_grad():
            for text in texts:
                ids = self.tokenizer.encode(text)
                if len(ids) < 2:
                    continue
                tokens = torch.tensor(ids, dtype=torch.long)
                logits, _captured = self.model(tokens[:-1], factors=lora)
                loss = torch.nn.functional.cross_entropy(logits, tokens[1:])
                total += float(loss) * (len(ids) - 1)
                count += len(ids) - 1
        if count == 0:
            return math.inf
        return math.exp(total / count)

    @torch.no_grad()
    def predict(
        self,
        sample: Sample,
        factors: dict[str, dict[str, torch.Tensor]] | None = None,
        dense: dict[str, torch.Tensor] | None = None,
        scale: float | None = None,
    ) -> tuple[str, dict[str, torch.Tensor]]:
        tokens, span, candidate_ids, _gold = self._encode_decision(sample)
        dense_deltas = None
        factor_module = None
        if factors is not None:
            factor_module = self._new_lora()
            for name, pair in factors.items():
                factor_module.factors[name].A.copy_(pair["A"])
                factor_module.factors[name].B.copy_(pair["B"])
        if dense is not None:
            dense_deltas = dense
        elif factors is not None and scale is not None and dense is None:
            dense_deltas = None
        logits, captured = self.model(tokens, span=span, factors=factor_module, dense=dense_deltas)
        gathered = logits[-1, candidate_ids]
        choice = int(torch.argmax(gathered))
        code = self.cards[sample.task].options[choice][0]
        return code, {name: value.cpu() for name, value in captured.items()}

    def predict_with_matrices(self, sample: Sample, matrices: dict[str, torch.Tensor] | None) -> tuple[str, dict[str, torch.Tensor]]:
        tokens, span, candidate_ids, _gold = self._encode_decision(sample)
        with torch.no_grad():
            logits, captured = self.model(tokens, span=span, dense=matrices)
        choice = int(torch.argmax(logits[-1, candidate_ids]))
        code = self.cards[sample.task].options[choice][0]
        return code, {name: value.cpu() for name, value in captured.items()}
