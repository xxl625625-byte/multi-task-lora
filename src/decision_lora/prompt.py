"""把口径卡和材料收成训练提示，并标出路由跨度。"""

from __future__ import annotations

from decision_lora.schema import TaskCard


def render_prompt(card: TaskCard, material: str) -> tuple[str, int, int]:
    goal = f"【判断目标】{card.goal}\n"
    rules = f"【判断规则】{card.rules}\n"
    body = f"【材料】{material}\n"
    options = " ".join(f"{code}{meaning}" for code, meaning in card.options)
    text = goal + rules + body + f"【选项】{options}\n"
    start = len(goal)
    end = len(goal + rules + body)
    if end <= start:
        raise ValueError(f"{card.name} 的规则和材料跨度为空")
    return text, start, end
