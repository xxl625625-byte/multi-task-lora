"""通用语料：一行一条，和判断任务分开存放。"""

from pathlib import Path


def load_general_corpus(path):
    """末尾约 10% 做早停，至少 2 条，并给训练至少留 2 条。"""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"找不到通用语料：{path}")
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        lines.append(text)
    if len(lines) < 4:
        raise ValueError(f"{path} 至少要 4 条通用文本，现在是 {len(lines)} 条")
    holdout = min(max(2, len(lines) // 10), len(lines) - 2)
    return lines[:-holdout], lines[-holdout:]
