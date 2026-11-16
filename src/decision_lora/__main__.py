"""命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from decision_lora.pipeline import Pipeline, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="多任务决策 LoRA 流水线")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--config", default="configs/demo.json")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / args.config)
    Pipeline(config, root).run()


if __name__ == "__main__":
    main()
