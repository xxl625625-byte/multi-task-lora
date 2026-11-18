"""命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from decision_lora.experiment import load_and_compare
from decision_lora.pipeline import Pipeline, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="多任务决策 LoRA 流水线")
    parser.add_argument("command", choices=["run", "compare"])
    parser.add_argument("--config", default="configs/demo.json")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config
    if args.command == "compare":
        load_and_compare(config_path, root)
        return
    Pipeline(load_config(config_path), root).run()


if __name__ == "__main__":
    main()
