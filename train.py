#!/usr/bin/env python3
"""微调入口。在本目录执行，或在 Notebook 里 %run。

  python train.py selftest
  python train.py run --model /path/Qwen3-0.6B --device cuda:1 --group 1 --batch-size 1 --max-length 8192 --epochs 3

业务 jsonl 放在 data/raw/。通用语料是 data/general/corpus.txt。分组是 configs/groups.json。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from decision_lora.finetune.cli import main

if __name__ == "__main__":
    main()
