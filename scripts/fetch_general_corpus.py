#!/usr/bin/env python3
"""从中文维基百科的官方接口抽取导语，写成通用语料。

不用爬页面。接口是 https://zh.wikipedia.org/w/api.php ，正文许可为 CC BY-SA 4.0。
在有外网的机器上执行，然后把 data/general/corpus.txt 拷到内网。
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://zh.wikipedia.org/w/api.php"
USER_AGENT = "decision-lora-corpus/0.1 (local general-corpus sample; educational)"
TARGET = 2000
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "general" / "corpus.txt"
BLOCKED = (
    "经营范围",
    "烟草许可",
    "道路运输许可",
    "食品许可",
    "建筑市场资质",
    "商务特许",
    "排污许可",
    "店铺名称",
    "批发商",
)


def fetch(params):
    url = API + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def clip(text):
    if len(text) <= 500:
        return text
    cut = text.rfind("。", 0, 500)
    if cut >= 80:
        return text[:cut + 1]
    return text[:500]


def main():
    seen = set()
    lines = []
    while len(lines) < TARGET:
        try:
            payload = fetch({
                "action": "query",
                "format": "json",
                "generator": "random",
                "grnnamespace": "0",
                "grnlimit": "20",
                "prop": "extracts",
                "explaintext": "1",
                "exintro": "1",
            })
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(10)
                continue
            raise
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            page_id = page.get("pageid")
            if page_id in seen:
                continue
            seen.add(page_id)
            text = " ".join((page.get("extract") or "").split())
            if len(text) < 80 or any(word in text for word in BLOCKED):
                continue
            lines.append(clip(text))
            if len(lines) >= TARGET:
                break
        print(f"已收集 {len(lines)} 条", flush=True)
        time.sleep(0.5)
    header = (
        "# 来源：中文维基百科随机条目导语，经 MediaWiki API 获取。\n"
        "# 许可：CC BY-SA 4.0，https://creativecommons.org/licenses/by-sa/4.0/\n"
        "# 署名：Wikipedia contributors，https://zh.wikipedia.org/\n"
        "# 一行一条。空行和 # 开头的行不参与训练。\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {OUTPUT} ，共 {len(lines)} 条")


if __name__ == "__main__":
    main()
