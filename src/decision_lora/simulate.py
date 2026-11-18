"""构造模拟样本。这些记录只用来跑通流程，不能用来下业务结论。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from decision_lora.schema import TaskCard, task_cards


@dataclass
class Sample:
    case_id: str
    task: str
    material: str
    answer: str
    variant: str

    def to_json(self) -> dict:
        return asdict(self)


GENERAL_CORPUS = [
    "长江是亚洲最长的河流，发源于青藏高原。",
    "光合作用把光能转成化学能，发生在植物的叶绿体中。",
    "人民币是中华人民共和国的法定货币。",
    "地球绕太阳公转一周大约是一年。",
    "汉字是表意文字，一个字常常对应一个音节。",
    "水在标准大气压下于零度结冰，一百度沸腾。",
]


_RAW: dict[str, list[tuple[str, str, str]]] = {
    "大宗商品判别": [
        ("bulk-1", "商品名称：螺纹钢", "A"),
        ("bulk-2", "商品名称：青云手机", "B"),
    ],
    "是否烟草许可": [
        ("tob-1", "企业名称：海淀商贸\n经营范围：卷烟零售", "A"),
        ("tob-2", "企业名称：新疆邸一餐饮管理有限公司\n经营范围：餐饮服务", "B"),
    ],
    "是否道路运输许可": [
        ("road-1", "企业名称：临港五金\n经营范围：道路货物运输", "A"),
        ("road-2", "企业名称：新疆邸一餐饮管理有限公司\n经营范围：餐饮服务", "B"),
    ],
    "是否食品许可": [
        ("food-1", "企业名称：新疆邸一餐饮管理有限公司\n经营范围：餐饮服务", "A"),
        ("food-2", "企业名称：临港五金\n经营范围：金属材料销售", "B"),
    ],
    "是否建筑市场资质": [
        ("build-1", "企业名称：华北粮油\n经营范围：房屋建筑工程施工", "A"),
        ("build-2", "企业名称：临港五金\n经营范围：金属材料销售", "B"),
    ],
    "是否商务特许许可": [
        ("fran-1", "企业名称：青云手机\n经营范围：商业特许经营", "A"),
        ("fran-2", "企业名称：新疆邸一餐饮管理有限公司\n经营范围：餐饮服务", "B"),
    ],
    "是否有排污许可": [
        ("pol-1", "企业名称：临港五金\n经营范围：金属表面处理", "A"),
        ("pol-2", "企业名称：新疆邸一餐饮管理有限公司\n经营范围：餐饮服务", "B"),
    ],
    "批发商身份识别": [
        ("whole-1", "企业名称：华北粮油\n经营范围：谷物批发", "A"),
        ("whole-2", "企业名称：武汉市江夏区水之缘茶饮店\n经营范围：甜品、饮料制售", "B"),
    ],
    "地址核验": [
        ("addr-1", "地址一：北京市海淀区中关村大街1号\n地址二：北京海淀区中关村大街1号", "G"),
        ("addr-2", "地址一：北京市海淀区中关村大街1号\n地址二：上海市浦东新区世纪大道8号", "A"),
    ],
    "店企核验": [
        (
            "store-1",
            "店铺信息：{\"dp_name\": \"海淀商贸\", \"tx_industry\": \"五金\"}\n"
            "工商信息：{\"gs_name\": \"海淀商贸有限公司\", \"opscope\": \"五金零售\", \"gs_industry\": \"零售\"}",
            "A",
        ),
        (
            "store-2",
            "店铺信息：{\"dp_name\": \"海淀商贸\", \"tx_industry\": \"五金\"}\n"
            "工商信息：{\"gs_name\": \"新疆邸一餐饮管理有限公司\", \"opscope\": \"餐饮服务\", \"gs_industry\": \"餐饮\"}",
            "B",
        ),
    ],
    "品牌名和店铺名一致性判别": [
        ("brand-1", "店铺名称：HUAWEI（授权店）\n品牌名称：华为HUAWEI", "A"),
        ("brand-2", "店铺名称：水之缘茶饮\n品牌名称：华为HUAWEI", "B"),
    ],
}

_NAME_SWAPS = {
    "海淀商贸": "景和商行",
    "临港五金": "东岸器材",
    "青云手机": "白露数码",
    "青云": "白露",
    "华北粮油": "南埔米业",
}


def _blank_material(card: TaskCard, material: str) -> str | None:
    prefix = card.decisive_prefix
    if prefix is None or prefix not in material or card.insufficient is None:
        return None
    lines = []
    changed = False
    for line in material.split("\n"):
        if line.startswith(prefix):
            lines.append(prefix + "（缺失）")
            changed = True
        else:
            lines.append(line)
    if not changed:
        return None
    return "\n".join(lines)


def _swap_material(material: str) -> str | None:
    swapped = material
    hit = False
    for source, target in _NAME_SWAPS.items():
        if source in swapped:
            swapped = swapped.replace(source, target)
            hit = True
    return swapped if hit else None


def build_samples(cards: dict[str, TaskCard] | None = None) -> list[Sample]:
    cards = cards or task_cards()
    samples: list[Sample] = []
    for name, rows in _RAW.items():
        card = cards[name]
        for index, (case_id, material, answer) in enumerate(rows):
            samples.append(Sample(case_id, name, material, answer, "original"))
            if index == 0:
                swapped = _swap_material(material)
                if swapped is not None:
                    samples.append(Sample(case_id, name, swapped, answer, "name_swap"))
            if index == 1:
                blank = _blank_material(card, material)
                if blank is not None:
                    samples.append(
                        Sample(case_id, name, blank, card.insufficient or answer, "field_blank")
                    )
    return samples


def write_jsonl(path: Path, samples: list[Sample] | None = None) -> Path:
    rows = samples or build_samples()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
    return path
