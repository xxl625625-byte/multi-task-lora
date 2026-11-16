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
    "地址核验": [
        (
            "addr-1",
            "店铺名：海淀商贸\n地址一：北京市海淀区中关村大街1号\n地址二：北京海淀区中关村大街1号",
            "B",
        ),
        (
            "addr-2",
            "店铺名：临港五金\n地址一：北京市海淀区中关村大街1号\n地址二：上海市浦东新区世纪大道8号",
            "A",
        ),
    ],
    "品牌名写法": [
        ("brand-1", "店铺名：青云手机\n名称一：青云手机\n名称二：青云 手机", "B"),
        ("brand-2", "店铺名：青云手机\n名称一：青云手机\n名称二：南山家电", "A"),
    ],
    "企业是否存在交集": [
        ("link-1", "企业甲：海淀商贸\n企业乙：临港五金\n关系证据：股东同为赵某", "B"),
        ("link-2", "企业甲：海淀商贸\n企业乙：临港五金\n关系证据：仅名称不同，无股东或地址记录", "A"),
    ],
    "店企核验": [
        ("store-1", "店铺名：海淀商贸\n执照主体：海淀商贸有限公司\n执照主体：海淀商贸有限公司", "B"),
        ("store-2", "店铺名：海淀商贸\n执照主体：临港五金厂", "A"),
    ],
    "品牌名授权": [
        ("auth-1", "店铺名：青云手机\n品牌：青云\n授权材料：品牌方授权书编号 QY-1", "B"),
        ("auth-2", "店铺名：青云手机\n品牌：青云\n授权材料：无", "A"),
    ],
    "大宗商品判别": [
        ("bulk-1", "商品名称：螺纹钢", "B"),
        ("bulk-2", "商品名称：青云手机", "A"),
    ],
    "敏感品名识别": [
        ("sens-1", "品名：盐酸", "B"),
        ("sens-2", "品名：螺纹钢", "A"),
    ],
    "中介机构识别": [
        ("agent-1", "企业名：海淀商贸\n经营证据：经营范围含商务代理", "B"),
        ("agent-2", "企业名：临港五金\n经营证据：经营范围只有金属零售", "A"),
    ],
    "批发商身份识别": [
        ("whole-1", "企业名：华北粮油\n经营范围：谷物批发", "B"),
        ("whole-2", "企业名：临港五金\n经营范围：五金零售", "A"),
    ],
    "是否需要许可": [
        ("lic-1", "经营活动：危险化学品经营", "B"),
        ("lic-2", "经营活动：日用百货零售", "A"),
    ],
    "经营范围覆盖": [
        ("scope-1", "经营范围：金属材料销售\n销项商品：螺纹钢", "B"),
        ("scope-2", "经营范围：金属材料销售\n销项商品：青云手机", "A"),
    ],
    "商品是否一致": [
        ("item-1", "店铺名：青云手机\n记录一：青云手机 128G\n记录二：青云 手机 128G", "B"),
        ("item-2", "店铺名：青云手机\n记录一：青云手机 128G\n记录二：青云手机 256G", "A"),
    ],
    "电销用户意图识别": [
        ("call-1", "坐席：考虑办一张卡吗？\n客户：利率多少，下周可以签。", "B"),
        ("call-2", "坐席：考虑办一张卡吗？\n客户：不需要，请不要再打。", "A"),
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
