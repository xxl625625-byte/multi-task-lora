"""十一项任务口径。文字与 baseline 的 tasks.py 一致，选项 A 是、B 否。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskCard:
    name: str
    goal: str
    rules: str
    options: tuple[tuple[str, str], ...]
    modality: str
    output_form: str
    tolerance: str
    carrier: str
    ready: bool
    high_cost: str | None
    insufficient: str | None
    decisive_prefix: str | None
    holdout: str | None


def _yes_no() -> tuple[tuple[str, str], ...]:
    return (("A", "是"), ("B", "否"))


def _license(name: str, activity: str) -> TaskCard:
    return TaskCard(
        name=name,
        goal=f"判断一家企业是否有{activity}",
        rules=f"从一家企业的经营范围判定这家企业是否有{activity}",
        options=_yes_no(),
        modality="structured",
        output_form="exclusive",
        tolerance="neutral",
        carrier="field",
        ready=True,
        high_cost="A",
        insufficient=None,
        decisive_prefix="经营范围：",
        holdout=None,
    )


_ADDRESS = (
    ("A", "不同省份"),
    ("B", "同省，不同市"),
    ("C", "同市，不同区县"),
    ("D", "同区县，不同街道或镇"),
    ("E", "同街道或镇，不同路或乡里"),
    ("F", "同路或乡里，不同小区或楼栋"),
    ("G", "同小区或楼栋，文字不完全一致"),
    ("H", "完全一致"),
    ("I", "无法判断"),
)


def task_cards() -> dict[str, TaskCard]:
    cards = [
        TaskCard(
            name="大宗商品判别",
            goal="判断一个商品是否是大宗商品",
            rules="判断一个商品是否是大宗商品，不用考虑该商品型号等具体规格，只考虑该商品本身的属性",
            options=_yes_no(),
            modality="structured",
            output_form="exclusive",
            tolerance="neutral",
            carrier="field",
            ready=True,
            high_cost="A",
            insufficient=None,
            decisive_prefix="商品名称：",
            holdout=None,
        ),
        _license("是否烟草许可", "烟草许可"),
        _license("是否道路运输许可", "道路运输许可"),
        _license("是否食品许可", "食品许可"),
        _license("是否建筑市场资质", "建筑市场资质"),
        _license("是否商务特许许可", "商务特许许可"),
        _license("是否有排污许可", "排污许可"),
        TaskCard(
            name="批发商身份识别",
            goal="判断一家企业是否是批发商",
            rules="企业名称或经营范围中含有批发字样，仅有销售、零售等不纳入批发商行列",
            options=_yes_no(),
            modality="structured",
            output_form="exclusive",
            tolerance="strict",
            carrier="field",
            ready=True,
            high_cost="A",
            insufficient=None,
            decisive_prefix="经营范围：",
            holdout=None,
        ),
        TaskCard(
            name="地址核验",
            goal="判断两个地址是否相同。",
            rules="将地址名称分省、市、区县、街道、路、楼等层次，判断两个地址是哪些层次相同，不要求字段完全一致",
            options=_ADDRESS,
            modality="structured",
            output_form="graded",
            tolerance="lenient",
            carrier="field",
            ready=True,
            high_cost=None,
            insufficient="I",
            decisive_prefix="地址二：",
            holdout=None,
        ),
        TaskCard(
            name="店企核验",
            goal="判断店铺和企业是否有关联",
            rules="根据店铺信息里面的店铺名称和行业，对比企业信息里面的企业名称、经营范围和工商行业，不要求字段完全一致，有相关性即可",
            options=_yes_no(),
            modality="structured",
            output_form="exclusive",
            tolerance="lenient",
            carrier="field",
            ready=True,
            high_cost="A",
            insufficient=None,
            decisive_prefix="工商信息：",
            holdout=None,
        ),
        TaskCard(
            name="品牌名和店铺名一致性判别",
            goal="判断品牌名称和店铺名称是否有关联",
            rules="品牌名称和店铺名称有相似度或有部分字段的重叠，不要求完全相同",
            options=_yes_no(),
            modality="structured",
            output_form="exclusive",
            tolerance="lenient",
            carrier="field",
            ready=True,
            high_cost="A",
            insufficient=None,
            decisive_prefix="店铺名称：",
            holdout=None,
        ),
    ]
    return {card.name: card for card in cards}


def task_order() -> tuple[str, ...]:
    return tuple(task_cards())


def cannot_merge(left: TaskCard, right: TaskCard) -> str | None:
    """返回不能合并的原因。可以合并时返回 None。"""
    if not left.ready or not right.ready:
        return "口径未写完"
    if left.output_form != right.output_form:
        return "输出形式不同"
    if left.carrier != right.carrier:
        return "证据载体不同"
    if {left.tolerance, right.tolerance} == {"lenient", "strict"}:
        return "容差方向相反"
    return None
