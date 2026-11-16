"""任务口径。硬约束和提示都从这里读，不从组名推断规则。"""

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


def _yn(extra: tuple[tuple[str, str], ...] = ()) -> tuple[tuple[str, str], ...]:
    return (("A", "否"), ("B", "是"), *extra)


def task_cards() -> dict[str, TaskCard]:
    insufficient = (("C", "证据不足"),)
    cards = [
        TaskCard(
            name="地址核验",
            goal="判断两个地址是不是同一地点。",
            rules="忽略明确缩写和重复的行政区前缀。楼栋号冲突则不是同一地点。决定性字段缺失时选证据不足。",
            options=_yn(insufficient),
            modality="structured",
            output_form="exclusive",
            tolerance="lenient",
            carrier="field",
            ready=True,
            high_cost="B",
            insufficient="C",
            decisive_prefix="地址二：",
            holdout=None,
        ),
        TaskCard(
            name="品牌名写法",
            goal="判断两个名称是不是同一种写法。",
            rules="只比较写法。空格、大小写和明确缩写可以忽略。主体或授权不在这道题里判断。",
            options=_yn(),
            modality="structured",
            output_form="exclusive",
            tolerance="lenient",
            carrier="field",
            ready=True,
            high_cost="B",
            insufficient=None,
            decisive_prefix="名称二：",
            holdout=None,
        ),
        TaskCard(
            name="企业是否存在交集",
            goal="判断两家企业在指定关系上是否有交集。",
            rules="只看股东或地址关系。同名不是同一实体。关系证据缺失时选证据不足。",
            options=_yn(insufficient),
            modality="structured",
            output_form="exclusive",
            tolerance="strict",
            carrier="field",
            ready=True,
            high_cost="B",
            insufficient="C",
            decisive_prefix="关系证据：",
            holdout=None,
        ),
        TaskCard(
            name="店企核验",
            goal="判断店铺和工商主体是不是同一主体。",
            rules="名称相近不够。需要执照主体或授权材料。材料缺失时选证据不足。",
            options=_yn(insufficient),
            modality="structured",
            output_form="exclusive",
            tolerance="strict",
            carrier="field",
            ready=True,
            high_cost="B",
            insufficient="C",
            decisive_prefix="执照主体：",
            holdout=None,
        ),
        TaskCard(
            name="品牌名授权",
            goal="判断店铺是否获得品牌授权。",
            rules="授权书或主体关系成立才选是。只看到名称相同，选否。",
            options=_yn(),
            modality="structured",
            output_form="exclusive",
            tolerance="strict",
            carrier="field",
            ready=True,
            high_cost="B",
            insufficient=None,
            decisive_prefix="授权材料：",
            holdout=None,
        ),
        TaskCard(
            name="大宗商品判别",
            goal="判断商品名称是不是大宗商品。",
            rules="能源、基础金属、矿产品、大宗农产品、大宗化工原料选是。品牌消费品、整机、软件和服务选否。不看数量。",
            options=_yn(),
            modality="structured",
            output_form="exclusive",
            tolerance="neutral",
            carrier="field",
            ready=True,
            high_cost="B",
            insufficient=None,
            decisive_prefix="商品名称：",
            holdout=None,
        ),
        TaskCard(
            name="敏感品名识别",
            goal="判断品名是否属于敏感清单。",
            rules="清单版本 S1：管制化学品名称选是。普通金属和日用品选否。",
            options=_yn(),
            modality="structured",
            output_form="exclusive",
            tolerance="neutral",
            carrier="field",
            ready=True,
            high_cost="A",
            insufficient=None,
            decisive_prefix="品名：",
            holdout=None,
        ),
        TaskCard(
            name="中介机构识别",
            goal="判断发票上下游企业是不是中介机构。",
            rules="需要经营范围或业务证据。不能只凭企业名称判断。",
            options=_yn(),
            modality="structured",
            output_form="exclusive",
            tolerance="neutral",
            carrier="field",
            ready=True,
            high_cost="B",
            insufficient=None,
            decisive_prefix="经营证据：",
            holdout=None,
        ),
        TaskCard(
            name="批发商身份识别",
            goal="判断企业登记的身份是不是批发。",
            rules="看登记的经营范围，不推断实际销量。范围同时含批发和零售时，登记含批发即选是。",
            options=_yn(),
            modality="structured",
            output_form="exclusive",
            tolerance="neutral",
            carrier="field",
            ready=True,
            high_cost="B",
            insufficient=None,
            decisive_prefix="经营范围：",
            holdout=None,
        ),
        TaskCard(
            name="是否需要许可",
            goal="判断这种经营是否需要许可。",
            rules="只判断是否属于需要许可的活动，不判断已经持有的证是否有效。",
            options=_yn(insufficient),
            modality="structured",
            output_form="exclusive",
            tolerance="neutral",
            carrier="field",
            ready=True,
            high_cost="A",
            insufficient="C",
            decisive_prefix="经营活动：",
            holdout=None,
        ),
        TaskCard(
            name="经营范围覆盖",
            goal="判断销项商品是否落在经营范围内。",
            rules="只判断范围覆盖，不判断实际经营真实性。每一种商品都被某个项目涵盖才选是。",
            options=_yn(insufficient),
            modality="structured",
            output_form="exclusive",
            tolerance="neutral",
            carrier="field",
            ready=True,
            high_cost="B",
            insufficient="C",
            decisive_prefix="销项商品：",
            holdout=None,
        ),
        TaskCard(
            name="商品是否一致",
            goal="判断两条商品记录是不是同一规格。",
            rules="材质、尺寸或型号冲突则不是同一规格。品牌缩写可以忽略。",
            options=_yn(insufficient),
            modality="structured",
            output_form="exclusive",
            tolerance="lenient",
            carrier="field",
            ready=True,
            high_cost="B",
            insufficient="C",
            decisive_prefix="记录二：",
            holdout="near",
        ),
        TaskCard(
            name="电销用户意图识别",
            goal="判断客户这句话是否包含购买意向。",
            rules="只看客户本人的话。多种意图可以分别判断。这里只问购买意向这一个二选一。",
            options=_yn(),
            modality="dialogue",
            output_form="multi_binary",
            tolerance="neutral",
            carrier="dialogue",
            ready=True,
            high_cost="B",
            insufficient=None,
            decisive_prefix="客户：",
            holdout="far",
        ),
    ]
    return {card.name: card for card in cards}


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
