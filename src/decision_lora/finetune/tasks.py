"""十一项任务的题目口径。原始 JSONL 只写材料字段和答案，这里补上判断目标和选项。"""

import json

TASK_ORDER = (
    "大宗商品判别",
    "是否烟草许可",
    "是否道路运输许可",
    "是否食品许可",
    "是否建筑市场资质",
    "是否商务特许许可",
    "是否有排污许可",
    "批发商身份识别",
    "地址核验",
    "店企核验",
    "品牌名和店铺名一致性判别",
)

BINARY_OPTIONS = (
    {"代号": "A", "名称": "是"},
    {"代号": "B", "名称": "否"},
)

ADDRESS_OPTIONS = (
    {"代号": "A", "名称": "不同省份"},
    {"代号": "B", "名称": "同省，不同市"},
    {"代号": "C", "名称": "同市，不同区县"},
    {"代号": "D", "名称": "同区县，不同街道或镇"},
    {"代号": "E", "名称": "同街道或镇，不同路或乡里"},
    {"代号": "F", "名称": "同路或乡里，不同小区或楼栋"},
    {"代号": "G", "名称": "同小区或楼栋，文字不完全一致"},
    {"代号": "H", "名称": "完全一致"},
    {"代号": "I", "名称": "无法判断"},
)

YES_ALIASES = {"是", "A是", "yes", "YES"}
NO_ALIASES = {"否", "B否", "no", "NO"}


def _text(row, *keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value).strip()
    return None


def _optional(row, *keys):
    """缺失、null、空字符串都视为空，别名里取第一个有内容的值。"""
    for key in keys:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            text = str(value).strip()
        else:
            text = ""
        if text:
            return text
    return ""


def _require_any(values, message):
    if not any(value.strip() for value in values):
        raise ValueError(message)


def build_goods(row):
    name = _optional(row, "商品名称")
    _require_any([name], "商品名称不能为空")
    return f"商品名称：{name}"


def build_license(row):
    company = _optional(row, "企业名称")
    scope = _optional(row, "经营范围")
    _require_any([company, scope], "企业名称和经营范围不能同时为空")
    return f"企业名称：{company}\n经营范围：{scope}"


def build_address(row):
    left = _optional(row, "地址一")
    right = _optional(row, "地址二")
    _require_any([left, right], "地址一和地址二不能同时为空")
    return f"地址一：{left}\n地址二：{right}"


def build_shop(row):
    structured_keys = (
        "店铺名称", "dp_name", "店铺行业", "tx_industry", "企业名称", "gs_name",
        "经营范围", "opscope", "工商行业", "gs_industry",
    )
    use_blob = not any(key in row for key in structured_keys) and (
        "店铺信息" in row or "工商信息" in row
    )
    if use_blob:
        shop_blob = _optional(row, "店铺信息")
        firm_blob = _optional(row, "工商信息")
        _require_any([shop_blob, firm_blob], "店企核验的材料字段不能同时为空")
        return f"店铺信息：{shop_blob}\n工商信息：{firm_blob}"
    shop_name = _optional(row, "店铺名称", "dp_name")
    shop_industry = _optional(row, "店铺行业", "tx_industry")
    company = _optional(row, "企业名称", "gs_name")
    scope = _optional(row, "经营范围", "opscope")
    industry = _optional(row, "工商行业", "gs_industry")
    _require_any(
        [shop_name, shop_industry, company, scope, industry],
        "店企核验的材料字段不能同时为空",
    )
    shop = json.dumps(
        {"dp_name": shop_name, "tx_industry": shop_industry},
        ensure_ascii=False, separators=(", ", ": "),
    )
    firm = json.dumps(
        {"gs_name": company, "opscope": scope, "gs_industry": industry},
        ensure_ascii=False, separators=(", ", ": "),
    )
    return f"店铺信息：{shop}\n工商信息：{firm}"


def build_brand(row):
    shop_name = _optional(row, "店铺名称")
    brand = _optional(row, "品牌名称")
    _require_any([shop_name, brand], "店铺名称和品牌名称不能同时为空")
    return f"店铺名称：{shop_name}\n品牌名称：{brand}"


def _binary(goal, rule, build):
    return {
        "题型": "是非题",
        "判断目标": goal,
        "判断规则": rule,
        "选项": BINARY_OPTIONS,
        "build": build,
    }


def _license(name):
    return _binary(
        f"判断一家企业是否有{name}",
        f"从一家企业的经营范围判定这家企业是否有{name}",
        build_license,
    )


TASKS = {
    "大宗商品判别": _binary(
        "判断一个商品是否是大宗商品",
        "判断一个商品是否是大宗商品，不用考虑该商品型号等具体规格，只考虑该商品本身的属性",
        build_goods,
    ),
    "是否烟草许可": _license("烟草许可"),
    "是否道路运输许可": _license("道路运输许可"),
    "是否食品许可": _license("食品许可"),
    "是否建筑市场资质": _license("建筑市场资质"),
    "是否商务特许许可": _license("商务特许许可"),
    "是否有排污许可": _license("排污许可"),
    "批发商身份识别": _binary(
        "判断一家企业是否是批发商",
        "企业名称或经营范围中含有批发字样，仅有销售、零售等不纳入批发商行列",
        build_license,
    ),
    "地址核验": {
        "题型": "分档题",
        "判断目标": "判断两个地址是否相同。",
        "判断规则": "将地址名称分省、市、区县、街道、路、楼等层次，判断两个地址是哪些层次相同，不要求字段完全一致",
        "选项": ADDRESS_OPTIONS,
        "build": build_address,
    },
    "店企核验": _binary(
        "判断店铺和企业是否有关联",
        "根据店铺信息里面的店铺名称和行业，对比企业信息里面的企业名称、经营范围和工商行业，不要求字段完全一致，有相关性即可",
        build_shop,
    ),
    "品牌名和店铺名一致性判别": _binary(
        "判断品牌名称和店铺名称是否有关联",
        "品牌名称和店铺名称有相似度或有部分字段的重叠，不要求完全相同",
        build_brand,
    ),
}


def normalize_answer(raw, spec):
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        raise ValueError("答案必须是代号或选项名称")
    text = str(raw).strip()
    codes = [item["代号"] for item in spec["选项"]]
    names = {item["名称"]: item["代号"] for item in spec["选项"]}
    if text in codes:
        return text
    if text in names:
        return names[text]
    if spec["题型"] == "是非题" and text in YES_ALIASES:
        return "A"
    if spec["题型"] == "是非题" and text in NO_ALIASES:
        return "B"
    shown = "、".join(codes)
    raise ValueError(f"答案 {text!r} 不在选项中，可选：{shown}，是非题也可写「是」或「否」")


def _key(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("分组键必须是字符串或整数")
    text = str(value).strip()
    return text or None


def expand_row(row, task_name=None):
    if not isinstance(row, dict):
        raise ValueError("每行必须是 JSON 对象")
    task = row.get("任务") or task_name
    if task not in TASKS:
        names = "、".join(TASK_ORDER)
        raise ValueError(f"任务未登记：{task}。当前接受：{names}")
    spec = TASKS[task]
    if isinstance(row.get("材料"), str) and row["材料"].strip():
        material = row["材料"].strip()
    else:
        material = spec["build"](row)
    if "答案" not in row or row["答案"] is None:
        raise ValueError("缺少字段：答案")
    return {
        "任务": task,
        "题型": spec["题型"],
        "判断目标": spec["判断目标"],
        "判断规则": spec["判断规则"],
        "材料": material,
        "选项": [dict(item) for item in spec["选项"]],
        "答案": normalize_answer(row["答案"], spec),
        "分组键": _key(row.get("分组键")) or _text(row, "企业名称", "gs_name"),
        "隔离键": _key(row.get("分组键")),
    }


def render_prompt(row):
    options = "；".join(item["代号"] + item["名称"] for item in row["选项"])
    return (
        f"【判断目标】{row['判断目标']}\n"
        f"【判断规则】{row['判断规则']}\n"
        f"【材料】\n{row['材料']}\n"
        f"【选项】{options}\n"
        "只回答代号，不要解释。"
    )
