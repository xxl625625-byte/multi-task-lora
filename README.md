# 多任务决策 LoRA

这个仓库实现《多任务决策 LoRA 训练方案》：硬约束和探针合并、通用语料投影相减、整题一次路由。当前没有业务数据，`data/simulated/samples.jsonl` 是虚构的几条记录，只用来把代码跑通。

基座是一个小型因果模型，模块名用 `q_proj`、`v_proj`、`o_proj`。残差、投影系数和路由不依赖具体基座。换成本地 Qwen 时，仍然只在这些模块上挂 LoRA，相减和路由的代码不用改。

模拟运行把开发集条数门槛从 200 降到 1，全量步数 T 设为 8。正式实验先改 `configs/demo.json`：`dev_floor_relaxed` 设为 false，`min_dev_for_merge` 设为 200，并写死正式的 T。两套数字不能放进同一张结论。

## 运行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest tests/test_core.py
.venv/bin/python -m decision_lora run --config configs/demo.json
```

在 `decision_lora` 目录下执行。报告写到 `outputs/report.json`。

组 LoRA 没有比冻结基座和共享 LoRA 高出 1 个百分点时，流水线停在门槛处，不训练通用语料。这是方案里的停止条件，不是运行失败。模拟样本很少，出现这个结果是正常的。

## 代码对应

| 方案步骤 | 代码 |
| --- | --- |
| 口径与硬约束 | `schema.py` |
| 模拟样本与案例切分 | `simulate.py`、`splits.py` |
| 探针合并 | `grouping.py` |
| 专用 LoRA、共享 LoRA、通用语料 | `train.py` |
| 投影系数、L1/L2/L0 | `residual.py`、`gates.py` |
| 整题路由 | `routing.py` |
| 执行顺序 | `pipeline.py` |
