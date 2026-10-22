# ExperimentRecord Schema 1.0

规范实现为 `src/researchbench/core/schema.py` 中的 dataclass 和校验器，无额外 Schema 依赖。

文件容器：`{"schema_version": "1.0", "records": [...]}`。每项如下：

| 字段 | 类型 / 约束 | 含义 |
|---|---|---|
| schema_version | 字符串，精确 `1.0` | 记录版本；YAML 中必须加引号 |
| run_id | 非空字符串，文件内唯一 | 一次独立训练 seed 的评估记录 |
| project / dataset / model | 非空字符串 | 项目、数据集版本和模型变体；未知填 Unverified |
| seed | 非负整数，不接受布尔值 | 随机种子；不是 episode/时间步编号 |
| split | 非空字符串 | train/validation/test 或显式自定义划分 |
| evaluation_id | 非空字符串 | 数据版本、划分、目标尺度、时域、评估样本集合等协议标识 |
| config | JSON 兼容 mapping | 模型配置，seed 单列；同模型跨 seed 必须一致 |
| metrics | 非空 mapping | 指标名 → value/direction/可用计数 |
| protocol | mapping | mode、missing、weighting、statistical_unit、reduction、metric_units |
| sources | 非空来源列表 | 文件绝对路径、SHA-256、role、format，数据文件另含 mapping/shape |
| data_kind | synthetic / real / unknown | 用户声明；不是真实性证明 |
| verification | 仅 Unverified | MVP 不提供科学真实性认证路径 |

## Metric

所有指标必须包含有限数值 `value` 和 `direction: min|max`。预计算模式只接收显式标量，不做自动归约。

prediction 模式由唯一的 `regression_metrics` 实现生成 MAE/RMSE，并记录：

- `n_total`: 全部选定元素数。
- `n_valid`: 有限目标/预测且正权重的元素数。
- `n_missing`: 目标或预测非有限的元素数。
- `n_zero_weight`: 有限目标/预测中的零权重数。
- `weight_sum`: 有效元素的正权重之和。

以上计数按缺失优先分类，互斥完备。计算过程不推断数据轴，shape 仅用于同形状校验和来源追踪。

## Protocol

prediction 模式：`mode=predictions`、`missing=error|omit`、`weighting=uniform|explicit`、`reduction=all_selected_elements`。`statistical_unit` 应明确描述一个输入元素的含义；跨 seed 汇总的统计单位另外固定为 seed。

precomputed 模式：`mode=precomputed`、`missing=not_applicable`、`weighting=reported`、`reduction=reported_scalar`。`statistical_unit` 必须描述标量的生成方法，例如每个训练 seed 在固定评估 episode 集合上的平均回报。该语义无法自动验证。

`metric_units` 必须覆盖且仅覆盖当前记录的指标名。协议完整 JSON 的 SHA-256 是 `protocol_id`，用于防止不同权重/缺失/单位声明混合；它无法证明实际数据采用了相同评估集合或权重含义。

## Source 与证据范围

`path` 指向实际读取的文件，`sha256` 为实际读取字节的哈希。`role` 为 data 或 config，`format` 为 csv/json/npz/yaml。数据 source 的 `mapping` 保留键或列选择，`shape` 保存所选数组形状。输入先读取字节，再用同一字节快照解析和哈希。NPZ 禁用 pickle。

`validate` 仅检查记录结构与数值约束。`validate --check-sources` 额外检查当前源文件字节是否与记录哈希一致，不重新计算指标。`run` 从配置重新读取并计算指标。来源不可用时，无法复核其真实性，状态仍为 Unverified。

## 升级策略

当前只接受 1.0，未知版本直接报错。未来破坏性字段/统计语义变化必须更新 Schema 版本，并提供显式迁移工具和迁移记录；禁止自动忽略未知字段或静默迁移。
