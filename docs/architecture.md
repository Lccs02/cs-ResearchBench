# MVP 接口与边界（实现前确定）

| 层 | 接口 | 输入 → 输出 | 约束 |
|---|---|---|---|
| core | ExperimentRecord.validate / read_records / write_records | 版本化 JSON ↔ 运行记录 | 未知版本、重复 run_id、非有限指标直接拒绝 |
| adapters | Adapter.read(run, base_dir, config_source) | 显式映射 → 单次运行记录 | CSV/JSON/NPZ，不推断键或轴；保留 SHA-256 |
| metrics | regression_metrics(target, prediction, weights, missing) | 同形状数组 → MAE/RMSE 与有效计数 | 全选定元素归约，无隐式广播 |
| analysis | summarize / ablations | 已校验记录 → 长格式汇总 | seed 为独立汇总单位；配对消融 |
| export | export_bundle | 已校验记录 → CSV/LaTeX/PNG/JSON | 全部结果保留 Unverified 与来源索引 |
| cli | ingest / validate / summarize / run | 本地文件 → 可复现产物 | 无 AI、网络、训练依赖 |

一份输入文件对应一次运行。模型配置必须剔除 seed（seed 单独记录）；同模型不同超参数应使用不同 model/variant 名称。evaluation_id 由调用者定义，应编码数据版本、划分、目标尺度、预测时域、预处理、评估样本集合及 episode/step 协议。同一 evaluation_id 的语义真实性仍为 Unverified，工具无法从标签证明测试集一致。

MVP 不支持自动挑选最优 checkpoint、不自动拼接文件、不自动解释多维轴、不做显著性检验或置信区间。RL 通过显式 precomputed 标量进入统一统计；不将时间步或 episode 当作独立训练 seed。项目专用适配器只声明接口，真实格式均为 Unverified。
