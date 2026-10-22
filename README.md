# ResearchBench

GitHub 仓库：[Lccs02/cs-ResearchBench](https://github.com/Lccs02/cs-ResearchBench)。

一个独立于 AI 的 Python 科研实验分析 MVP：显式读取已有实验输出，校验并计算指标，按随机种子汇总，进行配对消融比较，导出带来源信息的 CSV、LaTeX 和图形。

**所有示例都是固定随机种子的合成软件测试数据，不是论文结果。科学真实性默认且始终标注 `Unverified`。** “校验通过”仅表示满足 Schema 与计算约束，不能证明原始结果真实或评估协议一致。

## 安装与完整复现

要求 Python 3.10 或以上。以下命令从仓库根目录运行；不需要 AI、GPU、训练代码或真实科研数据。

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/generate_synthetic.py
researchbench run configs/synthetic.yaml --output outputs/demo --baseline baseline
researchbench validate outputs/demo/records.json --check-sources
```

输出目录必须不存在，避免覆盖已有结果。重复运行时使用新目录，例如 `outputs/demo-2`。

如需使用本次验收环境的精确依赖版本，在安装之前运行：

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e '.[dev]'
```

`requirements-tested.txt` 是已测试的 Python 3.12 环境快照，不承诺跨操作系统二进制逐位一致；项目依赖范围在 `pyproject.toml`。每份报告记录实际依赖版本。

分步运行与全流程等价：

```bash
researchbench ingest configs/synthetic.yaml --output outputs/records.json
researchbench validate outputs/records.json --check-sources
researchbench summarize outputs/records.json --output outputs/from-records --baseline baseline
python -m researchbench --help
```

## 模块接口

| 模块 | 主要接口 | 职责 |
|---|---|---|
| `core` | `ExperimentRecord`, `read_records`, `write_records` | Schema 1.0、字段校验、版本拒绝策略 |
| `adapters` | `Adapter.read`, `ConfigurableFileAdapter`, `ingest_config` | 读取映射及来源哈希 |
| `metrics` | `regression_metrics` | MAE/RMSE、权重和缺失值处理 |
| `analysis` | `summarize`, `ablations` | seed 汇总与配对消融 |
| `export` | `export_bundle` | 原子生成结果及 provenance |
| CLI | `ingest`, `validate`, `summarize`, `run` | 无 AI 的完整工作流 |

设计见 [docs/architecture.md](docs/architecture.md)。完整字段定义与升级策略见 [docs/schema.md](docs/schema.md)。

## 输入配置

一份文件对应一次运行。所有路径相对于 YAML 所在目录解析；不依赖终端当前目录。配置必须显式提供项目、数据集、模型、seed、split、evaluation_id、模型配置、输入映射、统计单位和单位。

```yaml
schema_version: '1.0'
runs:
  - run_id: example-model-42
    project: Unverified
    dataset: Unverified
    model: example-model
    seed: 42
    split: test
    evaluation_id: Unverified
    config: {}  # 模型配置；seed 使用独立字段
    data_kind: unknown  # synthetic / real / unknown，声明不等于真实性验证
    input:
      path: ./local_predictions.npz
      format: npz
      mapping:
        target: YOUR_CONFIRMED_TARGET_KEY
        prediction: YOUR_CONFIRMED_PREDICTION_KEY
      expected_shape: [100, 24, 8]  # 可选；示意形状，必须按真实文件修改
    evaluation:
      mode: predictions
      missing: error
      statistical_unit: selected scalar element
      units: Unverified
```

键名和形状示例不代表任何实际研究项目。不要直接把此片段用于 PACSR 等项目。

- **CSV**：映射值是列名或列名列表。字符串得到 `(N,)`；列表得到 `(N,K)`，不会压缩单列。CSV 自身无多维轴语义。
- **JSON**：映射值是顶层字面键名，或逐层访问路径，如 `[payload, forecasts, 0, values]`。不使用点号推断层级。
- **NPZ**：映射值是精确键名；`allow_pickle=False`，拒绝对象数组。
- **维度**：支持任意同形状数值数组；全部选定元素归约，不推测 sample/node/horizon/channel 轴，不隐式 reshape、transpose 或广播。细分通道/时域应通过明确预处理或新的 Adapter 实现并记录协议。
- **权重**：可选 `mapping.weights`，必须与目标数组完全同形状；跨 seed 汇总始终等权。
- **RL/预计算指标**：使用 `mode: precomputed`，mapping 每项读取一个标量，`directions` 指定 `min/max`，`units` 为指标到单位的映射。示例见 `configs/precomputed.example.yaml`。本 MVP 不从训练曲线自动挑选最佳 checkpoint，不自动把 episode 或 time step 平均为训练 seed 指标。

`evaluation_id` 必须区分数据版本、划分、目标尺度、预测时域、预处理、评估样本集合，以及 RL episode/step 协议。工具只能校验标签一致，不能证明这些事实。无法确认时填写 `Unverified` 并保留报告警示；不要据此作科研结论。

## 统计约定

给定有效的正权重元素集合 $V$、误差 $e_i=\hat y_i-y_i$：

$$\operatorname{MAE}=\frac{\sum_{i\in V}w_i|e_i|}{\sum_{i\in V}w_i},\qquad
\operatorname{RMSE}=\sqrt{\frac{\sum_{i\in V}w_i e_i^2}{\sum_{i\in V}w_i}}.$$

- 缺省权重为 1，非有限值（NaN/Inf）默认报错。显式 `missing: omit` 才成对删除目标/预测中的非有限元素。
- 权重必须有限且非负；零权重不参与计算。先计缺失，再计剩余有限元素中的零权重，确保 `n_total = n_valid + n_missing + n_zero_weight`。
- 没有有效正权重元素时拒绝计算，不返回 0。统计中保留有效数量和权重总和；指标只在明确的原始/归一化尺度计算，不做自动反归一化。
- 先得到每个 seed 的指标，再等权汇总。设 $S$ 个 seed 的结果为 $m_s$：

$$\bar m=\frac1S\sum_{s=1}^{S}m_s,\qquad
s_m=\sqrt{\frac{1}{S-1}\sum_{s=1}^{S}(m_s-\bar m)^2}.$$

- `ddof=1`，一个 seed 时标准差输出 `Unverified`，不填 0。误差棒为 seed 样本标准差，不是置信区间。不同 seed 的有效元素数可以不同，但仍等权；请检查 `metrics.csv` 的计数。
- 分组包括项目、数据集、split、evaluation_id、protocol、数据类型、指标、方向、单位和模型。同模型跨 seed 的模型配置必须一致；不同超参数使用不同模型/variant 名称。重复 seed 拒绝，防止伪重复。
- 消融严格匹配 seed 集合和协议；差值为 $d_s=m_s^{\mathrm{variant}}-m_s^{\mathrm{baseline}}$，报告配对差值的 mean/std。对于越小越好的指标，改善率为 $100(\bar m_b-\bar m_v)/|\bar m_b|$；越大越好时反转分子。基线均值为 0 时改善率为 `Unverified`。该百分比是均值的比值，不是各 seed 百分比的均值，也不代表显著性。

## 输出和溯源

| 文件 | 内容 |
|---|---|
| `records.json` | 完整记录、模型配置、协议、来源路径/哈希/映射/形状 |
| `metrics.csv` | 每次运行每项指标，run_id 和 source_ids |
| `summary.csv` | mean/std、seed 数、seed 列表和 run_ids |
| `ablations.csv` | 可选：配对差值与改善率、run_ids |
| `summary.tex` | 按相同协议/指标分表；使用 `booktabs`，转义特殊字符 |
| `summary_*.png` | 按相同协议/单位分图，含 Synthetic/Unverified 标记 |
| `manifest.json` | 各产物 SHA-256、运行到来源的索引、依赖版本、图形分组信息 |

来源路径是本地绝对路径；导出的报告本身可能暴露目录名与配置，分享前检查。CLI 不联网上传。`validate --check-sources` 检查当前源文件哈希；不加此参数时仅验证记录结构，不承诺原文件仍存在或未变。

## 扩展与限制

`adapters/base.py` 定义 Protocol，并提供 `PACSRAdapter`、`SatelliteMultiServiceAdapter`、`WorldModelRLAdapter` 的占位接口；调用会清晰报错，真实格式均为 **Unverified**。获取真实且可授权使用的样例后，实现 `read(...) -> ExperimentRecord`，复用 `metrics` 并保存全部来源。可通过 Python 的 `ingest_config(path, adapter=...)` 注入。MVP CLI 使用通用文件 Adapter。

`.agents/skills/research-analysis/SKILL.md` 只负责选择、配置和调用 CLI，不含指标实现。该 Skill 随仓库提供，不要求安装 AI 才能运行框架。

不包含训练、实验调度、数据库、Dashboard、自动选模、显著性检验或假定真实项目格式。无私有数据和大型权重；生成文件默认被 `.gitignore` 排除。GitHub Actions 在 Python 3.10/3.12 上安装、测试、构建 wheel 并运行完整示例。

## 当前验收状态

本地 Python 3.12 测试 **44 项通过**；干净环境安装、wheel 脱离源码目录运行、完整示例及 LaTeX 实际编译均通过。[GitHub Actions 验收运行](https://github.com/Lccs02/cs-ResearchBench/actions/runs/39821936090)中，Python 3.10 和 3.12 均 **44 项通过**，包括完整示例和独立 wheel 安装。详细命令与证据范围见 [docs/verification.md](docs/verification.md)。原始实验的科学真实性仍为 **Unverified**。
