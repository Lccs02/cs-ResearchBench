# MVP 验收记录

验收日期：2026-10-23（Asia/Shanghai）。本地 Python 3.12.14，精确依赖快照见 requirements-tested.txt。

## 已执行

| 检查 | 结果 |
|---|---|
| 新建 venv，`python -m pip install -e '.[dev]'` | 成功 |
| `python -m pytest -q` | **44 passed in 4.80s** |
| `python examples/generate_synthetic.py` | 生成 2 模型 × 3 seed 的 CSV/JSON/NPZ 合成输入 |
| `researchbench run configs/synthetic.yaml --output outputs/demo --baseline baseline` | 成功，6 runs |
| `researchbench validate outputs/demo/records.json --check-sources` | 成功，全部源哈希匹配 |
| `python -m build` | sdist 与 wheel 构建成功 |
| 独立 venv 安装 wheel，从 `/tmp` 运行完整示例及来源校验 | 成功，导入来自 site-packages |
| `pdflatex -interaction=nonstopmode -halt-on-error ...` | summary.tex 编译成功，1 页，无 overfull 警告 |
| Skill 格式校验与独立 CLI 使用 | 成功，无重复指标实现 |
| GitHub Actions / Python 3.10 | **44 passed in 4.05s**；安装、构建、完整示例、来源校验、独立 wheel 执行均成功 |
| GitHub Actions / Python 3.12 | **44 passed in 5.51s**；安装、构建、完整示例、来源校验、独立 wheel 执行均成功 |

远端证据：[Actions run 39821936090](https://github.com/Lccs02/cs-ResearchBench/actions/runs/39821936090)，被测源码提交 `d17bd02ae577f3238bfa5e431de63376c1a4c71e`。后续验收文档更新仅记录该结果，不改变源码；不将软件验收等同于实验真实性验证。

## 单元与端到端测试覆盖

- 手工可核查的加权 MAE/RMSE；任意维度、形状不匹配、零/负/无穷权重、NaN/Inf、空输入。
- CSV 列列表、JSON 嵌套路径、NPZ 显式键映射及禁止 pickle。
- 版本化记录往返、未知版本/字段/非法 seed/伪造 Verified 状态的拒绝。
- seed 等权、ddof=1、单 seed 的未定义 std、配对差值、零基线和最大化指标。
- 重复 seed、不一致模型配置、不匹配评估协议/seed 集合拒绝；split 分组隔离。
- CLI 完整产物、重新汇总的 CSV 一致性、来源篡改检测、产物哈希、输出覆盖拒绝、失败不留下报告。
- 项目专用 Adapter 对未知真实格式明确抛错，不伪造解析规则。

示例改善率是人为设定噪声尺度形成的预期软件测试行为，不能作为算法效果或论文结果。全部报告的科学状态保持 Unverified。

## 可复现命令

从仓库根目录安装后：

```bash
python -m pytest -q
python examples/generate_synthetic.py
researchbench run configs/synthetic.yaml --output outputs/new-check --baseline baseline
researchbench validate outputs/new-check/records.json --check-sources
python -m build
```

再次运行请选择新的输出目录。精确复刻依赖可先安装 requirements-tested.txt；浮点末位和图像字节跨平台不保证完全一致。
