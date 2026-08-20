# Code 10.4：JHBE 证据分析管线

本目录是配套 *Journal of Housing and the Built Environment* 修订稿的 Code 10.4 可复现分析版本。项目针对 41 个城市，比较政策文本表征、规则属性与 Dep 派生操作代理之间的关系，输出预测比较、测量有效性诊断和探索性模型归因结果。

需要明确：这些结果不是政策因果效应。随机森林性能、SHAP、R² Shapley 分解、城市排序变化和 discordant-pair 计数，都只能在各自标注的预测性或描述性证据范围内解释。

## 发布包包含什么

- Code 10.4 的完整 Python 脚本和单模块入口。
- 运行分析所需的 41 城输入工作簿。
- 可直接复现的 `outputs/combined_data_10.4.xlsx` 缓存。
- 已审计的 Dep operational bridge 和利率敏感性缓存。
- `tests/` 下的单元测试。

`Modelling Dep` 文件夹没有复制到本目录，因为它已经作为独立项目发布。本包只保留默认复现所需的已审计桥接表和缓存结果，避免重复发布同一内容。

## 环境和依赖

- Python 3.10 或更高版本；本地验证使用 Python 3.13。
- 可用的 `pip`。
- 缓存复现路径支持 Windows、macOS 和 Linux。
- 如果要从原始工作簿重新计算，还需要已单独发布的 Dep 引擎和本地中文 BERT 模型。

创建虚拟环境并安装依赖：

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 快速开始

在本目录根路径运行：

```bash
python -m pytest -q
python -B run_pipeline.py
```

第一条命令运行测试；第二条命令运行完整的缓存复现管线。程序读取随包提供的 41 城缓存，并将重新生成的表格、图形、日志和 LaTeX 片段写入 `outputs/`。

如果只想做快速烟测：

```bash
python -B run_pipeline.py --skip-shap --skip-shapley --skip-ablations
```

烟测只用于排查环境问题。因为主动跳过了若干证据模块，它不应被当作正式结果，也不应期待其通过完整的 requirements audit。

## 从原始数据重建

默认命令不需要 `Modelling Dep` 或 BERT 模型。如果需要从本地源工作簿重建 `combined_data_10.4.xlsx`，请提供已经单独发布的 `41_Cities_Dep` 目录和本地 `bert-base-chinese` 模型：

```powershell
# Windows PowerShell
$env:JHBE_DEP_DIR = "C:\path\to\Modelling Dep\41_Cities_Dep"
$env:JHBE_BERT_MODEL = "C:\path\to\bert-base-chinese"
python -B run_pipeline.py --force-rebuild --retune-m0 --optuna-trials 40
```

`JHBE_DEP_DIR` 必须直接指向包含 `multi_city_dep_model.py` 和 `city_catalog.json` 的目录。BERT 权重约 412 MB，本发布包没有重复携带；请提供与原始 local-files-only 加载方式兼容的模型目录。

## 主要入口

| 文件 | 用途 |
|---|---|
| `run_pipeline.py` | 运行完整 Code 10.4 证据管线 |
| `run_00_m0_optuna.py` | 只对 M0 调参并冻结一套 RF 参数 |
| `run_02_canonical.py` | 比较四种 PMC 表征与 M0 |
| `run_03_overlap.py` | 计算 P1 规则距离/文本距离诊断 |
| `run_11_pca_h1.py` | 历史文件名；当前实现运行 H1 属性桥接，不再执行 PCA |
| `run_14_shap_exploratory.py` | 生成探索性 SHAP 汇总 |
| `run_16_section5_assets.py` | 汇总 Section 5 表格、图形和资产清单 |

## 可复现契约

- 41 城横截面样本。
- `seed=42`。
- 随机森林使用重复交叉验证；实际参数契约会记录在首次运行后生成的 `outputs/logs/m0_frozen_rf_params.json`。
- Optuna 只用于 M0 调参；之后的模型复用同一冻结参数集。
- `IV3` 仅作描述性变量，`IV4` 仅用于机械重叠审计。
- SHAP 和 R² Shapley 是探索性模型归因量，不是效应估计或显著性检验。

## 数据和授权

本包含有政策文本、人工编码的政策属性和 41 城派生分析数据。将目录推送到公开仓库前，请先核对源材料的版权、再分发和隐私条件。这里没有擅自声明许可证；正式公开时，应根据代码和数据的实际权属补充合适的 license。

英文说明见 [`README.md`](README.md)。
