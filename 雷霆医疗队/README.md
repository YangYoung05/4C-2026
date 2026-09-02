# 雷霆医疗队分析代码

本目录保留“全球健康转型中的暴露-负担-响应失配诊断”的正式 Python 入口、方法文档和数据索引。大体积原始数据、清洗缓存、图片和报告产物不纳入 Git，由脚本在完整数据环境中重建。

## 分析框架

- 数据底座：统一国家标签、年份、健康负担、风险暴露、资源与政策数据。
- 模块 A/B/C：识别健康转型阶段、脆弱性类型、风险归因和压力-响应失配。
- 模块 D：组织政策数据、准因果候选、证据裁判与政策适配路径。
- 中国映射：将全球框架迁移到中国大陆 31 个省级行政区。

## 代码入口

| 顺序 | 脚本 | 作用 |
|---|---|---|
| A | [`a_foundation.py`](01_main/a_foundation.py) | 路径、字体、国家标签、目录和公共运行工具 |
| B | [`b_build_panels.py`](01_main/b_build_panels.py) | 数据清洗、数值修复和全球主面板构建 |
| C | [`c_run_abcd.py`](01_main/c_run_abcd.py) | 阶段识别、脆弱性分型、风险归因、失配和预测验证 |
| D | [`d_module_d_policy.py`](01_main/d_module_d_policy.py) | 政策适配、准因果增强、多政策组合和证据锁定 |
| E | [`e_china_mapping.py`](01_main/e_china_mapping.py) | 中国省级映射、来源审计和图表生成 |

方法、公式与阈值见 [`02_docs/分析方法与代码实现.md`](02_docs/分析方法与代码实现.md) 和 [`02_docs/分析标准与判定口径.md`](02_docs/分析标准与判定口径.md)。

## 环境与数据

需要 Python 3.9 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r 雷霆医疗队/requirements.txt
```

完整分析需要与 [`08_data_inventory/数据目录索引.md`](08_data_inventory/数据目录索引.md) 同构的外部数据。在仓库根目录运行时，可显式指定：

```bash
export THUNDER_RUNTIME_EXTERNAL_DATA_ROOT=/absolute/path/to/external_data
```

Heywhale 环境仍兼容 `/home/mw/project/雷霆医疗队` 和 `/home/mw/input/External_Data28782878/External Data`，但本地与 GitHub 克隆不依赖固定用户路径。

## 运行

详细顺序见 [`run_order.md`](run_order.md)。快速静态检查：

```bash
python3 -m py_compile 雷霆医疗队/01_main/*.py
python3 雷霆医疗队/01_main/a_foundation.py \
  --project-root 雷霆医疗队 --check
```

## 证据边界

- 主集成面板覆盖 2000-2024，但不是完整平衡面板，各数据源可用年份不同。
- 全球建模和地图展示口径需分开；台湾省在展示层纳入中国颜色，不伪造单独省级估计。
- 模块 D 的最强主证据是“严格准因果强候选”，不等于随机实验级强因果。
- 中国映射是应用层，不反向证明全球模块 D 的因果性。
