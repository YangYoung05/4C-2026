# 运行顺序

以仓库根目录为当前目录。完整重建需要 Python 3.9+、`requirements.txt` 中的依赖，以及数据索引中列出的外部数据。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r 雷霆医疗队/requirements.txt
export THUNDER_RUNTIME_EXTERNAL_DATA_ROOT=/absolute/path/to/external_data
```

## 静态与路径检查

```bash
python3 -m py_compile 雷霆医疗队/01_main/*.py
python3 雷霆医疗队/01_main/a_foundation.py \
  --project-root 雷霆医疗队 --check
```

路径检查会输出实际检测到的项目、input 和外部数据根目录。若 `external_data_root_exists` 为 `false`，可以完成代码静态校验，但不应启动全量分析。

## 全量分析

```bash
python3 雷霆医疗队/01_main/b_build_panels.py --build --audit
python3 雷霆医疗队/01_main/c_run_abcd.py --run --audit
python3 雷霆医疗队/01_main/d_module_d_policy.py --run --audit
python3 雷霆医疗队/01_main/e_china_mapping.py --data --audit --figures
```

| 顺序 | 输入/分析 | 生成目录 |
|---|---|---|
| B | 清洗多源数据，构建全球主面板 | `04_simulation/`、`09_data_clean/` |
| C | 阶段、脆弱性、风险归因、失配与预测验证 | `05_figures/`、`06_report_assets/b_global_abcd/` |
| D | 政策适配、准因果增强、证据裁判与锁定 | `05_figures/`、`06_report_assets/c_module_d_policy/` |
| E | 中国省级映射、数据边界审计和图表 | `05_figures/`、`06_report_assets/d_china_mapping/` |

上述生成目录已在 `.gitignore` 中排除，以避免重复提交大体积数据和派生文件。

## Heywhale

上传后如项目位于 `/home/mw/project/雷霆医疗队`，外部数据位于 `/home/mw/input/External_Data28782878/External Data`，无需额外设置路径。其他环境使用 `THUNDER_RUNTIME_PROJECT_ROOT`、`THUNDER_RUNTIME_INPUT_ROOT` 和 `THUNDER_RUNTIME_EXTERNAL_DATA_ROOT` 覆盖。

## 产物口径

- 缺少原始数据时脚本应明确报错，不静默伪造结果。
- OCR 候选字段只有通过字段级 QC 后才能进入核心评分。
- 模块 D 的严格表述与中国映射边界见项目 README。
