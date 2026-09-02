from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib
from matplotlib import font_manager
import pandas as pd

DECLARED_PROJECT_ROOT = Path("/home/mw/project/雷霆医疗队")
DECLARED_INPUT_ROOT = Path("/home/mw/input")
DECLARED_EXTERNAL_DATA_ROOT = DECLARED_INPUT_ROOT / "External_Data28782878" / "External Data"
RUNTIME_PROJECT_ROOT_ENV = "THUNDER_RUNTIME_PROJECT_ROOT"
RUNTIME_INPUT_ROOT_ENV = "THUNDER_RUNTIME_INPUT_ROOT"
RUNTIME_EXTERNAL_DATA_ROOT_ENV = "THUNDER_RUNTIME_EXTERNAL_DATA_ROOT"
PROJECT_MARKERS = ("01_main", "requirements.txt", "run_order.md")

REPORT_ASSET_SUBDIRS = {
    "index": "a_index",
    "global_abcd": "b_global_abcd",
    "module_d_policy": "c_module_d_policy",
    "china_mapping": "d_china_mapping",
    "audit_qc": "e_audit_qc",
    "report_manifest": "f_report_manifest",
}

CHINESE_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Arial Unicode MS",
]

REPORT_FIGURE_TITLE_SIZE = 22
REPORT_AXIS_TITLE_SIZE = 17
REPORT_SUBTITLE_SIZE = 13
REPORT_AXIS_LABEL_SIZE = 14
REPORT_TICK_LABEL_SIZE = 12
REPORT_LEGEND_SIZE = 11
REPORT_ANNOTATION_SIZE = 9

COUNTRY_LABEL_OVERRIDES = {
    "BGD": "孟加拉国",
    "FJI": "斐济",
    "HTI": "海地",
    "IND": "印度",
    "LAO": "老挝",
    "LKA": "斯里兰卡",
    "MMR": "缅甸",
    "PNG": "巴布亚新几内亚",
    "SLB": "所罗门群岛",
    "TON": "汤加",
    "VUT": "瓦努阿图",
    "WSM": "萨摩亚",
}


def _resolve_override(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _looks_like_project_root(path: Path) -> bool:
    return all((path / marker).exists() for marker in PROJECT_MARKERS)


def _discover_project_root() -> Path | None:
    source_root = Path(__file__).resolve().parents[1]
    if _looks_like_project_root(source_root):
        return source_root
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if _looks_like_project_root(candidate):
            return candidate
    return None


def _discover_input_root(project_root: Path) -> Path | None:
    for candidate in (
        project_root.parent / "input",
        project_root / "input",
        project_root.parent / "external_data",
        project_root.parent.parent / "input",
        project_root.parent.parent / "external_data",
    ):
        if candidate.exists():
            return candidate.resolve()
    return None


def _discover_external_data_root(project_root: Path, input_root: Path) -> Path | None:
    for candidate in (
        input_root / "External_Data28782878" / "External Data",
        input_root / "External_Data28782878",
        input_root / "External Data",
        project_root.parent / "external_data",
        project_root.parent / "input" / "External_Data28782878" / "External Data",
        project_root.parent / "input" / "External_Data28782878",
        project_root / "external_data",
        project_root.parent.parent / "external_data",
        project_root.parent.parent / "input" / "External_Data28782878" / "External Data",
        project_root.parent.parent / "input" / "External_Data28782878",
    ):
        if candidate.exists():
            return candidate.resolve()
    return None


def declared_project_root() -> Path:
    return DECLARED_PROJECT_ROOT


def declared_input_root() -> Path:
    return DECLARED_INPUT_ROOT


def declared_external_data_root() -> Path:
    return DECLARED_EXTERNAL_DATA_ROOT


def report_assets_root(project_root_or_report_dir: str | Path) -> Path:
    base = Path(project_root_or_report_dir)
    if base.name == "06_report_assets":
        return base
    return base / "06_report_assets"


def report_asset_category(file_name: str | Path) -> str:
    name = Path(file_name).name
    if name.startswith(("stage_", "vulnerability_", "risk_", "response_", "country_response_", "abc_")):
        return "global_abcd"
    if name.startswith(("policy_", "module_d_", "who_ncd_")):
        return "module_d_policy"
    if name.startswith("china_"):
        return "china_mapping"
    if name.startswith(("report_figure_", "visualization_", "visualization_", "naturalearth_")):
        return "report_manifest"
    if name.startswith((
        "report_logic_",
        "result_contamination_",
        "full_repo_",
        "six_item_",
        "cleaning_",
        "numeric_",
        "year_continuity_",
        "source_",
        "dbd_",
        "gbd_",
        "nbs_",
        "wdi_",
        "wb_hnp_",
        "external_",
        "panel_missingness_",
        "country_iso3_",
        "global_source_",
        "global_panel_",
        "phase123_",
        "analysis_panels_",
    )):
        return "audit_qc"
    return "audit_qc"


def report_asset_dir(project_root_or_report_dir: str | Path, category: str) -> Path:
    root = report_assets_root(project_root_or_report_dir)
    subdir = REPORT_ASSET_SUBDIRS[category]
    path = root / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_asset_path(project_root_or_report_dir: str | Path, file_name: str | Path) -> Path:
    name = Path(file_name).name
    path = report_asset_dir(project_root_or_report_dir, report_asset_category(name)) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def report_asset_glob(project_root_or_report_dir: str | Path, pattern: str) -> list[Path]:
    root = report_assets_root(project_root_or_report_dir)
    matches = list(root.glob(pattern))
    for subdir in REPORT_ASSET_SUBDIRS.values():
        matches.extend((root / subdir).glob(pattern))
    return sorted({path.resolve(): path for path in matches}.values(), key=lambda p: p.as_posix())


def ensure_report_asset_dirs(project_root_or_report_dir: str | Path) -> dict[str, Path]:
    root = report_assets_root(project_root_or_report_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {key: report_asset_dir(root, key) for key in REPORT_ASSET_SUBDIRS}


def clean_data_category(file_name: str) -> str:
    if file_name.startswith("cleaned_"):
        return "core_cleaned_panel"
    if file_name.startswith(("external_china_", "external_nbs_", "external_nhsa_", "external_chinapm25_")):
        return "china_mapping_clean_cache"
    if file_name.startswith((
        "external_ncd",
        "external_who",
        "external_mpower",
        "external_m_",
        "external_policy_d4",
        "external_wdi_tobacco",
        "external_smoking",
        "external_tobacco",
    )):
        return "module_d_policy_clean_cache"
    if file_name.startswith((
        "external_gbd",
        "external_dbd",
        "external_hdi",
        "external_hdr",
        "external_un",
        "external_wpp",
        "external_country",
    )):
        return "global_reference_clean_cache"
    return "other_external_clean_cache"


def collect_manifest_coverage(audit_dir: Path) -> dict[str, set[str]]:
    manifest_files = [
        "external_manifest.csv",
        "gbd_manifest.csv",
        "dbd_manifest.csv",
        "wdi_manifest.csv",
        "wb_hnp_manifest.csv",
        "nbs_manifest.csv",
        "numeric_cleaning_manifest.csv",
    ]
    coverage: dict[str, set[str]] = {}
    for manifest_name in manifest_files:
        manifest_path = audit_dir / manifest_name
        if not manifest_path.exists() or manifest_path.stat().st_size == 0:
            continue
        try:
            manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", low_memory=False)
        except Exception:
            continue
        for column in ("output_file", "output_path"):
            if column not in manifest.columns:
                continue
            for value in manifest[column].dropna().astype(str):
                if not value.endswith(".csv"):
                    continue
                coverage.setdefault(Path(value).name, set()).add(manifest_name)
    return coverage


def csv_shape_light(path: Path) -> tuple[int | None, int | None]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            row_count = sum(1 for _ in reader)
        return row_count, len(header or [])
    except Exception:
        return None, None


def write_clean_data_manifest(project_root_or_report_dir: str | Path) -> None:
    root = report_assets_root(project_root_or_report_dir)
    project_root = root.parent
    clean_dir = project_root / "09_data_clean"
    if not clean_dir.exists():
        return
    audit_dir = report_asset_dir(root, "audit_qc")
    coverage = collect_manifest_coverage(audit_dir)
    direct_module_outputs = {
        "external_china_nhsa_payment_policy_timeline.csv": "e_china_mapping.py:China NHSA DRG/DIP policy timeline",
        "external_china_scidb_city_pm25_2000_2024.csv": "e_china_mapping.py:ScienceDB city PM2.5 clean cache",
    }
    rows = []
    for path in sorted(clean_dir.glob("*.csv")):
        row_count, column_count = csv_shape_light(path)
        coverage_sources = sorted(coverage.get(path.name, set()))
        producer_note = ""
        manifest_status = "recorded_in_existing_manifest" if coverage_sources else "unrecorded"
        if not coverage_sources and path.name in direct_module_outputs:
            manifest_status = "direct_module_output_recorded_here"
            producer_note = direct_module_outputs[path.name]
        rows.append({
            "file_name": path.name,
            "relative_path": path.relative_to(project_root).as_posix(),
            "category": clean_data_category(path.name),
            "size_bytes": path.stat().st_size,
            "row_count": row_count,
            "column_count": column_count,
            "manifest_status": manifest_status,
            "source_manifest_files": ";".join(coverage_sources),
            "producer_note": producer_note,
        })
    out_csv = audit_dir / "clean_data_file_manifest.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "file_name",
            "relative_path",
            "category",
            "size_bytes",
            "row_count",
            "column_count",
            "manifest_status",
            "source_manifest_files",
            "producer_note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    status_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["manifest_status"])] = status_counts.get(str(row["manifest_status"]), 0) + 1
        category_counts[str(row["category"])] = category_counts.get(str(row["category"]), 0) + 1
    lines = [
        "# 09_data_clean 清洗数据索引",
        "",
        "`09_data_clean` 只保存由 `01_main` 流程读取、清洗、结构化或缓存的 CSV，不保存原始 PDF、ZIP、图片和报告结果。",
        "",
        f"当前 CSV 文件数：{len(rows)}",
        "",
        "## Manifest 状态",
        "",
        "| 状态 | 文件数 |",
        "|---|---:|",
        *[f"| `{key}` | {status_counts[key]} |" for key in sorted(status_counts)],
        "",
        "## 类型分布",
        "",
        "| 类型 | 文件数 |",
        "|---|---:|",
        *[f"| `{key}` | {category_counts[key]} |" for key in sorted(category_counts)],
        "",
        "完整清单见 `clean_data_file_manifest.csv`。",
    ]
    (audit_dir / "09_data_clean目录索引.md").write_text("\n".join(lines), encoding="utf-8")


def write_report_assets_index(project_root_or_report_dir: str | Path) -> None:
    root = report_assets_root(project_root_or_report_dir)
    ensure_report_asset_dirs(root)
    write_clean_data_manifest(root)
    rows = []
    for category, subdir in REPORT_ASSET_SUBDIRS.items():
        folder = root / subdir
        for path in sorted(folder.iterdir() if folder.exists() else []):
            if not path.is_file() or path.name.startswith("."):
                continue
            rows.append({
                "category": category,
                "subdir": subdir,
                "file_name": path.name,
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
            })
    index_dir = report_asset_dir(root, "index")
    csv_path = index_dir / "core_result_manifest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "subdir", "file_name", "relative_path", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 06_report_assets 目录索引",
        "",
        "本目录只保存报告和答辩可引用的结果表、证据表、审计表和清单文件。",
        "",
        "| 子目录 | 内容 |",
        "|---|---|",
        "| `a_index` | 目录索引和核心结果清单 |",
        "| `b_global_abcd` | 全球 A/B/C/D 前置结果：阶段、脆弱性、风险归因、响应失配和预测验证 |",
        "| `c_module_d_policy` | 模块 D 政策适配、准因果增强、多政策路径组合、证据裁判与锁定材料 |",
        "| `d_china_mapping` | 中国省级映射、政策响应、PM2.5、人口、疾病负担和边界材料 |",
        "| `e_audit_qc` | 数据源、清洗、缺失、年份、逻辑、污染和仓库审计 |",
        "| `f_report_manifest` | 报告图表清单、可视化字段字典和展示材料清单 |",
        "",
        f"当前索引文件数：{len(rows)}",
        "",
    ]
    (index_dir / "06_report_assets目录索引.md").write_text("\n".join(lines), encoding="utf-8")


def detect_project_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    override = _resolve_override(os.environ.get(RUNTIME_PROJECT_ROOT_ENV))
    if override is not None:
        return override
    if DECLARED_PROJECT_ROOT.exists():
        return DECLARED_PROJECT_ROOT.resolve()
    discovered = _discover_project_root()
    if discovered is not None:
        return discovered
    return DECLARED_PROJECT_ROOT


def detect_input_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser()
        if candidate != DECLARED_PROJECT_ROOT and not _looks_like_project_root(candidate):
            return candidate
    override = _resolve_override(os.environ.get(RUNTIME_INPUT_ROOT_ENV))
    if override is not None:
        return override
    if DECLARED_INPUT_ROOT.exists():
        return DECLARED_INPUT_ROOT.resolve()
    explicit_project_root = None
    if explicit is not None:
        candidate = explicit.expanduser()
        if candidate == DECLARED_PROJECT_ROOT or _looks_like_project_root(candidate):
            explicit_project_root = candidate
    project_root = detect_project_root(explicit_project_root)
    discovered = _discover_input_root(project_root)
    if discovered is not None:
        return discovered
    return DECLARED_INPUT_ROOT


def detect_external_data_root(explicit: Path | None = None, *, project_root: Path | None = None, input_root: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    override = _resolve_override(os.environ.get(RUNTIME_EXTERNAL_DATA_ROOT_ENV))
    if override is not None:
        return override
    if DECLARED_EXTERNAL_DATA_ROOT.exists():
        return DECLARED_EXTERNAL_DATA_ROOT.resolve()
    resolved_project_root = project_root.expanduser() if project_root is not None else detect_project_root()
    resolved_input_root = input_root.expanduser() if input_root is not None else detect_input_root(resolved_project_root)
    discovered = _discover_external_data_root(resolved_project_root, resolved_input_root)
    if discovered is not None:
        return discovered
    return DECLARED_EXTERNAL_DATA_ROOT


@lru_cache(maxsize=1)
def get_preferred_cjk_font() -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in CHINESE_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return None


def configure_matplotlib_fonts() -> bool:
    preferred = get_preferred_cjk_font()
    if preferred:
        matplotlib.rcParams["font.sans-serif"] = [preferred, "DejaVu Sans", "Arial", "sans-serif"]
    else:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams.update({
        "figure.titlesize": REPORT_FIGURE_TITLE_SIZE,
        "axes.titlesize": REPORT_AXIS_TITLE_SIZE,
        "axes.titlelocation": "center",
        "axes.labelsize": REPORT_AXIS_LABEL_SIZE,
        "xtick.labelsize": REPORT_TICK_LABEL_SIZE,
        "ytick.labelsize": REPORT_TICK_LABEL_SIZE,
        "legend.fontsize": REPORT_LEGEND_SIZE,
        "legend.title_fontsize": REPORT_LEGEND_SIZE,
    })
    return preferred is not None


def choose_text(chinese: str, english: str, use_chinese: bool) -> str:
    return chinese if use_chinese else english


def set_centered_title(ax, title: str, *, fontsize: int = REPORT_AXIS_TITLE_SIZE, pad: int = 12) -> None:
    ax.set_title(title, fontsize=fontsize, loc="center", pad=pad)


def set_centered_suptitle(fig, title: str, *, fontsize: int = REPORT_FIGURE_TITLE_SIZE, y: float = 0.98) -> None:
    fig.suptitle(title, fontsize=fontsize, x=0.5, y=y, ha="center")


@lru_cache(maxsize=8)
def load_country_zh_labels(project_root: str | Path) -> dict[str, str]:
    external_data_root = detect_external_data_root(project_root=Path(project_root))
    alias_path = external_data_root / "16_Project_Metadata_Registry" / "country_alias_zh.csv"
    if not alias_path.exists():
        return {}
    alias = pd.read_csv(alias_path, encoding="utf-8-sig")
    if not {"country_name_zh", "iso3"}.issubset(alias.columns):
        return {}
    alias["iso3"] = alias["iso3"].astype(str).str.upper().str.strip()
    alias["country_name_zh"] = alias["country_name_zh"].astype(str).str.strip()
    labels: dict[str, str] = {}
    for iso3, group in alias.dropna(subset=["iso3", "country_name_zh"]).groupby("iso3", sort=False):
        code = str(iso3).upper()
        override = COUNTRY_LABEL_OVERRIDES.get(code)
        if override:
            labels[code] = override
            continue
        names = [name for name in group["country_name_zh"].drop_duplicates().tolist() if name and name.lower() != "nan"]
        if names:
            labels[code] = sorted(names, key=len)[0]
    return labels


def country_display_name(iso3: object, labels: dict[str, str]) -> str:
    code = str(iso3).upper().strip()
    return labels.get(code, code)


def register_support_modules() -> None:
    current = sys.modules[__name__]
    sys.modules.setdefault("a_foundation", current)
    sys.modules.setdefault("foundation", current)
    sys.modules.setdefault("project_paths", current)
    sys.modules.setdefault("plot_style", current)


@contextlib.contextmanager
def runtime_context(script_name: str, args: Sequence[str] | None, project_root: Path):
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    old_project = os.environ.get(RUNTIME_PROJECT_ROOT_ENV)
    try:
        os.chdir(project_root)
        os.environ[RUNTIME_PROJECT_ROOT_ENV] = project_root.as_posix()
        sys.argv = [script_name, *(args or [])]
        yield
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        if old_project is None:
            os.environ.pop(RUNTIME_PROJECT_ROOT_ENV, None)
        else:
            os.environ[RUNTIME_PROJECT_ROOT_ENV] = old_project


def namespace_module_name(script_name: str) -> str:
    return Path(script_name).stem


def materialize_namespaces(builders: Mapping[str, Callable[[], dict[str, object]]]) -> dict[str, types.ModuleType]:
    register_support_modules()
    modules: dict[str, types.ModuleType] = {}
    for script_name, builder in builders.items():
        module_name = namespace_module_name(script_name)
        module = sys.modules.get(module_name)
        if module is None or not getattr(module, "__five_file_namespace__", False):
            namespace = builder()
            module = types.ModuleType(module_name)
            module.__dict__.update(namespace)
            module.__name__ = module_name
            module.__file__ = str(detect_project_root() / "01_main" / script_name)
            module.__five_file_namespace__ = True
            sys.modules[module_name] = module
        modules[script_name] = module
    return modules


def sanitize_generated_text_paths(project_root: Path) -> None:
    """Normalize generated text artifacts to declared Heywhale paths.

    Analysis scripts run locally with local paths, but the deliverable package is
    intended to be read on Heywhale. Keep code execution local-friendly while
    preventing regenerated CSV/JSON/MD manifests from leaking workstation paths.
    """
    local_project = project_root.resolve().as_posix()
    local_parent = project_root.resolve().parent.as_posix()
    local_input = (project_root.resolve().parent / "input").as_posix()
    local_external = (project_root.resolve().parent / "external_data").as_posix()
    replacements = [
        (local_project, DECLARED_PROJECT_ROOT.as_posix()),
        (local_external, DECLARED_EXTERNAL_DATA_ROOT.as_posix()),
        (local_input + "/External_Data28782878/External Data", DECLARED_EXTERNAL_DATA_ROOT.as_posix()),
        (local_input, DECLARED_INPUT_ROOT.as_posix()),
        (local_parent, DECLARED_PROJECT_ROOT.parent.as_posix()),
    ]
    text_roots = [
        project_root / "02_docs",
        project_root / "03_ai_logs",
        project_root / "06_report_assets",
        project_root / "08_data_inventory",
        project_root / "09_data_clean",
    ]
    text_files = [project_root / "README.md", project_root / "run_order.md", project_root / "requirements.txt"]
    skip_suffixes = {".png", ".jpg", ".jpeg", ".pdf", ".pptx", ".zip", ".mp4", ".mov", ".xlsx", ".pyc"}
    for text_root in text_roots:
        if text_root.exists():
            text_files.extend(path for path in text_root.rglob("*") if path.is_file())
    for path in text_files:
        if not path.exists() or path.suffix.lower() in skip_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        original = text
        for old, new in replacements:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")


def run_step_sequence(steps: Iterable[tuple[str, Sequence[str]]], builders: Mapping[str, Callable[[], dict[str, object]]], *, project_root: Path | None = None) -> None:
    root = detect_project_root(project_root)
    modules = materialize_namespaces(builders)
    for script_name, args in steps:
        module = modules[script_name]
        main = getattr(module, "main", None)
        if main is None:
            raise RuntimeError(f"Step has no main(): {script_name}")
        print("+ " + script_name + (" " + " ".join(args or []) if args else ""), flush=True)
        with runtime_context(script_name, args, root):
            main()
    write_report_assets_index(root)
    sanitize_generated_text_paths(root)

def _namespace_generate_country_alias_zh():
    __name__ = 'generate_country_alias_zh'
    import csv
    from collections import OrderedDict
    from pathlib import Path

    from foundation import detect_external_data_root, detect_project_root


    EXTRA_ALIASES = [
        ("美国", "USA"),
        ("美利坚合众国", "USA"),
        ("英国", "GBR"),
        ("俄罗斯", "RUS"),
        ("韩国", "KOR"),
        ("大韩民国", "KOR"),
        ("朝鲜", "PRK"),
        ("伊朗", "IRN"),
        ("叙利亚", "SYR"),
        ("阿拉伯叙利亚共和国", "SYR"),
        ("越南", "VNM"),
        ("老挝", "LAO"),
        ("缅甸", "MMR"),
        ("文莱", "BRN"),
        ("文莱达鲁萨兰国", "BRN"),
        ("刚果（金）", "COD"),
        ("刚果（布）", "COG"),
        ("刚果", "COG"),
        ("刚果民主共和国", "COD"),
        ("刚果共和国", "COG"),
        ("科特迪瓦", "CIV"),
        ("科特廸亚", "CIV"),
        ("捷克", "CZE"),
        ("捷克共和国", "CZE"),
        ("斯洛伐克", "SVK"),
        ("玻利维亚", "BOL"),
        ("玻利维亚国", "BOL"),
        ("委内瑞拉", "VEN"),
        ("坦桑尼亚", "TZA"),
        ("摩尔多瓦", "MDA"),
        ("巴勒斯坦", "PSE"),
        ("东帝汶民主共和国", "TLS"),
        ("伯利兹城", "BLZ"),
        ("卢森堡公国", "LUX"),
        ("厄立特里亚国", "ERI"),
        ("圣卢西亚岛", "LCA"),
        ("圣基茨和尼维斯联邦", "KNA"),
        ("圣多美和普林西比民主共和国", "STP"),
        ("圣马力诺共和国", "SMR"),
        ("多米尼加岛", "DMA"),
        ("安道尔共和国", "AND"),
        ("密克罗尼西亚联邦", "FSM"),
        ("巴林岛", "BHR"),
        ("帕劳共和国", "PLW"),
        ("摩纳哥公国", "MCO"),
        ("特立尼达拉岛和多巴哥", "TTO"),
        ("瑙鲁共和国", "NRU"),
        ("马尔他", "MLT"),
        ("黑山共和国", "MNE"),
        ("老挝人民民主共和国", "LAO"),
        ("朝鲜民主主义人民共和国", "PRK"),
        ("伊朗伊斯兰共和国", "IRN"),
        ("委内瑞拉玻利瓦尔共和国", "VEN"),
        ("叙利亚阿拉伯共和国", "SYR"),
        ("坦桑尼亚联合共和国", "TZA"),
        ("玻利维亚多民族国", "BOL"),
        ("摩尔多瓦共和国", "MDA"),
        ("中国香港特别行政区", "HKG"),
        ("中国澳门特别行政区", "MAC"),
        ("中国台湾", "TWN"),
        ("台湾", "TWN"),
        ("香港", "HKG"),
        ("澳门", "MAC"),
    ]
    def main() -> None:
        project_root = detect_project_root()
        external_data_root = detect_external_data_root(project_root=project_root)
        inventory_dir = external_data_root / "16_Project_Metadata_Registry"
        output_path = inventory_dir / "country_alias_zh.csv"
        if output_path.exists() and output_path.stat().st_size > 0:
            with output_path.open("r", encoding="utf-8-sig", newline="") as fh:
                row_count = max(0, sum(1 for _ in csv.DictReader(fh)))
            print(output_path.as_posix())
            print(row_count)
            return

        inventory_dir.mkdir(parents=True, exist_ok=True)
        lookup: "OrderedDict[str, str]" = OrderedDict()
        for name, code3 in EXTRA_ALIASES:
            lookup[name] = code3
        output_path.write_text(
            "country_name_zh,iso3\n" + "\n".join(f"{name},{code3}" for name, code3 in lookup.items()) + "\n",
            encoding="utf-8",
        )
        print(output_path.as_posix())
        print(len(lookup))


    if __name__ == "__main__":
        main()

    return locals()


NAMESPACE_BUILDERS = {"generate_country_alias_zh.py": _namespace_generate_country_alias_zh}


def check_paths(project_root: Path | None = None) -> dict[str, object]:
    root = detect_project_root(project_root)
    input_root = detect_input_root(root)
    external_data_root = detect_external_data_root(project_root=root, input_root=input_root)
    return {
        "declared_project_root": declared_project_root().as_posix(),
        "declared_input_root": declared_input_root().as_posix(),
        "declared_external_data_root": declared_external_data_root().as_posix(),
        "detected_project_root": root.as_posix(),
        "detected_input_root": input_root.as_posix(),
        "detected_external_data_root": external_data_root.as_posix(),
        "project_root_exists": root.exists(),
        "input_root_exists": input_root.exists(),
        "external_data_root_exists": external_data_root.exists(),
        "main_python_files": sorted(path.name for path in (root / "01_main").glob("*.py")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Foundation utilities for Thunder Medical Team project.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Print project/input path diagnostics.")
    parser.add_argument("--generate-country-alias", action="store_true", help="Rebuild country_alias_zh.csv from project metadata.")
    args = parser.parse_args()
    root = detect_project_root(args.project_root)
    if args.generate_country_alias:
        run_step_sequence([("generate_country_alias_zh.py", [])], NAMESPACE_BUILDERS, project_root=root)
    if args.check or not args.generate_country_alias:
        print(json.dumps(check_paths(root), ensure_ascii=False, indent=2))


register_support_modules()

if __name__ == "__main__":
    main()
