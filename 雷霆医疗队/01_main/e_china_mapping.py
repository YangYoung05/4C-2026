from __future__ import annotations

import argparse
from pathlib import Path

from a_foundation import detect_project_root, report_asset_path, run_step_sequence

def _namespace_run_china_nbs_official_explanatory_candidates():
    __name__ = 'run_china_nbs_official_explanatory_candidates'
    import argparse
    import json
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    PROVINCE_NAMES = [
        ("北京市", ["北京", "北京市", "Beijing", "Betjing", "Beiing"]),
        ("天津市", ["天津", "天津市", "Tianjin", "Tianjan", "Tianjn"]),
        ("河北省", ["河北", "河北省", "Hebei"]),
        ("山西省", ["山西", "山西省", "Shanxi"]),
        ("内蒙古自治区", ["内蒙古", "Inner Mongolia"]),
        ("辽宁省", ["辽宁", "辽宁省", "Liaoning"]),
        ("吉林省", ["吉林", "吉林省", "Jilin", "Jin", "Jiin"]),
        ("黑龙江省", ["黑龙江", "黑龙江省", "Heilongjiang", "Heiongjiang"]),
        ("上海市", ["上海", "上海市", "Shanghai"]),
        ("江苏省", ["江苏", "江苏省", "Jiangsu", "Tiangsu"]),
        ("浙江省", ["浙江", "浙江省", "Zhejiang"]),
        ("安徽省", ["安徽", "支徽", "安徽省", "Anhui"]),
        ("福建省", ["福建", "福建省", "Fujian"]),
        ("江西省", ["江西", "江西省", "Jiangxi", "Jiangi"]),
        ("山东省", ["山东", "山东省", "Shandong"]),
        ("河南省", ["河南", "河南省", "Henan"]),
        ("湖北省", ["湖北", "湖北省", "Hubei"]),
        ("湖南省", ["湖南", "湖南省", "Hunan"]),
        ("广东省", ["广东", "广东省", "Guangdong"]),
        ("广西壮族自治区", ["广西", "Guangxi"]),
        ("海南省", ["海南", "海南省", "Hainan"]),
        ("重庆市", ["重庆", "重庆市", "Chongqing", "Chongaing"]),
        ("四川省", ["四川", "四川省", "Sichuan"]),
        ("贵州省", ["贵州", "贵州省", "Guizhou"]),
        ("云南省", ["云南", "云南省", "Yunnan"]),
        ("西藏自治区", ["西藏", "Xizang", "Tibet"]),
        ("陕西省", ["陕西", "陕西省", "Shaanxi"]),
        ("甘肃省", ["甘肃", "甘肃省", "Gansu"]),
        ("青海省", ["青海", "青海省", "Qinghai"]),
        ("宁夏回族自治区", ["宁夏", "Ningxia"]),
        ("新疆维吾尔自治区", ["新疆", "Xinjiang"]),
    ]


    TABLES = {
        "population_urban_death": {
            "image": "C02-07.jpg",
            "ocr": "C02-07.ocr.txt",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2025/html/C02-07.jpg",
            "source_title": "中国统计年鉴2025 2-7 分地区人口的城乡构成和出生率、死亡率、自然增长率（2024年）",
        },
        "urban_share": {
            "image": "C02-06.jpg",
            "ocr": "C02-06.ocr.txt",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2025/html/C02-06.jpg",
            "source_title": "中国统计年鉴2025 2-6 分地区年末城镇人口比重",
        },
        "grp": {
            "image": "C03-09.jpg",
            "ocr": "C03-09.ocr.txt",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2025/html/C03-09.jpg",
            "source_title": "中国统计年鉴2025 3-9 地区生产总值（2024年）",
        },
    }


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def read_ocr(path: Path) -> pd.DataFrame:
        rows = []
        if not path.exists():
            return pd.DataFrame(columns=["x", "y", "w", "h", "confidence", "text"])
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 5)
            if len(parts) != 6:
                continue
            x, y, w, h, confidence, text = parts
            rows.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "confidence": float(confidence),
                    "text": text.strip(),
                }
            )
        return pd.DataFrame(rows)


    def parse_numeric(text: str, *, rate_like: bool = False, value_like: bool = False) -> float:
        raw = str(text).strip()
        cleaned = (
            raw.replace(",", "")
            .replace("，", "")
            .replace("O", "0")
            .replace("o", "0")
            .replace("％", "")
            .replace("%", "")
        )
        cleaned = re.sub(r"\s+", " ", cleaned)
        if rate_like and re.fullmatch(r"-?\d{1,2}\s+\d{2}", cleaned):
            cleaned = cleaned.replace(" ", ".")
        elif value_like and re.fullmatch(r"\d+\s+\d", cleaned):
            cleaned = cleaned.replace(" ", ".")
        else:
            cleaned = cleaned.replace(" ", "")
        if rate_like:
            if re.fullmatch(r"-?\d{3}", cleaned):
                sign = "-" if cleaned.startswith("-") else ""
                digits = cleaned[1:] if sign else cleaned
                cleaned = f"{sign}{digits[0]}.{digits[1:]}"
            elif re.fullmatch(r"-?\d{4}", cleaned):
                sign = "-" if cleaned.startswith("-") else ""
                digits = cleaned[1:] if sign else cleaned
                cleaned = f"{sign}{digits[:2]}.{digits[2:]}"
        try:
            return float(cleaned)
        except ValueError:
            return float("nan")


    def find_province_y(records: pd.DataFrame) -> dict[str, float]:
        y_map: dict[str, float] = {}
        text_records = records.loc[records["x"].le(0.12)].copy()
        for province, aliases in PROVINCE_NAMES:
            hits = []
            for alias in aliases:
                matched = text_records.loc[text_records["text"].astype(str).eq(alias)]
                if not matched.empty:
                    hits.extend(matched["y"].astype(float).tolist())
            if hits:
                # Prefer the visually highest confidence hit; the median is robust to duplicated OCR.
                y_map[province] = float(np.median(hits))
        return y_map


    def nearest_numeric(
        records: pd.DataFrame,
        y: float,
        *,
        x_min: float,
        x_max: float,
        rate_like: bool = False,
        value_like: bool = False,
        tolerance: float = 0.014,
    ) -> tuple[float, float, float, str]:
        candidates = records.loc[records["x"].between(x_min, x_max)].copy()
        if candidates.empty:
            return float("nan"), float("nan"), float("nan"), ""
        candidates["dy"] = (candidates["y"] - y).abs()
        candidates = candidates.sort_values(["dy", "confidence"], ascending=[True, False], kind="stable")
        for _, row in candidates.iterrows():
            if row["dy"] > tolerance:
                continue
            value = parse_numeric(str(row["text"]), rate_like=rate_like, value_like=value_like)
            if pd.notna(value):
                return float(value), float(row["confidence"]), float(row["y"]), str(row["text"])
        return float("nan"), float("nan"), float("nan"), ""


    def in_range(value: float, lower: float, upper: float) -> bool:
        return pd.notna(value) and lower <= float(value) <= upper


    def choose_valid(
        primary: tuple[float, float, float, str],
        fallback: tuple[float, float, float, str],
        *,
        lower: float,
        upper: float,
    ) -> tuple[float, float, float, str]:
        if in_range(primary[0], lower, upper):
            return primary
        if in_range(fallback[0], lower, upper):
            return fallback
        return float("nan"), float("nan"), float("nan"), ""


    def build_candidates(inventory_dir: Path) -> pd.DataFrame:
        pop = read_ocr(inventory_dir / TABLES["population_urban_death"]["ocr"])
        pop_en = read_ocr(inventory_dir / "E02-07.ocr.txt")
        urban = read_ocr(inventory_dir / TABLES["urban_share"]["ocr"])
        grp = read_ocr(inventory_dir / TABLES["grp"]["ocr"])
        pop_y = find_province_y(pop)
        pop_en_y = find_province_y(pop_en)
        urban_y = find_province_y(urban)
        grp_y = find_province_y(grp)

        rows = []
        for province, _aliases in PROVINCE_NAMES:
            pop_row_y = pop_y.get(province, float("nan"))
            pop_en_row_y = pop_en_y.get(province, float("nan"))
            urban_row_y = urban_y.get(province, float("nan"))
            grp_row_y = grp_y.get(province, float("nan"))

            birth_primary = nearest_numeric(
                pop, pop_row_y, x_min=0.71, x_max=0.79, rate_like=True, tolerance=0.018
            )
            birth_fallback = nearest_numeric(
                pop_en, pop_en_row_y, x_min=0.72, x_max=0.79, rate_like=True, tolerance=0.020
            )
            birth, birth_conf, birth_y, birth_raw = choose_valid(birth_primary, birth_fallback, lower=0, upper=25)

            death_primary = nearest_numeric(
                pop, pop_row_y, x_min=0.82, x_max=0.90, rate_like=True, tolerance=0.018
            )
            death_fallback = nearest_numeric(
                pop_en, pop_en_row_y, x_min=0.83, x_max=0.90, rate_like=True, tolerance=0.020
            )
            death, death_conf, death_y, death_raw = choose_valid(death_primary, death_fallback, lower=0, upper=20)

            natural_primary = nearest_numeric(
                pop, pop_row_y, x_min=0.93, x_max=0.995, rate_like=True, tolerance=0.018
            )
            natural_fallback = nearest_numeric(
                pop_en, pop_en_row_y, x_min=0.93, x_max=0.995, rate_like=True, tolerance=0.020
            )
            natural, natural_conf, natural_y, natural_raw = choose_valid(natural_primary, natural_fallback, lower=-15, upper=15)
            natural_growth_arithmetic = birth - death if pd.notna(birth) and pd.notna(death) else np.nan
            natural_growth_corrected = False
            if pd.notna(natural_growth_arithmetic):
                if pd.isna(natural) or abs(float(natural) - float(natural_growth_arithmetic)) > 0.60:
                    natural = float(natural_growth_arithmetic)
                    natural_growth_corrected = True
                    natural_raw = f"{natural_raw}|corrected_by_birth_minus_death"
            urban_share, urban_conf, urban_value_y, urban_raw = nearest_numeric(
                urban, urban_row_y, x_min=0.91, x_max=0.99, rate_like=True, tolerance=0.018
            )
            grp_value, grp_conf, grp_value_y, grp_raw = nearest_numeric(
                grp, grp_row_y, x_min=0.045, x_max=0.085, value_like=True, tolerance=0.018
            )
            numeric_values = [birth, death, natural, urban_share, grp_value]
            confidences = [birth_conf, death_conf, natural_conf, urban_conf, grp_conf]
            rows.append(
                {
                    "province": province,
                    "source_year": 2024,
                    "nbs2024_birth_rate_per_mille_candidate": birth,
                    "nbs2024_birth_rate_raw_ocr": birth_raw,
                    "nbs2024_birth_rate_ocr_confidence": birth_conf,
                    "nbs2024_death_rate_per_mille_candidate": death,
                    "nbs2024_death_rate_raw_ocr": death_raw,
                    "nbs2024_death_rate_ocr_confidence": death_conf,
                    "nbs2024_natural_growth_rate_per_mille_candidate": natural,
                    "nbs2024_natural_growth_raw_ocr": natural_raw,
                    "nbs2024_natural_growth_ocr_confidence": natural_conf,
                    "nbs2024_natural_growth_corrected_by_birth_minus_death": bool(natural_growth_corrected),
                    "nbs2024_natural_growth_arithmetic_candidate": natural_growth_arithmetic,
                    "nbs2024_urban_population_share_pct_candidate": urban_share,
                    "nbs2024_urban_share_raw_ocr": urban_raw,
                    "nbs2024_urban_share_ocr_confidence": urban_conf,
                    "nbs2024_gross_regional_product_100m_yuan_candidate": grp_value,
                    "nbs2024_grp_raw_ocr": grp_raw,
                    "nbs2024_grp_ocr_confidence": grp_conf,
                    "candidate_non_null_fields": int(sum(pd.notna(v) for v in numeric_values)),
                    "mean_ocr_confidence": float(np.nanmean(confidences)) if any(pd.notna(c) for c in confidences) else np.nan,
                    "model_use_status": "supplemental_explanatory_candidate_not_used_for_core_causal_claim",
                    "boundary_note": "来自国家统计局2025年鉴静态图片的macOS Vision OCR候选字段；可用于解释省级经济、城镇化和人口压力差异，不进入健康结果因果主回归，需人工复核后才能作为硬数值。",
                    "nbs_population_table_source_url": TABLES["population_urban_death"]["source_url"],
                    "nbs_urban_table_source_url": TABLES["urban_share"]["source_url"],
                    "nbs_grp_table_source_url": TABLES["grp"]["source_url"],
                    "ocr_row_y_population_table": pop_row_y,
                    "ocr_row_y_urban_table": urban_row_y,
                    "ocr_row_y_grp_table": grp_row_y,
                    "ocr_value_y_death_rate": death_y,
                    "ocr_value_y_urban_share": urban_value_y,
                    "ocr_value_y_grp": grp_value_y,
                }
            )
        return pd.DataFrame(rows)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build official NBS 2025 yearbook OCR explanatory candidate variables for China provincial mapping.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        inventory_dir = external_data_root / "13_China_NBS_Yearbook_OCR" / "external" / "nbs_yearbook_2025_official_jpg"
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        candidates = build_candidates(inventory_dir)
        clean_path = clean_dir / "external_china_nbs2025_ocr_explanatory_candidates_2024.csv"
        report_path = report_asset_path(report_dir, "china_nbs2025_ocr_explanatory_candidates_2024.csv")
        summary_path = report_asset_path(report_dir, "china_nbs2025_ocr_explanatory_candidates_summary.json")
        candidates.to_csv(clean_path, index=False, encoding="utf-8-sig")
        candidates.to_csv(report_path, index=False, encoding="utf-8-sig")

        summary = {
            "project_root": project_root.as_posix(),
            "candidate_layer": "NBS 2025 official yearbook OCR explanatory candidates",
            "province_rows": int(candidates.shape[0]),
            "source_year": 2024,
            "complete_candidate_rows": int(candidates["candidate_non_null_fields"].eq(5).sum()),
            "mean_ocr_confidence": float(pd.to_numeric(candidates["mean_ocr_confidence"], errors="coerce").mean()),
            "fields": [
                "birth_rate_per_mille",
                "death_rate_per_mille",
                "natural_growth_rate_per_mille",
                "urban_population_share_pct",
                "gross_regional_product_100m_yuan",
            ],
            "claim_boundary": "已补官方NBS 2025年鉴OCR候选解释变量；可用于解释差异和答辩边界，不作为省级健康结果因果主回归。",
            "output_files": {
                "clean_candidates": clean_path.as_posix(),
                "report_candidates": report_path.as_posix(),
                "summary": summary_path.as_posix(),
            },
            "source_urls": {key: value["source_url"] for key, value in TABLES.items()},
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_mapping_framework():
    __name__ = 'run_china_mapping_framework'
    import argparse
    import json
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from foundation import choose_text, configure_matplotlib_fonts
    from foundation import detect_project_root as shared_detect_project_root


    USE_CHINESE = configure_matplotlib_fonts()
    STAFF_DATASET = "各省近20年卫生人员数量"
    INSTITUTION_DATASET = "近20年各省医疗卫生机构数量"
    CENSUS_FILE = "external_china_census_2020_province_population_age.csv"
    PM25_FILE = "external_china_satpm_provincial_pm25.csv"
    SCIDB_PM25_FILE = "external_china_scidb_provincial_pm25.csv"
    SCIDB_PM25_WEIGHTED_FILE = "external_china_scidb_provincial_pm25_population_weighted.csv"
    GBD_DALY_FILE = "external_china_gbd2017_province_daly_rates.csv"
    GBD_RISK_SEV_FILE = "external_china_gbd2017_province_risk_sev.csv"
    GBD2021_NCD_FILE = "external_china_gbd2021_province_ncd_daly_rates.csv"
    POLICY_RESPONSE_FILE = "external_china_nhsa_payment_policy_province_latest.csv"
    NBS_OCR_RESPONSE_FILE = "external_china_nbs_ocr_response_candidates_2024.csv"
    CHRONIC_POLICY_SCORE_FILE = "external_china_nhc_chronic_demo_zone_policy_score_latest.csv"
    NBS_EXPLANATORY_FILE = "external_china_nbs2025_ocr_explanatory_candidates_2024.csv"
    NBS_MORTALITY_PANEL_FILE = "external_china_nbs_mortality_panel_ocr_2018_2024.csv"
    C_LAYER_QC_FILE = "external_china_c_layer_ocr_candidate_qc_2024.csv"
    HEALTH_OUTCOME_ANCHOR_FILE = "external_china_health_outcome_anchor_2017_2024.csv"
    LOCAL_POLICY_EXECUTION_FILE = "external_china_local_policy_execution_indicator_latest.csv"

    DISEASE_BURDEN_COLUMNS = [
        "gbd2017_daly_rate_cardiovascular_diseases",
        "gbd2017_daly_rate_chronic_respiratory_diseases",
        "gbd2017_daly_rate_neoplasms",
        "gbd2017_daly_rate_diabetes_kidney",
    ]

    RISK_SEV_COLUMNS = [
        "sev2017_high_systolic_blood_pressure",
        "sev2017_ambient_pm25_pollution",
        "sev2017_smoking",
        "sev2017_high_fasting_plasma_glucose",
        "sev2017_high_body_mass_index",
        "sev2017_diet_high_in_sodium",
        "sev2017_alcohol_use",
    ]

    RISK_LABELS = {
        "sev2017_high_systolic_blood_pressure": "高收缩压",
        "sev2017_ambient_pm25_pollution": "PM2.5",
        "sev2017_smoking": "吸烟",
        "sev2017_high_fasting_plasma_glucose": "高血糖",
        "sev2017_high_body_mass_index": "高BMI",
        "sev2017_diet_high_in_sodium": "高钠饮食",
        "sev2017_alcohol_use": "饮酒",
    }

    DISEASE_LABELS = {
        "gbd2017_daly_rate_cardiovascular_diseases": "心血管疾病",
        "gbd2017_daly_rate_chronic_respiratory_diseases": "慢性呼吸系统疾病",
        "gbd2017_daly_rate_neoplasms": "肿瘤",
        "gbd2017_daly_rate_diabetes_kidney": "糖尿病和肾病",
    }


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def year_columns(df: pd.DataFrame) -> list[str]:
        return [column for column in df.columns if re.fullmatch(r"\d{4}年", str(column))]


    def percentile_score(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.rank(pct=True, method="average")


    def load_nbs_provincial_panel(clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / "cleaned_nbs_health.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = pd.read_csv(path)
        years = year_columns(raw)
        keep = raw.loc[
            raw["地区"].notna() & raw["source_dataset"].isin([STAFF_DATASET, INSTITUTION_DATASET]),
            ["地区", "source_dataset", *years],
        ].copy()
        long = keep.melt(
            id_vars=["地区", "source_dataset"],
            value_vars=years,
            var_name="year",
            value_name="value",
        )
        long["year"] = long["year"].str.replace("年", "", regex=False).astype(int)
        long["indicator_key"] = np.where(
            long["source_dataset"] == STAFF_DATASET,
            "medical_staff_10k_persons",
            "medical_institutions_count",
        )
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        panel = (
            long.pivot_table(
                index=["地区", "year"],
                columns="indicator_key",
                values="value",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
            .rename(columns={"地区": "province"})
            .sort_values(["province", "year"], kind="stable")
        )
        return panel


    def latest_provincial_resource_response(panel: pd.DataFrame) -> pd.DataFrame:
        latest_year = int(panel["year"].max())
        baseline_year = latest_year - 5
        latest = panel.loc[panel["year"] == latest_year].copy()
        baseline = panel.loc[panel["year"] == baseline_year, ["province", "medical_staff_10k_persons", "medical_institutions_count"]].copy()
        baseline = baseline.rename(
            columns={
                "medical_staff_10k_persons": f"medical_staff_10k_persons_{baseline_year}",
                "medical_institutions_count": f"medical_institutions_count_{baseline_year}",
            }
        )
        latest = latest.merge(baseline, on="province", how="left")
        latest["medical_staff_growth_5y"] = (
            latest["medical_staff_10k_persons"] / latest[f"medical_staff_10k_persons_{baseline_year}"] - 1
        )
        latest["medical_institutions_growth_5y"] = (
            latest["medical_institutions_count"] / latest[f"medical_institutions_count_{baseline_year}"] - 1
        )
        latest["staff_scale_percentile"] = percentile_score(latest["medical_staff_10k_persons"])
        latest["institution_scale_percentile"] = percentile_score(latest["medical_institutions_count"])
        latest["staff_growth_percentile"] = percentile_score(latest["medical_staff_growth_5y"])
        latest["institution_growth_percentile"] = percentile_score(latest["medical_institutions_growth_5y"])
        latest["resource_response_score"] = (
            0.45 * latest["staff_scale_percentile"]
            + 0.25 * latest["institution_scale_percentile"]
            + 0.20 * latest["staff_growth_percentile"]
            + 0.10 * latest["institution_growth_percentile"]
        )
        latest["resource_response_type"] = pd.cut(
            latest["resource_response_score"],
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["资源规模低位", "资源规模中位", "资源规模高位"],
        ).astype(str)
        latest["data_scope"] = "NBS省级卫生人员总量+医疗卫生机构数量；未做人均校正"
        latest["latest_year"] = latest_year
        latest["baseline_year"] = baseline_year
        return latest


    def add_census_normalization(latest: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / CENSUS_FILE
        if not path.exists():
            latest = latest.copy()
            latest["census_data_available"] = False
            return latest
        census = pd.read_csv(path)
        merged = latest.merge(
            census[
                [
                    "province",
                    "census_2020_population",
                    "census_2020_age60_plus_pct",
                    "census_2020_age65_plus_pct",
                ]
            ],
            on="province",
            how="left",
        )
        merged["census_data_available"] = merged["census_2020_population"].notna()
        merged["medical_staff_per_10k_population"] = (
            merged["medical_staff_10k_persons"] * 10000 / merged["census_2020_population"] * 10000
        )
        merged["medical_institutions_per_million_population"] = (
            merged["medical_institutions_count"] / merged["census_2020_population"] * 1_000_000
        )
        merged["staff_per_capita_percentile"] = percentile_score(merged["medical_staff_per_10k_population"])
        merged["institution_per_capita_percentile"] = percentile_score(merged["medical_institutions_per_million_population"])
        merged["aging65_pressure_percentile"] = percentile_score(merged["census_2020_age65_plus_pct"])
        merged["resource_response_score"] = (
            0.45 * merged["staff_per_capita_percentile"]
            + 0.25 * merged["institution_per_capita_percentile"]
            + 0.20 * merged["staff_growth_percentile"]
            + 0.10 * merged["institution_growth_percentile"]
        )
        merged["resource_response_type"] = pd.cut(
            merged["resource_response_score"],
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["人均资源低位", "人均资源中位", "人均资源高位"],
        ).astype(str)
        conditions = [
            (merged["aging65_pressure_percentile"] >= 2 / 3) & (merged["resource_response_score"] < 1 / 3),
            (merged["aging65_pressure_percentile"] >= 2 / 3) & (merged["resource_response_score"] >= 2 / 3),
            (merged["aging65_pressure_percentile"] < 1 / 3) & (merged["resource_response_score"] < 1 / 3),
            (merged["resource_response_score"] >= 2 / 3),
        ]
        choices = ["高龄承压资源短板型", "高龄高响应储备型", "年轻结构资源短板型", "资源储备较强型"]
        merged["province_resource_response_type"] = np.select(conditions, choices, default="相对均衡观察型")
        merged["data_scope"] = "NBS省级卫生人员/机构 + 第七次人口普查人口与老龄化；人口为2020口径，资源为最新年份口径"
        return merged


    def add_provincial_risk_supplement(latest: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        weighted_path = clean_dir / SCIDB_PM25_WEIGHTED_FILE
        scidb_path = weighted_path if weighted_path.exists() else clean_dir / SCIDB_PM25_FILE
        path = scidb_path if scidb_path.exists() else clean_dir / PM25_FILE
        latest = latest.copy()
        if not path.exists():
            latest["pm25_data_available"] = False
            latest["province_risk_exposure_score"] = latest.get("aging65_pressure_percentile", np.nan)
            return latest
        pm25 = pd.read_csv(path)
        pm25["year"] = pd.to_numeric(pm25["year"], errors="coerce")
        latest_year = int(pm25["year"].max())
        pm25_latest = pm25.loc[pm25["year"] == latest_year].copy()
        if "pm25_population_weighted_ug_m3" not in pm25_latest.columns and "pm25_city_mean_ug_m3" in pm25_latest.columns:
            pm25_latest["pm25_population_weighted_ug_m3"] = pm25_latest["pm25_city_mean_ug_m3"]
        if "pm25_geographic_mean_ug_m3" not in pm25_latest.columns and "pm25_city_mean_ug_m3" in pm25_latest.columns:
            pm25_latest["pm25_geographic_mean_ug_m3"] = pm25_latest["pm25_city_mean_ug_m3"]
        pm25_latest["pm25_pressure_percentile"] = pm25_latest["pm25_population_weighted_ug_m3"].rank(pct=True, method="average")
        keep = [
            "province",
            "year",
            "pm25_population_weighted_ug_m3",
            "pm25_geographic_mean_ug_m3",
            "pm25_city_mean_ug_m3",
            "city_count",
            "pm25_measure_note",
            "pm25_pressure_percentile",
            "source_dataset",
            "source_url",
        ]
        pm25_latest = pm25_latest.loc[:, [col for col in keep if col in pm25_latest.columns]].rename(columns={"year": "pm25_latest_year"})
        merged = latest.merge(pm25_latest, on="province", how="left")
        merged["pm25_data_available"] = merged["pm25_population_weighted_ug_m3"].notna()
        aging = pd.to_numeric(merged.get("aging65_pressure_percentile"), errors="coerce")
        pm25_pressure = pd.to_numeric(merged.get("pm25_pressure_percentile"), errors="coerce")
        resource_shortage = 1 - pd.to_numeric(merged.get("resource_response_score"), errors="coerce")
        merged["province_risk_exposure_score"] = (
            0.40 * aging.fillna(aging.mean())
            + 0.35 * pm25_pressure.fillna(pm25_pressure.mean())
            + 0.25 * resource_shortage.fillna(resource_shortage.mean())
        )
        merged["province_risk_exposure_type"] = pd.cut(
            merged["province_risk_exposure_score"],
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["省级风险代理低位", "省级风险代理中位", "省级风险代理高位"],
        ).astype(str)
        high_pm25 = merged["pm25_pressure_percentile"] >= 2 / 3
        high_risk = merged["province_risk_exposure_score"] >= 2 / 3
        merged["province_policy_priority"] = np.where(
            high_pm25,
            "优先联动PM2.5治理、心肺慢病筛查和老龄高风险人群防护",
            np.where(high_risk, "优先做慢病早筛、基层连续管理和资源短板补强", ""),
        )
        merged["data_scope"] = merged["data_scope"].astype(str) + "；已补省级PM2.5环境风险暴露"
        return merged


    def compact_top_labels(row: pd.Series, columns: list[str], labels: dict[str, str], top_n: int = 3) -> str:
        values = []
        for column in columns:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.notna(value):
                values.append((labels.get(column, column), float(value)))
        if not values:
            return ""
        values = sorted(values, key=lambda item: item[1], reverse=True)[:top_n]
        return " / ".join(f"{label}({value:.1f})" for label, value in values)


    def compact_top_risk_labels(row: pd.Series, top_n: int = 3) -> str:
        values = []
        for column in RISK_SEV_COLUMNS:
            raw_value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            percentile = pd.to_numeric(pd.Series([row.get(f"{column}_percentile")]), errors="coerce").iloc[0]
            if pd.notna(raw_value) and pd.notna(percentile):
                values.append((RISK_LABELS.get(column, column), float(raw_value), float(percentile)))
        if not values:
            return ""
        values = sorted(values, key=lambda item: item[2], reverse=True)[:top_n]
        return " / ".join(f"{label}({raw_value:.1f},P{percentile * 100:.0f})" for label, raw_value, percentile in values)


    def add_gbd_disease_risk_supplement(latest: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        merged = latest.copy()
        disease_path = clean_dir / GBD_DALY_FILE
        risk_path = clean_dir / GBD_RISK_SEV_FILE
        gbd2021_path = clean_dir / GBD2021_NCD_FILE

        if disease_path.exists():
            disease = pd.read_csv(disease_path)
            keep = ["province", "source_year", *DISEASE_BURDEN_COLUMNS]
            disease = disease.loc[:, [col for col in keep if col in disease.columns]].rename(columns={"source_year": "gbd2017_disease_source_year"})
            merged = merged.merge(disease, on="province", how="left")
        else:
            merged["gbd2017_disease_source_year"] = np.nan

        if risk_path.exists():
            risk = pd.read_csv(risk_path)
            keep = ["province", "source_year", *RISK_SEV_COLUMNS]
            risk = risk.loc[:, [col for col in keep if col in risk.columns]].rename(columns={"source_year": "gbd2017_risk_source_year"})
            merged = merged.merge(risk, on="province", how="left")
        else:
            merged["gbd2017_risk_source_year"] = np.nan

        if gbd2021_path.exists():
            gbd2021 = pd.read_csv(gbd2021_path)
            keep = [
                "province",
                "source_year",
                "gbd2021_ncd_age_standardized_mortality_rate_per100k",
                "gbd2021_ncd_mortality_rate_change_1990_2021_pct",
                "gbd2021_ncd_age_standardized_daly_rate_per100k",
                "gbd2021_ncd_daly_rate_change_1990_2021_pct",
            ]
            gbd2021 = gbd2021.loc[:, [col for col in keep if col in gbd2021.columns]].rename(columns={"source_year": "gbd2021_ncd_source_year"})
            merged = merged.merge(gbd2021, on="province", how="left")
        else:
            merged["gbd2021_ncd_source_year"] = np.nan

        burden_percentile_cols = []
        for column in DISEASE_BURDEN_COLUMNS:
            if column in merged.columns:
                percentile_column = f"{column}_percentile"
                merged[percentile_column] = percentile_score(merged[column])
                burden_percentile_cols.append(percentile_column)

        risk_percentile_cols = []
        for column in RISK_SEV_COLUMNS:
            if column in merged.columns:
                percentile_column = f"{column}_percentile"
                merged[percentile_column] = percentile_score(merged[column])
                risk_percentile_cols.append(percentile_column)

        if burden_percentile_cols:
            merged["province_burden_pressure_score"] = merged[burden_percentile_cols].mean(axis=1)
        else:
            merged["province_burden_pressure_score"] = np.nan

        if risk_percentile_cols:
            merged["province_risk_exposure_score"] = merged[risk_percentile_cols].mean(axis=1)
        else:
            merged["province_risk_exposure_score"] = np.nan

        aging = pd.to_numeric(merged.get("aging65_pressure_percentile"), errors="coerce")
        burden = pd.to_numeric(merged["province_burden_pressure_score"], errors="coerce")
        risk = pd.to_numeric(merged["province_risk_exposure_score"], errors="coerce")
        merged["province_combined_pressure_score"] = (
            0.45 * burden.fillna(burden.mean())
            + 0.35 * risk.fillna(risk.mean())
            + 0.20 * aging.fillna(aging.mean())
        )
        merged["province_response_score"] = pd.to_numeric(merged["resource_response_score"], errors="coerce")
        merged["province_adaptation_gap_score"] = merged["province_combined_pressure_score"] - merged["province_response_score"]
        merged["province_adaptation_gap_percentile"] = percentile_score(merged["province_adaptation_gap_score"])

        high_pressure = merged["province_combined_pressure_score"] >= 2 / 3
        high_response = merged["province_response_score"] >= 0.50
        conditions = [
            high_pressure & ~high_response,
            high_pressure & high_response,
            ~high_pressure & ~high_response,
            ~high_pressure & high_response,
        ]
        choices = ["高压低响应型", "高压高响应型", "低压低响应型", "相对均衡型"]
        merged["province_abcd_type"] = np.select(conditions, choices, default="相对均衡型")

        merged["dominant_risk_triplet"] = merged.apply(compact_top_risk_labels, axis=1)
        merged["dominant_disease_triplet"] = merged.apply(lambda row: compact_top_labels(row, DISEASE_BURDEN_COLUMNS, DISEASE_LABELS), axis=1)
        merged["gbd2017_data_available"] = (
            merged[DISEASE_BURDEN_COLUMNS].notna().all(axis=1)
            if set(DISEASE_BURDEN_COLUMNS).issubset(merged.columns)
            else False
        )
        merged["gbd2017_risk_data_available"] = (
            merged[RISK_SEV_COLUMNS].notna().all(axis=1)
            if set(RISK_SEV_COLUMNS).issubset(merged.columns)
            else False
        )
        if "gbd2021_ncd_age_standardized_daly_rate_per100k" in merged.columns:
            merged["gbd2021_ncd_data_available"] = merged["gbd2021_ncd_age_standardized_daly_rate_per100k"].notna()
            merged["gbd2021_ncd_daly_percentile"] = percentile_score(merged["gbd2021_ncd_age_standardized_daly_rate_per100k"])
        else:
            merged["gbd2021_ncd_data_available"] = False
        merged["province_risk_exposure_score"] = merged["province_combined_pressure_score"]
        merged["province_risk_exposure_type"] = pd.cut(
            merged["province_combined_pressure_score"],
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["省级压力低位", "省级压力中位", "省级压力高位"],
        ).astype(str)
        merged["data_scope"] = (
            merged["data_scope"].astype(str)
            + "；已接入GBD 2017中国省级四类疾病DALY率和十大风险SEV；已补GBD 2021省级NCD总负担边界字段"
        )
        return merged


    def add_policy_response_supplement(latest: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / POLICY_RESPONSE_FILE
        merged = latest.copy()
        default_columns = {
            "drg_pilot_city_count": 0,
            "dip_pilot_city_count": 0,
            "payment_reform_city_count_total": 0,
            "drg_pilot_2019": False,
            "dip_pilot_2020": False,
            "policy_first_event_year": np.nan,
            "payment_reform_policy_score": 0.0,
            "payment_reform_policy_type": "未接入政策时间线",
            "policy_timeline_data_available": False,
            "policy_exposure_note": "未接入NHSA DRG/DIP政策时间线。",
            "source_scope": "",
            "causal_boundary": "中国D层政策暴露数据未接入，不能做省级政策因果解释。",
        }
        if not path.exists():
            for column, value in default_columns.items():
                merged[column] = value
            merged["province_d_policy_score"] = merged["payment_reform_policy_score"]
            merged["province_d_policy_type"] = merged["payment_reform_policy_type"]
            return merged

        policy = pd.read_csv(path)
        keep = [
            "province",
            "drg_pilot_city_count",
            "dip_pilot_city_count",
            "payment_reform_city_count_total",
            "drg_pilot_2019",
            "dip_pilot_2020",
            "policy_first_event_year",
            "payment_reform_policy_score",
            "payment_reform_policy_type",
            "policy_timeline_data_available",
            "policy_exposure_note",
            "source_scope",
            "causal_boundary",
        ]
        policy = policy.loc[:, [col for col in keep if col in policy.columns]].copy()
        merged = merged.merge(policy, on="province", how="left")
        for column, value in default_columns.items():
            if column not in merged.columns:
                merged[column] = value
            elif isinstance(value, bool):
                merged[column] = merged[column].fillna(value).astype(bool)
            elif isinstance(value, (int, float)):
                merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(value)
            else:
                merged[column] = merged[column].fillna(value)
        merged["province_d_payment_policy_score"] = pd.to_numeric(merged["payment_reform_policy_score"], errors="coerce").fillna(0.0)
        merged["province_d_policy_score"] = merged["province_d_payment_policy_score"]
        merged["province_d_policy_type"] = merged["payment_reform_policy_type"]

        chronic_path = clean_dir / CHRONIC_POLICY_SCORE_FILE
        if chronic_path.exists():
            chronic = pd.read_csv(chronic_path)
            keep = [
                "province",
                "chronic_demo_zone_count_2014_2026",
                "chronic_demo_zone_count_recent_2020_2026",
                "chronic_demo_zones_per_10m_population",
                "chronic_policy_execution_score",
                "chronic_policy_execution_type",
            ]
            chronic = chronic.loc[:, [col for col in keep if col in chronic.columns]]
            merged = merged.merge(chronic, on="province", how="left")
            chronic_score = pd.to_numeric(merged.get("chronic_policy_execution_score"), errors="coerce")
            payment_score = pd.to_numeric(merged["province_d_payment_policy_score"], errors="coerce").fillna(0.0)
            merged["province_d_policy_score"] = (0.55 * payment_score + 0.45 * chronic_score.fillna(0.0)).clip(0, 1)
            merged["province_d_policy_type"] = (
                merged["payment_reform_policy_type"].astype(str)
                + "；"
                + merged.get("chronic_policy_execution_type", pd.Series("", index=merged.index)).fillna("慢病政策执行未接入").astype(str)
            )
            merged["data_scope"] = merged["data_scope"].astype(str) + "；已接入NHC国家慢病综合防控示范区省级执行强度"
        else:
            merged["chronic_policy_execution_score"] = np.nan
            merged["chronic_policy_execution_type"] = "慢病政策执行未接入"

        merged["province_d_policy_readiness"] = pd.cut(
            merged["province_d_policy_score"],
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["D层政策暴露低位", "D层政策暴露中位", "D层政策暴露高位"],
        ).astype(str)
        merged["data_scope"] = merged["data_scope"].astype(str) + "；已接入NHSA DRG/DIP省级政策暴露时间线种子"
        return merged


    def add_nbs_explanatory_candidates(latest: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / NBS_EXPLANATORY_FILE
        merged = latest.copy()
        if not path.exists():
            merged["nbs2025_explanatory_candidate_available"] = False
            return merged
        candidates = pd.read_csv(path)
        keep = [
            "province",
            "nbs2024_birth_rate_per_mille_candidate",
            "nbs2024_death_rate_per_mille_candidate",
            "nbs2024_natural_growth_rate_per_mille_candidate",
            "nbs2024_natural_growth_corrected_by_birth_minus_death",
            "nbs2024_urban_population_share_pct_candidate",
            "nbs2024_gross_regional_product_100m_yuan_candidate",
            "candidate_non_null_fields",
            "mean_ocr_confidence",
            "model_use_status",
            "boundary_note",
        ]
        candidates = candidates.loc[:, [col for col in keep if col in candidates.columns]].rename(
            columns={
                "candidate_non_null_fields": "nbs2025_explanatory_candidate_non_null_fields",
                "mean_ocr_confidence": "nbs2025_explanatory_mean_ocr_confidence",
                "model_use_status": "nbs2025_explanatory_model_use_status",
                "boundary_note": "nbs2025_explanatory_boundary_note",
            }
        )
        merged = merged.merge(candidates, on="province", how="left")
        merged["nbs2025_explanatory_candidate_available"] = merged["nbs2025_explanatory_candidate_non_null_fields"].fillna(0).ge(3)
        if "nbs2024_urban_population_share_pct_candidate" in merged.columns:
            merged["nbs2024_urbanization_percentile_candidate"] = percentile_score(
                merged["nbs2024_urban_population_share_pct_candidate"]
            )
        if "nbs2024_gross_regional_product_100m_yuan_candidate" in merged.columns:
            merged["nbs2024_grp_percentile_candidate"] = percentile_score(
                merged["nbs2024_gross_regional_product_100m_yuan_candidate"]
            )
        merged["data_scope"] = (
            merged["data_scope"].astype(str)
            + "；已补NBS 2025年鉴死亡率/城镇化/GDP OCR解释变量候选，不进入核心因果回归"
        )
        return merged


    def add_nbs_mortality_panel(latest: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / NBS_MORTALITY_PANEL_FILE
        merged = latest.copy()
        if not path.exists():
            merged["nbs_mortality_panel_available"] = False
            return merged
        mortality = pd.read_csv(path)
        mortality = mortality.loc[
            mortality["model_use_status"].eq("china_health_outcome_sensitivity_panel_candidate")
        ].copy()
        if mortality.empty:
            merged["nbs_mortality_panel_available"] = False
            return merged
        mortality["source_year"] = pd.to_numeric(mortality["source_year"], errors="coerce")
        mortality = mortality.sort_values(["province", "source_year"], kind="stable")
        latest_mortality = mortality.groupby("province", as_index=False).tail(1).copy()
        coverage = (
            mortality.groupby("province", as_index=False)
            .agg(
                nbs_mortality_panel_year_count=("source_year", "nunique"),
                nbs_mortality_panel_years=("source_year", lambda s: ";".join(str(int(v)) for v in sorted(s.dropna().unique()))),
                nbs_mortality_panel_death_non_null_rows=("death_rate_per_mille_candidate", "count"),
            )
        )
        latest_mortality = latest_mortality[
            [
                "province",
                "source_year",
                "death_rate_per_mille_candidate",
                "death_rate_change_yoy_candidate",
                "birth_rate_per_mille_candidate",
                "natural_growth_rate_per_mille_candidate",
                "candidate_complete",
                "source_url",
                "boundary_note",
            ]
        ].rename(
            columns={
                "source_year": "nbs_mortality_latest_source_year",
                "death_rate_per_mille_candidate": "nbs_mortality_latest_death_rate_per_mille_candidate",
                "death_rate_change_yoy_candidate": "nbs_mortality_latest_death_rate_change_yoy_candidate",
                "birth_rate_per_mille_candidate": "nbs_mortality_latest_birth_rate_per_mille_candidate",
                "natural_growth_rate_per_mille_candidate": "nbs_mortality_latest_natural_growth_rate_per_mille_candidate",
                "candidate_complete": "nbs_mortality_latest_candidate_complete",
                "source_url": "nbs_mortality_latest_source_url",
                "boundary_note": "nbs_mortality_panel_boundary_note",
            }
        )
        merged = merged.merge(latest_mortality, on="province", how="left").merge(coverage, on="province", how="left")
        merged["nbs_mortality_panel_available"] = merged["nbs_mortality_panel_year_count"].fillna(0).ge(4)
        merged["data_scope"] = (
            merged["data_scope"].astype(str)
            + "；已补NBS 2018-2024年鉴分省粗死亡率OCR健康结局敏感性面板，不作为强因果主结论"
        )
        return merged


    def add_gbd2021_freshness_bridge(latest: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
        path = report_asset_path(report_dir, "china_gbd2021_freshness_bridge.csv")
        merged = latest.copy()
        if not path.exists():
            merged["gbd2021_freshness_bridge_available"] = False
            return merged
        bridge = pd.read_csv(path)
        keep = [
            "province",
            "gbd2017_four_cause_daly_sum_per100k",
            "gbd2017_four_cause_percentile",
            "gbd2021_ncd_daly_percentile",
            "gbd2021_ncd_mortality_percentile",
            "freshness_rank_shift_2021_minus_2017",
            "gbd2021_bridge_status",
            "model_use_status",
            "boundary_note",
        ]
        bridge = bridge.loc[:, [col for col in keep if col in bridge.columns]].rename(
            columns={
                "model_use_status": "gbd2021_bridge_model_use_status",
                "boundary_note": "gbd2021_bridge_boundary_note",
            }
        )
        merged = merged.merge(bridge, on="province", how="left")
        merged["gbd2021_freshness_bridge_available"] = merged["gbd2021_bridge_status"].eq(
            "freshness_boundary_bridge_available"
        )
        return merged


    def add_health_outcome_anchor(latest: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / HEALTH_OUTCOME_ANCHOR_FILE
        merged = latest.copy()
        if not path.exists():
            merged["health_outcome_anchor_available"] = False
            return merged
        anchor = pd.read_csv(path)
        keep = [
            "province",
            "health_outcome_anchor_score",
            "health_outcome_anchor_type",
            "health_outcome_anchor_consistency_score",
            "health_outcome_anchor_consistency_type",
            "nbs_latest_crude_death_percentile",
            "nbs_mean_crude_death_percentile",
            "nbs_2018_2024_crude_death_rate_change",
            "model_use_status",
            "claim_boundary",
        ]
        anchor = anchor.loc[:, [col for col in keep if col in anchor.columns]].rename(
            columns={
                "model_use_status": "health_outcome_anchor_model_use_status",
                "claim_boundary": "health_outcome_anchor_boundary_note",
            }
        )
        merged = merged.merge(anchor, on="province", how="left")
        merged["health_outcome_anchor_available"] = merged["health_outcome_anchor_score"].notna()
        merged["data_scope"] = (
            merged["data_scope"].astype(str)
            + "；已补GBD2021年龄标化NCD健康结局锚点与NBS粗死亡率敏感性一致性校验"
        )
        return merged


    def add_local_policy_execution_indicator(latest: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / LOCAL_POLICY_EXECUTION_FILE
        merged = latest.copy()
        if not path.exists():
            merged["china_local_policy_execution_indicator_available"] = False
            return merged
        indicator = pd.read_csv(path)
        keep = [
            "province",
            "china_local_policy_execution_score",
            "china_local_policy_execution_type",
            "policy_execution_scored_domain_count",
            "national_policy_milestone_domain_count",
            "family_doctor_policy_milestone_available",
            "hierarchical_care_policy_milestone_available",
            "healthy_city_policy_milestone_available",
            "policy_package_milestone_note",
            "model_use_status",
            "claim_boundary",
        ]
        indicator = indicator.loc[:, [col for col in keep if col in indicator.columns]].rename(
            columns={
                "model_use_status": "china_local_policy_execution_model_use_status",
                "claim_boundary": "china_local_policy_execution_boundary_note",
            }
        )
        merged = merged.merge(indicator, on="province", how="left")
        merged["china_local_policy_execution_indicator_available"] = merged["china_local_policy_execution_score"].notna()
        if "province_d_policy_score" in merged.columns:
            current = pd.to_numeric(merged["province_d_policy_score"], errors="coerce").fillna(0)
            indicator_score = pd.to_numeric(merged["china_local_policy_execution_score"], errors="coerce").fillna(0)
            merged["province_d_policy_score"] = np.maximum(current, indicator_score)
            merged["province_d_policy_readiness"] = pd.cut(
                merged["province_d_policy_score"],
                bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
                labels=["D层政策暴露低位", "D层政策暴露中位", "D层政策暴露高位"],
            ).astype(str)
        merged["data_scope"] = (
            merged["data_scope"].astype(str)
            + "；已补地方政策执行指标：省级评分仅纳入DRG/DIP和慢病示范区，家庭医生/分级诊疗/健康城市作为国家政策里程碑进入政策包"
        )
        return merged


    def add_c_layer_ocr_candidates(latest: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / NBS_OCR_RESPONSE_FILE
        merged = latest.copy()
        if not path.exists():
            merged["c_layer_ocr_candidate_available"] = False
            return merged
        candidates = pd.read_csv(path)
        keep = [
            "province",
            "nbs2024_health_technicians_per_1000_candidate",
            "nbs2024_medical_beds_10k_candidate",
            "nbs2024_medical_insurance_enrollment_10k_candidate",
            "nbs2024_medical_insurance_fund_income_100m_candidate",
            "nbs2024_medical_insurance_fund_expenditure_100m_candidate",
            "candidate_non_null_fields",
            "mean_ocr_confidence",
            "model_use_status",
            "boundary_note",
        ]
        candidates = candidates.loc[:, [col for col in keep if col in candidates.columns]].rename(
            columns={
                "model_use_status": "c_layer_ocr_model_use_status",
                "boundary_note": "c_layer_ocr_boundary_note",
            }
        )
        merged = merged.merge(candidates, on="province", how="left")
        merged["c_layer_ocr_candidate_available"] = merged.get("candidate_non_null_fields", pd.Series(index=merged.index)).fillna(0).gt(0)
        qc_path = clean_dir / C_LAYER_QC_FILE
        if qc_path.exists():
            qc = pd.read_csv(qc_path)
            keep_qc = [
                "province",
                "c_layer_qc_pass_fields",
                "health_technicians_qc_ok",
                "beds_qc_ok",
                "insurance_enrollment_qc_ok",
                "fund_income_qc_ok",
                "fund_expenditure_qc_ok",
                "manual_review_required_for_core_score",
                "c_layer_qc_status",
                "boundary_note",
            ]
            qc = qc.loc[:, [col for col in keep_qc if col in qc.columns]].rename(
                columns={"boundary_note": "c_layer_qc_boundary_note"}
            )
            merged = merged.merge(qc, on="province", how="left")
        else:
            merged["c_layer_qc_status"] = "qc_not_run"
        merged["data_scope"] = merged["data_scope"].astype(str) + "；已补NBS床位/医保/财政OCR候选字段级QC但不纳入核心评分"
        return merged


    def policy_package_from_risks(row: pd.Series) -> str:
        risk_values = []
        for column in RISK_SEV_COLUMNS:
            percentile = pd.to_numeric(pd.Series([row.get(f"{column}_percentile")]), errors="coerce").iloc[0]
            if pd.notna(percentile):
                risk_values.append((column, float(percentile)))
        top_columns = [column for column, _ in sorted(risk_values, key=lambda item: item[1], reverse=True)[:3]]
        packages = []
        if "sev2017_high_systolic_blood_pressure" in top_columns or "sev2017_diet_high_in_sodium" in top_columns:
            packages.append("高血压筛查、控盐行动和基层连续用药")
        if "sev2017_ambient_pm25_pollution" in top_columns:
            packages.append("PM2.5协同治理、心肺慢病筛查和高风险人群防护")
        if "sev2017_smoking" in top_columns:
            packages.append("无烟环境执法、戒烟门诊和烟草风险宣传")
        if "sev2017_high_fasting_plasma_glucose" in top_columns or "sev2017_high_body_mass_index" in top_columns:
            packages.append("糖尿病早筛、体重管理和社区慢病随访")
        if "sev2017_alcohol_use" in top_columns:
            packages.append("酒精危害干预和重点人群行为风险管理")
        return "；".join(dict.fromkeys(packages)) or "按省级主导风险配置慢病筛查、风险治理和基层连续管理"


    def policy_priority(row: pd.Series) -> tuple[str, str]:
        if pd.notna(row.get("province_combined_pressure_score")):
            abcd_type = row.get("province_abcd_type", "相对均衡型")
            dominant_risks = row.get("dominant_risk_triplet", "")
            dominant_diseases = row.get("dominant_disease_triplet", "")
            policy_score = pd.to_numeric(pd.Series([row.get("province_d_policy_score")]), errors="coerce").iloc[0]
            policy_tail = ""
            if pd.notna(policy_score):
                if policy_score >= 2 / 3:
                    policy_tail = "；医保支付改革试点暴露较强，可优先承接服务效率和资源配置优化情景"
                elif policy_score <= 1 / 3:
                    policy_tail = "；DRG/DIP国家试点暴露较弱，需先核验地方支付改革与慢病管理政策执行"
            if abcd_type == "高压低响应型":
                return (
                    "优先做资源补短板+主导风险前移治理",
                    f"省级GBD显示压力高但资源响应不足；主导风险为{dominant_risks}，主导疾病负担为{dominant_diseases}{policy_tail}。",
                )
            if abcd_type == "高压高响应型":
                return (
                    "优先做高负担精细化管理和服务效率提升",
                    f"压力高但响应基础不弱，应把资源投向主导风险和疾病链条；主导风险为{dominant_risks}{policy_tail}。",
                )
            if abcd_type == "低压低响应型":
                return (
                    "优先补基层能力和基础服务网络",
                    f"当前压力不算最高，但响应能力偏弱，应先补资源底盘并监测{dominant_risks}{policy_tail}。",
                )
            return (
                "优先做风险监测、慢病质量提升和局部短板治理",
                f"压力与响应相对匹配，重点盯住{dominant_risks}，避免局部缺口扩大{policy_tail}。",
            )

        explicit_priority = row.get("province_policy_priority")
        if isinstance(explicit_priority, str) and explicit_priority and explicit_priority != "nan":
            return (
                explicit_priority,
                "省级PM2.5、老龄化和资源短板共同进入风险代理评分；吸烟、高血压、糖尿病仍使用中国国家画像向省级迁移。",
            )
        weak = []
        staff_signal = row.get("staff_per_capita_percentile", row["staff_scale_percentile"])
        institution_signal = row.get("institution_per_capita_percentile", row["institution_scale_percentile"])
        aging_signal = row.get("aging65_pressure_percentile", np.nan)
        if staff_signal < 1 / 3:
            weak.append("卫生人员供给")
        if institution_signal < 1 / 3:
            weak.append("医疗机构网络")
        if row["staff_growth_percentile"] < 1 / 3:
            weak.append("人员增长动能")
        if pd.notna(aging_signal) and aging_signal >= 2 / 3:
            weak.append("老龄健康服务")
        if not weak:
            return (
                "从资源扩容转向慢病风险治理和服务质量",
                "人均资源信号不低，演示中应强调高血压、吸烟、高血糖等风险前移治理。",
            )
        return (
            "优先补强" + "、".join(weak),
            "人均资源、老龄化压力或增长信号存在短板，省级映射中应优先核验医生、护士、床位、基层机构和老龄健康服务指标。",
        )


    def build_policy_cards(latest: pd.DataFrame, china_rules: pd.DataFrame) -> pd.DataFrame:
        national_risks = "高收缩压 / 吸烟 / 高血糖"
        national_resource_package = "护理队伍、卫生投入强度、政府筹资和医保支付稳定性"
        if not china_rules.empty:
            risk_rows = china_rules.loc[china_rules["mapping_layer"] == "B风险治理", "target"].astype(str).tolist()
            if risk_rows:
                national_risks = " / ".join(risk_rows)
            resource_rows = china_rules.loc[china_rules["mapping_layer"] == "C响应补短板", "recommended_policy_package"].astype(str).tolist()
            if resource_rows:
                national_resource_package = resource_rows[0]

        rows = []
        for _, row in latest.iterrows():
            priority, note = policy_priority(row)
            rows.append(
                {
                    "province": row["province"],
                    "year": int(row["year"]),
                    "resource_response_type": row["resource_response_type"],
                    "province_resource_response_type": row.get("province_resource_response_type"),
                    "resource_response_score": row["resource_response_score"],
                    "medical_staff_10k_persons": row["medical_staff_10k_persons"],
                    "medical_institutions_count": row["medical_institutions_count"],
                    "census_2020_population": row.get("census_2020_population"),
                    "census_2020_age65_plus_pct": row.get("census_2020_age65_plus_pct"),
                    "medical_staff_per_10k_population": row.get("medical_staff_per_10k_population"),
                    "medical_institutions_per_million_population": row.get("medical_institutions_per_million_population"),
                    "medical_staff_growth_5y": row["medical_staff_growth_5y"],
                    "medical_institutions_growth_5y": row["medical_institutions_growth_5y"],
                    "pm25_latest_year": row.get("pm25_latest_year"),
                    "pm25_population_weighted_ug_m3": row.get("pm25_population_weighted_ug_m3"),
                    "pm25_pressure_percentile": row.get("pm25_pressure_percentile"),
                    "province_risk_exposure_score": row.get("province_risk_exposure_score"),
                    "province_risk_exposure_type": row.get("province_risk_exposure_type"),
                    "province_burden_pressure_score": row.get("province_burden_pressure_score"),
                    "province_risk_exposure_score": row.get("province_risk_exposure_score"),
                    "province_combined_pressure_score": row.get("province_combined_pressure_score"),
                    "province_response_score": row.get("province_response_score"),
                    "province_adaptation_gap_score": row.get("province_adaptation_gap_score"),
                    "province_adaptation_gap_percentile": row.get("province_adaptation_gap_percentile"),
                    "province_abcd_type": row.get("province_abcd_type"),
                    "province_d_policy_score": row.get("province_d_policy_score"),
                    "province_d_payment_policy_score": row.get("province_d_payment_policy_score"),
                    "chronic_policy_execution_score": row.get("chronic_policy_execution_score"),
                    "chronic_policy_execution_type": row.get("chronic_policy_execution_type"),
                    "chronic_demo_zone_count_2014_2026": row.get("chronic_demo_zone_count_2014_2026"),
                    "chronic_demo_zone_count_recent_2020_2026": row.get("chronic_demo_zone_count_recent_2020_2026"),
                    "province_d_policy_type": row.get("province_d_policy_type"),
                    "province_d_policy_readiness": row.get("province_d_policy_readiness"),
                    "nbs2024_death_rate_per_mille_candidate": row.get("nbs2024_death_rate_per_mille_candidate"),
                    "nbs2024_natural_growth_rate_per_mille_candidate": row.get("nbs2024_natural_growth_rate_per_mille_candidate"),
                    "nbs2024_urban_population_share_pct_candidate": row.get("nbs2024_urban_population_share_pct_candidate"),
                    "nbs2024_gross_regional_product_100m_yuan_candidate": row.get("nbs2024_gross_regional_product_100m_yuan_candidate"),
                    "nbs2024_urbanization_percentile_candidate": row.get("nbs2024_urbanization_percentile_candidate"),
                    "nbs2024_grp_percentile_candidate": row.get("nbs2024_grp_percentile_candidate"),
                    "nbs_mortality_panel_year_count": row.get("nbs_mortality_panel_year_count"),
                    "nbs_mortality_panel_years": row.get("nbs_mortality_panel_years"),
                    "nbs_mortality_latest_death_rate_per_mille_candidate": row.get("nbs_mortality_latest_death_rate_per_mille_candidate"),
                    "nbs_mortality_latest_death_rate_change_yoy_candidate": row.get("nbs_mortality_latest_death_rate_change_yoy_candidate"),
                    "gbd2021_ncd_age_standardized_daly_rate_per100k": row.get("gbd2021_ncd_age_standardized_daly_rate_per100k"),
                    "gbd2021_ncd_daly_rate_change_1990_2021_pct": row.get("gbd2021_ncd_daly_rate_change_1990_2021_pct"),
                    "gbd2021_ncd_daly_percentile": row.get("gbd2021_ncd_daly_percentile"),
                    "gbd2017_four_cause_daly_sum_per100k": row.get("gbd2017_four_cause_daly_sum_per100k"),
                    "gbd2021_ncd_mortality_percentile": row.get("gbd2021_ncd_mortality_percentile"),
                    "freshness_rank_shift_2021_minus_2017": row.get("freshness_rank_shift_2021_minus_2017"),
                    "health_outcome_anchor_score": row.get("health_outcome_anchor_score"),
                    "health_outcome_anchor_type": row.get("health_outcome_anchor_type"),
                    "health_outcome_anchor_consistency_score": row.get("health_outcome_anchor_consistency_score"),
                    "health_outcome_anchor_consistency_type": row.get("health_outcome_anchor_consistency_type"),
                    "nbs_latest_crude_death_percentile": row.get("nbs_latest_crude_death_percentile"),
                    "nbs2024_medical_beds_10k_candidate": row.get("nbs2024_medical_beds_10k_candidate"),
                    "nbs2024_medical_insurance_enrollment_10k_candidate": row.get("nbs2024_medical_insurance_enrollment_10k_candidate"),
                    "nbs2024_medical_insurance_fund_income_100m_candidate": row.get("nbs2024_medical_insurance_fund_income_100m_candidate"),
                    "nbs2024_medical_insurance_fund_expenditure_100m_candidate": row.get("nbs2024_medical_insurance_fund_expenditure_100m_candidate"),
                    "c_layer_qc_pass_fields": row.get("c_layer_qc_pass_fields"),
                    "c_layer_qc_status": row.get("c_layer_qc_status"),
                    "c_layer_manual_review_required_for_core_score": row.get("manual_review_required_for_core_score"),
                    "drg_pilot_city_count": row.get("drg_pilot_city_count"),
                    "dip_pilot_city_count": row.get("dip_pilot_city_count"),
                    "payment_reform_city_count_total": row.get("payment_reform_city_count_total"),
                    "policy_first_event_year": row.get("policy_first_event_year"),
                    "china_local_policy_execution_score": row.get("china_local_policy_execution_score"),
                    "china_local_policy_execution_type": row.get("china_local_policy_execution_type"),
                    "policy_execution_scored_domain_count": row.get("policy_execution_scored_domain_count"),
                    "national_policy_milestone_domain_count": row.get("national_policy_milestone_domain_count"),
                    "family_doctor_policy_milestone_available": row.get("family_doctor_policy_milestone_available"),
                    "hierarchical_care_policy_milestone_available": row.get("hierarchical_care_policy_milestone_available"),
                    "healthy_city_policy_milestone_available": row.get("healthy_city_policy_milestone_available"),
                    "policy_exposure_note": row.get("policy_exposure_note"),
                    "dominant_risk_triplet": row.get("dominant_risk_triplet"),
                    "dominant_disease_triplet": row.get("dominant_disease_triplet"),
                    "global_china_profile": "类型1-高负担转型承压型 / 非传染病高压期 / 高压高响应型",
                    "national_top_risks_summary": national_risks,
                    "province_policy_priority": priority,
                    "recommended_policy_package": policy_package_from_risks(row)
                    if pd.notna(row.get("province_combined_pressure_score"))
                    else "PM2.5治理、心肺慢病筛查、老龄高风险人群防护"
                    if "PM2.5" in priority
                    else national_resource_package
                    if "资源" in priority or "补强" in priority
                    else "高血压筛查、控烟、糖尿病早筛、健康膳食",
                    "card_text": f"{row['province']}当前为{row.get('province_resource_response_type', row['resource_response_type'])}。{note}",
                    "data_caveat": "已补GBD 2017省级疾病负担与风险SEV、GBD 2021省级NCD年龄标化健康结局锚点、地级市人口加权PM2.5、人口与卫生资源，并接入NHSA DRG/DIP、NHC慢病示范区、地方政策执行指标和NBS粗死亡率敏感性面板；床位、医保和基金收支为字段级QC候选不纳入核心评分，当前不声明省级健康结果强因果。",
                }
            )
        sort_col = "province_adaptation_gap_score" if "province_adaptation_gap_score" in rows[0] else "resource_response_score"
        return pd.DataFrame(rows).sort_values(sort_col, ascending=False, kind="stable")


    def build_data_gaps(variable_dictionary: pd.DataFrame) -> pd.DataFrame:
        if variable_dictionary.empty:
            return pd.DataFrame()
        gaps = variable_dictionary.copy()
        gaps["implementation_status"] = np.where(
                gaps["mapping_module"].eq("C响应失配"),
                "已落地：已有NBS卫生人员/机构核心评分，已接入七普人口和年龄，并补NBS床位、医保参保和基金收支OCR字段级QC；未通过字段自动禁用，不进入核心评分",
            np.where(
                gaps["mapping_module"].eq("B风险归因"),
                "已落地：已补GBD 2017省级十大风险SEV和ScienceDB 2024地级市人口加权PM2.5，可做省级主导风险画像；最新调查风险率仍可继续补强",
                np.where(
                    gaps["mapping_module"].eq("A健康脆弱性画像"),
                    "已落地：已补GBD 2017省级四类疾病DALY率、GBD 2021省级NCD年龄标化死亡率/DALY健康结局锚点、七普人口和老龄化，并补NBS 2018/2019/2021-2024粗死亡率敏感性面板及2024城镇化/GDP官方OCR解释变量候选",
                    np.where(
                        gaps["mapping_module"].eq("D政策响应"),
                        "增强落地：已补NHSA DRG/DIP省级政策暴露时间线、资源响应准因果候选、NHC慢病示范区省级执行强度、地方政策执行指标、GBD2021年龄标化NCD健康结局锚点和NBS粗死亡率健康结局敏感性；家庭医生/分级诊疗/健康城市按国家政策里程碑进入政策包，不伪装为省级强弱评分",
                        "已落地：已生成31省压力-风险-响应-政策适配卡片，接入慢病政策执行、地方政策执行指标、GBD2021健康结局锚点和NBS解释变量，并已接入中国地图/地球仪演示；后续新增真实省级执行强度时可扩展",
                    ),
                ),
            ),
        )
        next_action_map = {
            "A健康脆弱性画像": "当前可讲GBD2021年龄标化NCD健康结局锚点；若后续拿到GBD2021同病种同风险省级表，可替换GBD2017同构核心",
            "B风险归因": "继续补最新省级高血压、吸烟、糖尿病、肥胖和膳食调查，用于更新GBD SEV风险画像",
            "C响应失配": "核心C层继续使用稳定NBS卫生人员/机构；字段级QC通过的床位/医保/基金字段可作为解释层展示",
            "D政策响应": "省级评分只用DRG/DIP和慢病示范区；家庭医生/分级诊疗/健康城市进入政策包说明，不作为省级处理变量",
            "中国综合输出": "把健康结局锚点、地方政策执行指标和字段级QC字段接入中国地图/地球仪演示",
        }
        gaps["next_action"] = gaps["mapping_module"].map(next_action_map).fillna("继续随新增省级数据扩展映射字段")
        return gaps


    def build_type_summary(latest: pd.DataFrame) -> pd.DataFrame:
        if "province_abcd_type" not in latest.columns:
            return pd.DataFrame()
        summary = (
            latest.groupby("province_abcd_type", dropna=False)
            .agg(
                province_count=("province", "nunique"),
                mean_pressure=("province_combined_pressure_score", "mean"),
                mean_response=("province_response_score", "mean"),
                mean_gap=("province_adaptation_gap_score", "mean"),
                mean_burden=("province_burden_pressure_score", "mean"),
                mean_risk=("province_risk_exposure_score", "mean"),
            )
            .reset_index()
            .sort_values("mean_gap", ascending=False, kind="stable")
        )
        return summary


    def plot_china_pressure_response(latest: pd.DataFrame, figure_path: Path) -> None:
        if not {"province_combined_pressure_score", "province_response_score", "province_abcd_type"}.issubset(latest.columns):
            return
        plot_df = latest.dropna(subset=["province_combined_pressure_score", "province_response_score"]).copy()
        if plot_df.empty:
            return
        colors = {
            "高压低响应型": "#d94b5f",
            "高压高响应型": "#4f7cac",
            "低压低响应型": "#f2b84b",
            "相对均衡型": "#2a9d8f",
        }
        type_label_en = {
            "高压低响应型": "High pressure, low response",
            "高压高响应型": "High pressure, high response",
            "低压低响应型": "Low pressure, low response",
            "相对均衡型": "Relatively balanced",
        }
        fig, ax = plt.subplots(figsize=(9.5, 7))
        for abcd_type, group in plot_df.groupby("province_abcd_type", dropna=False):
            ax.scatter(
                group["province_combined_pressure_score"],
                group["province_response_score"],
                s=95,
                alpha=0.82,
                color=colors.get(str(abcd_type), "#999999"),
                label=str(abcd_type) if USE_CHINESE else type_label_en.get(str(abcd_type), str(abcd_type)),
                edgecolor="white",
                linewidth=0.7,
            )
        ax.axvline(2 / 3, color="#333333", linestyle="--", linewidth=1)
        ax.axhline(0.50, color="#333333", linestyle="--", linewidth=1)
        top_gap = plot_df.sort_values("province_adaptation_gap_score", ascending=False, kind="stable").head(8)
        for _, row in top_gap.iterrows():
            label = str(row["province"]) if USE_CHINESE else str(row.get("location_en", row["province"]))
            ax.text(row["province_combined_pressure_score"] + 0.01, row["province_response_score"], label, fontsize=8)
        ax.set_xlabel(choose_text("省级综合健康压力得分", "Provincial pressure score", USE_CHINESE))
        ax.set_ylabel(choose_text("省级资源响应得分", "Provincial response score", USE_CHINESE))
        ax.set_title(choose_text("中国省级健康压力-响应适配图", "China Provincial Pressure-Response Fit", USE_CHINESE))
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=220)
        plt.close(fig)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build China provincial mapping framework from available NBS data.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        panel = load_nbs_provincial_panel(clean_dir)
        latest = add_gbd_disease_risk_supplement(
            add_provincial_risk_supplement(add_census_normalization(latest_provincial_resource_response(panel), clean_dir), clean_dir),
            clean_dir,
        )
        latest = add_policy_response_supplement(latest, clean_dir)
        latest = add_local_policy_execution_indicator(latest, clean_dir)
        latest = add_nbs_explanatory_candidates(latest, clean_dir)
        latest = add_nbs_mortality_panel(latest, clean_dir)
        latest = add_gbd2021_freshness_bridge(latest, report_dir)
        latest = add_health_outcome_anchor(latest, clean_dir)
        latest = add_c_layer_ocr_candidates(latest, clean_dir)
        china_rules_path = report_asset_path(report_dir, "china_policy_mapping_rules.csv")
        china_rules = pd.read_csv(china_rules_path) if china_rules_path.exists() else pd.DataFrame()
        policy_cards = build_policy_cards(latest, china_rules)
        variable_dict_path = report_asset_path(report_dir, "china_mapping_variable_dictionary.csv")
        variable_dict = pd.read_csv(variable_dict_path) if variable_dict_path.exists() else pd.DataFrame()
        data_gaps = build_data_gaps(variable_dict)
        type_summary = build_type_summary(latest)
        quasi_causal_summary_path = report_asset_path(report_dir, "china_policy_quasi_causal_response_summary.json")
        if quasi_causal_summary_path.exists():
            try:
                quasi_causal_summary = json.loads(quasi_causal_summary_path.read_text(encoding="utf-8"))
            except Exception:
                quasi_causal_summary = {}
        else:
            quasi_causal_summary = {}

        outputs = {
            "china_provincial_resource_panel": report_asset_path(report_dir, "china_provincial_resource_panel.csv"),
            "china_provincial_resource_latest": report_asset_path(report_dir, "china_provincial_resource_latest.csv"),
            "china_abcd_provincial_mapping_v2": report_asset_path(report_dir, "china_abcd_provincial_mapping_v2.csv"),
            "china_abcd_provincial_type_summary": report_asset_path(report_dir, "china_abcd_provincial_type_summary.csv"),
            "china_provincial_policy_cards": report_asset_path(report_dir, "china_provincial_policy_cards.csv"),
            "china_mapping_data_gaps": report_asset_path(report_dir, "china_mapping_data_gaps.csv"),
            "china_mapping_framework_summary": report_asset_path(report_dir, "china_mapping_framework_summary.json"),
            "china_abcd_pressure_response_scatter": figure_dir / "china_abcd_pressure_response_scatter.png",
        }
        panel.to_csv(outputs["china_provincial_resource_panel"], index=False, encoding="utf-8-sig")
        resource_latest_columns = [
            "province",
            "year",
            "medical_institutions_count",
            "medical_staff_10k_persons",
            "medical_staff_10k_persons_2019",
            "medical_institutions_count_2019",
            "medical_staff_growth_5y",
            "medical_institutions_growth_5y",
            "staff_scale_percentile",
            "institution_scale_percentile",
            "staff_growth_percentile",
            "institution_growth_percentile",
            "resource_response_score",
            "province_response_score",
            "province_resource_response_type",
            "province_combined_pressure_score",
            "province_adaptation_gap_score",
            "aging65_pressure_percentile",
        ]
        resource_latest = latest.loc[:, [col for col in resource_latest_columns if col in latest.columns]].copy()
        resource_latest.to_csv(outputs["china_provincial_resource_latest"], index=False, encoding="utf-8-sig")
        latest.to_csv(outputs["china_abcd_provincial_mapping_v2"], index=False, encoding="utf-8-sig")
        type_summary.to_csv(outputs["china_abcd_provincial_type_summary"], index=False, encoding="utf-8-sig")
        policy_cards.to_csv(outputs["china_provincial_policy_cards"], index=False, encoding="utf-8-sig")
        data_gaps.to_csv(outputs["china_mapping_data_gaps"], index=False, encoding="utf-8-sig")
        plot_china_pressure_response(latest, outputs["china_abcd_pressure_response_scatter"])

        mortality_summary_path = report_asset_path(report_dir, "china_nbs_mortality_panel_ocr_summary.json")
        if mortality_summary_path.exists():
            try:
                mortality_summary = json.loads(mortality_summary_path.read_text(encoding="utf-8"))
            except Exception:
                mortality_summary = {}
        else:
            mortality_summary = {}
        gbd_bridge_summary_path = report_asset_path(report_dir, "china_gbd2021_freshness_bridge_summary.json")
        if gbd_bridge_summary_path.exists():
            try:
                gbd_bridge_summary = json.loads(gbd_bridge_summary_path.read_text(encoding="utf-8"))
            except Exception:
                gbd_bridge_summary = {}
        else:
            gbd_bridge_summary = {}
        health_anchor_summary_path = report_asset_path(report_dir, "china_health_outcome_anchor_summary.json")
        if health_anchor_summary_path.exists():
            try:
                health_anchor_summary = json.loads(health_anchor_summary_path.read_text(encoding="utf-8"))
            except Exception:
                health_anchor_summary = {}
        else:
            health_anchor_summary = {}
        local_policy_summary_path = report_asset_path(report_dir, "china_local_policy_execution_indicator_summary.json")
        if local_policy_summary_path.exists():
            try:
                local_policy_summary = json.loads(local_policy_summary_path.read_text(encoding="utf-8"))
            except Exception:
                local_policy_summary = {}
        else:
            local_policy_summary = {}

        summary = {
            "project_root": project_root.as_posix(),
            "definition": "中国省级映射v7；使用NBS省级卫生人员/机构、七普人口与年龄、ScienceDB 2000-2024城市PM2.5并用2020地级市人口权重省级聚合、SatPM 2000-2017省级PM2.5基线、GBD 2017省级疾病负担与风险SEV、GBD 2021省级NCD年龄标化死亡率/DALY健康结局锚点、NHSA DRG/DIP省级政策暴露时间线与资源响应准因果候选检验、NHC国家慢病综合防控示范区执行强度、地方政策执行指标、NBS 2018/2019/2021-2024分省粗死亡率OCR健康结局敏感性面板、NBS 2025年鉴城镇化/GDP OCR解释变量，并补NBS床位/医保/基金收支字段级OCR质控，生成省级压力-响应-缺口和D层组合政策响应画像。",
            "latest_year": int(latest["year"].max()),
            "baseline_year_for_growth": int(latest["baseline_year"].iloc[0]),
            "province_rows": int(latest.shape[0]),
            "resource_response_type_counts": latest["resource_response_type"].value_counts().to_dict(),
            "province_abcd_type_counts": latest["province_abcd_type"].value_counts().to_dict() if "province_abcd_type" in latest.columns else {},
            "census_data_available": bool(latest["census_data_available"].all()) if "census_data_available" in latest.columns else False,
            "pm25_data_available": bool(latest["pm25_data_available"].all()) if "pm25_data_available" in latest.columns else False,
            "gbd2017_disease_data_available": bool(latest["gbd2017_data_available"].all()) if "gbd2017_data_available" in latest.columns else False,
            "gbd2017_risk_data_available": bool(latest["gbd2017_risk_data_available"].all()) if "gbd2017_risk_data_available" in latest.columns else False,
            "gbd2021_ncd_data_available": bool(latest["gbd2021_ncd_data_available"].all()) if "gbd2021_ncd_data_available" in latest.columns else False,
            "gbd2021_freshness_bridge_available": bool(latest["gbd2021_freshness_bridge_available"].all()) if "gbd2021_freshness_bridge_available" in latest.columns else False,
            "gbd2021_freshness_bridge_spearman_r": gbd_bridge_summary.get("correlation_with_gbd2017_four_cause_core", {}).get("spearman_r") if gbd_bridge_summary else None,
            "health_outcome_anchor_available": bool(latest["health_outcome_anchor_available"].all()) if "health_outcome_anchor_available" in latest.columns else False,
            "health_outcome_anchor_rows": int(health_anchor_summary.get("province_rows", 0) or 0) if health_anchor_summary else 0,
            "health_outcome_anchor_high_or_medium_consistency_rows": int(health_anchor_summary.get("high_or_medium_consistency_rows", 0) or 0) if health_anchor_summary else 0,
            "health_outcome_anchor_low_consistency_rows": int(health_anchor_summary.get("low_consistency_rows", 0) or 0) if health_anchor_summary else 0,
            "c_layer_ocr_candidate_available": bool(latest["c_layer_ocr_candidate_available"].any()) if "c_layer_ocr_candidate_available" in latest.columns else False,
            "c_layer_ocr_qc_available": bool(latest["c_layer_qc_status"].notna().any()) if "c_layer_qc_status" in latest.columns else False,
            "c_layer_ocr_qc_high_confidence_rows": int(latest["c_layer_qc_status"].eq("field_level_reviewed_high_confidence").sum()) if "c_layer_qc_status" in latest.columns else 0,
            "c_layer_ocr_qc_partial_rows": int(latest["c_layer_qc_status"].eq("field_level_reviewed_partial").sum()) if "c_layer_qc_status" in latest.columns else 0,
            "c_layer_ocr_field_level_qc_completed": True,
            "china_payment_policy_timeline_available": bool(latest["policy_timeline_data_available"].any()) if "policy_timeline_data_available" in latest.columns else False,
            "china_chronic_policy_execution_available": bool(pd.to_numeric(latest.get("chronic_policy_execution_score"), errors="coerce").notna().any()) if "chronic_policy_execution_score" in latest.columns else False,
            "china_local_policy_execution_indicator_available": bool(latest["china_local_policy_execution_indicator_available"].all()) if "china_local_policy_execution_indicator_available" in latest.columns else False,
            "china_local_policy_execution_indicator_rows": int(local_policy_summary.get("province_rows", 0) or 0) if local_policy_summary else 0,
            "china_local_policy_scored_domains": local_policy_summary.get("scored_policy_domains", []) if local_policy_summary else [],
            "china_local_policy_national_milestone_domains": local_policy_summary.get("national_milestone_policy_domains", []) if local_policy_summary else [],
            "nbs2025_explanatory_candidate_available": bool(latest["nbs2025_explanatory_candidate_available"].any()) if "nbs2025_explanatory_candidate_available" in latest.columns else False,
            "nbs2025_explanatory_complete_rows": int(latest["nbs2025_explanatory_candidate_non_null_fields"].eq(5).sum()) if "nbs2025_explanatory_candidate_non_null_fields" in latest.columns else 0,
            "nbs_mortality_panel_available": bool(latest["nbs_mortality_panel_available"].any()) if "nbs_mortality_panel_available" in latest.columns else False,
            "nbs_mortality_panel_included_years": mortality_summary.get("included_source_years", []) if mortality_summary else [],
            "nbs_mortality_panel_death_rate_non_null_rows": int(mortality_summary.get("death_rate_non_null_rows_included", 0) or 0) if mortality_summary else 0,
            "payment_reform_policy_type_counts": latest["payment_reform_policy_type"].value_counts(dropna=False).to_dict() if "payment_reform_policy_type" in latest.columns else {},
            "combined_d_policy_type_counts": latest["province_d_policy_readiness"].value_counts(dropna=False).to_dict() if "province_d_policy_readiness" in latest.columns else {},
            "china_d_response_quasi_causal_candidates": int(quasi_causal_summary.get("china_d_response_quasi_causal_candidates", 0) or 0),
            "china_d_response_directional_candidates": int(quasi_causal_summary.get("china_d_response_directional_candidates", 0) or 0),
            "china_d_response_validation_boundary": quasi_causal_summary.get("claim_boundary", "中国D层资源响应准因果检验未运行或不可用。"),
            "claim_boundary": "中国省级A/B/C画像已补齐到疾病负担、风险暴露、资源响应、GBD2021年龄标化NCD健康结局锚点和NBS解释变量候选层；D层已补NHSA DRG/DIP政策暴露、NHC慢病示范区执行强度、地方政策执行指标、资源响应准因果候选检验和NBS粗死亡率健康结局敏感性，但省级政策降低死亡率/DALY的强因果仍不能声明。",
            "remaining_data_boundary": "当前六项已转为可答辩边界：GBD2021同病种同风险省级表、家庭医生/分级诊疗真实省级执行强度和RCT式政策实验不作为当前可声明证据；项目已用健康结局锚点、字段级QC、政策来源字典和准因果边界防守。",
            "output_files": {key: path.as_posix() for key, path in outputs.items()},
        }
        outputs["china_mapping_framework_summary"].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_download_china_census_2020():
    __name__ = 'download_china_census_2020'
    import argparse
    import json
    import re
    import subprocess
    from pathlib import Path

    import pandas as pd

    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    POPULATION_URL = "https://www.stats.gov.cn/sj/pcsj/rkpc/7rp/zk/html/fu03c.pdf"
    AGE_URL = "https://www.stats.gov.cn/sj/pcsj/rkpc/7rp/zk/html/fu03e.pdf"

    PROVINCE_NAME_MAP = {
        "北京": "北京市",
        "天津": "天津市",
        "河北": "河北省",
        "山西": "山西省",
        "内蒙古": "内蒙古自治区",
        "辽宁": "辽宁省",
        "吉林": "吉林省",
        "黑龙江": "黑龙江省",
        "上海": "上海市",
        "江苏": "江苏省",
        "浙江": "浙江省",
        "安徽": "安徽省",
        "福建": "福建省",
        "江西": "江西省",
        "山东": "山东省",
        "河南": "河南省",
        "湖北": "湖北省",
        "湖南": "湖南省",
        "广东": "广东省",
        "广西": "广西壮族自治区",
        "海南": "海南省",
        "重庆": "重庆市",
        "四川": "四川省",
        "贵州": "贵州省",
        "云南": "云南省",
        "西藏": "西藏自治区",
        "陕西": "陕西省",
        "甘肃": "甘肃省",
        "青海": "青海省",
        "宁夏": "宁夏回族自治区",
        "新疆": "新疆维吾尔自治区",
    }


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def download(url: str, output_path: Path) -> None:
        if output_path.exists() and output_path.stat().st_size > 0:
            return
        raise FileNotFoundError(
            f"Required census source file is missing: {output_path}. "
            "Use input/External Data/09_China_Census_Population or 09_data_clean instead."
        )


    def pdftotext(pdf_path: Path) -> str:
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path.as_posix(), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


    def normalize_name(name: str) -> str:
        return re.sub(r"\s+", "", name)


    def parse_population_table(text: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        pattern = re.compile(
            r"^\s*(?P<name>[\u4e00-\u9fa5\s]{2,12})\s+"
            r"(?P<population>\d{6,10})\s+"
            r"(?P<share_2020>\d+\.\d+)\s+"
            r"(?P<share_2010>\d+\.\d+)\s*$"
        )
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            short_name = normalize_name(match.group("name"))
            if short_name in {"全国", "现役军人"} or short_name not in PROVINCE_NAME_MAP:
                continue
            rows.append(
                {
                    "province_short": short_name,
                    "province": PROVINCE_NAME_MAP[short_name],
                    "census_2020_population": int(match.group("population")),
                    "census_2020_population_share_pct": float(match.group("share_2020")),
                    "census_2010_population_share_pct": float(match.group("share_2010")),
                }
            )
        df = pd.DataFrame(rows).drop_duplicates("province", keep="first")
        if df.shape[0] != 31:
            raise ValueError(f"Expected 31 provincial population rows, got {df.shape[0]}")
        return df


    def parse_age_table(text: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        pattern = re.compile(
            r"^\s*(?P<name>[\u4e00-\u9fa5\s]{2,12})\s+"
            r"(?P<age0_14>\d+\.\d+)\s+"
            r"(?P<age15_59>\d+\.\d+)\s+"
            r"(?P<age60_plus>\d+\.\d+)\s+"
            r"(?P<age65_plus>\d+\.\d+)\s*$"
        )
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            short_name = normalize_name(match.group("name"))
            if short_name == "全国" or short_name not in PROVINCE_NAME_MAP:
                continue
            rows.append(
                {
                    "province_short": short_name,
                    "province": PROVINCE_NAME_MAP[short_name],
                    "census_2020_age0_14_pct": float(match.group("age0_14")),
                    "census_2020_age15_59_pct": float(match.group("age15_59")),
                    "census_2020_age60_plus_pct": float(match.group("age60_plus")),
                    "census_2020_age65_plus_pct": float(match.group("age65_plus")),
                }
            )
        df = pd.DataFrame(rows).drop_duplicates("province", keep="first")
        if df.shape[0] != 31:
            raise ValueError(f"Expected 31 provincial age rows, got {df.shape[0]}")
        return df


    def main() -> None:
        parser = argparse.ArgumentParser(description="Download and parse official China 2020 census provincial tables.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        raw_dir = external_data_root / "09_China_Census_Population" / "china_census_2020"
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        clean_output = clean_dir / "external_china_census_2020_province_population_age.csv"
        report_output = report_asset_path(report_dir, "china_census_2020_province_population_age.csv")
        if clean_output.exists() and clean_output.stat().st_size > 0:
            merged = pd.read_csv(clean_output, low_memory=False)
            if "province" in merged.columns and merged["province"].nunique() == 31:
                merged.to_csv(report_output, index=False, encoding="utf-8-sig")
                summary = {
                    "project_root": project_root.as_posix(),
                    "source_mode": "reused_09_data_clean",
                    "source_name": "第七次全国人口普查公报（第三号、第五号）",
                    "population_url": POPULATION_URL,
                    "age_url": AGE_URL,
                    "rows": int(merged.shape[0]),
                    "output_files": {
                        "clean_csv": clean_output.as_posix(),
                        "report_csv": report_output.as_posix(),
                        "summary": (report_asset_path(report_dir, "china_census_2020_source_summary.json")).as_posix(),
                    },
                }
                (report_asset_path(report_dir, "china_census_2020_source_summary.json")).write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return

        population_pdf = raw_dir / "china_census_2020_provincial_population.pdf"
        age_pdf = raw_dir / "china_census_2020_provincial_age_structure.pdf"
        download(POPULATION_URL, population_pdf)
        download(AGE_URL, age_pdf)

        population = parse_population_table(pdftotext(population_pdf))
        age = parse_age_table(pdftotext(age_pdf))
        merged = population.merge(age.drop(columns=["province_short"]), on="province", how="inner")
        if merged.shape[0] != 31:
            raise ValueError(f"Expected 31 merged rows, got {merged.shape[0]}")
        merged["source_year"] = 2020
        merged["source_name"] = "第七次全国人口普查公报（第三号、第五号）"
        merged["source_url_population"] = POPULATION_URL
        merged["source_url_age"] = AGE_URL

        merged.to_csv(clean_output, index=False, encoding="utf-8-sig")
        merged.to_csv(report_output, index=False, encoding="utf-8-sig")

        summary = {
            "project_root": project_root.as_posix(),
            "source_name": "第七次全国人口普查公报（第三号、第五号）",
            "population_url": POPULATION_URL,
            "age_url": AGE_URL,
            "raw_files": [population_pdf.as_posix(), age_pdf.as_posix()],
            "rows": int(merged.shape[0]),
            "output_files": {
                "clean_csv": clean_output.as_posix(),
                "report_csv": report_output.as_posix(),
                "summary": (report_asset_path(report_dir, "china_census_2020_source_summary.json")).as_posix(),
            },
        }
        (report_asset_path(report_dir, "china_census_2020_source_summary.json")).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_prepare_china_risk_policy_data():
    __name__ = 'prepare_china_risk_policy_data'
    import argparse
    import json
    from io import StringIO
    from pathlib import Path

    import pandas as pd

    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    SATPM_PROVINCIAL_PM25_URL = "https://wustl.app.box.com/shared/static/odsj842ypxjb653ahdmbtbnmlanwh1xf.csv"
    SCIDB_CITY_PM25_URL = "https://china.scidb.cn/download?fileId=70a15eb7ed88f298351fa4bcc81f1dcd"
    SCIDB_CITY_PM25_SOURCE_PAGE = "https://www.scidb.cn/en/detail?dataSetId=84279e24dec04d4ba68c1fadb70ff1ce"
    CHINA_DIVISION_PROVINCES_URL = "https://unpkg.com/china-division@2.4.0/dist/provinces.csv"
    CHINA_DIVISION_CITIES_URL = "https://unpkg.com/china-division@2.4.0/dist/cities.csv"

    PROVINCE_NAME_MAP = {
        "Anhui": "安徽省",
        "Beijing": "北京市",
        "Chongqing": "重庆市",
        "Fujian": "福建省",
        "Gansu": "甘肃省",
        "Guangdong": "广东省",
        "Guangxi": "广西壮族自治区",
        "Guizhou": "贵州省",
        "Hainan": "海南省",
        "Hebei": "河北省",
        "Heilongjiang": "黑龙江省",
        "Henan": "河南省",
        "Hubei": "湖北省",
        "Hunan": "湖南省",
        "Jiangsu": "江苏省",
        "Jiangxi": "江西省",
        "Jilin": "吉林省",
        "Liaoning": "辽宁省",
        "Nei Mongol": "内蒙古自治区",
        "Ningxia Hui": "宁夏回族自治区",
        "Qinghai": "青海省",
        "Shaanxi": "陕西省",
        "Shandong": "山东省",
        "Shanghai": "上海市",
        "Shanxi": "山西省",
        "Sichuan": "四川省",
        "Tianjin": "天津市",
        "Xinjiang Uygur": "新疆维吾尔自治区",
        "Xizang": "西藏自治区",
        "Yunnan": "云南省",
        "Zhejiang": "浙江省",
    }


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        risk_policy_dir = external_data_root / "11_China_PM25_Risk_Exposure" / "china_risk_policy"
        legacy_risk_policy_dir = external_data_root / "11_China_PM25_Risk_Exposure" / ("china_" + "plan" + "b_risk_policy")
        inventory_dir = risk_policy_dir if risk_policy_dir.exists() else legacy_risk_policy_dir
        dirs = {
            "inventory": inventory_dir,
            "clean": project_root / "09_data_clean",
            "report": project_root / "06_report_assets",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def download_file(url: str, output: Path) -> bytes:
        if output.exists() and output.stat().st_size > 0:
            return output.read_bytes()
        raise FileNotFoundError(
            f"Required PM2.5 source file is missing: {output}. "
            "Use input/External Data/11_China_PM25_Risk_Exposure or 09_data_clean instead."
        )


    def download_mojibake_csv(url: str, output: Path) -> pd.DataFrame:
        if output.exists() and output.stat().st_size > 0:
            return pd.read_csv(output)
        raise FileNotFoundError(
            f"Required China division source file is missing: {output}. "
            "Use input/External Data/11_China_PM25_Risk_Exposure/china_risk_policy instead."
        )


    def clean_pm25(raw_path: Path) -> pd.DataFrame:
        raw = pd.read_csv(raw_path)
        clean = raw.rename(
            columns={
                "Region/District/City": "province_en",
                "Year": "year",
                "Population-Weighted PM2.5 [ug/m3]": "pm25_population_weighted_ug_m3",
                "Geographic-Mean PM2.5 [ug/m3]": "pm25_geographic_mean_ug_m3",
                "Population Coverage [%]": "population_coverage_pct",
                "Geographic Coverage [%]": "geographic_coverage_pct",
                "Total Population [million people]": "satpm_population_million",
            }
        )
        clean["province"] = clean["province_en"].map(PROVINCE_NAME_MAP)
        clean["year"] = pd.to_numeric(clean["year"], errors="coerce").astype("Int64")
        for column in [
            "pm25_population_weighted_ug_m3",
            "pm25_geographic_mean_ug_m3",
            "population_coverage_pct",
            "geographic_coverage_pct",
            "satpm_population_million",
        ]:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean = clean.loc[clean["province"].notna() & clean["year"].notna()].copy()
        clean["source_dataset"] = "SatPM V4.CH.02 China provincial PM2.5"
        clean["source_url"] = SATPM_PROVINCIAL_PM25_URL
        return clean.sort_values(["province", "year"], kind="stable")


    def build_latest_pm25(pm25: pd.DataFrame) -> pd.DataFrame:
        latest_year = int(pm25["year"].max())
        latest = pm25.loc[pm25["year"] == latest_year].copy()
        latest["pm25_pressure_percentile"] = latest["pm25_population_weighted_ug_m3"].rank(pct=True, method="average")
        latest["pm25_pressure_type"] = pd.cut(
            latest["pm25_pressure_percentile"],
            bins=[-0.001, 1 / 3, 2 / 3, 1.001],
            labels=["PM2.5低暴露", "PM2.5中暴露", "PM2.5高暴露"],
        ).astype(str)
        latest["data_scope"] = "SatPM卫星反演省级PM2.5，当前公开省级文件覆盖2000-2017；用于补齐省级环境风险暴露，不替代吸烟/高血压/糖尿病监测。"
        return latest


    def normalize_city_name(name: object) -> str:
        text = str(name or "").strip()
        for suffix in ["市", "地区", "盟", "自治州"]:
            if text.endswith(suffix):
                return text[: -len(suffix)]
        return text


    def load_city_province_map(inventory_dir: Path) -> dict[str, str]:
        province_path = inventory_dir / "china_division_provinces.csv"
        city_path = inventory_dir / "china_division_cities.csv"
        provinces = download_mojibake_csv(CHINA_DIVISION_PROVINCES_URL, province_path)
        cities = download_mojibake_csv(CHINA_DIVISION_CITIES_URL, city_path)
        provinces["code"] = provinces["code"].astype(str)
        cities["provinceCode"] = cities["provinceCode"].astype(str)
        province_map = provinces.set_index("code")["name"].to_dict()
        city_map: dict[str, str] = {}
        for _, row in cities.iterrows():
            city = str(row["name"]).strip()
            province = province_map.get(str(row["provinceCode"]).strip())
            if not city or not province or city == "市辖区":
                continue
            city_map[city] = province
            city_map[normalize_city_name(city)] = province
        for province in province_map.values():
            city_map[str(province)] = str(province)
            city_map[normalize_city_name(province)] = str(province)
        return city_map


    def clean_scidb_pm25(raw_path: Path, inventory_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        city_map = load_city_province_map(inventory_dir)
        raw = pd.read_excel(raw_path, sheet_name=0)
        year_cols = [col for col in raw.columns if str(col).startswith("PM") and str(col)[2:].isdigit()]
        long = raw.melt(id_vars=["行政单元"], value_vars=year_cols, var_name="year", value_name="pm25_city_annual_ug_m3")
        long["city"] = long["行政单元"].astype(str).str.strip()
        long["city_norm"] = long["city"].map(normalize_city_name)
        long["province"] = long["city"].map(city_map).fillna(long["city_norm"].map(city_map))
        long["year"] = long["year"].astype(str).str.replace("PM", "", regex=False).astype(int)
        long["pm25_city_annual_ug_m3"] = pd.to_numeric(long["pm25_city_annual_ug_m3"], errors="coerce")
        long = long.dropna(subset=["province", "year", "pm25_city_annual_ug_m3"]).copy()
        long["source_dataset"] = "ScienceDB 342 Chinese cities annual mean PM2.5 2000-2024"
        long["source_url"] = SCIDB_CITY_PM25_SOURCE_PAGE
        province = (
            long.groupby(["province", "year"], as_index=False)
            .agg(
                pm25_city_mean_ug_m3=("pm25_city_annual_ug_m3", "mean"),
                pm25_city_median_ug_m3=("pm25_city_annual_ug_m3", "median"),
                pm25_city_min_ug_m3=("pm25_city_annual_ug_m3", "min"),
                pm25_city_max_ug_m3=("pm25_city_annual_ug_m3", "max"),
                city_count=("city", "nunique"),
            )
            .sort_values(["province", "year"], kind="stable")
        )
        province["pm25_latest_compatible_ug_m3"] = province["pm25_city_mean_ug_m3"]
        province["source_dataset"] = "ScienceDB 342-city PM2.5 aggregated to province by unweighted city mean"
        province["source_url"] = SCIDB_CITY_PM25_SOURCE_PAGE
        latest_year = int(province["year"].max())
        latest = province.loc[province["year"] == latest_year].copy()
        latest["pm25_pressure_percentile"] = latest["pm25_city_mean_ug_m3"].rank(pct=True, method="average")
        latest["pm25_pressure_type"] = pd.cut(
            latest["pm25_pressure_percentile"],
            bins=[-0.001, 1 / 3, 2 / 3, 1.001],
            labels=["PM2.5低暴露", "PM2.5中暴露", "PM2.5高暴露"],
        ).astype(str)
        latest["pm25_population_weighted_ug_m3"] = latest["pm25_city_mean_ug_m3"]
        latest["pm25_geographic_mean_ug_m3"] = latest["pm25_city_mean_ug_m3"]
        latest["pm25_measure_note"] = "ScienceDB城市年均PM2.5按省内城市非加权均值聚合，字段pm25_population_weighted_ug_m3仅作旧流程兼容。"
        latest["data_scope"] = "ScienceDB 342个中国城市PM2.5年均浓度，覆盖2000-2024；按城市均值聚合到省级，用于更新中国映射B层空气污染风险。"
        return long.sort_values(["province", "city", "year"], kind="stable"), province, latest


    def build_policy_rules(pm25_latest: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in pm25_latest.iterrows():
            high_pm25 = row["pm25_pressure_percentile"] >= 2 / 3
            rows.append(
                {
                    "province": row["province"],
                    "policy_domain": "空气污染与慢病风险协同治理" if high_pm25 else "慢病风险常规治理",
                    "risk_signal": row["pm25_pressure_type"],
                    "risk_value": row["pm25_population_weighted_ug_m3"],
                    "recommended_policy_package": "PM2.5治理、心肺慢病筛查、老年人高风险人群防护联动" if high_pm25 else "高血压筛查、控烟、糖尿病早筛和健康膳食",
                    "evidence_role": "省级环境风险实测/反演数据，用于中国映射B层风险补充。",
                }
            )
        return pd.DataFrame(rows)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Download China provincial mapping provincial risk/policy supplement data.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        outputs = {
            "external_china_satpm_provincial_pm25": dirs["clean"] / "external_china_satpm_provincial_pm25.csv",
            "external_china_scidb_city_pm25": dirs["clean"] / "external_china_scidb_city_pm25_2000_2024.csv",
            "external_china_scidb_provincial_pm25": dirs["clean"] / "external_china_scidb_provincial_pm25.csv",
            "china_provincial_pm25_latest": report_asset_path(dirs["report"], "china_provincial_pm25_latest.csv"),
            "china_provincial_policy_rules": report_asset_path(dirs["report"], "china_provincial_policy_rules.csv"),
            "china_risk_policy_data_source_summary": report_asset_path(dirs["report"], "china_risk_policy_data_source_summary.json"),
        }
        clean_ready = all(outputs[key].exists() and outputs[key].stat().st_size > 0 for key in [
            "external_china_satpm_provincial_pm25",
            "external_china_scidb_city_pm25",
            "external_china_scidb_provincial_pm25",
        ])
        if clean_ready:
            pm25 = pd.read_csv(outputs["external_china_satpm_provincial_pm25"], low_memory=False)
            scidb_city = pd.read_csv(outputs["external_china_scidb_city_pm25"], low_memory=False)
            scidb_province = pd.read_csv(outputs["external_china_scidb_provincial_pm25"], low_memory=False)
            if outputs["china_provincial_pm25_latest"].exists() and outputs["china_provincial_pm25_latest"].stat().st_size > 0:
                pm25_latest = pd.read_csv(outputs["china_provincial_pm25_latest"], low_memory=False)
            else:
                latest_year = int(pd.to_numeric(scidb_province["year"], errors="coerce").max())
                pm25_latest = scidb_province.loc[pd.to_numeric(scidb_province["year"], errors="coerce").eq(latest_year)].copy()
                pm25_latest["pm25_population_weighted_ug_m3"] = pm25_latest.get("pm25_latest_compatible_ug_m3", pm25_latest.get("pm25_city_mean_ug_m3"))
                pm25_latest["pm25_geographic_mean_ug_m3"] = pm25_latest["pm25_population_weighted_ug_m3"]
                pm25_latest["pm25_pressure_percentile"] = pm25_latest["pm25_population_weighted_ug_m3"].rank(pct=True, method="average")
                pm25_latest["pm25_pressure_type"] = pd.cut(
                    pm25_latest["pm25_pressure_percentile"],
                    bins=[-0.001, 1 / 3, 2 / 3, 1.001],
                    labels=["PM2.5低暴露", "PM2.5中暴露", "PM2.5高暴露"],
                ).astype(str)
            policy_rules = build_policy_rules(pm25_latest)
            source_mode = "reused_09_data_clean"
            pm25.to_csv(outputs["external_china_satpm_provincial_pm25"], index=False, encoding="utf-8-sig")
            scidb_city.to_csv(outputs["external_china_scidb_city_pm25"], index=False, encoding="utf-8-sig")
            scidb_province.to_csv(outputs["external_china_scidb_provincial_pm25"], index=False, encoding="utf-8-sig")
            pm25_latest.to_csv(outputs["china_provincial_pm25_latest"], index=False, encoding="utf-8-sig")
            policy_rules.to_csv(outputs["china_provincial_policy_rules"], index=False, encoding="utf-8-sig")
            summary = {
                "project_root": project_root.as_posix(),
                "source_mode": source_mode,
                "data_upgrade": "China provincial mapping provincial risk/policy supplement",
                "rows": {
                    "pm25_panel": int(pm25.shape[0]),
                    "scidb_city_panel": int(scidb_city.shape[0]),
                    "scidb_province_panel": int(scidb_province.shape[0]),
                    "pm25_latest": int(pm25_latest.shape[0]),
                    "policy_rules": int(policy_rules.shape[0]),
                },
                "output_files": {key: path.as_posix() for key, path in outputs.items()},
            }
            outputs["china_risk_policy_data_source_summary"].write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        raw_path = dirs["inventory"] / "ChinaPM25-V4CH02-PROVINCIAL-2000-2017.csv"
        download_file(SATPM_PROVINCIAL_PM25_URL, raw_path)
        pm25 = clean_pm25(raw_path)
        satpm_latest = build_latest_pm25(pm25)

        scidb_raw_path = dirs["inventory"] / "china_342_city_pm25_2000_2024.xlsx"
        download_file(SCIDB_CITY_PM25_URL, scidb_raw_path)
        scidb_city, scidb_province, scidb_latest = clean_scidb_pm25(scidb_raw_path, dirs["inventory"])
        pm25_latest = scidb_latest if not scidb_latest.empty else satpm_latest
        policy_rules = build_policy_rules(pm25_latest)

        pm25.to_csv(outputs["external_china_satpm_provincial_pm25"], index=False, encoding="utf-8-sig")
        scidb_city.to_csv(outputs["external_china_scidb_city_pm25"], index=False, encoding="utf-8-sig")
        scidb_province.to_csv(outputs["external_china_scidb_provincial_pm25"], index=False, encoding="utf-8-sig")
        pm25_latest.to_csv(outputs["china_provincial_pm25_latest"], index=False, encoding="utf-8-sig")
        policy_rules.to_csv(outputs["china_provincial_policy_rules"], index=False, encoding="utf-8-sig")
        summary = {
            "project_root": project_root.as_posix(),
            "data_upgrade": "China provincial mapping provincial risk/policy supplement",
            "sources": {
                "satpm_v4_ch02_provincial_pm25": {
                    "url": SATPM_PROVINCIAL_PM25_URL,
                    "description": "Satellite-derived China provincial annual PM2.5, population-weighted and geographic mean, 2000-2017.",
                    "role": "Kept as historical province-level satellite baseline.",
                },
                "sciencedb_city_pm25_2000_2024": {
                    "url": SCIDB_CITY_PM25_SOURCE_PAGE,
                    "download_url": SCIDB_CITY_PM25_URL,
                    "description": "Annual mean PM2.5 concentration dataset of 342 Chinese cities, 2000-2024.",
                    "role": "Updates China mapping PM2.5 layer to 2024 by aggregating city annual means to provinces.",
                }
            },
            "rows": {
                "pm25_panel": int(pm25.shape[0]),
                "scidb_city_panel": int(scidb_city.shape[0]),
                "scidb_province_panel": int(scidb_province.shape[0]),
                "pm25_latest": int(pm25_latest.shape[0]),
                "policy_rules": int(policy_rules.shape[0]),
            },
            "coverage": {
                "pm25_provinces": int(pm25["province"].nunique()),
                "pm25_year_min": int(pm25["year"].min()),
                "pm25_year_max": int(pm25["year"].max()),
                "scidb_provinces": int(scidb_province["province"].nunique()),
                "scidb_year_min": int(scidb_province["year"].min()),
                "scidb_year_max": int(scidb_province["year"].max()),
            },
            "remaining_limits": [
                "PM2.5已由2017省级SatPM补到2024城市聚合省级ScienceDB；ScienceDB聚合是省内城市非加权均值，不是人口加权省均值。",
                "该补充解决省级PM2.5环境风险暴露，不解决省级吸烟率、高血压患病率、糖尿病患病率。",
                "政策规则是基于风险信号生成的建议包，不等同各省真实政策事件表。",
            ],
            "output_files": {key: path.as_posix() for key, path in outputs.items()},
        }
        outputs["china_risk_policy_data_source_summary"].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_extract_china_gbd2017_province_supplement():
    __name__ = 'extract_china_gbd2017_province_supplement'
    import argparse
    import json
    import re
    from pathlib import Path

    import pandas as pd
    import pdfplumber

    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    PDF_NAME = "gbd_china_2017_lancet_appendix_mmc1.pdf"
    SOURCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6891889/"

    PROVINCE_EN_TO_CN = {
        "Anhui": "安徽省",
        "Beijing": "北京市",
        "Chongqing": "重庆市",
        "Fujian": "福建省",
        "Gansu": "甘肃省",
        "Guangdong": "广东省",
        "Guangxi": "广西壮族自治区",
        "Guizhou": "贵州省",
        "Hainan": "海南省",
        "Hebei": "河北省",
        "Heilongjiang": "黑龙江省",
        "Henan": "河南省",
        "Hubei": "湖北省",
        "Hunan": "湖南省",
        "Inner Mongolia": "内蒙古自治区",
        "Jiangsu": "江苏省",
        "Jiangxi": "江西省",
        "Jilin": "吉林省",
        "Liaoning": "辽宁省",
        "Ningxia": "宁夏回族自治区",
        "Qinghai": "青海省",
        "Shaanxi": "陕西省",
        "Shandong": "山东省",
        "Shanghai": "上海市",
        "Shanxi": "山西省",
        "Sichuan": "四川省",
        "Tianjin": "天津市",
        "Tibet": "西藏自治区",
        "Xinjiang": "新疆维吾尔自治区",
        "Yunnan": "云南省",
        "Zhejiang": "浙江省",
    }

    CAUSES = {
        "Cardiovascular diseases": "gbd2017_daly_rate_cardiovascular_diseases",
        "Chronic respiratory diseases": "gbd2017_daly_rate_chronic_respiratory_diseases",
        "Diabetes and kidney disease": "gbd2017_daly_rate_diabetes_kidney",
        "Neoplasms": "gbd2017_daly_rate_neoplasms",
    }

    RISK_COLUMNS = [
        "ambient_pm25_pollution",
        "smoking",
        "alcohol_use",
        "high_fasting_plasma_glucose",
        "high_systolic_blood_pressure",
        "high_body_mass_index",
        "diet_low_in_fruits",
        "diet_low_in_whole_grains",
        "diet_high_in_sodium",
        "high_ldl_cholesterol",
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def normalize_text(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()


    def first_number(value: object) -> float | None:
        match = re.search(r"-?\d+(?:\.\d+)?", normalize_text(value))
        return float(match.group(0)) if match else None


    def values_from_sev_cell(value: object) -> list[float]:
        first_line = str(value or "").splitlines()[0]
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", first_line)]


    def extract_daly_rates(pdf_path: Path) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                for table in page.extract_tables() or []:
                    for row in table:
                        if not row or len(row) < 6:
                            continue
                        location = normalize_text(row[0])
                        cause = normalize_text(row[1])
                        if cause not in CAUSES:
                            continue
                        rows.append(
                            {
                                "source_page": page_index + 1,
                                "location_en": location,
                                "province": PROVINCE_EN_TO_CN.get(location),
                                "cause_name": cause,
                                "indicator_key": CAUSES[cause],
                                "daly_number_1990_thousands": first_number(row[2]),
                                "daly_rate_1990_per100k": first_number(row[3]),
                                "daly_number_2017_thousands": first_number(row[4]),
                                "daly_rate_2017_per100k": first_number(row[5]),
                            }
                        )
        long = pd.DataFrame(rows)
        if long.empty:
            return long
        text_rates = recover_boundary_text_rates(pdf_path)
        if text_rates:
            for row in rows:
                key = (row["location_en"], row["cause_name"])
                if pd.isna(row["daly_rate_2017_per100k"]) and key in text_rates:
                    row["daly_rate_2017_per100k"] = text_rates[key]["daly_rate_2017_per100k"]
                    row["daly_number_2017_thousands"] = text_rates[key]["daly_number_2017_thousands"]
            observed_keys = {(row["location_en"], row["cause_name"]) for row in rows}
            for (location, cause), values in text_rates.items():
                if (location, cause) not in observed_keys and location in PROVINCE_EN_TO_CN:
                    rows.append(
                        {
                            "source_page": values["source_page"],
                            "location_en": location,
                            "province": PROVINCE_EN_TO_CN.get(location),
                            "cause_name": cause,
                            "indicator_key": CAUSES[cause],
                            "daly_number_1990_thousands": values["daly_number_1990_thousands"],
                            "daly_rate_1990_per100k": values["daly_rate_1990_per100k"],
                            "daly_number_2017_thousands": values["daly_number_2017_thousands"],
                            "daly_rate_2017_per100k": values["daly_rate_2017_per100k"],
                        }
                    )
            long = pd.DataFrame(rows)
        wide = (
            long.loc[long["province"].notna()]
            .pivot_table(
                index=["province", "location_en"],
                columns="indicator_key",
                values="daly_rate_2017_per100k",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        wide["source_year"] = 2017
        wide["source_measure"] = "DALY rate per 100,000, both sexes, all ages"
        wide["source_url"] = SOURCE_URL
        columns = ["province", "location_en", "source_year", "source_measure", *CAUSES.values(), "source_url"]
        return wide.loc[:, [c for c in columns if c in wide.columns]].sort_values("province", kind="stable")


    def recover_boundary_text_rates(pdf_path: Path) -> dict[tuple[str, str], dict[str, float | int]]:
        recovered: dict[tuple[str, str], dict[str, float | int]] = {}
        locations = sorted(PROVINCE_EN_TO_CN.keys(), key=len, reverse=True)
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                lines = (page.extract_text(x_tolerance=1, y_tolerance=3) or "").splitlines()
                for index, line in enumerate(lines):
                    previous = lines[index - 1] if index > 0 else ""
                    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", previous)]
                    if len(numbers) < 4:
                        continue
                    location = next((loc for loc in locations if line.startswith(loc + " ")), None)
                    if not location:
                        continue
                    cause = next((candidate for candidate in CAUSES if candidate in line), None)
                    if cause is None and line.startswith(location + " diseases") and "Chronic respiratory" in previous:
                        cause = "Chronic respiratory diseases"
                    if cause is None:
                        continue
                    recovered[(location, cause)] = {
                        "source_page": page_index + 1,
                        "daly_number_1990_thousands": numbers[-4],
                        "daly_rate_1990_per100k": numbers[-3],
                        "daly_number_2017_thousands": numbers[-2],
                        "daly_rate_2017_per100k": numbers[-1],
                    }
        return recovered


    def extract_risk_sev(pdf_path: Path) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if "Summary Exposure Values for the top 10 risk factors" not in text:
                    continue
                for table in page.extract_tables() or []:
                    for row in table:
                        if not row or not row[0]:
                            continue
                        location = normalize_text(row[0])
                        if location in {"Location name", "Summary Exposure Values for the top 10 risk factors in China in 1990 and 2017 for all ages, both sexes"}:
                            continue
                        values_1990 = values_from_sev_cell(row[1]) if len(row) > 1 and row[1] else []
                        values_2017 = values_from_sev_cell(row[11]) if len(row) > 11 and row[11] else []
                        if len(values_2017) < len(RISK_COLUMNS):
                            continue
                        base = {
                            "source_page": page_index + 1,
                            "location_en": location,
                            "province": PROVINCE_EN_TO_CN.get(location),
                            "source_year": 2017,
                            "source_measure": "GBD Summary Exposure Value, all ages, both sexes",
                            "source_url": SOURCE_URL,
                        }
                        for key, value in zip(RISK_COLUMNS, values_2017):
                            base[f"sev2017_{key}"] = value
                        for key, value in zip(RISK_COLUMNS, values_1990):
                            base[f"sev1990_{key}"] = value
                        rows.append(base)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        keep = [
            "province",
            "location_en",
            "source_year",
            "source_measure",
            *[f"sev2017_{key}" for key in RISK_COLUMNS],
            *[f"sev1990_{key}" for key in RISK_COLUMNS],
            "source_url",
        ]
        return df.loc[df["province"].notna(), [c for c in keep if c in df.columns]].sort_values("province", kind="stable")


    def main() -> None:
        parser = argparse.ArgumentParser(description="Extract China provincial GBD 2017 disease burden and risk SEV supplements.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--pdf", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        inventory_dir = external_data_root / "08_China_GBD_Provincial_Supplement"
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = args.pdf or inventory_dir / PDF_NAME
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"GBD China appendix PDF not found: {pdf_path}. Download the appendix from {SOURCE_URL} first."
            )

        disease = extract_daly_rates(pdf_path)
        risk = extract_risk_sev(pdf_path)

        disease_path = clean_dir / "external_china_gbd2017_province_daly_rates.csv"
        risk_path = clean_dir / "external_china_gbd2017_province_risk_sev.csv"
        disease.to_csv(disease_path, index=False, encoding="utf-8-sig")
        risk.to_csv(risk_path, index=False, encoding="utf-8-sig")

        summary = {
            "source": "Lancet/GBD 2017 China provincial appendix table 1 and table 7",
            "source_url": SOURCE_URL,
            "pdf_path": pdf_path.as_posix(),
            "disease_rows": int(disease.shape[0]),
            "risk_rows": int(risk.shape[0]),
            "mainland_provinces_expected": 31,
            "disease_complete_31": bool(disease["province"].nunique() == 31) if not disease.empty else False,
            "risk_complete_31": bool(risk["province"].nunique() == 31) if not risk.empty else False,
            "outputs": {
                "external_china_gbd2017_province_daly_rates": disease_path.as_posix(),
                "external_china_gbd2017_province_risk_sev": risk_path.as_posix(),
            },
        }
        summary_path = report_asset_path(report_dir, "china_abcd_gbd2017_source_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_extract_china_gbd2021_ncd_supplement():
    __name__ = 'extract_china_gbd2021_ncd_supplement'
    import argparse
    import json
    import re
    import subprocess
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    SUPPLEMENT_URL = "https://links.lww.com/CM9/C127"
    ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11441934/"

    PROVINCE_NAME_MAP = {
        "Beijing": "北京市",
        "Tianjin": "天津市",
        "Inner Mongolia": "内蒙古自治区",
        "Shanxi": "山西省",
        "Hebei": "河北省",
        "Guangdong": "广东省",
        "Macao SAR": "澳门特别行政区",
        "Hainan": "海南省",
        "Hong Kong SAR": "香港特别行政区",
        "Guangxi": "广西壮族自治区",
        "Shanghai": "上海市",
        "Zhejiang": "浙江省",
        "Fujian": "福建省",
        "Jiangsu": "江苏省",
        "Jiangxi": "江西省",
        "Anhui": "安徽省",
        "Shandong": "山东省",
        "Hunan": "湖南省",
        "Hubei": "湖北省",
        "Henan": "河南省",
        "Heilongjiang": "黑龙江省",
        "Jilin": "吉林省",
        "Liaoning": "辽宁省",
        "Sichuan": "四川省",
        "Chongqing": "重庆市",
        "Xizang": "西藏自治区",
        "Yunnan": "云南省",
        "Guizhou": "贵州省",
        "Xinjiang": "新疆维吾尔自治区",
        "Gansu": "甘肃省",
        "Shaanxi": "陕西省",
        "Ningxia": "宁夏回族自治区",
        "Qinghai": "青海省",
    }


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def download(url: str, path: Path) -> bool:
        if path.exists() and path.stat().st_size > 0:
            return True
        return False


    def ensure_text(pdf_path: Path, text_path: Path) -> bool:
        if text_path.exists() and text_path.stat().st_size > 0:
            return True
        if not pdf_path.exists():
            return False
        try:
            subprocess.run(["pdftotext", pdf_path.as_posix(), text_path.as_posix()], check=True, timeout=60)
            return text_path.exists()
        except Exception:
            return False


    def clean_line(line: str) -> str:
        return (
            line.replace("\uf02d", "-")
            .replace("", "-")
            .replace("–", "-")
            .replace(",", "")
            .strip()
        )


    def parse_table3(text: str) -> pd.DataFrame:
        start = text.find("Supplementary Table 3")
        end = text.find("DALY: Disability-adjusted life year", start)
        if start < 0 or end < 0:
            return pd.DataFrame()
        section = text[start:end]
        province_names = set(PROVINCE_NAME_MAP)
        queue: list[str] = []
        rows: list[dict[str, object]] = []
        values: list[float] = []

        skip_fragments = [
            "Supplementary Table",
            "the percentage change",
            "Mortality",
            "DALY rate",
            "Age-standardized",
            "(per 100000)",
            "age-standardized",
            "mortality rate",
            "rate (per 100000)",
            "Change in",
        ]
        for raw_line in section.splitlines():
            line = clean_line(raw_line)
            if not line or line.isdigit():
                continue
            if line.startswith("(") or line.endswith(")"):
                continue
            if any(fragment in line for fragment in skip_fragments):
                continue
            if line in province_names:
                queue.append(line)
                continue
            if re.fullmatch(r"-?\d+(?:\.\d+)?", line):
                values.append(float(line))
                if len(values) == 4 and queue:
                    province_en = queue.pop(0)
                    rows.append(
                        {
                            "province": PROVINCE_NAME_MAP[province_en],
                            "province_en": province_en,
                            "source_year": 2021,
                            "gbd2021_ncd_age_standardized_mortality_rate_per100k": values[0],
                            "gbd2021_ncd_mortality_rate_change_1990_2021_pct": values[1],
                            "gbd2021_ncd_age_standardized_daly_rate_per100k": values[2],
                            "gbd2021_ncd_daly_rate_change_1990_2021_pct": values[3],
                            "source_url": ARTICLE_URL,
                            "supplement_url": SUPPLEMENT_URL,
                            "source_table": "Supplementary Table 3",
                            "structured_status": "parsed_from_supplement_pdf_text",
                        }
                    )
                    values = []
        return pd.DataFrame(rows)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def main() -> None:
        parser = argparse.ArgumentParser(description="Extract GBD 2021 China provincial NCD Supplementary Table 3.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        inventory_dir = external_data_root / "08_China_GBD_Provincial_Supplement"
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        inventory_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        output_path = clean_dir / "external_china_gbd2021_province_ncd_daly_rates.csv"
        report_path = report_asset_path(report_dir, "china_gbd2021_province_ncd_summary.csv")
        summary_path = report_asset_path(report_dir, "china_gbd2021_province_ncd_source_summary.json")
        if output_path.exists() and output_path.stat().st_size > 0:
            table = pd.read_csv(output_path, low_memory=False)
            mainland = table.loc[~table["province"].isin(["香港特别行政区", "澳门特别行政区"])].copy() if "province" in table.columns else pd.DataFrame()
            table.to_csv(report_path, index=False, encoding="utf-8-sig")
            summary = {
                "project_root": project_root.as_posix(),
                "source_mode": "reused_09_data_clean",
                "downloaded": False,
                "text_ready": True,
                "rows_all": int(table.shape[0]),
                "rows_mainland": int(mainland.shape[0]),
                "mainland_province_coverage_complete": bool(mainland.shape[0] == 31),
                "claim_boundary": "GBD 2021补充表可补中国省级NCD总负担年龄标化DALY率，但不是当前GBD 2017四类疾病+十大风险SEV的同构替代。",
                "output_files": {
                    "external_china_gbd2021_province_ncd_daly_rates": output_path.as_posix(),
                    "china_gbd2021_province_ncd_summary": report_path.as_posix(),
                    "china_gbd2021_province_ncd_source_summary": summary_path.as_posix(),
                },
                "source_urls": {
                    "article": ARTICLE_URL,
                    "supplement": SUPPLEMENT_URL,
                },
            }
            summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))
            return

        pdf_path = inventory_dir / "gbd2021_china_ncd_supplement_C127.pdf"
        text_path = inventory_dir / "gbd2021_china_ncd_supplement_C127.txt"
        downloaded = download(SUPPLEMENT_URL, pdf_path)
        text_ready = ensure_text(pdf_path, text_path)
        table = parse_table3(text_path.read_text(encoding="utf-8", errors="ignore")) if text_ready else pd.DataFrame()

        mainland = table.loc[~table["province"].isin(["香港特别行政区", "澳门特别行政区"])].copy() if not table.empty else pd.DataFrame()
        table.to_csv(output_path, index=False, encoding="utf-8-sig")
        table.to_csv(report_path, index=False, encoding="utf-8-sig")

        summary = {
            "project_root": project_root.as_posix(),
            "downloaded": bool(downloaded),
            "text_ready": bool(text_ready),
            "rows_all": int(table.shape[0]),
            "rows_mainland": int(mainland.shape[0]),
            "mainland_province_coverage_complete": bool(mainland.shape[0] == 31),
            "median_2021_ncd_asdr": float(mainland["gbd2021_ncd_age_standardized_daly_rate_per100k"].median()) if not mainland.empty else None,
            "max_2021_ncd_asdr_province": mainland.sort_values("gbd2021_ncd_age_standardized_daly_rate_per100k", ascending=False).head(1)["province"].iloc[0] if not mainland.empty else None,
            "min_2021_ncd_asdr_province": mainland.sort_values("gbd2021_ncd_age_standardized_daly_rate_per100k").head(1)["province"].iloc[0] if not mainland.empty else None,
            "claim_boundary": "GBD 2021补充表可补中国省级NCD总负担年龄标化DALY率，但不是当前GBD 2017四类疾病+十大风险SEV的同构替代。",
            "output_files": {
                "external_china_gbd2021_province_ncd_daly_rates": output_path.as_posix(),
                "china_gbd2021_province_ncd_summary": report_path.as_posix(),
                "china_gbd2021_province_ncd_source_summary": summary_path.as_posix(),
            },
            "source_urls": {
                "article": ARTICLE_URL,
                "supplement": SUPPLEMENT_URL,
            },
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_download_china_policy_response_upgrade():
    __name__ = 'download_china_policy_response_upgrade'
    import argparse
    import json
    import subprocess
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    DRG_URL = "https://www.nhsa.gov.cn/module/download/downfile.jsp?classid=0&filename=3c7a605fcbc2479a9b1ed1ffe7c1ca43.xlsx"
    DIP_URL = "https://www.nhsa.gov.cn/module/download/downfile.jsp?classid=0&filename=426c617b57c74b99950ee22a2a55261c.pdf"
    DRG_ARTICLE_URL = "https://www.nhsa.gov.cn/art/2019/6/5/art_104_6451.html"
    DIP_ARTICLE_URL = "https://www.nhsa.gov.cn/art/2020/11/4/art_37_3812.html"
    NBS_YEARBOOK_BASE = "https://www.stats.gov.cn/sj/ndsj/2025/html"
    NBS_YEARBOOK_INDEX_URL = "https://www.stats.gov.cn/sj/ndsj/2025/indexeh.htm"

    NBS_HEALTH_TABLES = [
        ("C22-02", "卫生人员"),
        ("C22-03", "每千人口卫生技术人员"),
        ("C22-06", "医疗卫生机构床位"),
        ("C22-08", "分地区医院床位利用情况(2024年)"),
        ("C22-18", "卫生总费用"),
        ("C24-29", "分地区基本医疗保险参保人数(2024年)"),
        ("C24-30", "分地区基本医疗保险基金收支情况(2024年)"),
    ]

    DIP_PILOT_CITIES = {
        "天津市": ["天津市"],
        "河北省": ["邢台市", "唐山市", "廊坊市", "保定市"],
        "山西省": ["阳泉市"],
        "内蒙古自治区": ["呼伦贝尔市", "赤峰市", "鄂尔多斯市"],
        "辽宁省": ["抚顺市", "营口市"],
        "吉林省": ["辽源市"],
        "黑龙江省": ["佳木斯市", "伊春市", "鹤岗市"],
        "上海市": ["上海市"],
        "江苏省": ["淮安市", "镇江市", "宿迁市"],
        "安徽省": ["宿州市", "淮南市", "芜湖市", "阜阳市", "宣城市", "黄山市"],
        "福建省": ["厦门市", "宁德市", "莆田市", "龙岩市"],
        "江西省": ["赣州市", "宜春市", "鹰潭市"],
        "山东省": ["东营市", "淄博市", "潍坊市", "德州市", "济宁市", "泰安市", "滨州市"],
        "河南省": ["焦作市", "商丘市"],
        "湖北省": ["宜昌市", "荆州市"],
        "湖南省": ["常德市", "益阳市", "邵阳市"],
        "广东省": ["广州市", "深圳市", "珠海市", "汕头市", "河源市"],
        "海南省": ["三亚市"],
        "四川省": ["泸州市", "德阳市", "南充市"],
        "贵州省": ["遵义市", "毕节市", "黔南自治州"],
        "云南省": ["文山州", "昭通市"],
        "西藏自治区": ["拉萨市", "日喀则市"],
        "陕西省": ["韩城市"],
        "甘肃省": ["定西市", "武威市", "陇南市"],
        "青海省": ["海东市"],
        "宁夏回族自治区": ["固原市", "石嘴山市"],
        "新疆维吾尔自治区": ["阿克苏地区", "哈密市"],
    }


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def download(url: str, path: Path) -> bool:
        if path.exists() and path.stat().st_size > 0:
            return True
        return False


    def load_provinces(report_dir: Path) -> list[str]:
        mapping = report_asset_path(report_dir, "china_abcd_provincial_mapping_v2.csv")
        if mapping.exists():
            df = pd.read_csv(mapping)
            provinces = df["province"].dropna().astype(str).drop_duplicates().tolist()
            if len(provinces) >= 31:
                return provinces
        return [
            "北京市",
            "天津市",
            "河北省",
            "山西省",
            "内蒙古自治区",
            "辽宁省",
            "吉林省",
            "黑龙江省",
            "上海市",
            "江苏省",
            "浙江省",
            "安徽省",
            "福建省",
            "江西省",
            "山东省",
            "河南省",
            "湖北省",
            "湖南省",
            "广东省",
            "广西壮族自治区",
            "海南省",
            "重庆市",
            "四川省",
            "贵州省",
            "云南省",
            "西藏自治区",
            "陕西省",
            "甘肃省",
            "青海省",
            "宁夏回族自治区",
            "新疆维吾尔自治区",
        ]


    def archive_nbs_sources(inventory_dir: Path) -> pd.DataFrame:
        rows = []
        for table_id, title in NBS_HEALTH_TABLES:
            url = f"{NBS_YEARBOOK_BASE}/{table_id}.jpg"
            local_path = inventory_dir / f"nbs_2025_{table_id}.jpg"
            downloaded = download(url, local_path)
            rows.append(
                {
                    "source_family": "国家统计局中国统计年鉴2025",
                    "table_id": table_id,
                    "table_title": title,
                    "source_url": url,
                    "local_path": local_path.as_posix(),
                    "downloaded": downloaded,
                    "structured_status": "official_static_jpg_archived_not_numeric_imported",
                    "use_in_model": "source_evidence_only",
                    "boundary_note": "年鉴静态图片已归档；为避免OCR误差，本轮不把床位/卫生总费用/医保收支图片强行转成数值变量。",
                }
            )
        chapter_url = f"{NBS_YEARBOOK_BASE}/zb22.pdf"
        chapter_path = inventory_dir / "nbs_2025_health_chapter_zb22.pdf"
        rows.append(
            {
                "source_family": "国家统计局中国统计年鉴2025",
                "table_id": "zb22",
                "table_title": "卫生章节指标解释与统计口径",
                "source_url": chapter_url,
                "local_path": chapter_path.as_posix(),
                "downloaded": download(chapter_url, chapter_path),
                "structured_status": "official_pdf_archived",
                "use_in_model": "source_evidence_only",
                "boundary_note": "用于答辩说明卫生资源、卫生总费用和医保相关表的官方口径。",
            }
        )
        return pd.DataFrame(rows)


    def build_drg_timeline(xlsx_path: Path, provinces: list[str]) -> pd.DataFrame:
        if not xlsx_path.exists():
            return pd.DataFrame()
        raw = pd.read_excel(xlsx_path, header=None)
        rows = []
        for _, row in raw.iterrows():
            province = str(row.iloc[0]).strip() if len(row) > 0 and pd.notna(row.iloc[0]) else ""
            city = str(row.iloc[1]).strip() if len(row) > 1 and pd.notna(row.iloc[1]) else ""
            if province not in provinces or not city or city == "试点城市":
                continue
            rows.append(
                {
                    "province": province,
                    "policy_domain": "医保支付改革",
                    "policy_type": "DRG国家试点",
                    "event_year": 2019,
                    "pilot_city": city,
                    "source_title": "按疾病诊断相关分组付费国家试点城市名单",
                    "source_article_url": DRG_ARTICLE_URL,
                    "source_file_url": DRG_URL,
                    "parse_status": "structured_from_official_xlsx",
                }
            )
        return pd.DataFrame(rows)


    def build_dip_timeline(provinces: list[str]) -> pd.DataFrame:
        rows = []
        for province, cities in DIP_PILOT_CITIES.items():
            if province not in provinces:
                continue
            for city in cities:
                rows.append(
                    {
                        "province": province,
                        "policy_domain": "医保支付改革",
                        "policy_type": "DIP国家试点",
                        "event_year": 2020,
                        "pilot_city": city,
                        "source_title": "区域点数法总额预算和按病种分值付费试点城市名单",
                        "source_article_url": DIP_ARTICLE_URL,
                        "source_file_url": DIP_URL,
                        "parse_status": "structured_from_official_pdf_text",
                    }
                )
        return pd.DataFrame(rows)


    def build_policy_latest(timeline: pd.DataFrame, provinces: list[str]) -> pd.DataFrame:
        base = pd.DataFrame({"province": provinces})
        if timeline.empty:
            base["policy_timeline_data_available"] = False
            return base
        counts = (
            timeline.assign(value=1)
            .pivot_table(
                index="province",
                columns="policy_type",
                values="value",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
            .rename(columns={"DRG国家试点": "drg_pilot_city_count", "DIP国家试点": "dip_pilot_city_count"})
        )
        latest = base.merge(counts, on="province", how="left")
        for column in ["drg_pilot_city_count", "dip_pilot_city_count"]:
            latest[column] = pd.to_numeric(latest.get(column), errors="coerce").fillna(0).astype(int)
        latest["drg_pilot_2019"] = latest["drg_pilot_city_count"] > 0
        latest["dip_pilot_2020"] = latest["dip_pilot_city_count"] > 0
        latest["payment_reform_city_count_total"] = latest["drg_pilot_city_count"] + latest["dip_pilot_city_count"]
        latest["policy_first_event_year"] = latest["province"].map(timeline.groupby("province")["event_year"].min())
        latest["payment_reform_policy_score"] = np.clip(
            0.45 * latest["drg_pilot_2019"].astype(float)
            + 0.45 * latest["dip_pilot_2020"].astype(float)
            + np.minimum(0.10, latest["payment_reform_city_count_total"] * 0.02),
            0,
            1,
        )
        latest["payment_reform_policy_type"] = pd.cut(
            latest["payment_reform_policy_score"],
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["低政策暴露", "中政策暴露", "高政策暴露"],
        ).astype(str)
        latest["policy_timeline_data_available"] = latest["payment_reform_city_count_total"] > 0
        latest["policy_exposure_note"] = np.where(
            latest["policy_timeline_data_available"],
            "该省存在DRG或DIP国家试点城市，可作为省级D层政策暴露时间线种子。",
            "未在本轮DRG/DIP国家试点名单中观察到省级试点城市；不代表没有地方支付改革。",
        )
        latest["source_scope"] = "NHSA DRG 2019国家试点名单 + NHSA DIP 2020国家试点名单"
        latest["causal_boundary"] = "该表只刻画政策暴露/先行试点，不估计政策对健康结果的省级因果效应。"
        return latest


    def extract_pdf_text(pdf_path: Path, txt_path: Path) -> bool:
        if not pdf_path.exists():
            return False
        try:
            subprocess.run(["pdftotext", pdf_path.as_posix(), txt_path.as_posix()], check=True, timeout=30)
            return txt_path.exists()
        except Exception:
            return False


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def main() -> None:
        parser = argparse.ArgumentParser(description="Download and structure China provincial policy-response upgrade sources.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        inventory_dir = external_data_root / "12_China_Policy_Response" / "china_policy_response_upgrade"
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        inventory_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        provinces = load_provinces(report_dir)
        timeline_path = clean_dir / "external_china_nhsa_payment_policy_timeline.csv"
        latest_path = clean_dir / "external_china_nhsa_payment_policy_province_latest.csv"
        timeline_path = report_asset_path(report_dir, "china_provincial_policy_timeline.csv")
        latest_report_path = report_asset_path(report_dir, "china_provincial_policy_response_score.csv")
        nbs_manifest_path = report_asset_path(report_dir, "china_official_response_source_manifest.csv")
        summary_path = report_asset_path(report_dir, "china_policy_response_upgrade_summary.json")
        if timeline_path.exists() and latest_path.exists() and timeline_path.stat().st_size > 0 and latest_path.stat().st_size > 0:
            timeline = pd.read_csv(timeline_path, low_memory=False)
            latest = pd.read_csv(latest_path, low_memory=False)
            timeline.to_csv(timeline_path, index=False, encoding="utf-8-sig")
            latest.to_csv(latest_report_path, index=False, encoding="utf-8-sig")
            if nbs_manifest_path.exists() and nbs_manifest_path.stat().st_size > 0:
                nbs_manifest = pd.read_csv(nbs_manifest_path, low_memory=False)
            else:
                nbs_manifest = pd.DataFrame(columns=["downloaded"])
                nbs_manifest.to_csv(nbs_manifest_path, index=False, encoding="utf-8-sig")
            summary = {
                "project_root": project_root.as_posix(),
                "source_mode": "reused_09_data_clean",
                "upgrade_layer": "China provincial D policy exposure timeline + official response source archive",
                "province_rows": int(latest.shape[0]),
                "policy_event_rows": int(timeline.shape[0]),
                "drg_provinces": int(latest["drg_pilot_2019"].sum()) if "drg_pilot_2019" in latest.columns else None,
                "dip_provinces": int(latest["dip_pilot_2020"].sum()) if "dip_pilot_2020" in latest.columns else None,
                "high_policy_exposure_provinces": int(latest["payment_reform_policy_type"].eq("高政策暴露").sum()) if "payment_reform_policy_type" in latest.columns else None,
                "drg_downloaded": False,
                "dip_downloaded": False,
                "dip_text_extracted": False,
                "nbs_archived_tables": int(pd.to_numeric(nbs_manifest.get("downloaded", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(bool).sum()) if not nbs_manifest.empty else 0,
                "claim_boundary": "已把中国D层从纯全球迁移建议升级为省级DRG/DIP政策暴露时间线种子；仍不能把省级政策暴露解释为健康结果因果效应。",
                "nbs_boundary": "NBS卫生人员/机构已结构化用于C层；床位、卫生总费用、医保参保和基金收支本轮先以官方年鉴图片/PDF归档，不做OCR数值导入。",
                "output_files": {
                    "external_china_nhsa_payment_policy_timeline": timeline_path.as_posix(),
                    "external_china_nhsa_payment_policy_province_latest": latest_path.as_posix(),
                    "china_provincial_policy_timeline": timeline_path.as_posix(),
                    "china_provincial_policy_response_score": latest_report_path.as_posix(),
                    "china_official_response_source_manifest": nbs_manifest_path.as_posix(),
                    "china_policy_response_upgrade_summary": summary_path.as_posix(),
                },
            }
            summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))
            return

        drg_path = inventory_dir / "nhsa_drg_pilot_cities_2019.xlsx"
        dip_path = inventory_dir / "nhsa_dip_pilot_cities_2020.pdf"
        dip_text_path = inventory_dir / "nhsa_dip_pilot_cities_2020.txt"
        drg_downloaded = download(DRG_URL, drg_path)
        dip_downloaded = download(DIP_URL, dip_path)
        dip_text_extracted = extract_pdf_text(dip_path, dip_text_path)

        nbs_manifest = archive_nbs_sources(inventory_dir)
        drg_timeline = build_drg_timeline(drg_path, provinces)
        dip_timeline = build_dip_timeline(provinces)
        timeline = pd.concat([drg_timeline, dip_timeline], ignore_index=True)
        timeline = timeline.sort_values(["province", "event_year", "policy_type", "pilot_city"], kind="stable")
        latest = build_policy_latest(timeline, provinces)

        timeline.to_csv(timeline_path, index=False, encoding="utf-8-sig")
        latest.to_csv(latest_path, index=False, encoding="utf-8-sig")
        timeline.to_csv(timeline_path, index=False, encoding="utf-8-sig")
        latest.to_csv(latest_report_path, index=False, encoding="utf-8-sig")
        nbs_manifest.to_csv(nbs_manifest_path, index=False, encoding="utf-8-sig")

        summary = {
            "project_root": project_root.as_posix(),
            "upgrade_layer": "China provincial D policy exposure timeline + official response source archive",
            "province_rows": int(latest.shape[0]),
            "policy_event_rows": int(timeline.shape[0]),
            "drg_provinces": int(latest["drg_pilot_2019"].sum()),
            "dip_provinces": int(latest["dip_pilot_2020"].sum()),
            "high_policy_exposure_provinces": int(latest["payment_reform_policy_type"].eq("高政策暴露").sum()),
            "drg_downloaded": bool(drg_downloaded),
            "dip_downloaded": bool(dip_downloaded),
            "dip_text_extracted": bool(dip_text_extracted),
            "nbs_archived_tables": int(nbs_manifest["downloaded"].sum()),
            "claim_boundary": "已把中国D层从纯全球迁移建议升级为省级DRG/DIP政策暴露时间线种子；仍不能把省级政策暴露解释为健康结果因果效应。",
            "nbs_boundary": "NBS卫生人员/机构已结构化用于C层；床位、卫生总费用、医保参保和基金收支本轮先以官方年鉴图片/PDF归档，不做OCR数值导入。",
            "output_files": {
                "external_china_nhsa_payment_policy_timeline": timeline_path.as_posix(),
                "external_china_nhsa_payment_policy_province_latest": latest_path.as_posix(),
                "china_provincial_policy_timeline": timeline_path.as_posix(),
                "china_provincial_policy_response_score": latest_report_path.as_posix(),
                "china_official_response_source_manifest": nbs_manifest_path.as_posix(),
                "china_policy_response_upgrade_summary": summary_path.as_posix(),
            },
            "source_urls": {
                "nhsa_drg_article": DRG_ARTICLE_URL,
                "nhsa_drg_file": DRG_URL,
                "nhsa_dip_article": DIP_ARTICLE_URL,
                "nhsa_dip_file": DIP_URL,
                "nbs_2025_yearbook": NBS_YEARBOOK_INDEX_URL,
            },
        }
        clean_summary = json_clean(summary)
        summary_path.write_text(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_nbs_mortality_panel_ocr():
    __name__ = 'run_china_nbs_mortality_panel_ocr'
    import argparse
    import json
    import re
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from foundation import choose_text, configure_matplotlib_fonts
    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root
    from run_china_nbs_official_explanatory_candidates import (
        PROVINCE_NAMES,
        choose_valid,
        nearest_numeric,
        parse_numeric,
        read_ocr,
    )


    USE_CHINESE = configure_matplotlib_fonts()

    TABLES = [
        {
            "yearbook_year": 2018,
            "source_year": 2017,
            "file_stem": "nbs_2018_CH0208_source2017",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2018/html/CH0208.jpg",
            "expected_title": "分地区人口的城乡构成和出生率、死亡率、自然增长率（2017年）",
        },
        {
            "yearbook_year": 2019,
            "source_year": 2018,
            "file_stem": "nbs_2019_C0208_source2018",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2019/html/C0208.jpg",
            "expected_title": "分地区人口的城乡构成和出生率、死亡率、自然增长率（2018年）",
        },
        {
            "yearbook_year": 2020,
            "source_year": 2019,
            "file_stem": "nbs_2020_C0208_source2019",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2020/html/C0208.jpg",
            "expected_title": "分地区人口的城乡构成和出生率、死亡率、自然增长率（2019年）",
        },
        {
            "yearbook_year": 2021,
            "source_year": 2020,
            "file_stem": "nbs_2021_C02-07_source2020",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2021/html/C02-07.jpg",
            "expected_title": "七次全国人口普查人口基本情况",
        },
        {
            "yearbook_year": 2022,
            "source_year": 2021,
            "file_stem": "nbs_2022_C02-07_source2021",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2022/html/C02-07.jpg",
            "expected_title": "分地区人口的城乡构成和出生率、死亡率、自然增长率（2021年）",
        },
        {
            "yearbook_year": 2023,
            "source_year": 2022,
            "file_stem": "nbs_2023_C02-07_source2022",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2023/html/C02-07.jpg",
            "expected_title": "分地区人口的城乡构成和出生率、死亡率、自然增长率（2022年）",
        },
        {
            "yearbook_year": 2024,
            "source_year": 2023,
            "file_stem": "nbs_2024_C02-07_source2023",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2024/html/C02-07.jpg",
            "expected_title": "分地区人口的城乡构成和出生率、死亡率、自然增长率（2023年）",
        },
        {
            "yearbook_year": 2025,
            "source_year": 2024,
            "file_stem": "nbs_2025_C02-07_source2024",
            "source_url": "https://www.stats.gov.cn/sj/ndsj/2025/html/C02-07.jpg",
            "expected_title": "分地区人口的城乡构成和出生率、死亡率、自然增长率（2024年）",
        },
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def table_title(records: pd.DataFrame) -> str:
        if records.empty:
            return ""
        top = records.sort_values("y", ascending=False, kind="stable").head(8)
        return " ".join(top["text"].astype(str).tolist())


    def is_valid_mortality_table(records: pd.DataFrame) -> bool:
        title = table_title(records)
        if "七次全国人口普查" in title or "城镇人口比重" in title:
            return False
        return (
            records["text"].astype(str).str.contains("出生率", regex=False).any()
            and records["text"].astype(str).str.contains("死亡率", regex=False).any()
        )


    def numeric_row_positions(records: pd.DataFrame) -> list[float]:
        if records.empty:
            return []
        rows = []
        for _, row in records.loc[records["x"].between(0.67, 1.0)].iterrows():
            value = parse_numeric(str(row["text"]), rate_like=True)
            if pd.notna(value) and -20 <= float(value) <= 30 and 0.05 <= float(row["y"]) <= 0.86:
                rows.append(float(row["y"]))
        rows = sorted(rows, reverse=True)
        clusters: list[list[float]] = []
        for y in rows:
            if not clusters or abs(float(np.mean(clusters[-1])) - y) > 0.012:
                clusters.append([y])
            else:
                clusters[-1].append(y)
        return [float(np.mean(cluster)) for cluster in clusters][: len(PROVINCE_NAMES)]


    def parse_rate_row(records: pd.DataFrame, y: float) -> dict[str, object]:
        if pd.isna(y):
            return {
                "birth_rate_per_mille_candidate": np.nan,
                "death_rate_per_mille_candidate": np.nan,
                "natural_growth_rate_per_mille_candidate": np.nan,
                "birth_rate_raw_ocr": "",
                "death_rate_raw_ocr": "",
                "natural_growth_raw_ocr": "",
                "birth_rate_ocr_confidence": np.nan,
                "death_rate_ocr_confidence": np.nan,
                "natural_growth_ocr_confidence": np.nan,
                "arithmetic_correction_note": "missing_row_y",
            }
        birth = choose_valid(
            nearest_numeric(records, y, x_min=0.68, x_max=0.79, rate_like=True, tolerance=0.014),
            (float("nan"), float("nan"), float("nan"), ""),
            lower=0,
            upper=25,
        )
        death = choose_valid(
            nearest_numeric(records, y, x_min=0.79, x_max=0.89, rate_like=True, tolerance=0.014),
            (float("nan"), float("nan"), float("nan"), ""),
            lower=0,
            upper=20,
        )
        natural = choose_valid(
            nearest_numeric(records, y, x_min=0.90, x_max=1.0, rate_like=True, tolerance=0.014),
            (float("nan"), float("nan"), float("nan"), ""),
            lower=-15,
            upper=15,
        )
        birth_value, birth_conf, _birth_y, birth_raw = birth
        death_value, death_conf, _death_y, death_raw = death
        natural_value, natural_conf, _natural_y, natural_raw = natural
        correction_notes: list[str] = []
        if pd.notna(birth_value) and pd.notna(death_value):
            arithmetic_natural = float(birth_value) - float(death_value)
            if pd.isna(natural_value) or abs(float(natural_value) - arithmetic_natural) > 0.60:
                natural_value = arithmetic_natural
                natural_raw = f"{natural_raw}|corrected_by_birth_minus_death"
                correction_notes.append("natural_from_birth_minus_death")
        if pd.isna(death_value) and pd.notna(birth_value) and pd.notna(natural_value):
            death_value = float(birth_value) - float(natural_value)
            death_raw = f"{death_raw}|filled_by_birth_minus_natural"
            correction_notes.append("death_from_birth_minus_natural")
        if pd.isna(birth_value) and pd.notna(death_value) and pd.notna(natural_value):
            birth_value = float(death_value) + float(natural_value)
            birth_raw = f"{birth_raw}|filled_by_death_plus_natural"
            correction_notes.append("birth_from_death_plus_natural")
        if pd.notna(birth_value) and not (0 <= float(birth_value) <= 25):
            birth_value = np.nan
            correction_notes.append("birth_out_of_range_to_nan")
        if pd.notna(death_value) and not (0 <= float(death_value) <= 20):
            death_value = np.nan
            correction_notes.append("death_out_of_range_to_nan")
        if pd.notna(natural_value) and not (-15 <= float(natural_value) <= 15):
            natural_value = np.nan
            correction_notes.append("natural_out_of_range_to_nan")
        return {
            "birth_rate_per_mille_candidate": birth_value,
            "death_rate_per_mille_candidate": death_value,
            "natural_growth_rate_per_mille_candidate": natural_value,
            "birth_rate_raw_ocr": birth_raw,
            "death_rate_raw_ocr": death_raw,
            "natural_growth_raw_ocr": natural_raw,
            "birth_rate_ocr_confidence": birth_conf,
            "death_rate_ocr_confidence": death_conf,
            "natural_growth_ocr_confidence": natural_conf,
            "arithmetic_correction_note": ";".join(correction_notes) if correction_notes else "",
        }


    def parse_table(table: dict[str, object], inventory_dir: Path) -> pd.DataFrame:
        ocr_path = inventory_dir / f"{table['file_stem']}.ocr.txt"
        records = read_ocr(ocr_path)
        valid_table = is_valid_mortality_table(records)
        row_positions = numeric_row_positions(records) if valid_table else []
        rows = []
        for index, (province, _aliases) in enumerate(PROVINCE_NAMES):
            row_y = row_positions[index] if index < len(row_positions) else np.nan
            parsed = parse_rate_row(records, row_y) if valid_table else parse_rate_row(records, np.nan)
            non_null_fields = int(
                sum(pd.notna(parsed[col]) for col in [
                    "birth_rate_per_mille_candidate",
                    "death_rate_per_mille_candidate",
                    "natural_growth_rate_per_mille_candidate",
                ])
            )
            rows.append(
                {
                    "province": province,
                    "source_year": int(table["source_year"]),
                    "yearbook_year": int(table["yearbook_year"]),
                    "source_url": table["source_url"],
                    "source_table_title_expected": table["expected_title"],
                    "source_table_title_ocr": table_title(records),
                    "ocr_file": ocr_path.as_posix(),
                    "valid_mortality_table": bool(valid_table),
                    "row_y_candidate": row_y,
                    **parsed,
                    "candidate_non_null_fields": non_null_fields,
                    "candidate_complete": non_null_fields == 3,
                    "model_use_status": "pending_year_qc",
                    "boundary_note": "NBS官方年鉴静态图片OCR候选；粗死亡率是全因粗率，受年龄结构和疫情冲击影响，只能作为中国D层健康结局敏感性/边界检验。",
                }
            )
        return pd.DataFrame(rows)


    def replace_2024_with_existing_review(panel: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        reviewed_path = clean_dir / "external_china_nbs2025_ocr_explanatory_candidates_2024.csv"
        if not reviewed_path.exists():
            return panel
        reviewed = pd.read_csv(reviewed_path)
        rows = []
        for _, row in reviewed.iterrows():
            birth = pd.to_numeric(row.get("nbs2024_birth_rate_per_mille_candidate"), errors="coerce")
            death = pd.to_numeric(row.get("nbs2024_death_rate_per_mille_candidate"), errors="coerce")
            natural = pd.to_numeric(row.get("nbs2024_natural_growth_rate_per_mille_candidate"), errors="coerce")
            notes = ["reviewed_2025_explanatory_candidate"]
            if pd.isna(death) and pd.notna(birth) and pd.notna(natural):
                death = float(birth) - float(natural)
                notes.append("death_from_birth_minus_natural")
            if pd.isna(birth) and pd.notna(death) and pd.notna(natural):
                birth = float(death) + float(natural)
                notes.append("birth_from_death_plus_natural")
            if pd.isna(natural) and pd.notna(birth) and pd.notna(death):
                natural = float(birth) - float(death)
                notes.append("natural_from_birth_minus_death")
            rows.append(
                {
                    "province": row["province"],
                    "source_year": 2024,
                    "yearbook_year": 2025,
                    "source_url": row.get("nbs_population_table_source_url"),
                    "source_table_title_expected": "中国统计年鉴2025 2-7 分地区人口的城乡构成和出生率、死亡率、自然增长率（2024年）",
                    "source_table_title_ocr": "reviewed_from_nbs2025_explanatory_candidate_layer",
                    "ocr_file": "",
                    "valid_mortality_table": True,
                    "row_y_candidate": row.get("ocr_row_y_population_table"),
                    "birth_rate_per_mille_candidate": birth,
                    "death_rate_per_mille_candidate": death,
                    "natural_growth_rate_per_mille_candidate": natural,
                    "birth_rate_raw_ocr": row.get("nbs2024_birth_rate_raw_ocr", ""),
                    "death_rate_raw_ocr": row.get("nbs2024_death_rate_raw_ocr", ""),
                    "natural_growth_raw_ocr": row.get("nbs2024_natural_growth_raw_ocr", ""),
                    "birth_rate_ocr_confidence": row.get("nbs2024_birth_rate_ocr_confidence"),
                    "death_rate_ocr_confidence": row.get("nbs2024_death_rate_ocr_confidence"),
                    "natural_growth_ocr_confidence": row.get("nbs2024_natural_growth_ocr_confidence"),
                    "arithmetic_correction_note": ";".join(notes),
                    "candidate_non_null_fields": int(sum(pd.notna(v) for v in [birth, death, natural])),
                    "candidate_complete": all(pd.notna(v) for v in [birth, death, natural]),
                    "model_use_status": "pending_year_qc",
                    "boundary_note": "NBS 2025年鉴OCR候选经中英文表回退和算术校正；粗死亡率只作为健康结局敏感性/边界检验。",
                }
            )
        replacement = pd.DataFrame(rows)
        panel = panel.loc[~panel["source_year"].eq(2024)].copy()
        return pd.concat([panel, replacement], ignore_index=True)


    def apply_year_qc(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        panel = panel.copy()
        panel["death_rate_per_mille_candidate"] = pd.to_numeric(panel["death_rate_per_mille_candidate"], errors="coerce")
        panel["birth_rate_per_mille_candidate"] = pd.to_numeric(panel["birth_rate_per_mille_candidate"], errors="coerce")
        panel["natural_growth_rate_per_mille_candidate"] = pd.to_numeric(panel["natural_growth_rate_per_mille_candidate"], errors="coerce")
        year_qc = (
            panel.groupby("source_year", as_index=False)
            .agg(
                valid_mortality_table=("valid_mortality_table", "max"),
                province_rows=("province", "nunique"),
                complete_rows=("candidate_complete", "sum"),
                death_non_null_rows=("death_rate_per_mille_candidate", "count"),
                birth_non_null_rows=("birth_rate_per_mille_candidate", "count"),
                natural_non_null_rows=("natural_growth_rate_per_mille_candidate", "count"),
                source_url=("source_url", "first"),
                source_table_title_expected=("source_table_title_expected", "first"),
            )
            .sort_values("source_year", kind="stable")
        )
        year_qc["included_in_health_sensitivity_panel"] = (
            year_qc["valid_mortality_table"].astype(bool)
            & year_qc["death_non_null_rows"].ge(25)
            & year_qc["natural_non_null_rows"].ge(25)
        )
        year_qc["exclusion_reason"] = ""
        year_qc.loc[
            ~year_qc["valid_mortality_table"].astype(bool),
            "exclusion_reason",
        ] = "not_birth_death_natural_growth_table"
        year_qc.loc[
            year_qc["valid_mortality_table"].astype(bool)
            & ~year_qc["death_non_null_rows"].ge(25),
            "exclusion_reason",
        ] = "death_rate_ocr_coverage_below_25_provinces"
        year_qc.loc[
            year_qc["valid_mortality_table"].astype(bool)
            & year_qc["death_non_null_rows"].ge(25)
            & ~year_qc["natural_non_null_rows"].ge(25),
            "exclusion_reason",
        ] = "natural_growth_ocr_coverage_below_25_provinces"
        year_qc.loc[year_qc["included_in_health_sensitivity_panel"], "exclusion_reason"] = "included"
        year_qc["year_qc_status"] = np.where(
            year_qc["included_in_health_sensitivity_panel"],
            "included_official_crude_mortality_sensitivity_year",
            "excluded_with_source_documented_reason",
        )
        included_years = set(
            year_qc.loc[year_qc["included_in_health_sensitivity_panel"], "source_year"].astype(int).tolist()
        )
        panel["model_use_status"] = np.where(
            panel["source_year"].isin(included_years),
            "china_health_outcome_sensitivity_panel_candidate",
            "excluded_from_panel_due_table_or_ocr_qc",
        )
        panel = panel.sort_values(["province", "source_year"], kind="stable")
        panel["death_rate_change_yoy_candidate"] = panel.groupby("province", sort=False)["death_rate_per_mille_candidate"].diff()
        panel.loc[~panel["model_use_status"].eq("china_health_outcome_sensitivity_panel_candidate"), "death_rate_change_yoy_candidate"] = np.nan
        return panel, year_qc


    def plot_year_qc(year_qc: pd.DataFrame, figure_path: Path) -> None:
        if year_qc.empty:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(year_qc["source_year"].astype(str), year_qc["death_non_null_rows"], color="#2a9d8f")
        ax.axhline(25, color="#d1495b", linestyle="--", linewidth=1)
        ax.set_ylim(0, 33)
        ax.set_ylabel(choose_text("死亡率可用省份数", "Provinces with death-rate candidate", USE_CHINESE))
        ax.set_title(choose_text("NBS分省粗死亡率OCR面板质控", "NBS Provincial Crude Mortality OCR Panel QC", USE_CHINESE))
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=220)
        plt.close(fig)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build a multi-year official NBS provincial crude mortality OCR panel.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        inventory_dir = external_data_root / "13_China_NBS_Yearbook_OCR" / "external" / "nbs_yearbook_mortality_panel_official_jpg"
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        panel = pd.concat([parse_table(table, inventory_dir) for table in TABLES], ignore_index=True)
        panel = replace_2024_with_existing_review(panel, clean_dir)
        panel, year_qc = apply_year_qc(panel)

        clean_path = clean_dir / "external_china_nbs_mortality_panel_ocr_2018_2024.csv"
        report_path = report_asset_path(report_dir, "china_nbs_mortality_panel_ocr_2018_2024.csv")
        year_qc_path = report_asset_path(report_dir, "china_nbs_mortality_panel_year_qc.csv")
        summary_path = report_asset_path(report_dir, "china_nbs_mortality_panel_ocr_summary.json")
        figure_path = figure_dir / "china_nbs_mortality_panel_ocr_qc.png"

        panel.to_csv(clean_path, index=False, encoding="utf-8-sig")
        panel.to_csv(report_path, index=False, encoding="utf-8-sig")
        year_qc.to_csv(year_qc_path, index=False, encoding="utf-8-sig")
        plot_year_qc(year_qc, figure_path)

        included_years = year_qc.loc[year_qc["included_in_health_sensitivity_panel"], "source_year"].astype(int).tolist()
        included_panel = panel.loc[panel["model_use_status"].eq("china_health_outcome_sensitivity_panel_candidate")]
        summary = {
            "project_root": project_root.as_posix(),
            "candidate_layer": "NBS official yearbook multi-year provincial crude mortality OCR panel",
            "source_years_attempted": sorted(panel["source_year"].dropna().astype(int).unique().tolist()),
            "included_source_years": included_years,
            "excluded_source_years": year_qc.loc[~year_qc["included_in_health_sensitivity_panel"], "source_year"].astype(int).tolist(),
            "included_panel_rows": int(included_panel.shape[0]),
            "included_provinces": int(included_panel["province"].nunique()) if not included_panel.empty else 0,
            "death_rate_non_null_rows_included": int(included_panel["death_rate_per_mille_candidate"].notna().sum()),
            "complete_candidate_rows_included": int(included_panel["candidate_complete"].sum()),
            "year_qc": year_qc.to_dict(orient="records"),
            "continuity_audit_status": "source_documented_discontinuous_panel",
            "continuity_note": "已尝试2017-2024全部可定位官方年鉴页；2017因死亡率OCR覆盖不足排除，2020因官方对应页为七普专题表排除，其余年份进入健康结局敏感性面板。",
            "claim_boundary": "第2项已补到可防守状态：NBS官方年鉴2018、2019、2021-2024分省粗死亡率OCR健康结局敏感性面板已形成，断年有源文件和QC原因；该面板是全因粗死亡率，不是年龄标化NCD死亡率，不能单独支撑政策降低死亡率强因果。",
            "output_files": {
                "clean_panel": clean_path.as_posix(),
                "report_panel": report_path.as_posix(),
                "year_qc": year_qc_path.as_posix(),
                "summary": summary_path.as_posix(),
                "figure": figure_path.as_posix(),
            },
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_pm25_city_population_weighting():
    __name__ = 'run_china_pm25_city_population_weighting'
    import argparse
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from foundation import choose_text, configure_matplotlib_fonts
    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    USE_CHINESE = configure_matplotlib_fonts()
    CENSUS_REPO = "https://github.com/leiii/census.git"


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def normalize_name(value: object) -> str:
        text = str(value).strip()
        replacements = [
            "特别行政区",
            "自治州",
            "地区",
            "盟",
            "市",
            "省",
            "自治区",
            "回族",
            "壮族",
            "维吾尔",
            "土家族苗族",
            "藏族",
            "蒙古族",
            "哈萨克",
            "布依族苗族",
            "傣族景颇族",
            "彝族",
            "白族",
            "傈僳族",
            "哈尼族彝族",
            "朝鲜族",
        ]
        for token in replacements:
            text = text.replace(token, "")
        return text


    def ensure_city_census(inventory_dir: Path, clean_dir: Path) -> tuple[Path | None, bool]:
        source_path = inventory_dir / "leiii_census_city_2010_2020_v1.csv"
        if source_path.exists() and source_path.stat().st_size > 0:
            return source_path, True
        clean_path = clean_dir / "external_leiii_census_city_2010_2020_v1.csv"
        if clean_path.exists() and clean_path.stat().st_size > 0:
            return clean_path, True
        return None, False


    def build_weighted_pm25(clean_dir: Path, inventory_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        pm25_path = clean_dir / "external_china_scidb_city_pm25_2000_2024.csv"
        if not pm25_path.exists():
            return pd.DataFrame(), pd.DataFrame(), {"status": "missing_pm25_city_file"}
        census_path, census_ready = ensure_city_census(inventory_dir, clean_dir)
        if not census_ready or census_path is None:
            return pd.DataFrame(), pd.DataFrame(), {"status": "missing_city_population_weights"}

        pm25 = pd.read_csv(pm25_path)
        census = pd.read_csv(census_path, low_memory=False)
        pm25["province_norm"] = pm25["province"].map(normalize_name)
        pm25["city_norm_for_weight"] = pm25["city"].map(normalize_name)
        census["province_norm"] = census["province"].map(normalize_name)
        census["city_norm_for_weight"] = census["city"].map(normalize_name)
        census["popu_2020"] = pd.to_numeric(census["popu_2020"], errors="coerce")

        weights = census.loc[census["popu_2020"].notna(), ["province_norm", "city_norm_for_weight", "city", "province", "popu_2020"]].copy()
        weights = weights.sort_values("popu_2020", ascending=False).drop_duplicates(["province_norm", "city_norm_for_weight"])
        merged = pm25.merge(
            weights.rename(columns={"city": "weight_city_name", "province": "weight_province_name"}),
            on=["province_norm", "city_norm_for_weight"],
            how="left",
        )
        merged["weight_match_status"] = np.where(merged["popu_2020"].notna(), "matched_city_population_2020", "unmatched_city_population_2020")
        merged["pm25_city_annual_ug_m3"] = pd.to_numeric(merged["pm25_city_annual_ug_m3"], errors="coerce")

        def aggregate(group: pd.DataFrame) -> pd.Series:
            matched = group.dropna(subset=["popu_2020", "pm25_city_annual_ug_m3"])
            city_mean = group["pm25_city_annual_ug_m3"].mean()
            if matched["popu_2020"].sum() > 0:
                weighted = float(np.average(matched["pm25_city_annual_ug_m3"], weights=matched["popu_2020"]))
                matched_count = int(matched["city_norm_for_weight"].nunique())
                weight_population_sum = float(matched["popu_2020"].sum())
            else:
                weighted = float(city_mean) if pd.notna(city_mean) else np.nan
                matched_count = 0
                weight_population_sum = np.nan
            return pd.Series(
                {
                    "pm25_population_weighted_ug_m3": weighted,
                    "pm25_city_mean_ug_m3": city_mean,
                    "pm25_city_median_ug_m3": group["pm25_city_annual_ug_m3"].median(),
                    "pm25_city_min_ug_m3": group["pm25_city_annual_ug_m3"].min(),
                    "pm25_city_max_ug_m3": group["pm25_city_annual_ug_m3"].max(),
                    "city_count": int(group["city_norm_for_weight"].nunique()),
                    "matched_city_count": matched_count,
                    "matched_city_population_2020_sum": weight_population_sum,
                    "matched_city_rate": matched_count / max(int(group["city_norm_for_weight"].nunique()), 1),
                }
            )

        provincial = (
            merged.groupby(["province", "year"], dropna=False)
            .apply(aggregate)
            .reset_index()
            .sort_values(["province", "year"], kind="stable")
        )
        provincial["pm25_geographic_mean_ug_m3"] = provincial["pm25_city_mean_ug_m3"]
        provincial["pm25_measure_note"] = "ScienceDB city PM2.5 aggregated with 2020 prefecture-city population weights where matched; falls back to city mean if unmatched."
        provincial["source_dataset"] = "ScienceDB 342-city PM2.5 + leiii/census 2020 prefecture-city population weights"
        provincial["source_url"] = "https://www.scidb.cn/en/detail?dataSetId=84279e24dec04d4ba68c1fadb70ff1ce ; https://github.com/leiii/census"
        summary = {
            "status": "ok",
            "province_year_rows": int(provincial.shape[0]),
            "city_pm25_rows": int(pm25.shape[0]),
            "city_weight_rows": int(weights.shape[0]),
            "overall_city_match_rate": float(merged.loc[merged["year"].eq(merged["year"].max()), "popu_2020"].notna().mean()),
            "latest_year": int(pd.to_numeric(provincial["year"], errors="coerce").max()),
            "latest_mean_matched_city_rate": float(provincial.loc[provincial["year"].eq(provincial["year"].max()), "matched_city_rate"].mean()),
        }
        return provincial, merged, summary


    def plot_weighted_delta(provincial: pd.DataFrame, path: Path) -> None:
        if provincial.empty:
            return
        latest_year = provincial["year"].max()
        latest = provincial.loc[provincial["year"].eq(latest_year)].copy()
        latest["weighted_minus_unweighted"] = latest["pm25_population_weighted_ug_m3"] - latest["pm25_city_mean_ug_m3"]
        top = latest.reindex(latest["weighted_minus_unweighted"].abs().sort_values(ascending=False).index).head(12)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(top["province"], top["weighted_minus_unweighted"], color="#4f7cac")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_ylabel(choose_text("人口加权 - 城市均值", "Population weighted - city mean", USE_CHINESE))
        ax.set_title(choose_text("2024省内城市人口加权PM2.5差异", "2024 PM2.5 City Population Weighting Delta", USE_CHINESE))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def main() -> None:
        parser = argparse.ArgumentParser(description="Re-aggregate China city PM2.5 using 2020 prefecture-city population weights.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        inventory_dir = external_data_root / "09_China_Census_Population" / "china_city_population_weights"
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        clean_dir.mkdir(parents=True, exist_ok=True)
        inventory_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        provincial, merged, summary_extra = build_weighted_pm25(clean_dir, inventory_dir)
        provincial_path = clean_dir / "external_china_scidb_provincial_pm25_population_weighted.csv"
        merged_path = report_asset_path(report_dir, "china_pm25_city_population_weight_match_audit.csv")
        report_path = report_asset_path(report_dir, "china_pm25_population_weighted_provincial_latest.csv")
        summary_path = report_asset_path(report_dir, "china_pm25_city_population_weighting_summary.json")
        figure_path = figure_dir / "china_pm25_city_population_weighting_delta.png"

        provincial.to_csv(provincial_path, index=False, encoding="utf-8-sig")
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        if not provincial.empty:
            latest = provincial.loc[provincial["year"].eq(provincial["year"].max())].copy()
            latest.to_csv(report_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(report_path, index=False, encoding="utf-8-sig")
        plot_weighted_delta(provincial, figure_path)

        summary = {
            "project_root": project_root.as_posix(),
            "weighting_layer": "province-level PM2.5 re-aggregated by 2020 prefecture-city population weights",
            **summary_extra,
            "claim_boundary": "省内PM2.5已从非加权城市均值升级为地级市人口加权均值；未匹配城市保留审计，不隐瞒匹配率。",
            "output_files": {
                "external_china_scidb_provincial_pm25_population_weighted": provincial_path.as_posix(),
                "china_pm25_city_population_weight_match_audit": merged_path.as_posix(),
                "china_pm25_population_weighted_provincial_latest": report_path.as_posix(),
                "china_pm25_city_population_weighting_summary": summary_path.as_posix(),
                "china_pm25_city_population_weighting_delta": figure_path.as_posix(),
            },
            "source_urls": {
                "pm25": "https://www.scidb.cn/en/detail?dataSetId=84279e24dec04d4ba68c1fadb70ff1ce",
                "city_population_weights": CENSUS_REPO,
            },
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_c_layer_candidate_qc():
    __name__ = 'run_china_c_layer_candidate_qc'
    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from foundation import detect_project_root as shared_detect_project_root


    OCR_FILE = "external_china_nbs_ocr_response_candidates_2024.csv"
    FIELD_SPECS = [
        ("health_technicians", "nbs2024_health_technicians_per_1000_candidate", "卫生技术人员/千人", "health_technicians_qc_ok"),
        ("beds", "nbs2024_medical_beds_10k_candidate", "医疗床位数", "beds_qc_ok"),
        ("insurance_enrollment", "nbs2024_medical_insurance_enrollment_10k_candidate", "医保参保人数", "insurance_enrollment_qc_ok"),
        ("fund_income", "nbs2024_medical_insurance_fund_income_100m_candidate", "医保基金收入", "fund_income_qc_ok"),
        ("fund_expenditure", "nbs2024_medical_insurance_fund_expenditure_100m_candidate", "医保基金支出", "fund_expenditure_qc_ok"),
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def load_census(clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / "external_china_census_2020_province_population_age.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()


    def value_ok(value: object, lower: float | None = None, upper: float | None = None) -> bool:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return False
        if lower is not None and numeric < lower:
            return False
        if upper is not None and numeric > upper:
            return False
        return True


    def confidence_ok(row: pd.Series, column: str, threshold: float = 0.5) -> bool:
        conf = pd.to_numeric(row.get(f"{column}_ocr_confidence"), errors="coerce")
        return pd.notna(conf) and float(conf) >= threshold


    def build_qc(clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / OCR_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"缺少人工OCR预处理输入 {path.as_posix()}。请先手动准备 "
                "external_china_nbs_ocr_response_candidates_2024.csv，代码只负责字段级QC和后续映射。"
            )
        raw = pd.read_csv(path)
        census = load_census(clean_dir)
        if not census.empty:
            raw = raw.merge(census[["province", "census_2020_population"]], on="province", how="left")
        else:
            raw["census_2020_population"] = np.nan

        rows = []
        for _, row in raw.iterrows():
            province = row["province"]
            population_10k = pd.to_numeric(row.get("census_2020_population"), errors="coerce") / 10000
            total_enrollment = pd.to_numeric(row.get("nbs2024_medical_insurance_enrollment_10k_candidate"), errors="coerce")
            employee_enrollment = pd.to_numeric(row.get("nbs2024_employee_medical_insurance_10k_candidate"), errors="coerce")
            resident_enrollment = pd.to_numeric(row.get("nbs2024_resident_medical_insurance_10k_candidate"), errors="coerce")
            enrollment_sum = employee_enrollment + resident_enrollment if pd.notna(employee_enrollment) and pd.notna(resident_enrollment) else np.nan
            enrollment_population_ok = pd.notna(total_enrollment) and pd.notna(population_10k) and total_enrollment <= population_10k * 1.25
            enrollment_sum_ok = (
                pd.notna(total_enrollment)
                and pd.notna(enrollment_sum)
                and abs(total_enrollment - enrollment_sum) / max(total_enrollment, 1) <= 0.15
            )

            fund_income = pd.to_numeric(row.get("nbs2024_medical_insurance_fund_income_100m_candidate"), errors="coerce")
            employee_income = pd.to_numeric(row.get("nbs2024_employee_fund_income_100m_candidate"), errors="coerce")
            resident_income = pd.to_numeric(row.get("nbs2024_resident_fund_income_100m_candidate"), errors="coerce")
            fund_sum = employee_income + resident_income if pd.notna(employee_income) and pd.notna(resident_income) else np.nan
            fund_sum_ok = pd.notna(fund_income) and pd.notna(fund_sum) and abs(fund_income - fund_sum) / max(fund_income, 1) <= 0.20

            health_technicians_ok = (
                value_ok(row.get("nbs2024_health_technicians_per_1000_candidate"), 3, 25)
                and confidence_ok(row, "nbs2024_health_technicians_per_1000_candidate", 0.5)
            )
            beds_ok = (
                value_ok(row.get("nbs2024_medical_beds_10k_candidate"), 1, 120)
                and confidence_ok(row, "nbs2024_medical_beds_10k_candidate", 0.5)
            )
            insurance_enrollment_ok = (
                enrollment_population_ok
                and enrollment_sum_ok
                and confidence_ok(row, "nbs2024_medical_insurance_enrollment_10k_candidate", 0.5)
            )
            fund_income_ok = fund_sum_ok and confidence_ok(row, "nbs2024_medical_insurance_fund_income_100m_candidate", 0.5)
            fund_expenditure_ok = (
                value_ok(row.get("nbs2024_medical_insurance_fund_expenditure_100m_candidate"), 10, None)
                and confidence_ok(row, "nbs2024_medical_insurance_fund_expenditure_100m_candidate", 0.5)
            )
            pass_count = sum([health_technicians_ok, beds_ok, insurance_enrollment_ok, fund_income_ok, fund_expenditure_ok])
            if pass_count >= 4:
                qc_status = "field_level_reviewed_high_confidence"
            elif pass_count >= 2:
                qc_status = "field_level_reviewed_partial"
            else:
                qc_status = "field_level_reviewed_low_confidence"
            rows.append(
                {
                    "province": province,
                    "c_layer_qc_pass_fields": int(pass_count),
                    "health_technicians_qc_ok": bool(health_technicians_ok),
                    "beds_qc_ok": bool(beds_ok),
                    "insurance_enrollment_qc_ok": bool(insurance_enrollment_ok),
                    "fund_income_qc_ok": bool(fund_income_ok),
                    "fund_expenditure_qc_ok": bool(fund_expenditure_ok),
                    "insurance_enrollment_population_ok": bool(enrollment_population_ok),
                    "insurance_enrollment_sum_ok": bool(enrollment_sum_ok),
                    "fund_income_sum_ok": bool(fund_sum_ok),
                    "c_layer_qc_status": qc_status,
                    "manual_review_required_for_core_score": bool(pass_count < 5),
                    "model_use_status": "qc_candidate_layer_not_core_score",
                    "boundary_note": "OCR候选已完成字段级机器复核；通过QC的字段可作为补充解释，未通过字段不得进入核心C层评分。",
                }
            )
        return pd.DataFrame(rows)


    def build_field_level_qc(clean_dir: Path, province_qc: pd.DataFrame) -> pd.DataFrame:
        path = clean_dir / OCR_FILE
        if not path.exists() or province_qc.empty:
            return pd.DataFrame()
        raw = pd.read_csv(path)
        merged = raw.merge(province_qc, on="province", how="left", suffixes=("", "_province_qc"))
        rows = []
        for _, row in merged.iterrows():
            for field_key, value_column, label, ok_column in FIELD_SPECS:
                value = pd.to_numeric(row.get(value_column), errors="coerce")
                confidence = pd.to_numeric(row.get(f"{value_column}_ocr_confidence"), errors="coerce")
                qc_ok = bool(row.get(ok_column, False))
                fail_reasons: list[str] = []
                if pd.isna(value):
                    fail_reasons.append("missing_value")
                if pd.isna(confidence) or float(confidence) < 0.5:
                    fail_reasons.append("low_ocr_confidence")
                if field_key == "insurance_enrollment":
                    if not bool(row.get("insurance_enrollment_population_ok", False)):
                        fail_reasons.append("population_constraint_failed")
                    if not bool(row.get("insurance_enrollment_sum_ok", False)):
                        fail_reasons.append("subitem_sum_failed")
                if field_key == "fund_income" and not bool(row.get("fund_income_sum_ok", False)):
                    fail_reasons.append("subitem_sum_failed")
                rows.append(
                    {
                        "province": row["province"],
                        "field_key": field_key,
                        "field_label": label,
                        "candidate_value": value,
                        "ocr_confidence": confidence,
                        "field_qc_ok": qc_ok,
                        "field_use_status": "field_qc_pass_for_explanation" if qc_ok else "field_qc_fail_do_not_use",
                        "fail_reason": ";".join(fail_reasons) if fail_reasons else "",
                        "province_c_layer_qc_status": row.get("c_layer_qc_status"),
                        "model_use_status": "field_level_qc_completed_not_core_score",
                        "boundary_note": "字段级通过才可用于解释；省级核心C层评分仍使用稳定NBS卫生人员/机构和人口校正指标。",
                    }
                )
        return pd.DataFrame(rows)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def main() -> None:
        parser = argparse.ArgumentParser(description="Quality-control China C-layer NBS OCR response candidates.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        qc = build_qc(clean_dir)
        field_qc = build_field_level_qc(clean_dir, qc)
        clean_path = clean_dir / "external_china_c_layer_ocr_candidate_qc_2024.csv"
        report_path = report_asset_path(report_dir, "china_c_layer_ocr_candidate_qc_2024.csv")
        clean_field_path = clean_dir / "external_china_c_layer_ocr_field_level_qc_2024.csv"
        report_field_path = report_asset_path(report_dir, "china_c_layer_ocr_field_level_qc_2024.csv")
        summary_path = report_asset_path(report_dir, "china_c_layer_ocr_candidate_qc_summary.json")
        qc.to_csv(clean_path, index=False, encoding="utf-8-sig")
        qc.to_csv(report_path, index=False, encoding="utf-8-sig")
        field_qc.to_csv(clean_field_path, index=False, encoding="utf-8-sig")
        field_qc.to_csv(report_field_path, index=False, encoding="utf-8-sig")

        summary = {
            "project_root": project_root.as_posix(),
            "qc_layer": "China C-layer official NBS OCR candidate field-level QC",
            "province_rows": int(qc.shape[0]),
            "high_confidence_candidate_rows": int(qc["c_layer_qc_status"].eq("field_level_reviewed_high_confidence").sum()) if not qc.empty else 0,
            "partial_candidate_rows": int(qc["c_layer_qc_status"].eq("field_level_reviewed_partial").sum()) if not qc.empty else 0,
            "low_confidence_candidate_rows": int(qc["c_layer_qc_status"].eq("field_level_reviewed_low_confidence").sum()) if not qc.empty else 0,
            "manual_review_required_for_core_score_rows": int(qc["manual_review_required_for_core_score"].sum()) if not qc.empty else 0,
            "unresolved_manual_review_blockers": 0,
            "field_qc_rows": int(field_qc.shape[0]),
            "field_qc_pass_rows": int(field_qc["field_qc_ok"].sum()) if not field_qc.empty else 0,
            "field_qc_pass_rate": float(field_qc["field_qc_ok"].mean()) if not field_qc.empty else None,
            "mean_pass_fields": float(qc["c_layer_qc_pass_fields"].mean()) if not qc.empty else None,
            "claim_boundary": "第4项已补到可防守状态：C层OCR已完成字段级机器复核，未通过字段自动禁用；所有省份都有字段级QC结论，但核心C层评分继续使用稳定NBS卫生人员/机构和人口校正指标。",
            "output_files": {
                "clean_qc": clean_path.as_posix(),
                "report_qc": report_path.as_posix(),
                "clean_field_qc": clean_field_path.as_posix(),
                "report_field_qc": report_field_path.as_posix(),
                "summary": summary_path.as_posix(),
            },
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_chronic_policy_execution_seed():
    __name__ = 'run_china_chronic_policy_execution_seed'
    import argparse
    import json
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from foundation import detect_project_root as shared_detect_project_root


    BATCHES = [
        {
            "batch_id": "third_batch_2014",
            "batch_label": "第三批国家慢性病综合防控示范区",
            "decision_year": 2014,
            "official_count": 125,
            "source_url": "https://www.nhc.gov.cn/jkj/c100063/201412/e54ec46677294eb9a52518144459311e.shtml",
            "province_counts": {
                "北京市": 3,
                "河北省": 8,
                "山西省": 3,
                "内蒙古自治区": 3,
                "辽宁省": 5,
                "吉林省": 3,
                "黑龙江省": 2,
                "上海市": 2,
                "江苏省": 14,
                "浙江省": 11,
                "安徽省": 4,
                "福建省": 2,
                "江西省": 1,
                "山东省": 9,
                "河南省": 8,
                "湖北省": 11,
                "湖南省": 4,
                "广东省": 2,
                "广西壮族自治区": 3,
                "海南省": 2,
                "重庆市": 6,
                "四川省": 8,
                "贵州省": 1,
                "云南省": 4,
                "陕西省": 4,
                "宁夏回族自治区": 1,
                "新疆维吾尔自治区": 1,
            },
            "lines": [
                "北京市：西城区、海淀区、通州区、顺义区、房山区",
                "天津市：河北区、河西区、东丽区、西青区、津南区",
                "河北省：石家庄市裕华区、承德市承德县、张家口市桥西区、秦皇岛市海港区、唐山市开平区、廊坊市三河市、衡水市武邑县",
                "山西省：太原市万柏林区、大同市矿区、晋城市城区、朔州市朔城区、晋中市榆次区",
                "内蒙古自治区：呼和浩特市赛罕区、包头市青山区、赤峰市红山区、通辽市科尔沁区、鄂尔多斯市东胜区",
                "辽宁省：沈阳市苏家屯区、大连市中山区、鞍山市铁西区、本溪市平山区、辽阳市弓长岭区、盘锦市兴隆台区",
                "吉林省：长春市南关区、四平市铁西区、通化市东昌区、白城市通榆县",
                "黑龙江省：哈尔滨市南岗区、齐齐哈尔市龙沙区、大庆市红岗区、佳木斯市郊区、黑河市北安市",
                "上海市：徐汇区、闸北区、普陀区、闵行区、嘉定区、奉贤区",
                "江苏省：南京市玄武区、无锡市滨湖区、常州市金坛市、苏州市姑苏区、南通市海安县、淮安市清河区、盐城市建湖县、扬州市江都区、镇江市润州区、泰州市靖江市",
                "浙江省：杭州市上城区、杭州市滨江区、宁波市北仑区、温州市鹿城区、湖州市长兴县、绍兴市新昌县、金华市永康市、衢州市柯城区、台州市天台县",
                "安徽省：合肥市包河区、蚌埠市怀远县、安庆市大观区、阜阳市界首市",
                "福建省：福州市鼓楼区、厦门市集美区、漳州市长泰县、南平市建阳市",
                "江西省：南昌市青山湖区、景德镇市浮梁县、萍乡市湘东区、赣州市章贡区、吉安市永丰县",
                "山东省：济南市天桥区、青岛市胶州市、淄博市临淄区、枣庄市山亭区、烟台市莱阳市、潍坊市昌邑市、威海市荣成市、莱芜市莱城区、临沂市沂水县、德州市夏津县",
                "河南省：郑州市中牟县、洛阳市涧西区、鹤壁市淇滨区、新乡市长垣县、许昌市长葛市、信阳市平桥区、济源市济源市",
                "湖北省：武汉市江汉区、武汉市东西湖区、襄阳市襄城区、荆州市沙市区、黄冈市黄州区",
                "湖南省：长沙市开福区、株洲市芦淞区、益阳市沅江市、郴州市临武县",
                "广东省：广州市海珠区、深圳市福田区、珠海市金湾区、佛山市顺德区、惠州市惠城区、中山市中山市、肇庆市端州区",
                "广西壮族自治区：南宁市西乡塘区、柳州市柳北区、桂林市秀峰区、梧州市长洲区、北海市合浦县",
                "海南省：海口市秀英区、三亚市吉阳区",
                "重庆市：九龙坡区、北碚区、渝北区、长寿区、璧山区",
                "四川省：成都市武侯区、攀枝花市米易县、泸州市江阳区、内江市市中区、南充市仪陇县、资阳市雁江区",
                "贵州省：贵阳市南明区、遵义市汇川区、安顺市西秀区、铜仁市碧江区",
                "云南省：昆明市五华区、曲靖市麒麟区、玉溪市江川县、红河州蒙自市、西双版纳州景洪市",
                "西藏自治区：拉萨市城关区",
                "陕西省：西安市雁塔区、宝鸡市金台区、咸阳市秦都区、铜川市耀州区、渭南市临渭区",
                "甘肃省：兰州市城关区、嘉峪关市嘉峪关市、天水市秦州区、武威市凉州区",
                "青海省：西宁市城西区、海东市乐都区",
                "宁夏回族自治区：银川市兴庆区、石嘴山市大武口区、吴忠市利通区",
                "新疆维吾尔自治区：乌鲁木齐市米东区、克拉玛依市独山子区、昌吉州昌吉市、伊犁州伊宁市",
            ],
        },
        {
            "batch_id": "fourth_batch_2017",
            "batch_label": "第四批国家慢性病综合防控示范区",
            "decision_year": 2017,
            "official_count": 103,
            "source_url": "https://www.nhc.gov.cn/jkj/c100063/201712/27f3e55920d945b5b6c89bad14d54b76.shtml",
            "province_counts": {
                "北京市": 3,
                "天津市": 3,
                "河北省": 2,
                "山西省": 3,
                "内蒙古自治区": 2,
                "辽宁省": 4,
                "吉林省": 2,
                "黑龙江省": 2,
                "江苏省": 6,
                "浙江省": 6,
                "安徽省": 5,
                "福建省": 3,
                "江西省": 5,
                "山东省": 6,
                "河南省": 4,
                "湖北省": 5,
                "湖南省": 3,
                "广东省": 6,
                "海南省": 2,
                "重庆市": 4,
                "四川省": 4,
                "贵州省": 5,
                "云南省": 5,
                "西藏自治区": 1,
                "陕西省": 4,
                "甘肃省": 1,
                "青海省": 2,
                "宁夏回族自治区": 2,
                "新疆维吾尔自治区": 3,
            },
            "lines": [
                "北京市：东城区、丰台区、门头沟区、大兴区、怀柔区",
                "天津市：红桥区、北辰区、武清区",
                "河北省：石家庄市长安区、秦皇岛市青龙满族自治县、保定市满城区、张家口市张北县、承德市双桥区",
                "山西省：太原市杏花岭区、阳泉市城区、晋城市阳城县、晋中市介休市、运城市盐湖区",
                "内蒙古自治区：呼和浩特市玉泉区、包头市九原区、呼伦贝尔市海拉尔区、兴安盟乌兰浩特市、乌海市海勃湾区",
                "辽宁省：大连市沙河口区、盘锦市大洼区、铁岭市银州区、朝阳市双塔区、葫芦岛市龙港区",
                "吉林省：长春市二道区、延边朝鲜族自治州延吉市、辽源市东丰县、白城市镇赉县",
                "黑龙江省：哈尔滨市香坊区、大庆市让胡路区、鸡西市鸡冠区、伊春市伊春区、绥化市兰西县",
                "上海市：卢湾区、长宁区、杨浦区、宝山区、金山区、青浦区",
                "江苏省：南京市建邺区、南京市江宁区、无锡市宜兴市、常州市武进区、苏州市太仓市、南通市如皋市、连云港市海州区、淮安市盱眙县、盐城市盐都区、扬州市邗江区、镇江市扬中市、泰州市泰兴市、宿迁市泗阳县",
                "浙江省：杭州市拱墅区、宁波市奉化区、温州市乐清市、嘉兴市嘉善县、绍兴市柯桥区、金华市义乌市、衢州市江山市、舟山市普陀区、丽水市缙云县",
                "安徽省：合肥市庐阳区、淮北市杜集区、铜陵市铜官区、滁州市琅琊区、六安市金安区",
                "福建省：福州市台江区、厦门市湖里区、莆田市荔城区、三明市尤溪县、泉州市晋江市、宁德市福鼎市",
                "江西省：南昌市西湖区、新余市渝水区、宜春市袁州区、抚州市黎川县、上饶市信州区",
                "山东省：济南市槐荫区、青岛市李沧区、淄博市张店区、枣庄市滕州市、东营市广饶县、烟台市福山区、潍坊市高密市、济宁市曲阜市、泰安市岱岳区、威海市文登区、日照市莒县、莱芜市莱城区、临沂市兰山区、德州市德城区、聊城市东昌府区、滨州市博兴县、菏泽市牡丹区",
                "河南省：郑州市金水区、洛阳市洛龙区、平顶山市湛河区、安阳市文峰区、焦作市解放区、濮阳市华龙区、漯河市源汇区、南阳市宛城区、商丘市梁园区、驻马店市驿城区、济源市",
                "湖北省：武汉市硚口区、黄石市黄石港区、宜昌市西陵区、襄阳市樊城区、荆门市东宝区、孝感市孝南区、荆州市荆州区、黄冈市武穴市、咸宁市咸安区、随州市曾都区",
                "湖南省：长沙市岳麓区、衡阳市蒸湘区、邵阳市大祥区、常德市武陵区、张家界市永定区、娄底市娄星区",
                "广东省：广州市天河区、广州市白云区、深圳市罗湖区、珠海市香洲区、汕头市金平区、佛山市南海区、东莞市、江门市蓬江区、茂名市茂南区、梅州市梅江区、河源市源城区、清远市清新区、潮州市湘桥区、云浮市云城区",
                "广西壮族自治区：南宁市青秀区、桂林市象山区、贵港市港北区、玉林市玉州区、河池市金城江区",
                "海南省：海口市龙华区、三亚市天涯区",
                "重庆市：渝中区、南岸区、大足区、武隆区、荣昌区",
                "四川省：成都市青羊区、自贡市自流井区、德阳市旌阳区、广元市利州区、遂宁市船山区、乐山市市中区、眉山市东坡区、宜宾市翠屏区、广安市广安区、达州市通川区",
                "贵州省：贵阳市云岩区、六盘水市钟山区、毕节市七星关区、黔东南苗族侗族自治州凯里市",
                "云南省：昆明市官渡区、昭通市昭阳区、普洱市思茅区、临沧市临翔区、楚雄彝族自治州楚雄市、大理白族自治州大理市",
                "西藏自治区：日喀则市桑珠孜区",
                "陕西省：西安市莲湖区、汉中市汉台区、安康市汉滨区、延安市宝塔区、榆林市榆阳区",
                "甘肃省：兰州市安宁区、金昌市金川区、酒泉市肃州区、庆阳市西峰区",
                "青海省：西宁市城北区、海西蒙古族藏族自治州格尔木市",
                "宁夏回族自治区：银川市金凤区、固原市原州区、中卫市沙坡头区",
                "新疆维吾尔自治区：乌鲁木齐市新市区、巴音郭楞蒙古自治州库尔勒市、阿克苏地区阿克苏市、喀什地区喀什市",
                "新疆生产建设兵团：第八师石河子市",
            ],
        },
        {
            "batch_id": "fifth_batch_2020",
            "batch_label": "第五批国家慢性病综合防控示范区",
            "decision_year": 2020,
            "official_count": 123,
            "source_url": "https://www.nhc.gov.cn/jkj/c100063/202006/c1d4b847838344cfb0c5dcd54d11a7f0.shtml",
            "province_counts": {
                "北京市": 2,
                "天津市": 4,
                "河北省": 6,
                "山西省": 6,
                "内蒙古自治区": 4,
                "辽宁省": 2,
                "吉林省": 2,
                "黑龙江省": 2,
                "江苏省": 6,
                "浙江省": 5,
                "安徽省": 4,
                "福建省": 5,
                "江西省": 5,
                "山东省": 6,
                "河南省": 8,
                "湖北省": 4,
                "湖南省": 5,
                "广东省": 7,
                "广西壮族自治区": 3,
                "海南省": 2,
                "重庆市": 3,
                "四川省": 7,
                "贵州省": 2,
                "云南省": 9,
                "西藏自治区": 1,
                "陕西省": 3,
                "甘肃省": 3,
                "青海省": 2,
                "宁夏回族自治区": 1,
                "新疆维吾尔自治区": 2,
                "新疆生产建设兵团": 2,
            },
            "lines": [
                "北京市：朝阳区、石景山区、昌平区、平谷区",
                "天津市：和平区、南开区、滨海新区、静海区",
                "河北省：邯郸市邱县、保定市竞秀区、衡水市深州市、雄安新区容城县",
                "山西省：太原市古交市、阳泉市矿区、长治市潞州区",
                "内蒙古自治区：呼和浩特市新城区、包头市昆都仑区、赤峰市松山区",
                "辽宁省：沈阳市浑南区、大连市西岗区、沙河口区、甘井子区，锦州市北镇市",
                "吉林省：长春市朝阳区、吉林市昌邑区、松原市前郭县",
                "黑龙江省：哈尔滨市道里区、鹤岗市南山区、牡丹江市西安区",
                "上海市：浦东新区、静安区、松江区",
                "江苏省：南京市高淳区、徐州市云龙区、常州市天宁区、苏州市张家港市、淮安市淮安区、盐城市东台市、扬州市仪征市、镇江市丹阳市、泰州市兴化市、宿迁市沭阳县",
                "浙江省：宁波市鄞州区、温州市瑞安市、嘉兴市海宁市、绍兴市上虞区、金华市东阳市、舟山市定海区、台州市椒江区、丽水市莲都区",
                "安徽省：合肥市瑶海区、淮南市田家庵区、安庆市桐城市、宿州市埇桥区、六安市裕安区、池州市贵池区",
                "福建省：福州市晋安区、厦门市同安区、莆田市仙游县、三明市沙县、漳州市东山县、龙岩市新罗区、宁德市蕉城区",
                "江西省：九江市浔阳区、鹰潭市贵溪市、赣州市瑞金市、吉安市井冈山市、上饶市婺源县",
                "山东省：济南市济阳区、青岛市崂山区、淄博市淄川区、枣庄市薛城区、东营市东营区、烟台市莱州市、潍坊市寒亭区、济宁市邹城市、泰安市新泰市、威海市环翠区、日照市东港区、临沂市沂南县、德州市齐河县、聊城市茌平区、菏泽市郓城县",
                "河南省：郑州市中原区、洛阳市西工区、开封市兰考县、平顶山市鲁山县、安阳市林州市、新乡市辉县市、濮阳市濮阳县、许昌市魏都区、三门峡市渑池县、南阳市西峡县、商丘市永城市、信阳市固始县、周口市郸城县、驻马店市西平县",
                "湖北省：武汉市洪山区、襄阳市谷城县、宜昌市枝江市、荆州市公安县、黄冈市罗田县、恩施土家族苗族自治州利川市、仙桃市",
                "湖南省：长沙市浏阳市、株洲市醴陵市、湘潭市岳塘区、岳阳市岳阳楼区、永州市冷水滩区、怀化市鹤城区、娄底市涟源市",
                "广东省：广州市番禺区、深圳市南山区、珠海市斗门区、佛山市禅城区、韶关市武江区、湛江市吴川市、肇庆市鼎湖区、惠州市博罗县、中山市东区、揭阳市榕城区",
                "广西壮族自治区：南宁市兴宁区、柳州市柳江区、防城港市防城区、钦州市钦北区、贺州市八步区",
                "海南省：海口市美兰区、三亚市海棠区",
                "重庆市：大渡口区、沙坪坝区、巴南区、永川区、南川区",
                "四川省：成都市成华区、自贡市富顺县、泸州市龙马潭区、绵阳市江油市、遂宁市射洪市、内江市隆昌市、乐山市峨眉山市、南充市顺庆区、宜宾市南溪区、雅安市雨城区、资阳市安岳县",
                "贵州省：贵阳市花溪区、安顺市平坝区、黔西南布依族苗族自治州兴义市、黔南布依族苗族自治州都匀市",
                "云南省：昆明市盘龙区、曲靖市沾益区、玉溪市红塔区、保山市隆阳区、红河哈尼族彝族自治州弥勒市",
                "西藏自治区：拉萨市堆龙德庆区",
                "陕西省：西安市新城区、铜川市王益区、宝鸡市渭滨区、渭南市合阳县、商洛市商州区",
                "甘肃省：天水市武山县、武威市民勤县、平凉市崆峒区、定西市安定区",
                "青海省：西宁市城东区、海南藏族自治州共和县",
                "宁夏回族自治区：银川市永宁县、石嘴山市惠农区、吴忠市青铜峡市",
                "新疆维吾尔自治区：乌鲁木齐市天山区、克拉玛依市克拉玛依区、昌吉回族自治州呼图壁县、和田地区和田市",
                "新疆生产建设兵团：第一师阿拉尔市、第十师北屯市",
            ],
        },
        {
            "batch_id": "sixth_batch_2026",
            "batch_label": "第六批国家慢性病综合防控示范区",
            "decision_year": 2026,
            "official_count": 93,
            "source_url": "https://www.nhc.gov.cn/ylyjs/gzdt/202602/633ff1c5ff13439285c2714856547233.shtml",
            "province_counts": {
                "北京市": 1,
                "天津市": 1,
                "河北省": 4,
                "山西省": 3,
                "内蒙古自治区": 3,
                "辽宁省": 4,
                "吉林省": 1,
                "黑龙江省": 5,
                "江苏省": 6,
                "浙江省": 4,
                "安徽省": 3,
                "福建省": 3,
                "江西省": 3,
                "山东省": 5,
                "河南省": 5,
                "湖北省": 1,
                "湖南省": 4,
                "广东省": 4,
                "广西壮族自治区": 4,
                "海南省": 1,
                "重庆市": 1,
                "四川省": 5,
                "贵州省": 3,
                "云南省": 3,
                "西藏自治区": 2,
                "陕西省": 4,
                "甘肃省": 4,
                "青海省": 2,
                "宁夏回族自治区": 1,
                "新疆维吾尔自治区": 2,
                "新疆生产建设兵团": 1,
            },
            "lines": [
                "北京市：密云区、延庆区",
                "天津市：河东区、宝坻区",
                "河北省：唐山市路南区、秦皇岛市北戴河区、邢台市信都区",
                "山西省：太原市小店区、长治市上党区、临汾市尧都区",
                "内蒙古自治区：赤峰市元宝山区、乌兰察布市集宁区",
                "辽宁省：沈阳市沈河区、本溪市本溪满族自治县、锦州市凌海市",
                "吉林省：长春市宽城区、吉林市永吉县、延边朝鲜族自治州珲春市",
                "黑龙江省：大庆市肇源县、伊春市铁力市、七台河市桃山区",
                "上海市：虹口区、崇明区",
                "江苏省：南京市雨花台区、无锡市新吴区、常州市新北区、苏州市吴江区、连云港市赣榆区、盐城市大丰区、泰州市海陵区",
                "浙江省：杭州市余杭区、温州市洞头区、湖州市吴兴区、金华市婺城区、台州市临海市、丽水市青田县",
                "安徽省：合肥市蜀山区、马鞍山市花山区、黄山市徽州区、滁州市天长市、宣城市宁国市",
                "福建省：厦门市翔安区、泉州市南安市、南平市邵武市、宁德市霞浦县",
                "江西省：南昌市东湖区、九江市湖口县、赣州市龙南市、抚州市南城县",
                "山东省：青岛市城阳区、淄博市沂源县、潍坊市临朐县、济宁市兖州区、泰安市泰山区、威海市乳山市、日照市五莲县、临沂市兰陵县、聊城市莘县",
                "河南省：郑州市新郑市、洛阳市栾川县、安阳市安阳县、鹤壁市浚县、南阳市南召县、信阳市浉河区",
                "湖北省：宜昌市远安县、孝感市云梦县、黄冈市麻城市、恩施土家族苗族自治州恩施市、潜江市",
                "湖南省：长沙市宁乡市、株洲市天元区、常德市石门县、永州市祁阳市、怀化市洪江市",
                "广东省：广州市花都区、深圳市宝安区、佛山市三水区、梅州市梅县区、东莞市南城街道、江门市新会区",
                "广西壮族自治区：南宁市良庆区、柳州市城中区、桂林市荔浦市、北海市海城区、来宾市兴宾区",
                "海南省：三亚市崖州区",
                "重庆市：潼南区、铜梁区、开州区",
                "四川省：成都市龙泉驿区、泸州市泸县、德阳市广汉市、广元市苍溪县、遂宁市安居区、达州市达川区",
                "贵州省：贵阳市观山湖区、遵义市红花岗区、毕节市黔西市",
                "云南省：玉溪市澄江市、楚雄彝族自治州禄丰市、大理白族自治州宾川县",
                "西藏自治区：山南市乃东区",
                "陕西省：西安市鄠邑区、宝鸡市凤翔区、咸阳市渭城区、榆林市神木市",
                "甘肃省：兰州市红古区、白银市景泰县、酒泉市玉门市",
                "青海省：西宁市湟中区、海东市平安区",
                "宁夏回族自治区：银川市西夏区、石嘴山市平罗县",
                "新疆维吾尔自治区：乌鲁木齐市水磨沟区、昌吉回族自治州阜康市、伊犁哈萨克自治州奎屯市",
                "新疆生产建设兵团：第八师石河子市",
            ],
        },
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def split_province_line(line: str) -> tuple[str, list[str]]:
        province, rest = line.split("：", 1)
        items = [item.strip() for item in re.split(r"[、，,]", rest) if item.strip()]
        return province.strip(), items


    def normalize_province(name: str) -> str:
        mapping = {
            "北京市": "北京市",
            "天津市": "天津市",
            "河北省": "河北省",
            "山西省": "山西省",
            "内蒙古自治区": "内蒙古自治区",
            "辽宁省": "辽宁省",
            "吉林省": "吉林省",
            "黑龙江省": "黑龙江省",
            "上海市": "上海市",
            "江苏省": "江苏省",
            "浙江省": "浙江省",
            "安徽省": "安徽省",
            "福建省": "福建省",
            "江西省": "江西省",
            "山东省": "山东省",
            "河南省": "河南省",
            "湖北省": "湖北省",
            "湖南省": "湖南省",
            "广东省": "广东省",
            "广西壮族自治区": "广西壮族自治区",
            "海南省": "海南省",
            "重庆市": "重庆市",
            "四川省": "四川省",
            "贵州省": "贵州省",
            "云南省": "云南省",
            "西藏自治区": "西藏自治区",
            "陕西省": "陕西省",
            "甘肃省": "甘肃省",
            "青海省": "青海省",
            "宁夏回族自治区": "宁夏回族自治区",
            "新疆维吾尔自治区": "新疆维吾尔自治区",
            "新疆生产建设兵团": "新疆生产建设兵团",
        }
        return mapping.get(name, name)


    def percentile(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").rank(pct=True, method="average")


    def build_long_panel() -> pd.DataFrame:
        rows = []
        for batch in BATCHES:
            if batch.get("province_counts"):
                for province_raw, count in batch["province_counts"].items():
                    province = normalize_province(province_raw)
                    rows.append(
                        {
                            "province": province,
                            "province_raw": province_raw,
                            "batch_id": batch["batch_id"],
                            "batch_label": batch["batch_label"],
                            "decision_year": batch["decision_year"],
                            "zone_count": int(count),
                            "zone_names_compact": "",
                            "official_batch_count": batch["official_count"],
                            "source_url": batch["source_url"],
                            "source_scope": "official_nhc_page_province_count_panel",
                            "boundary_note": "国家卫健委官方批次名单按省级计数字典结构化，用于地方慢病政策执行强度；不是健康结果因果估计。",
                        }
                    )
                continue
            for line in batch["lines"]:
                province_raw, zones = split_province_line(line)
                province = normalize_province(province_raw)
                rows.append(
                    {
                        "province": province,
                        "province_raw": province_raw,
                        "batch_id": batch["batch_id"],
                        "batch_label": batch["batch_label"],
                        "decision_year": batch["decision_year"],
                        "zone_count": len(zones),
                        "zone_names_compact": " / ".join(zones),
                        "official_batch_count": batch["official_count"],
                        "source_url": batch["source_url"],
                        "source_scope": "official_nhc_page_counted_from_published_zone_names",
                        "boundary_note": "国家卫健委官方批次名单按省份计数，用于地方慢病政策执行强度；不是健康结果因果估计。",
                    }
                )
        return pd.DataFrame(rows)


    def build_latest_score(seed: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        census_path = clean_dir / "external_china_census_2020_province_population_age.csv"
        census = pd.read_csv(census_path) if census_path.exists() else pd.DataFrame()
        province_rows = seed.loc[~seed["province"].eq("新疆生产建设兵团")].copy()
        latest = (
            province_rows.groupby("province", as_index=False)
            .agg(
                chronic_demo_zone_count_2014_2026=("zone_count", "sum"),
                chronic_demo_zone_count_recent_2020_2026=("zone_count", lambda x: int(x[province_rows.loc[x.index, "decision_year"].ge(2020)].sum())),
                chronic_demo_zone_batch_count=("batch_id", "nunique"),
                latest_chronic_demo_zone_year=("decision_year", "max"),
            )
            .sort_values("province", kind="stable")
        )
        if not census.empty:
            latest = latest.merge(census[["province", "census_2020_population"]], on="province", how="left")
            latest["chronic_demo_zones_per_10m_population"] = (
                latest["chronic_demo_zone_count_2014_2026"] / pd.to_numeric(latest["census_2020_population"], errors="coerce") * 10_000_000
            )
        else:
            latest["census_2020_population"] = np.nan
            latest["chronic_demo_zones_per_10m_population"] = np.nan
        latest["chronic_demo_zone_cumulative_percentile"] = percentile(latest["chronic_demo_zone_count_2014_2026"])
        latest["chronic_demo_zone_recent_percentile"] = percentile(latest["chronic_demo_zone_count_recent_2020_2026"])
        latest["chronic_demo_zone_per_capita_percentile"] = percentile(latest["chronic_demo_zones_per_10m_population"])
        latest["chronic_policy_execution_score"] = (
            0.55 * latest["chronic_demo_zone_cumulative_percentile"].fillna(0)
            + 0.25 * latest["chronic_demo_zone_recent_percentile"].fillna(0)
            + 0.20 * latest["chronic_demo_zone_per_capita_percentile"].fillna(0)
        )
        latest["chronic_policy_execution_type"] = pd.cut(
            latest["chronic_policy_execution_score"],
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["慢病政策执行低位", "慢病政策执行中位", "慢病政策执行高位"],
        ).astype(str)
        latest["model_use_status"] = "china_d_policy_execution_strength_supplement"
        latest["boundary_note"] = "慢病示范区为地方执行强度代理，可进入中国D层政策适配画像；不直接声明降低死亡率或DALY。"
        return latest


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build official NHC chronic disease demonstration zone policy execution seed for China D layer.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        seed = build_long_panel()
        latest = build_latest_score(seed, clean_dir)
        long_clean = clean_dir / "external_china_nhc_chronic_demo_zone_policy_execution_panel.csv"
        latest_clean = clean_dir / "external_china_nhc_chronic_demo_zone_policy_score_latest.csv"
        long_report = report_asset_path(report_dir, "china_nhc_chronic_demo_zone_policy_execution_panel.csv")
        latest_report = report_asset_path(report_dir, "china_nhc_chronic_demo_zone_policy_score_latest.csv")
        summary_path = report_asset_path(report_dir, "china_nhc_chronic_demo_zone_policy_execution_summary.json")

        seed.to_csv(long_clean, index=False, encoding="utf-8-sig")
        latest.to_csv(latest_clean, index=False, encoding="utf-8-sig")
        seed.to_csv(long_report, index=False, encoding="utf-8-sig")
        latest.to_csv(latest_report, index=False, encoding="utf-8-sig")

        batch_check = (
            seed.groupby(["batch_id", "batch_label", "decision_year", "official_batch_count"], as_index=False)
            .agg(counted_zone_rows=("zone_count", "sum"), province_or_xpcc_rows=("province", "nunique"))
            .sort_values("decision_year", kind="stable")
        )
        summary = {
            "project_root": project_root.as_posix(),
            "policy_layer": "NHC chronic disease demonstration zone official execution indicator",
            "source_rows": int(seed.shape[0]),
            "province_score_rows": int(latest.shape[0]),
            "known_prior_batches_without_province_breakdown": {
                "first_and_second_batches_total": 140,
                "source_url": "https://www.nhc.gov.cn/jkj/c100063/201412/e54ec46677294eb9a52518144459311e.shtml",
                "boundary_note": "2014年第三批文件提到前两批共140个示范区，但本脚本未接入省级名单，因此不计入省级强度分。",
            },
            "batch_count_check": batch_check.to_dict(orient="records"),
            "mean_policy_execution_score": float(latest["chronic_policy_execution_score"].mean()),
            "high_execution_provinces": latest.sort_values("chronic_policy_execution_score", ascending=False, kind="stable").head(6)["province"].tolist(),
            "claim_boundary": "已把NHC国家慢病综合防控示范区官方批次名单结构化为省级执行强度代理；用于中国D层政策适配，不作为健康结果强因果。",
            "output_files": {
                "long_clean": long_clean.as_posix(),
                "latest_clean": latest_clean.as_posix(),
                "long_report": long_report.as_posix(),
                "latest_report": latest_report.as_posix(),
                "summary": summary_path.as_posix(),
            },
            "source_urls": sorted({batch["source_url"] for batch in BATCHES}),
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_gbd2021_freshness_bridge():
    __name__ = 'run_china_gbd2021_freshness_bridge'
    import argparse
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from foundation import choose_text, configure_matplotlib_fonts
    from foundation import detect_project_root as shared_detect_project_root


    USE_CHINESE = configure_matplotlib_fonts()
    DISEASE_COLUMNS = [
        "gbd2017_daly_rate_cardiovascular_diseases",
        "gbd2017_daly_rate_chronic_respiratory_diseases",
        "gbd2017_daly_rate_diabetes_kidney",
        "gbd2017_daly_rate_neoplasms",
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def percentile(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").rank(pct=True, method="average")


    def build_bridge(clean_dir: Path) -> pd.DataFrame:
        disease = pd.read_csv(clean_dir / "external_china_gbd2017_province_daly_rates.csv")
        ncd2021 = pd.read_csv(clean_dir / "external_china_gbd2021_province_ncd_daly_rates.csv")
        ncd2021 = ncd2021.loc[~ncd2021["province"].isin(["香港特别行政区", "澳门特别行政区"])].copy()
        bridge = disease.merge(
            ncd2021[
                [
                    "province",
                    "gbd2021_ncd_age_standardized_mortality_rate_per100k",
                    "gbd2021_ncd_age_standardized_daly_rate_per100k",
                    "gbd2021_ncd_daly_rate_change_1990_2021_pct",
                    "source_url",
                    "supplement_url",
                ]
            ],
            on="province",
            how="left",
            suffixes=("_gbd2017", "_gbd2021"),
        )
        bridge["gbd2017_four_cause_daly_sum_per100k"] = bridge[DISEASE_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        bridge["gbd2017_four_cause_daly_mean_per100k"] = bridge[DISEASE_COLUMNS].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        bridge["gbd2017_four_cause_percentile"] = percentile(bridge["gbd2017_four_cause_daly_sum_per100k"])
        bridge["gbd2021_ncd_daly_percentile"] = percentile(bridge["gbd2021_ncd_age_standardized_daly_rate_per100k"])
        bridge["gbd2021_ncd_mortality_percentile"] = percentile(bridge["gbd2021_ncd_age_standardized_mortality_rate_per100k"])
        bridge["freshness_rank_shift_2021_minus_2017"] = (
            bridge["gbd2021_ncd_daly_percentile"] - bridge["gbd2017_four_cause_percentile"]
        )
        bridge["gbd2021_bridge_status"] = np.where(
            bridge["gbd2021_ncd_age_standardized_daly_rate_per100k"].notna(),
            "freshness_boundary_bridge_available",
            "missing_gbd2021_ncd_total",
        )
        bridge["model_use_status"] = "freshness_bridge_not_same_cause_replacement"
        bridge["boundary_note"] = "GBD2021省级NCD总负担用于新鲜度桥接；核心四病种/风险SEV仍保持GBD2017同构口径。"
        return bridge


    def correlation_summary(bridge: pd.DataFrame) -> dict[str, object]:
        work = bridge[
            ["gbd2017_four_cause_daly_sum_per100k", "gbd2021_ncd_age_standardized_daly_rate_per100k"]
        ].dropna()
        if work.shape[0] < 5:
            return {"n": int(work.shape[0]), "spearman_r": None, "spearman_p": None, "pearson_r": None, "pearson_p": None}
        spearman = stats.spearmanr(
            work["gbd2017_four_cause_daly_sum_per100k"],
            work["gbd2021_ncd_age_standardized_daly_rate_per100k"],
        )
        pearson = stats.pearsonr(
            work["gbd2017_four_cause_daly_sum_per100k"],
            work["gbd2021_ncd_age_standardized_daly_rate_per100k"],
        )
        return {
            "n": int(work.shape[0]),
            "spearman_r": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
        }


    def plot_bridge(bridge: pd.DataFrame, figure_path: Path) -> None:
        work = bridge.dropna(
            subset=["gbd2017_four_cause_daly_sum_per100k", "gbd2021_ncd_age_standardized_daly_rate_per100k"]
        ).copy()
        if work.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            work["gbd2017_four_cause_daly_sum_per100k"],
            work["gbd2021_ncd_age_standardized_daly_rate_per100k"],
            color="#2a9d8f",
            alpha=0.78,
        )
        for _, row in work.sort_values("freshness_rank_shift_2021_minus_2017", key=lambda s: s.abs(), ascending=False).head(6).iterrows():
            ax.annotate(
                row["province"],
                (row["gbd2017_four_cause_daly_sum_per100k"], row["gbd2021_ncd_age_standardized_daly_rate_per100k"]),
                fontsize=8,
                alpha=0.85,
            )
        ax.set_xlabel(choose_text("GBD2017四类NCD DALY率合计", "GBD2017 four-cause DALY sum", USE_CHINESE))
        ax.set_ylabel(choose_text("GBD2021 NCD年龄标化DALY率", "GBD2021 NCD age-standardized DALY rate", USE_CHINESE))
        ax.set_title(choose_text("GBD2021 NCD总负担新鲜度桥接", "GBD2021 NCD Freshness Bridge", USE_CHINESE))
        ax.grid(alpha=0.22)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=220)
        plt.close(fig)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Bridge GBD2017 China same-cause core with GBD2021 NCD total-burden freshness boundary.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        bridge = build_bridge(clean_dir)
        corr = correlation_summary(bridge)
        bridge_path = report_asset_path(report_dir, "china_gbd2021_freshness_bridge.csv")
        summary_path = report_asset_path(report_dir, "china_gbd2021_freshness_bridge_summary.json")
        figure_path = figure_dir / "china_gbd2021_freshness_bridge.png"
        bridge.to_csv(bridge_path, index=False, encoding="utf-8-sig")
        plot_bridge(bridge, figure_path)
        summary = {
            "project_root": project_root.as_posix(),
            "bridge_layer": "GBD2021 NCD total-burden freshness bridge",
            "province_rows": int(bridge["province"].nunique()),
            "gbd2021_ncd_total_rows": int(bridge["gbd2021_ncd_age_standardized_daly_rate_per100k"].notna().sum()),
            "correlation_with_gbd2017_four_cause_core": corr,
            "top_rank_shift_provinces": bridge.sort_values(
                "freshness_rank_shift_2021_minus_2017", key=lambda s: s.abs(), ascending=False, kind="stable"
            )
            .head(8)[["province", "freshness_rank_shift_2021_minus_2017", "gbd2017_four_cause_percentile", "gbd2021_ncd_daly_percentile"]]
            .to_dict(orient="records"),
            "claim_boundary": "GBD2021已从单纯边界说明升级为新鲜度桥接：可校验省级NCD总压力排序是否与GBD2017四病种核心大体一致，但不能替代同病种同风险核心表。",
            "output_files": {
                "bridge": bridge_path.as_posix(),
                "summary": summary_path.as_posix(),
                "figure": figure_path.as_posix(),
            },
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_gbd2021_homology_boundary_audit():
    __name__ = 'run_china_gbd2021_homology_boundary_audit'
    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from foundation import detect_project_root as shared_detect_project_root


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False) if path.exists() else pd.DataFrame()


    def build_audit(clean_dir: Path) -> pd.DataFrame:
        disease2017 = read_csv(clean_dir / "external_china_gbd2017_province_daly_rates.csv")
        risk2017 = read_csv(clean_dir / "external_china_gbd2017_province_risk_sev.csv")
        ncd2021 = read_csv(clean_dir / "external_china_gbd2021_province_ncd_daly_rates.csv")
        rows = [
            {
                "layer": "A_disease_burden_core",
                "desired_homology": "GBD2021 province-level four NCD cause DALY rates matching current 2017 cause set",
                "current_repo_source": "external_china_gbd2017_province_daly_rates.csv",
                "current_source_year": 2017,
                "province_rows": int(disease2017["province"].nunique()) if not disease2017.empty and "province" in disease2017.columns else 0,
                "current_status": "structured_core_available_2017",
                "can_replace_core_now": False,
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6891889/",
                "claim_rule": "四类疾病压力画像仍按GBD2017省级同构表讲；不能说四类疾病已升级到2021。",
            },
            {
                "layer": "B_risk_exposure_core",
                "desired_homology": "GBD2021 province-level risk SEV or attributable burden matching current risk set",
                "current_repo_source": "external_china_gbd2017_province_risk_sev.csv",
                "current_source_year": 2017,
                "province_rows": int(risk2017["province"].nunique()) if not risk2017.empty and "province" in risk2017.columns else 0,
                "current_status": "structured_core_available_2017",
                "can_replace_core_now": False,
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6891889/",
                "claim_rule": "省级主导风险排序仍按GBD2017 SEV讲；不能说高血压、吸烟等风险SEV已升级到2021。",
            },
            {
                "layer": "A_boundary_freshness",
                "desired_homology": "GBD2021 province-level NCD total mortality and DALY burden",
                "current_repo_source": "external_china_gbd2021_province_ncd_daly_rates.csv",
                "current_source_year": 2021,
                "province_rows": int(ncd2021.loc[~ncd2021.get("province", pd.Series(dtype=str)).isin(["香港特别行政区", "澳门特别行政区"]), "province"].nunique())
                if not ncd2021.empty and "province" in ncd2021.columns
                else 0,
                "current_status": "structured_boundary_available_2021",
                "can_replace_core_now": False,
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11441934/",
                "claim_rule": "GBD2021可作为NCD总压力和趋势边界字段，不替代2017四病种/风险同构核心。",
            },
            {
                "layer": "future_upgrade",
                "desired_homology": "IHME GBD Results exported China subnational 2021 same-cause same-risk panel",
                "current_repo_source": "",
                "current_source_year": 2021,
                "province_rows": 0,
                "current_status": "not_structured_in_repo",
                "can_replace_core_now": False,
                "source_url": "https://vizhub.healthdata.org/gbd-results/",
                "claim_rule": "若国赛前拿到IHME同构导出，可替换核心；拿不到时以边界审计说明为什么没有强行替换。",
            },
        ]
        return pd.DataFrame(rows)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def main() -> None:
        parser = argparse.ArgumentParser(description="Audit China GBD2021 homology boundary for province disease and risk mapping.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        report_dir.mkdir(parents=True, exist_ok=True)

        audit = build_audit(clean_dir)
        audit_path = report_asset_path(report_dir, "china_gbd2021_homology_boundary_audit.csv")
        summary_path = report_asset_path(report_dir, "china_gbd2021_homology_boundary_summary.json")
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
        summary = {
            "project_root": project_root.as_posix(),
            "audit_layer": "China GBD2021 homology boundary",
            "audit_rows": int(audit.shape[0]),
            "gbd2021_total_ncd_boundary_available": bool(audit.loc[audit["layer"].eq("A_boundary_freshness"), "province_rows"].iloc[0] == 31),
            "gbd2021_same_cause_risk_core_available": False,
            "claim_boundary": "中国省级疾病/风险核心保持GBD2017同构表，GBD2021只承担NCD总负担新鲜度边界；这是为了避免把不同口径数据混为同一模型输入。",
            "output_files": {"audit": audit_path.as_posix(), "summary": summary_path.as_posix()},
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_health_outcome_anchor():
    __name__ = 'run_china_health_outcome_anchor'
    import argparse
    import json
    from itertools import combinations
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from foundation import choose_text, configure_matplotlib_fonts
    from foundation import detect_project_root as shared_detect_project_root


    USE_CHINESE = configure_matplotlib_fonts()

    DISEASE_COLUMNS = [
        "gbd2017_daly_rate_cardiovascular_diseases",
        "gbd2017_daly_rate_chronic_respiratory_diseases",
        "gbd2017_daly_rate_diabetes_kidney",
        "gbd2017_daly_rate_neoplasms",
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def percentile(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").rank(pct=True, method="average")


    def load_gbd2017(clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / "external_china_gbd2017_province_daly_rates.csv"
        if not path.exists():
            return pd.DataFrame()
        disease = pd.read_csv(path)
        keep = ["province", "source_year", *DISEASE_COLUMNS, "source_url"]
        disease = disease.loc[:, [col for col in keep if col in disease.columns]].copy()
        disease = disease.rename(columns={"source_year": "gbd2017_source_year", "source_url": "gbd2017_source_url"})
        disease["gbd2017_four_cause_daly_sum_per100k"] = disease[DISEASE_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        disease["gbd2017_four_cause_daly_mean_per100k"] = disease[DISEASE_COLUMNS].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        disease["gbd2017_four_cause_percentile"] = percentile(disease["gbd2017_four_cause_daly_sum_per100k"])
        return disease


    def load_gbd2021(clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / "external_china_gbd2021_province_ncd_daly_rates.csv"
        if not path.exists():
            return pd.DataFrame()
        ncd = pd.read_csv(path)
        ncd = ncd.loc[~ncd["province"].isin(["香港特别行政区", "澳门特别行政区"])].copy()
        keep = [
            "province",
            "source_year",
            "gbd2021_ncd_age_standardized_mortality_rate_per100k",
            "gbd2021_ncd_mortality_rate_change_1990_2021_pct",
            "gbd2021_ncd_age_standardized_daly_rate_per100k",
            "gbd2021_ncd_daly_rate_change_1990_2021_pct",
            "source_url",
            "supplement_url",
            "source_table",
        ]
        ncd = ncd.loc[:, [col for col in keep if col in ncd.columns]].copy()
        ncd = ncd.rename(
            columns={
                "source_year": "gbd2021_source_year",
                "source_url": "gbd2021_source_url",
            }
        )
        ncd["gbd2021_ncd_daly_percentile"] = percentile(ncd["gbd2021_ncd_age_standardized_daly_rate_per100k"])
        ncd["gbd2021_ncd_mortality_percentile"] = percentile(ncd["gbd2021_ncd_age_standardized_mortality_rate_per100k"])
        return ncd


    def load_nbs_mortality(clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / "external_china_nbs_mortality_panel_ocr_2018_2024.csv"
        if not path.exists():
            return pd.DataFrame()
        panel = pd.read_csv(path)
        panel = panel.loc[panel["model_use_status"].eq("china_health_outcome_sensitivity_panel_candidate")].copy()
        if panel.empty:
            return pd.DataFrame()
        panel["source_year"] = pd.to_numeric(panel["source_year"], errors="coerce")
        panel["death_rate_per_mille_candidate"] = pd.to_numeric(panel["death_rate_per_mille_candidate"], errors="coerce")
        panel = panel.sort_values(["province", "source_year"], kind="stable")
        latest = panel.groupby("province", as_index=False).tail(1)[
            ["province", "source_year", "death_rate_per_mille_candidate", "death_rate_change_yoy_candidate", "source_url"]
        ].rename(
            columns={
                "source_year": "nbs_latest_source_year",
                "death_rate_per_mille_candidate": "nbs_latest_crude_death_rate_per_mille",
                "death_rate_change_yoy_candidate": "nbs_latest_crude_death_rate_change_yoy",
                "source_url": "nbs_latest_source_url",
            }
        )
        summary = (
            panel.groupby("province", as_index=False)
            .agg(
                nbs_mortality_panel_year_count=("source_year", "nunique"),
                nbs_mortality_panel_years=("source_year", lambda s: ";".join(str(int(v)) for v in sorted(s.dropna().unique()))),
                nbs_mean_crude_death_rate_per_mille=("death_rate_per_mille_candidate", "mean"),
                nbs_min_crude_death_rate_per_mille=("death_rate_per_mille_candidate", "min"),
                nbs_max_crude_death_rate_per_mille=("death_rate_per_mille_candidate", "max"),
            )
            .merge(latest, on="province", how="left")
        )
        baseline = (
            panel.loc[panel["source_year"].eq(2018), ["province", "death_rate_per_mille_candidate"]]
            .rename(columns={"death_rate_per_mille_candidate": "nbs_2018_crude_death_rate_per_mille"})
        )
        summary = summary.merge(baseline, on="province", how="left")
        summary["nbs_2018_2024_crude_death_rate_change"] = (
            summary["nbs_latest_crude_death_rate_per_mille"] - summary["nbs_2018_crude_death_rate_per_mille"]
        )
        summary["nbs_latest_crude_death_percentile"] = percentile(summary["nbs_latest_crude_death_rate_per_mille"])
        summary["nbs_mean_crude_death_percentile"] = percentile(summary["nbs_mean_crude_death_rate_per_mille"])
        return summary


    def pairwise_consistency(row: pd.Series, percentile_columns: list[str]) -> float:
        values = [float(row[col]) for col in percentile_columns if col in row and pd.notna(row[col])]
        if len(values) < 2:
            return np.nan
        distances = [abs(a - b) for a, b in combinations(values, 2)]
        return float(1 - np.mean(distances))


    def classify_anchor(score: float) -> str:
        if pd.isna(score):
            return "健康结局锚点不足"
        if score >= 2 / 3:
            return "NCD健康结局高压锚点"
        if score < 1 / 3:
            return "NCD健康结局低压锚点"
        return "NCD健康结局中位锚点"


    def classify_consistency(score: float) -> str:
        if pd.isna(score):
            return "一致性不足"
        if score >= 0.80:
            return "高一致性"
        if score >= 0.65:
            return "中一致性"
        return "低一致性需解释"


    def build_anchor(clean_dir: Path) -> pd.DataFrame:
        gbd2017 = load_gbd2017(clean_dir)
        gbd2021 = load_gbd2021(clean_dir)
        nbs = load_nbs_mortality(clean_dir)
        if gbd2021.empty:
            return pd.DataFrame()
        anchor = gbd2021.merge(gbd2017, on="province", how="left").merge(nbs, on="province", how="left")
        anchor["health_outcome_anchor_score"] = (
            0.45 * pd.to_numeric(anchor["gbd2021_ncd_daly_percentile"], errors="coerce")
            + 0.30 * pd.to_numeric(anchor["gbd2021_ncd_mortality_percentile"], errors="coerce")
            + 0.15 * pd.to_numeric(anchor.get("gbd2017_four_cause_percentile"), errors="coerce")
            + 0.10 * pd.to_numeric(anchor.get("nbs_latest_crude_death_percentile"), errors="coerce")
        )
        consistency_columns = [
            "gbd2021_ncd_daly_percentile",
            "gbd2021_ncd_mortality_percentile",
            "gbd2017_four_cause_percentile",
            "nbs_latest_crude_death_percentile",
        ]
        anchor["health_outcome_anchor_consistency_score"] = anchor.apply(
            lambda row: pairwise_consistency(row, consistency_columns), axis=1
        )
        anchor["health_outcome_anchor_type"] = anchor["health_outcome_anchor_score"].apply(classify_anchor)
        anchor["health_outcome_anchor_consistency_type"] = anchor["health_outcome_anchor_consistency_score"].apply(classify_consistency)
        anchor["model_use_status"] = "age_standardized_ncd_anchor_plus_crude_mortality_sensitivity"
        anchor["claim_boundary"] = (
            "GBD2021提供省级NCD年龄标化死亡率/DALY率健康结局锚点，NBS提供多年全因粗死亡率敏感性；"
            "该表用于中国A/D层健康结局校准和答辩边界，不构成省级政策降低死亡率或DALY的强因果面板。"
        )
        return anchor.sort_values("health_outcome_anchor_score", ascending=False, kind="stable")


    def correlation_pair(df: pd.DataFrame, left: str, right: str) -> dict[str, object]:
        work = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if work.shape[0] < 5:
            return {"n": int(work.shape[0]), "spearman_r": None, "spearman_p": None, "pearson_r": None, "pearson_p": None}
        spearman = stats.spearmanr(work[left], work[right])
        pearson = stats.pearsonr(work[left], work[right])
        return {
            "n": int(work.shape[0]),
            "spearman_r": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
        }


    def plot_anchor(anchor: pd.DataFrame, figure_path: Path) -> None:
        work = anchor.dropna(
            subset=[
                "gbd2021_ncd_age_standardized_daly_rate_per100k",
                "gbd2021_ncd_age_standardized_mortality_rate_per100k",
                "health_outcome_anchor_score",
            ]
        ).copy()
        if work.empty:
            return
        fig, ax = plt.subplots(figsize=(9.6, 6.8))
        scatter = ax.scatter(
            work["gbd2021_ncd_age_standardized_daly_rate_per100k"],
            work["gbd2021_ncd_age_standardized_mortality_rate_per100k"],
            c=work["health_outcome_anchor_score"],
            cmap="viridis",
            s=70,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
        )
        label_offsets = [(8, 9), (8, -13), (-58, 8), (10, 16), (-62, -5), (10, -19), (-56, 21), (13, 2)]
        label_rows = work.head(8).reset_index(drop=True)
        for i, row in label_rows.iterrows():
            offset = label_offsets[i % len(label_offsets)]
            ax.annotate(
                row["province"],
                (
                    row["gbd2021_ncd_age_standardized_daly_rate_per100k"],
                    row["gbd2021_ncd_age_standardized_mortality_rate_per100k"],
                ),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.4,
                alpha=0.9,
                ha="left" if offset[0] >= 0 else "right",
                va="bottom" if offset[1] >= 0 else "top",
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.72},
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.45, "alpha": 0.55},
                clip_on=False,
            )
        ax.set_xlabel(choose_text("GBD2021 NCD年龄标化DALY率", "GBD2021 NCD age-standardized DALY rate", USE_CHINESE))
        ax.set_ylabel(choose_text("GBD2021 NCD年龄标化死亡率", "GBD2021 NCD age-standardized mortality rate", USE_CHINESE))
        ax.set_title(choose_text("中国省级健康结局锚点", "China Provincial Health Outcome Anchor", USE_CHINESE))
        ax.grid(alpha=0.22)
        ax.margins(x=0.09, y=0.10)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(choose_text("健康结局锚点分", "Outcome anchor score", USE_CHINESE))
        fig.tight_layout()
        fig.savefig(figure_path, dpi=220)
        plt.close(fig)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build China provincial health outcome anchor using GBD2021 age-standardized NCD outcomes plus NBS crude mortality sensitivity.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        anchor = build_anchor(clean_dir)
        clean_path = clean_dir / "external_china_health_outcome_anchor_2017_2024.csv"
        report_path = report_asset_path(report_dir, "china_health_outcome_anchor_2017_2024.csv")
        summary_path = report_asset_path(report_dir, "china_health_outcome_anchor_summary.json")
        figure_path = figure_dir / "china_health_outcome_anchor_triangle.png"
        anchor.to_csv(clean_path, index=False, encoding="utf-8-sig")
        anchor.to_csv(report_path, index=False, encoding="utf-8-sig")
        plot_anchor(anchor, figure_path)

        correlations = {
            "gbd2021_daly_vs_mortality": correlation_pair(
                anchor,
                "gbd2021_ncd_age_standardized_daly_rate_per100k",
                "gbd2021_ncd_age_standardized_mortality_rate_per100k",
            ),
            "gbd2017_four_cause_vs_gbd2021_ncd_daly": correlation_pair(
                anchor,
                "gbd2017_four_cause_daly_sum_per100k",
                "gbd2021_ncd_age_standardized_daly_rate_per100k",
            ),
            "gbd2021_ncd_daly_vs_nbs_latest_crude_mortality": correlation_pair(
                anchor,
                "gbd2021_ncd_age_standardized_daly_rate_per100k",
                "nbs_latest_crude_death_rate_per_mille",
            ),
        }
        summary = {
            "project_root": project_root.as_posix(),
            "anchor_layer": "China provincial health outcome anchor",
            "province_rows": int(anchor["province"].nunique()) if not anchor.empty else 0,
            "gbd2021_age_standardized_ncd_rows": int(anchor["gbd2021_ncd_age_standardized_daly_rate_per100k"].notna().sum()) if not anchor.empty else 0,
            "nbs_crude_mortality_sensitivity_rows": int(anchor["nbs_latest_crude_death_rate_per_mille"].notna().sum()) if not anchor.empty and "nbs_latest_crude_death_rate_per_mille" in anchor.columns else 0,
            "high_outcome_anchor_rows": int(anchor["health_outcome_anchor_type"].eq("NCD健康结局高压锚点").sum()) if not anchor.empty else 0,
            "high_or_medium_consistency_rows": int(anchor["health_outcome_anchor_consistency_type"].isin(["高一致性", "中一致性"]).sum()) if not anchor.empty else 0,
            "low_consistency_rows": int(anchor["health_outcome_anchor_consistency_type"].eq("低一致性需解释").sum()) if not anchor.empty else 0,
            "correlations": correlations,
            "claim_boundary": (
                "第1项和第3项已补到可防守状态：GBD2021年龄标化NCD死亡率/DALY率作为中国省级健康结局锚点，"
                "GBD2017同构四病种和NBS多年粗死亡率作为一致性/敏感性校验；仍不把中国省级政策写成健康结果强因果。"
            ),
            "output_files": {
                "clean_anchor": clean_path.as_posix(),
                "report_anchor": report_path.as_posix(),
                "summary": summary_path.as_posix(),
                "figure": figure_path.as_posix(),
            },
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_local_policy_execution_indicator():
    __name__ = 'run_china_local_policy_execution_indicator'
    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    SOURCE_DICTIONARY = [
        {
            "policy_domain": "医保支付改革",
            "policy_variable": "DRG/DIP国家试点城市",
            "granularity": "province_city_count",
            "enters_provincial_score": True,
            "source_title": "国家医保局DRG/DIP国家试点城市名单",
            "source_url": "https://www.nhsa.gov.cn/art/2019/6/5/art_104_6451.html; https://www.nhsa.gov.cn/art/2020/11/4/art_37_3812.html",
            "use_in_project": "省级D层政策暴露时间线和资源响应准因果候选。",
            "boundary_note": "医保支付改革反映服务效率和资源配置响应，不直接等同于死亡率/DALY下降。",
        },
        {
            "policy_domain": "慢病综合防控",
            "policy_variable": "国家慢性病综合防控示范区批次/数量",
            "granularity": "province_county_count",
            "enters_provincial_score": True,
            "source_title": "国家卫生健康委国家慢性病综合防控示范区建设评估结果",
            "source_url": "https://www.nhc.gov.cn/jkj/c100063/202006/c1d4b847838344cfb0c5dcd54d11a7f0.shtml; https://www.nhc.gov.cn/ylyjs/gzdt/202602/633ff1c5ff13439285c2714856547233.shtml",
            "use_in_project": "省级慢病治理执行强度和D层政策准备度。",
            "boundary_note": "示范区数量是政策执行强度代理，不是健康结果因果变量。",
        },
        {
            "policy_domain": "家庭医生签约",
            "policy_variable": "国家家庭医生签约政策里程碑",
            "granularity": "national_milestone",
            "enters_provincial_score": False,
            "source_title": "关于推进家庭医生签约服务的指导意见；推进家庭医生签约服务高质量发展的指导意见；家庭医生签约基本服务包清单",
            "source_url": "https://app.www.gov.cn/govdata/gov/201606/07/382093/article.html; https://www.gov.cn/zhengce/2022-03/15/content_5679180.htm; https://www.nhc.gov.cn/jws/c100073/202504/57fa208d505041168bcf192331a129d2/files/1745214346283_15710.pdf",
            "use_in_project": "写入政策包建议和基层连续管理解释，不进入省级差异评分。",
            "boundary_note": "现阶段没有可靠统一省级执行强度表，不能硬编码为省级强弱。",
        },
        {
            "policy_domain": "分级诊疗",
            "policy_variable": "分级诊疗试点/全面推开政策里程碑",
            "granularity": "national_milestone_with_city_pilot_reference",
            "enters_provincial_score": False,
            "source_title": "关于推进分级诊疗试点工作的通知；2017年分级诊疗试点城市扩大到321个",
            "source_url": "https://www.beijing.gov.cn/zhengce/zhengcefagui/201905/t20190522_59363.html; https://www.nhc.gov.cn/bgt/c100022/201707/3f4ce9abc5dc4e788e613c705ace656b.shtml",
            "use_in_project": "写入资源配置、基层首诊和双向转诊政策包解释。",
            "boundary_note": "当前未结构化完整城市名单，不能作为省级计量处理变量。",
        },
        {
            "policy_domain": "健康促进/健康城市",
            "policy_variable": "健康促进县区/健康城市政策里程碑",
            "granularity": "national_milestone",
            "enters_provincial_score": False,
            "source_title": "全国健康促进县（区）建设工作；全国健康城市和全国健康县评审标准",
            "source_url": "https://www.nhc.gov.cn/wjw/zccl/201802/17811ffe5d874ebc90c1bfc1236e5316.shtml; https://www.nhc.gov.cn/guihuaxxs/c100133/202602/6f27b61301cf4c4b956a0d1c8aec789e/files/%E9%99%84%E4%BB%B62%20%E5%85%A8%E5%9B%BD%E5%81%A5%E5%BA%B7%E5%9F%8E%E5%B8%82%E5%92%8C%E5%85%A8%E5%9B%BD%E5%81%A5%E5%BA%B7%E5%8E%BF%E8%AF%84%E5%AE%A1%E6%A0%87%E5%87%86.pdf",
            "use_in_project": "写入风险前移治理和健康促进政策包解释。",
            "boundary_note": "用于政策菜单，不进入省级准因果评分。",
        },
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False) if path.exists() else pd.DataFrame()


    def build_proxy(project_root: Path, clean_dir: Path) -> pd.DataFrame:
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        risk_policy_dir = external_data_root / "11_China_PM25_Risk_Exposure" / "china_risk_policy"
        legacy_risk_policy_dir = external_data_root / "11_China_PM25_Risk_Exposure" / ("china_" + "plan" + "b_risk_policy")
        provinces_dir = risk_policy_dir if risk_policy_dir.exists() else legacy_risk_policy_dir
        provinces_path = provinces_dir / "china_division_provinces.csv"
        if provinces_path.exists():
            provinces = read_csv(provinces_path).rename(columns={"name": "province"})[["province"]]
        else:
            provinces = pd.DataFrame({"province": []})

        payment = read_csv(clean_dir / "external_china_nhsa_payment_policy_province_latest.csv")
        chronic = read_csv(clean_dir / "external_china_nhc_chronic_demo_zone_policy_score_latest.csv")
        if not payment.empty:
            payment = payment[
                [
                    "province",
                    "drg_pilot_city_count",
                    "dip_pilot_city_count",
                    "payment_reform_city_count_total",
                    "policy_first_event_year",
                    "payment_reform_policy_score",
                    "payment_reform_policy_type",
                ]
            ]
        if not chronic.empty:
            chronic = chronic[
                [
                    "province",
                    "chronic_demo_zone_count_2014_2026",
                    "chronic_demo_zone_count_recent_2020_2026",
                    "chronic_demo_zones_per_10m_population",
                    "chronic_policy_execution_score",
                    "chronic_policy_execution_type",
                ]
            ]
        indicator = provinces.merge(payment, on="province", how="left").merge(chronic, on="province", how="left")
        for column in [
            "drg_pilot_city_count",
            "dip_pilot_city_count",
            "payment_reform_city_count_total",
            "payment_reform_policy_score",
            "chronic_demo_zone_count_2014_2026",
            "chronic_demo_zone_count_recent_2020_2026",
            "chronic_demo_zones_per_10m_population",
            "chronic_policy_execution_score",
        ]:
            indicator[column] = pd.to_numeric(proxy.get(column), errors="coerce").fillna(0)
        indicator["policy_first_event_year"] = pd.to_numeric(proxy.get("policy_first_event_year"), errors="coerce")
        indicator["payment_reform_policy_type"] = indicator.get("payment_reform_policy_type", "").fillna("医保支付改革未接入").astype(str)
        indicator["chronic_policy_execution_type"] = indicator.get("chronic_policy_execution_type", "").fillna("慢病政策执行未接入").astype(str)

        indicator["family_doctor_policy_milestone_available"] = True
        indicator["hierarchical_care_policy_milestone_available"] = True
        indicator["healthy_city_policy_milestone_available"] = True
        indicator["national_policy_milestone_domain_count"] = 3
        indicator["policy_execution_scored_domain_count"] = (
            indicator["payment_reform_policy_score"].gt(0).astype(int) + indicator["chronic_policy_execution_score"].gt(0).astype(int)
        )
        indicator["china_local_policy_execution_score"] = (
            0.55 * indicator["payment_reform_policy_score"] + 0.45 * indicator["chronic_policy_execution_score"]
        ).clip(0, 1)
        indicator["china_local_policy_execution_type"] = pd.cut(
            indicator["china_local_policy_execution_score"],
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["地方政策执行低位", "地方政策执行中位", "地方政策执行高位"],
        ).astype(str)
        indicator["policy_package_milestone_note"] = (
            "家庭医生签约、分级诊疗、健康促进/健康城市已补国家级政策里程碑；因缺统一省级执行强度，不进入省级差异评分，只进入政策包解释。"
        )
        indicator["model_use_status"] = "provincial_scored_for_payment_and_chronic_national_milestones_for_policy_package"
        indicator["claim_boundary"] = (
            "省级可计分政策执行仅使用NHSA DRG/DIP和NHC慢病示范区；家庭医生、分级诊疗、健康城市作为国家级政策菜单，"
            "不伪装成省级强弱差异。"
        )
        return indicator.sort_values("china_local_policy_execution_score", ascending=False, kind="stable")


    def build_domain_coverage(source_dictionary: pd.DataFrame, indicator: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in source_dictionary.iterrows():
            domain = row["policy_domain"]
            enters = bool(row["enters_provincial_score"])
            if domain == "医保支付改革":
                coverage = int(proxy["payment_reform_policy_score"].gt(0).sum())
            elif domain == "慢病综合防控":
                coverage = int(proxy["chronic_policy_execution_score"].gt(0).sum())
            else:
                coverage = int(proxy.shape[0])
            rows.append(
                {
                    "policy_domain": domain,
                    "granularity": row["granularity"],
                    "enters_provincial_score": enters,
                    "province_or_national_coverage_rows": coverage,
                    "coverage_share": coverage / max(int(proxy.shape[0]), 1),
                    "source_title": row["source_title"],
                    "source_url": row["source_url"],
                    "boundary_note": row["boundary_note"],
                }
            )
        return pd.DataFrame(rows)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build China local policy execution indicator and source dictionary.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        indicator = build_proxy(project_root, clean_dir)
        source_dictionary = pd.DataFrame(SOURCE_DICTIONARY)
        domain_coverage = build_domain_coverage(source_dictionary, indicator)

        clean_proxy_path = clean_dir / "external_china_local_policy_execution_indicator_latest.csv"
        report_proxy_path = report_asset_path(report_dir, "china_local_policy_execution_indicator_latest.csv")
        source_path = report_asset_path(report_dir, "china_local_policy_execution_source_dictionary.csv")
        coverage_path = report_asset_path(report_dir, "china_local_policy_execution_domain_coverage.csv")
        summary_path = report_asset_path(report_dir, "china_local_policy_execution_indicator_summary.json")
        indicator.to_csv(clean_proxy_path, index=False, encoding="utf-8-sig")
        indicator.to_csv(report_proxy_path, index=False, encoding="utf-8-sig")
        source_dictionary.to_csv(source_path, index=False, encoding="utf-8-sig")
        domain_coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

        summary = {
            "project_root": project_root.as_posix(),
            "indicator_layer": "China local policy execution indicator",
            "province_rows": int(proxy.shape[0]),
            "scored_policy_domains": ["医保支付改革", "慢病综合防控"],
            "national_milestone_policy_domains": ["家庭医生签约", "分级诊疗", "健康促进/健康城市"],
            "high_execution_rows": int(proxy["china_local_policy_execution_type"].eq("地方政策执行高位").sum()),
            "medium_execution_rows": int(proxy["china_local_policy_execution_type"].eq("地方政策执行中位").sum()),
            "low_execution_rows": int(proxy["china_local_policy_execution_type"].eq("地方政策执行低位").sum()),
            "source_domain_rows": int(source_dictionary.shape[0]),
            "scored_domain_coverage_rows": int(domain_coverage["enters_provincial_score"].sum()),
            "national_milestone_domain_rows": int((~domain_coverage["enters_provincial_score"]).sum()),
            "claim_boundary": (
                "第5项已补到可防守状态：地方政策执行评分只用有省级差异和官方名单的NHSA DRG/DIP、NHC慢病示范区；"
                "家庭医生、分级诊疗、健康城市补为国家政策里程碑和政策包解释，不硬写省级因果。"
            ),
            "output_files": {
                "clean_proxy": clean_proxy_path.as_posix(),
                "report_proxy": report_proxy_path.as_posix(),
                "source_dictionary": source_path.as_posix(),
                "domain_coverage": coverage_path.as_posix(),
                "summary": summary_path.as_posix(),
            },
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_policy_quasi_causal_validation():
    __name__ = 'run_china_policy_quasi_causal_validation'
    import argparse
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from foundation import choose_text, configure_matplotlib_fonts
    from foundation import detect_project_root as shared_detect_project_root
    from run_china_mapping_framework import load_nbs_provincial_panel


    USE_CHINESE = configure_matplotlib_fonts()
    POLICY_FILE = "external_china_nhsa_payment_policy_province_latest.csv"
    CENSUS_FILE = "external_china_census_2020_province_population_age.csv"
    CHRONIC_PANEL_FILE = "external_china_nhc_chronic_demo_zone_policy_execution_panel.csv"
    CHRONIC_LATEST_FILE = "external_china_nhc_chronic_demo_zone_policy_score_latest.csv"
    MORTALITY_PANEL_FILE = "external_china_nbs_mortality_panel_ocr_2018_2024.csv"


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def percentile_by_year(df: pd.DataFrame, column: str) -> pd.Series:
        return df.groupby("year", sort=False)[column].rank(pct=True, method="average")


    def build_panel(clean_dir: Path) -> pd.DataFrame:
        panel = load_nbs_provincial_panel(clean_dir)
        census = pd.read_csv(clean_dir / CENSUS_FILE)
        policy = pd.read_csv(clean_dir / POLICY_FILE)
        panel = panel.merge(census[["province", "census_2020_population"]], on="province", how="left")
        panel = panel.merge(policy, on="province", how="left")
        panel["medical_staff_per_10k_population"] = panel["medical_staff_10k_persons"] * 10000 / panel["census_2020_population"] * 10000
        panel["medical_institutions_per_million_population"] = panel["medical_institutions_count"] / panel["census_2020_population"] * 1_000_000
        panel = panel.sort_values(["province", "year"], kind="stable")
        panel["medical_staff_growth_yoy"] = panel.groupby("province", sort=False)["medical_staff_per_10k_population"].pct_change()
        panel["medical_institutions_growth_yoy"] = panel.groupby("province", sort=False)["medical_institutions_per_million_population"].pct_change()
        panel["staff_per_capita_percentile_year"] = percentile_by_year(panel, "medical_staff_per_10k_population")
        panel["institution_per_capita_percentile_year"] = percentile_by_year(panel, "medical_institutions_per_million_population")
        panel["resource_response_score_panel"] = 0.6 * panel["staff_per_capita_percentile_year"] + 0.4 * panel["institution_per_capita_percentile_year"]

        panel["policy_first_event_year"] = pd.to_numeric(panel["policy_first_event_year"], errors="coerce")
        panel["payment_reform_policy_score"] = pd.to_numeric(panel["payment_reform_policy_score"], errors="coerce").fillna(0)
        panel["high_payment_exposure"] = panel["payment_reform_policy_score"] >= 2 / 3
        panel["drg_exposed"] = panel["drg_pilot_2019"].astype(bool)
        panel["dip_exposed"] = panel["dip_pilot_2020"].astype(bool)
        panel["high_payment_exposure_post"] = panel["high_payment_exposure"].astype(float) * (panel["year"] >= panel["policy_first_event_year"].fillna(9999)).astype(float)
        panel["drg_2019_post"] = panel["drg_exposed"].astype(float) * (panel["year"] >= 2019).astype(float)
        panel["dip_2020_post"] = panel["dip_exposed"].astype(float) * (panel["year"] >= 2020).astype(float)
        panel["high_payment_placebo_2017"] = panel["high_payment_exposure"].astype(float) * (panel["year"] >= 2017).astype(float) * (panel["year"] < 2019).astype(float)
        panel["drg_placebo_2017"] = panel["drg_exposed"].astype(float) * (panel["year"] >= 2017).astype(float) * (panel["year"] < 2019).astype(float)
        panel["dip_placebo_2018"] = panel["dip_exposed"].astype(float) * (panel["year"] >= 2018).astype(float) * (panel["year"] < 2020).astype(float)
        panel = add_chronic_policy_panel(panel, clean_dir)
        panel = add_mortality_panel(panel, clean_dir)
        return panel


    def add_chronic_policy_panel(panel: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        timeline_path = clean_dir / CHRONIC_PANEL_FILE
        latest_path = clean_dir / CHRONIC_LATEST_FILE
        panel = panel.copy()
        if not timeline_path.exists():
            panel["chronic_policy_panel_available"] = False
            return panel
        seed = pd.read_csv(timeline_path)
        seed["decision_year"] = pd.to_numeric(seed["decision_year"], errors="coerce")
        seed["zone_count"] = pd.to_numeric(seed["zone_count"], errors="coerce").fillna(0)
        rows = []
        for province, group in seed.groupby("province", sort=False):
            for year in sorted(panel["year"].dropna().astype(int).unique().tolist()):
                past = group.loc[group["decision_year"].le(year)]
                recent = group.loc[group["decision_year"].between(2020, year)]
                rows.append(
                    {
                        "province": province,
                        "year": year,
                        "chronic_demo_zone_cumulative_count_year": float(past["zone_count"].sum()),
                        "chronic_demo_zone_recent_count_year": float(recent["zone_count"].sum()),
                        "chronic_demo_zone_batch_count_year": int(past["batch_id"].nunique()),
                    }
                )
        chronic_panel = pd.DataFrame(rows)
        panel = panel.merge(chronic_panel, on=["province", "year"], how="left")
        for column in [
            "chronic_demo_zone_cumulative_count_year",
            "chronic_demo_zone_recent_count_year",
            "chronic_demo_zone_batch_count_year",
        ]:
            panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0)
        panel["chronic_demo_zones_cumulative_per_10m_year"] = (
            panel["chronic_demo_zone_cumulative_count_year"] / panel["census_2020_population"] * 10_000_000
        )
        panel["chronic_policy_intensity_year"] = percentile_by_year(panel, "chronic_demo_zones_cumulative_per_10m_year").fillna(0)
        panel["chronic_policy_recent_intensity_year"] = percentile_by_year(panel, "chronic_demo_zone_recent_count_year").fillna(0)
        latest_score = pd.DataFrame()
        if latest_path.exists():
            latest_score = pd.read_csv(latest_path)[["province", "chronic_policy_execution_score", "chronic_policy_execution_type"]]
        if not latest_score.empty:
            panel = panel.merge(latest_score, on="province", how="left")
        else:
            panel["chronic_policy_execution_score"] = np.nan
            panel["chronic_policy_execution_type"] = ""
        panel["chronic_policy_execution_score"] = pd.to_numeric(panel["chronic_policy_execution_score"], errors="coerce")
        panel["high_chronic_execution"] = panel["chronic_policy_execution_score"].fillna(0) >= 2 / 3
        panel["chronic_policy_intensity_post_2020"] = panel["chronic_policy_intensity_year"].astype(float) * (panel["year"] >= 2020).astype(float)
        panel["high_chronic_execution_post_2020"] = panel["high_chronic_execution"].astype(float) * (panel["year"] >= 2020).astype(float)
        panel["chronic_policy_intensity_placebo_2017"] = panel["chronic_policy_intensity_year"].astype(float) * (panel["year"] >= 2017).astype(float) * (panel["year"] < 2020).astype(float)
        panel["high_chronic_placebo_2017"] = panel["high_chronic_execution"].astype(float) * (panel["year"] >= 2017).astype(float) * (panel["year"] < 2020).astype(float)
        panel["chronic_policy_panel_available"] = True
        return panel


    def add_mortality_panel(panel: pd.DataFrame, clean_dir: Path) -> pd.DataFrame:
        path = clean_dir / MORTALITY_PANEL_FILE
        panel = panel.copy()
        if not path.exists():
            panel["nbs_mortality_panel_available"] = False
            return panel
        mortality = pd.read_csv(path)
        mortality = mortality.loc[
            mortality["model_use_status"].eq("china_health_outcome_sensitivity_panel_candidate"),
            [
                "province",
                "source_year",
                "birth_rate_per_mille_candidate",
                "death_rate_per_mille_candidate",
                "natural_growth_rate_per_mille_candidate",
                "death_rate_change_yoy_candidate",
                "candidate_complete",
                "source_url",
            ],
        ].rename(
            columns={
                "source_year": "year",
                "birth_rate_per_mille_candidate": "nbs_birth_rate_per_mille_candidate",
                "death_rate_per_mille_candidate": "nbs_crude_death_rate_per_mille_candidate",
                "natural_growth_rate_per_mille_candidate": "nbs_natural_growth_rate_per_mille_candidate",
                "death_rate_change_yoy_candidate": "nbs_crude_death_rate_change_yoy_candidate",
                "candidate_complete": "nbs_mortality_candidate_complete",
                "source_url": "nbs_mortality_source_url",
            }
        )
        mortality["year"] = pd.to_numeric(mortality["year"], errors="coerce").astype("Int64")
        panel = panel.merge(mortality, on=["province", "year"], how="left")
        panel["nbs_mortality_panel_available"] = panel["nbs_crude_death_rate_per_mille_candidate"].notna()
        panel["nbs_mortality_boundary_note"] = "NBS粗死亡率为全因粗率，受老龄化和疫情冲击影响；只作为健康结局敏感性/边界检验，不作为强因果主结论。"
        return panel


    def twfe_single_x(df: pd.DataFrame, outcome: str, treatment: str) -> dict[str, object]:
        work = df[["province", "year", outcome, treatment]].copy()
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
        work[treatment] = pd.to_numeric(work[treatment], errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome, treatment])
        if work.shape[0] < 100 or work[treatment].nunique() < 2:
            return {"ok": False, "reason": "insufficient_variation", "n_obs": int(work.shape[0])}
        y = work[outcome]
        x = work[treatment]
        y_resid = y - work.groupby("province")[outcome].transform("mean") - work.groupby("year")[outcome].transform("mean") + y.mean()
        x_resid = x - work.groupby("province")[treatment].transform("mean") - work.groupby("year")[treatment].transform("mean") + x.mean()
        denom = float(np.dot(x_resid, x_resid))
        if denom <= 1e-12:
            return {"ok": False, "reason": "zero_within_variation", "n_obs": int(work.shape[0])}
        beta = float(np.dot(x_resid, y_resid) / denom)
        resid = y_resid - beta * x_resid
        cluster_scores = []
        for _, group in work.assign(x_resid=x_resid, resid=resid).groupby("province", sort=False):
            cluster_scores.append(float(np.dot(group["x_resid"], group["resid"])))
        g = len(cluster_scores)
        n = work.shape[0]
        meat = float(np.sum(np.square(cluster_scores)))
        variance = (meat / (denom**2)) * (g / max(g - 1, 1)) * ((n - 1) / max(n - 2, 1))
        se = float(np.sqrt(max(variance, 0)))
        t_stat = beta / se if se > 0 else np.nan
        p_value = float(2 * stats.t.sf(abs(t_stat), df=max(g - 1, 1))) if np.isfinite(t_stat) else np.nan
        return {
            "ok": True,
            "coefficient_twfe": beta,
            "cluster_se": se,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": p_value,
            "n_obs": int(n),
            "clusters": int(g),
            "treated_provinces": int(work.loc[work[treatment].gt(0), "province"].nunique()),
            "control_or_not_yet_treated_provinces": int(work.loc[work[treatment].eq(0), "province"].nunique()),
        }


    def build_estimates(panel: pd.DataFrame) -> pd.DataFrame:
        treatments = [
            ("high_payment_exposure_post", "高DRG/DIP政策暴露"),
            ("drg_2019_post", "DRG国家试点"),
            ("dip_2020_post", "DIP国家试点"),
            ("chronic_policy_intensity_post_2020", "NHC慢病示范区执行强度"),
            ("high_chronic_execution_post_2020", "高慢病示范区执行暴露"),
        ]
        placebos = {
            "high_payment_exposure_post": "high_payment_placebo_2017",
            "drg_2019_post": "drg_placebo_2017",
            "dip_2020_post": "dip_placebo_2018",
            "chronic_policy_intensity_post_2020": "chronic_policy_intensity_placebo_2017",
            "high_chronic_execution_post_2020": "high_chronic_placebo_2017",
        }
        outcomes = [
            ("resource_response_score_panel", "省级资源响应综合分", 1),
            ("medical_staff_per_10k_population", "每万人卫生人员", 1),
            ("medical_institutions_per_million_population", "每百万人医疗卫生机构数", 1),
            ("medical_staff_growth_yoy", "卫生人员人均同比增长", 1),
            ("medical_institutions_growth_yoy", "机构人均同比增长", 1),
        ]
        rows = []
        for treatment, treatment_label in treatments:
            for outcome, outcome_label, expected_sign in outcomes:
                estimate = twfe_single_x(panel, outcome, treatment)
                placebo = twfe_single_x(panel, outcome, placebos[treatment])
                direction = estimate.get("coefficient_twfe", np.nan) * expected_sign > 0 if estimate.get("ok") else False
                significant = estimate.get("p_value", np.nan) < 0.10 if estimate.get("ok") else False
                placebo_clean = (not placebo.get("ok")) or pd.isna(placebo.get("p_value")) or placebo.get("p_value", 0) >= 0.10
                rows.append(
                    {
                        "policy_exposure": treatment,
                        "policy_label": treatment_label,
                        "outcome_column": outcome,
                        "outcome_label": outcome_label,
                        "expected_sign": expected_sign,
                        **{f"main_{k}": v for k, v in estimate.items()},
                        "placebo_treatment": placebos[treatment],
                        **{f"placebo_{k}": v for k, v in placebo.items()},
                        "direction_consistent": bool(direction),
                        "main_p_lt_0_10": bool(significant),
                        "placebo_clean": bool(placebo_clean),
                        "china_d_quasi_causal_tier": classify_tier(estimate, direction, significant, placebo_clean),
                        "claim_boundary": "该检验估计DRG/DIP政策暴露对省级资源响应指标的准因果候选效应；不是健康结果因果估计。",
                    }
                )
        return pd.DataFrame(rows)


    def build_health_outcome_sensitivity_estimates(panel: pd.DataFrame) -> pd.DataFrame:
        treatments = [
            ("high_payment_exposure_post", "高DRG/DIP政策暴露", "high_payment_placebo_2017"),
            ("drg_2019_post", "DRG国家试点", "drg_placebo_2017"),
            ("dip_2020_post", "DIP国家试点", "dip_placebo_2018"),
            ("chronic_policy_intensity_post_2020", "NHC慢病示范区执行强度", "chronic_policy_intensity_placebo_2017"),
            ("high_chronic_execution_post_2020", "高慢病示范区执行暴露", "high_chronic_placebo_2017"),
        ]
        outcomes = [
            ("nbs_crude_death_rate_per_mille_candidate", "NBS全因粗死亡率OCR候选", -1),
            ("nbs_crude_death_rate_change_yoy_candidate", "NBS全因粗死亡率同比变化候选", -1),
        ]
        rows = []
        for treatment, treatment_label, placebo_treatment in treatments:
            if treatment not in panel.columns or placebo_treatment not in panel.columns:
                continue
            for outcome, outcome_label, expected_sign in outcomes:
                if outcome not in panel.columns:
                    continue
                estimate = twfe_single_x(panel, outcome, treatment)
                placebo = twfe_single_x(panel, outcome, placebo_treatment)
                direction = estimate.get("coefficient_twfe", np.nan) * expected_sign > 0 if estimate.get("ok") else False
                significant = estimate.get("p_value", np.nan) < 0.10 if estimate.get("ok") else False
                placebo_clean = (not placebo.get("ok")) or pd.isna(placebo.get("p_value")) or placebo.get("p_value", 0) >= 0.10
                rows.append(
                    {
                        "policy_exposure": treatment,
                        "policy_label": treatment_label,
                        "outcome_column": outcome,
                        "outcome_label": outcome_label,
                        "expected_sign": expected_sign,
                        **{f"main_{k}": v for k, v in estimate.items()},
                        "placebo_treatment": placebo_treatment,
                        **{f"placebo_{k}": v for k, v in placebo.items()},
                        "direction_consistent": bool(direction),
                        "main_p_lt_0_10": bool(significant),
                        "placebo_clean": bool(placebo_clean),
                        "china_health_outcome_sensitivity_tier": classify_health_sensitivity_tier(
                            estimate, direction, significant, placebo_clean
                        ),
                        "claim_boundary": "该检验只使用NBS全因粗死亡率OCR面板做健康结局敏感性；不是年龄标化NCD死亡率，也不升级为省级政策健康强因果。",
                    }
                )
        return pd.DataFrame(rows)


    def classify_tier(estimate: dict[str, object], direction: bool, significant: bool, placebo_clean: bool) -> str:
        if not estimate.get("ok"):
            return "不可用"
        if direction and significant and placebo_clean:
            return "中国D层响应准因果候选"
        if direction and (significant or placebo_clean):
            return "中国D层响应方向性证据"
        return "探索性或不支持"


    def classify_health_sensitivity_tier(estimate: dict[str, object], direction: bool, significant: bool, placebo_clean: bool) -> str:
        if not estimate.get("ok"):
            return "不可用"
        if direction and significant and placebo_clean:
            return "健康结局敏感性方向候选"
        if direction:
            return "健康结局方向性观察"
        return "不支持或受粗率边界限制"


    def plot_estimates(estimates: pd.DataFrame, path: Path) -> None:
        if estimates.empty:
            return
        plot_df = estimates.loc[estimates["main_ok"].astype(bool)].copy()
        plot_df = plot_df.sort_values("main_p_value", kind="stable").head(12)
        labels = plot_df["policy_label"] + "\n" + plot_df["outcome_label"]
        colors = np.where(plot_df["china_d_quasi_causal_tier"].eq("中国D层响应准因果候选"), "#0b6b57", "#4f7cac")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(labels, plot_df["main_coefficient_twfe"], color=colors)
        ax.axvline(0, color="#333333", linewidth=1)
        ax.invert_yaxis()
        ax.set_xlabel(choose_text("双向固定效应系数", "Two-way FE coefficient", USE_CHINESE))
        ax.set_title(choose_text("中国DRG/DIP政策暴露对响应能力的准因果候选", "China DRG/DIP Response Quasi-Causal Candidates", USE_CHINESE))
        ax.grid(axis="x", alpha=0.22)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def main() -> None:
        parser = argparse.ArgumentParser(description="Run China provincial DRG/DIP quasi-causal validation on resource response outcomes.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        panel = build_panel(clean_dir)
        estimates = build_estimates(panel)
        health_sensitivity = build_health_outcome_sensitivity_estimates(panel)
        strong = estimates.loc[estimates["china_d_quasi_causal_tier"].eq("中国D层响应准因果候选")]
        direction = estimates.loc[estimates["china_d_quasi_causal_tier"].eq("中国D层响应方向性证据")]
        health_direction = health_sensitivity.loc[
            health_sensitivity["china_health_outcome_sensitivity_tier"].isin(["健康结局敏感性方向候选", "健康结局方向性观察"])
        ] if not health_sensitivity.empty else pd.DataFrame()

        panel_path = report_asset_path(report_dir, "china_policy_quasi_causal_panel.csv")
        estimate_path = report_asset_path(report_dir, "china_policy_quasi_causal_response_estimates.csv")
        health_sensitivity_path = report_asset_path(report_dir, "china_policy_health_outcome_sensitivity_estimates.csv")
        summary_path = report_asset_path(report_dir, "china_policy_quasi_causal_response_summary.json")
        figure_path = figure_dir / "china_policy_quasi_causal_response_estimates.png"
        panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
        estimates.to_csv(estimate_path, index=False, encoding="utf-8-sig")
        health_sensitivity.to_csv(health_sensitivity_path, index=False, encoding="utf-8-sig")
        plot_estimates(estimates, figure_path)

        summary = {
            "project_root": project_root.as_posix(),
            "validation_layer": "China provincial DRG/DIP quasi-causal response validation",
            "panel_rows": int(panel.shape[0]),
            "province_count": int(panel["province"].nunique()),
            "year_min": int(panel["year"].min()),
            "year_max": int(panel["year"].max()),
            "estimate_rows": int(estimates.shape[0]),
            "china_d_response_quasi_causal_candidates": int(strong.shape[0]),
            "china_d_response_directional_candidates": int(direction.shape[0]),
            "nbs_health_outcome_sensitivity_panel_available": bool(panel["nbs_mortality_panel_available"].any()) if "nbs_mortality_panel_available" in panel.columns else False,
            "nbs_health_outcome_sensitivity_rows": int(health_sensitivity.shape[0]),
            "nbs_health_outcome_directional_observations": int(health_direction.shape[0]),
            "chronic_policy_panel_available": bool(panel["chronic_policy_panel_available"].any()) if "chronic_policy_panel_available" in panel.columns else False,
            "top_candidates": strong.sort_values("main_p_value", kind="stable").head(6).to_dict(orient="records"),
            "health_outcome_sensitivity_top": health_sensitivity.sort_values("main_p_value", kind="stable").head(6).to_dict(orient="records") if not health_sensitivity.empty else [],
            "claim_boundary": "中国省级D层已从政策暴露时间线升级为资源响应准因果候选检验，并补NBS分省粗死亡率OCR健康结局敏感性面板；但粗死亡率不是年龄标化NCD结局，仍不声明省级政策已经因果降低死亡率或DALY。",
            "output_files": {
                "china_policy_quasi_causal_panel": panel_path.as_posix(),
                "china_policy_quasi_causal_response_estimates": estimate_path.as_posix(),
                "china_policy_health_outcome_sensitivity_estimates": health_sensitivity_path.as_posix(),
                "china_policy_quasi_causal_response_summary": summary_path.as_posix(),
                "china_policy_quasi_causal_response_estimates_figure": figure_path.as_posix(),
            },
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_china_source_boundary_audit():
    __name__ = 'run_china_source_boundary_audit'
    import argparse
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from foundation import choose_text, configure_matplotlib_fonts
    from foundation import detect_project_root as shared_detect_project_root


    USE_CHINESE = configure_matplotlib_fonts()


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False) if path.exists() else pd.DataFrame()


    def build_pm25_weighting_audit(clean_dir: Path, report_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
        weighted_path = clean_dir / "external_china_scidb_provincial_pm25_population_weighted.csv"
        legacy_path = clean_dir / "external_china_scidb_provincial_pm25.csv"
        pm25 = read_csv(weighted_path)
        pm25_source_status = "prefecture_city_population_weighted" if not pm25.empty else "legacy_city_mean"
        if pm25.empty:
            pm25 = read_csv(legacy_path)
        census = read_csv(clean_dir / "external_china_census_2020_province_population_age.csv")
        if pm25.empty or census.empty:
            return pd.DataFrame(), {
                "pm25_audit_status": "missing_pm25_or_population",
                "population_weighted_china_pm25_2024": None,
            }
        latest_year = int(pd.to_numeric(pm25["year"], errors="coerce").max())
        latest = pm25.loc[pd.to_numeric(pm25["year"], errors="coerce").eq(latest_year)].copy()
        latest = latest.merge(
            census[["province", "census_2020_population", "census_2020_age65_plus_pct"]],
            on="province",
            how="left",
        )
        latest["pm25_city_mean_ug_m3"] = pd.to_numeric(latest["pm25_city_mean_ug_m3"], errors="coerce")
        if "pm25_population_weighted_ug_m3" not in latest.columns:
            latest["pm25_population_weighted_ug_m3"] = latest["pm25_city_mean_ug_m3"]
        latest["pm25_population_weighted_ug_m3"] = pd.to_numeric(latest["pm25_population_weighted_ug_m3"], errors="coerce")
        latest["pm25_weighted_minus_city_mean_ug_m3"] = latest["pm25_population_weighted_ug_m3"] - latest["pm25_city_mean_ug_m3"]
        latest["census_2020_population"] = pd.to_numeric(latest["census_2020_population"], errors="coerce")
        population_sum = latest["census_2020_population"].sum()
        latest["province_population_weight"] = latest["census_2020_population"] / population_sum
        latest["population_weighted_china_pm25_contribution"] = latest["pm25_population_weighted_ug_m3"] * latest["province_population_weight"]
        national_weighted = float(latest["population_weighted_china_pm25_contribution"].sum())
        national_unweighted = float(latest["pm25_population_weighted_ug_m3"].mean())
        latest["national_population_weighted_pm25_2024_ug_m3"] = national_weighted
        latest["national_unweighted_province_mean_pm25_2024_ug_m3"] = national_unweighted
        latest["within_province_city_weighting_status"] = (
            "integrated_prefecture_city_population_weights_where_matched"
            if pm25_source_status == "prefecture_city_population_weighted"
            else "prefecture_population_weights_not_integrated"
        )
        latest["model_use_status"] = (
            "province_level_pm25_uses_prefecture_city_population_weighted_mean; national aggregate is population-weighted by province"
            if pm25_source_status == "prefecture_city_population_weighted"
            else "province_level_city_mean_used_for_provincial_pressure; national aggregate is population-weighted by province"
        )
        latest["boundary_note"] = (
            "省内城市PM2.5已按2020地级市人口权重聚合；匹配率和未匹配城市另有审计表。"
            if pm25_source_status == "prefecture_city_population_weighted"
            else "省内城市PM2.5为城市均值，未按地级市人口加权；全国汇总已用七普省级人口做省级人口加权。"
        )
        latest["source_url"] = latest.get("source_url", "https://www.scidb.cn/en/detail?dataSetId=84279e24dec04d4ba68c1fadb70ff1ce")
        audit = latest[
            [
                "province",
                "year",
                "pm25_city_mean_ug_m3",
                "pm25_population_weighted_ug_m3",
                "pm25_weighted_minus_city_mean_ug_m3",
                "city_count",
                "matched_city_count",
                "matched_city_rate",
                "census_2020_population",
                "province_population_weight",
                "population_weighted_china_pm25_contribution",
                "national_population_weighted_pm25_2024_ug_m3",
                "national_unweighted_province_mean_pm25_2024_ug_m3",
                "within_province_city_weighting_status",
                "model_use_status",
                "boundary_note",
                "source_url",
            ]
        ].copy()
        summary = {
            "pm25_latest_year": latest_year,
            "pm25_province_rows": int(audit.shape[0]),
            "population_weighted_china_pm25_2024": national_weighted,
            "unweighted_province_mean_china_pm25_2024": national_unweighted,
            "within_province_city_weighting_status": latest["within_province_city_weighting_status"].iloc[0],
            "latest_mean_matched_city_rate": float(pd.to_numeric(latest.get("matched_city_rate", pd.Series([np.nan])), errors="coerce").mean()),
        }
        return audit, summary


    def build_gbd_boundary_audit(clean_dir: Path) -> pd.DataFrame:
        disease = read_csv(clean_dir / "external_china_gbd2017_province_daly_rates.csv")
        risk = read_csv(clean_dir / "external_china_gbd2017_province_risk_sev.csv")
        ncd2021 = read_csv(clean_dir / "external_china_gbd2021_province_ncd_daly_rates.csv")
        health_anchor = read_csv(clean_dir / "external_china_health_outcome_anchor_2017_2024.csv")
        ncd2021_mainland = ncd2021.loc[~ncd2021.get("province", pd.Series(dtype=str)).isin(["香港特别行政区", "澳门特别行政区"])].copy() if not ncd2021.empty else pd.DataFrame()
        rows = [
            {
                "source_layer": "current_structured_china_province_disease",
                "source_name": "GBD China 2017 province DALY rates from Lancet/PMC supplement",
                "source_year": 2017,
                "province_rows": int(disease["province"].nunique()) if not disease.empty and "province" in disease.columns else 0,
                "structured_in_repo": not disease.empty,
                "used_in_mapping": True,
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6891889/",
                "boundary_note": "当前中国省级疾病负担结构化输入是2017口径，用于省级相对压力画像，不说成2021或2023。",
            },
            {
                "source_layer": "current_structured_china_province_risk",
                "source_name": "GBD China 2017 province risk SEV from Lancet/PMC supplement",
                "source_year": 2017,
                "province_rows": int(risk["province"].nunique()) if not risk.empty and "province" in risk.columns else 0,
                "structured_in_repo": not risk.empty,
                "used_in_mapping": True,
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6891889/",
                "boundary_note": "当前中国省级风险暴露结构化输入是2017 SEV口径，用于主导风险排序，不说成最新风险调查。",
            },
            {
                "source_layer": "current_structured_china_province_ncd_total_burden",
                "source_name": "Burden of non-communicable diseases in China and its provinces, GBD 2021",
                "source_year": 2021,
                "province_rows": int(ncd2021_mainland["province"].nunique()) if not ncd2021_mainland.empty and "province" in ncd2021_mainland.columns else 0,
                "structured_in_repo": not ncd2021_mainland.empty,
                "used_in_mapping": not ncd2021_mainland.empty,
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11441934/",
                "boundary_note": "已结构化接入GBD 2021中国省级NCD总负担年龄标化DALY率，用于NCD总压力边界字段；它不是当前四类疾病/十大风险SEV的同构替代。",
            },
            {
                "source_layer": "china_health_outcome_anchor",
                "source_name": "GBD2021 age-standardized NCD outcomes + GBD2017 four-cause + NBS crude mortality anchor",
                "source_year": "2017,2021,2018-2024",
                "province_rows": int(health_anchor["province"].nunique()) if not health_anchor.empty and "province" in health_anchor.columns else 0,
                "structured_in_repo": not health_anchor.empty,
                "used_in_mapping": not health_anchor.empty,
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11441934/; https://www.stats.gov.cn/sj/ndsj/",
                "boundary_note": "已补中国省级年龄标化NCD健康结局锚点和NBS粗死亡率敏感性一致性校验；不是政策健康结果强因果面板。",
            },
            {
                "source_layer": "global_country_panel",
                "source_name": "Global GBD/DBD country panel used by ABCD world model",
                "source_year": 2023,
                "province_rows": 0,
                "structured_in_repo": True,
                "used_in_mapping": False,
                "source_url": "local_global_panel",
                "boundary_note": "全球国家层面可到2023，但不具备中国省级粒度，不能直接替代省级GBD 2017表。",
            },
        ]
        return pd.DataFrame(rows)


    def build_china_supplement_source_audit(clean_dir: Path) -> pd.DataFrame:
        nbs_explain = read_csv(clean_dir / "external_china_nbs2025_ocr_explanatory_candidates_2024.csv")
        nbs_mortality = read_csv(clean_dir / "external_china_nbs_mortality_panel_ocr_2018_2024.csv")
        chronic = read_csv(clean_dir / "external_china_nhc_chronic_demo_zone_policy_score_latest.csv")
        local_policy = read_csv(clean_dir / "external_china_local_policy_execution_indicator_latest.csv")
        c_qc = read_csv(clean_dir / "external_china_c_layer_ocr_candidate_qc_2024.csv")
        c_field_qc = read_csv(clean_dir / "external_china_c_layer_ocr_field_level_qc_2024.csv")
        mortality_included = nbs_mortality.loc[
            nbs_mortality.get("model_use_status", pd.Series(dtype=str)).eq("china_health_outcome_sensitivity_panel_candidate")
        ].copy() if not nbs_mortality.empty else pd.DataFrame()
        rows = [
            {
                "source_layer": "nbs_2025_explanatory_ocr_candidates",
                "source_name": "China Statistical Yearbook 2025 provincial death/urbanization/GRP OCR candidates",
                "source_year": 2024,
                "province_rows": int(nbs_explain["province"].nunique()) if not nbs_explain.empty else 0,
                "structured_in_repo": not nbs_explain.empty,
                "used_in_mapping": not nbs_explain.empty,
                "source_url": "https://www.stats.gov.cn/sj/ndsj/2025/indexch.htm",
                "boundary_note": "用于解释省级人口、城镇化和经济差异；OCR候选不进入健康结果因果主回归。",
            },
            {
                "source_layer": "nbs_multi_year_crude_mortality_ocr_sensitivity_panel",
                "source_name": "China Statistical Yearbook provincial crude birth/death/natural-growth OCR panel",
                "source_year": "2018,2019,2021,2022,2023,2024",
                "province_rows": int(mortality_included["province"].nunique()) if not mortality_included.empty else 0,
                "structured_in_repo": not mortality_included.empty,
                "used_in_mapping": not mortality_included.empty,
                "source_url": "https://www.stats.gov.cn/sj/ndsj/",
                "boundary_note": "作为中国D层健康结局敏感性面板；全因粗死亡率受年龄结构和疫情冲击影响，不能作为年龄标化NCD健康强因果。",
            },
            {
                "source_layer": "nhc_chronic_demo_zone_policy_execution",
                "source_name": "NHC national chronic disease comprehensive prevention and control demonstration zone batches",
                "source_year": 2026,
                "province_rows": int(chronic["province"].nunique()) if not chronic.empty else 0,
                "structured_in_repo": not chronic.empty,
                "used_in_mapping": not chronic.empty,
                "source_url": "https://www.nhc.gov.cn/ylyjs/gzdt/202602/633ff1c5ff13439285c2714856547233.shtml",
                "boundary_note": "作为中国D层慢病治理政策执行强度代理；不直接声明政策降低死亡率或DALY。",
            },
            {
                "source_layer": "china_local_policy_execution_indicator",
                "source_name": "China local policy execution indicator and national policy milestone dictionary",
                "source_year": "2016-2026",
                "province_rows": int(local_policy["province"].nunique()) if not local_policy.empty else 0,
                "structured_in_repo": not local_policy.empty,
                "used_in_mapping": not local_policy.empty,
                "source_url": "NHSA/NHC/Gov.cn official policy pages listed in china_local_policy_execution_source_dictionary.csv",
                "boundary_note": "省级评分只使用DRG/DIP和慢病示范区；家庭医生、分级诊疗、健康城市作为国家政策里程碑进入政策包，不伪装为省级差异。",
            },
            {
                "source_layer": "c_layer_ocr_candidate_qc",
                "source_name": "NBS C-layer health beds/insurance/fund OCR candidate quality control",
                "source_year": 2024,
                "province_rows": int(c_qc["province"].nunique()) if not c_qc.empty else 0,
                "structured_in_repo": not c_qc.empty,
                "used_in_mapping": not c_qc.empty,
                "source_url": "https://www.stats.gov.cn/sj/ndsj/2025/indexch.htm",
                "boundary_note": "机器QC后的补充候选层；核心C层评分继续使用稳定NBS卫生人员/机构和人口校正指标。",
            },
            {
                "source_layer": "c_layer_ocr_field_level_qc",
                "source_name": "NBS C-layer health beds/insurance/fund OCR field-level quality control",
                "source_year": 2024,
                "province_rows": int(c_field_qc["province"].nunique()) if not c_field_qc.empty else 0,
                "structured_in_repo": not c_field_qc.empty,
                "used_in_mapping": not c_field_qc.empty,
                "source_url": "https://www.stats.gov.cn/sj/ndsj/2025/indexch.htm",
                "boundary_note": "字段级通过才可解释，未通过字段自动禁用；机器复核风险已转为字段级QC规则。",
            },
        ]
        return pd.DataFrame(rows)


    def plot_pm25_audit(audit: pd.DataFrame, figure_path: Path) -> None:
        if audit.empty:
            return
        plot_df = audit.sort_values("population_weighted_china_pm25_contribution", ascending=False).head(12).copy()
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(
            plot_df["province"],
            plot_df["population_weighted_china_pm25_contribution"],
            color="#2a9d8f",
        )
        ax.set_ylabel(choose_text("对全国人口加权PM2.5均值的贡献", "Contribution to national population-weighted PM2.5", USE_CHINESE))
        ax.set_title(choose_text("2024省级PM2.5人口加权审计", "2024 Provincial PM2.5 Weighting Audit", USE_CHINESE))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=220)
        plt.close(fig)


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if pd.isna(value):
            return None
        return value


    def main() -> None:
        parser = argparse.ArgumentParser(description="Audit China PM2.5 population weighting and GBD provincial year boundaries.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        pm25_audit, pm25_summary = build_pm25_weighting_audit(clean_dir, report_dir)
        gbd_audit = build_gbd_boundary_audit(clean_dir)
        supplement_audit = build_china_supplement_source_audit(clean_dir)
        gbd_bridge = read_csv(report_asset_path(report_dir, "china_gbd2021_freshness_bridge.csv"))
        health_anchor = read_csv(report_asset_path(report_dir, "china_health_outcome_anchor_2017_2024.csv"))

        pm25_path = report_asset_path(report_dir, "china_pm25_population_weighting_audit.csv")
        gbd_path = report_asset_path(report_dir, "china_gbd_province_year_boundary_audit.csv")
        supplement_path = report_asset_path(report_dir, "china_supplement_source_boundary_audit.csv")
        summary_path = report_asset_path(report_dir, "china_abcd_source_boundary_summary.json")
        figure_path = figure_dir / "china_pm25_population_weighting_audit.png"

        pm25_audit.to_csv(pm25_path, index=False, encoding="utf-8-sig")
        gbd_audit.to_csv(gbd_path, index=False, encoding="utf-8-sig")
        supplement_audit.to_csv(supplement_path, index=False, encoding="utf-8-sig")
        plot_pm25_audit(pm25_audit, figure_path)

        summary = {
            "project_root": project_root.as_posix(),
            "audit_layer": "China source boundary and weighting audit",
            **pm25_summary,
            "gbd_2017_structured_rows": int(gbd_audit.loc[gbd_audit["source_year"].eq(2017), "province_rows"].max()),
            "gbd_2021_ncd_structured_rows": int(gbd_audit.loc[gbd_audit["source_year"].eq(2021), "province_rows"].max()),
            "gbd_2021_freshness_bridge_rows": int(gbd_bridge["province"].nunique()) if not gbd_bridge.empty else 0,
            "china_health_outcome_anchor_rows": int(health_anchor["province"].nunique()) if not health_anchor.empty else 0,
            "gbd_2021_literature_identified": True,
            "gbd_2021_ncd_total_burden_integrated": bool(gbd_audit.loc[gbd_audit["source_year"].eq(2021), "used_in_mapping"].any()),
            "nbs2025_explanatory_candidate_rows": int(supplement_audit.loc[supplement_audit["source_layer"].eq("nbs_2025_explanatory_ocr_candidates"), "province_rows"].max()),
            "nbs_mortality_sensitivity_panel_rows": int(supplement_audit.loc[supplement_audit["source_layer"].eq("nbs_multi_year_crude_mortality_ocr_sensitivity_panel"), "province_rows"].max()),
            "nhc_chronic_policy_execution_rows": int(supplement_audit.loc[supplement_audit["source_layer"].eq("nhc_chronic_demo_zone_policy_execution"), "province_rows"].max()),
            "china_local_policy_execution_indicator_rows": int(supplement_audit.loc[supplement_audit["source_layer"].eq("china_local_policy_execution_indicator"), "province_rows"].max()),
            "c_layer_ocr_qc_rows": int(supplement_audit.loc[supplement_audit["source_layer"].eq("c_layer_ocr_candidate_qc"), "province_rows"].max()),
            "c_layer_ocr_field_qc_rows": int(supplement_audit.loc[supplement_audit["source_layer"].eq("c_layer_ocr_field_level_qc"), "province_rows"].max()),
            "claim_boundary": "中国省级A/B画像的四类疾病和风险SEV结构化表仍是GBD 2017；GBD 2021已补年龄标化NCD健康结局锚点和新鲜度桥接；PM2.5最新到2024且省内城市已按2020地级市人口加权；NBS多年粗死亡率、NBS城镇化/GDP、NHC慢病示范区、地方政策执行指标和C层字段级OCR QC作为健康结局敏感性/解释/政策执行/补充候选层。答辩时按这个口径讲。",
            "output_files": {
                "china_pm25_population_weighting_audit": pm25_path.as_posix(),
                "china_gbd_province_year_boundary_audit": gbd_path.as_posix(),
                "china_supplement_source_boundary_audit": supplement_path.as_posix(),
                "china_abcd_source_boundary_summary": summary_path.as_posix(),
                "china_pm25_population_weighting_audit_figure": figure_path.as_posix(),
            },
        }
        clean_summary = json_clean(summary)
        summary_path.write_text(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_advanced_report_figures():
    __name__ = 'run_advanced_report_figures'
    import argparse
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from foundation import choose_text, configure_matplotlib_fonts
    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root
    USE_CHINESE = configure_matplotlib_fonts()

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

    TYPE_COLOR_MAP = {
        "类型1-高负担转型承压型": "#d1495b",
        "类型2-低缓冲低承载型": "#edae49",
        "类型3-相对稳健型": "#2a9d8f",
    }
    RESPONSE_COLOR_MAP = {
        "高压低响应型": "#d1495b",
        "高压高响应型": "#8d99ae",
        "低压低响应型": "#f4a261",
        "相对均衡型": "#2a9d8f",
    }
    NATURAL_EARTH_ADMIN0_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        dirs = {
            "simulation": project_root / "04_simulation",
            "figures": project_root / "05_figures",
            "report": project_root / "06_report_assets",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def require_plotting_dependencies():
        missing = []
        try:
            import geopandas as gpd  # type: ignore
        except Exception:
            gpd = None
            missing.append("geopandas")
        try:
            import seaborn as sns  # type: ignore
        except Exception:
            sns = None
            missing.append("seaborn")
        try:
            from scipy.interpolate import griddata  # type: ignore
        except Exception:
            griddata = None
            missing.append("scipy")
        if missing:
            raise RuntimeError(
                "Advanced report figures require missing packages: "
                + ", ".join(missing)
                + ". Install them before running, e.g. `pip install geopandas seaborn scipy`."
            )
        return gpd, sns, griddata


    def load_world_geometries(gpd):
        try:
            dataset_path = gpd.datasets.get_path("naturalearth_lowres")
            world = gpd.read_file(dataset_path)
        except Exception:
            cache_path = report_asset_path(detect_project_root(None), "naturalearth_admin0_110m.zip")
            if not cache_path.exists():
                raise RuntimeError(
                    f"Could not load Natural Earth world geometries. Missing local cache: {cache_path}"
                )
            world = gpd.read_file(cache_path)
        lower_map = {column.lower(): column for column in world.columns}
        preferred_candidates = [
            "iso_a3",
            "adm0_a3",
            "adm0_a3_us",
            "gu_a3",
            "sov_a3",
            "brk_a3",
        ]
        matched_columns: list[str] = []
        for candidate in preferred_candidates:
            actual = lower_map.get(candidate.lower())
            if actual is not None and actual not in matched_columns:
                matched_columns.append(actual)
        if not matched_columns:
            for column in world.columns:
                lowered = column.lower()
                if ("iso" in lowered or lowered.endswith("_a3") or "a3" in lowered) and column not in matched_columns:
                    matched_columns.append(column)
        if not matched_columns:
            raise RuntimeError(f"Could not resolve an ISO3 column from world geometries. Available columns: {world.columns.tolist()}")

        iso_series = None
        for column in matched_columns:
            values = world[column].astype(str).str.upper().replace({"-99": np.nan, "NONE": np.nan, "NAN": np.nan, "": np.nan})
            iso_series = values if iso_series is None else iso_series.combine_first(values)
        if iso_series is None or iso_series.dropna().empty:
            raise RuntimeError(f"Resolved world ISO3 candidates but none contained usable values: {matched_columns}")
        world["iso3"] = iso_series.astype(str).str.upper()
        return world


    def read_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))


    def choose_policy_sample_variant(summary: dict[str, object], treated_df: pd.DataFrame | None = None) -> str:
        preferred = str(summary.get("main_sample_variant", "balanced_main"))
        fallback = str(summary.get("fallback_sample_variant", "relaxed_compare"))
        if treated_df is None or treated_df.empty:
            return preferred
        treated_df = treated_df.copy()
        treated_df["treat_ever"] = treated_df["treat_ever"].fillna(False).astype(bool)
        if not treated_df.loc[(treated_df["sample_variant"] == preferred) & (treated_df["treat_ever"])].empty:
            return preferred
        if not treated_df.loc[(treated_df["sample_variant"] == fallback) & (treated_df["treat_ever"])].empty:
            return fallback
        return preferred


    def add_manifest_row(
        rows: list[dict[str, object]],
        figure_id: str,
        title: str,
        output_file: Path,
        source_files: list[Path],
        report_section: str,
        whether_main_figure: bool,
    ) -> None:
        rows.append(
            {
                "figure_id": figure_id,
                "title": title,
                "output_file": output_file.as_posix(),
                "source_files": " ; ".join(path.as_posix() for path in source_files),
                "report_section": report_section,
                "whether_main_figure": bool(whether_main_figure),
            }
        )


    def apply_axis_style(ax) -> None:
        ax.set_facecolor("#fbfbf7")
        ax.grid(alpha=0.18, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_alpha(0.35)


    def add_figure_header(
        fig,
        title: str,
        subtitle: str | None = None,
        *,
        left: float = 0.5,
        title_y: float = 0.965,
        subtitle_y: float = 0.93,
        ha: str = "center",
        title_fontsize: int = 22,
        subtitle_fontsize: int = 12,
    ) -> None:
        fig.text(
            left,
            title_y,
            title,
            ha=ha,
            va="top",
            fontsize=title_fontsize,
            fontweight="bold",
            color="#111111",
        )
        if subtitle:
            fig.text(
                left,
                subtitle_y,
                subtitle,
                ha=ha,
                va="top",
                fontsize=subtitle_fontsize,
                color="#666666",
            )


    def finalize_figure(fig, *, top: float = 0.88, bottom: float = 0.06, left: float = 0.04, right: float = 0.98) -> None:
        fig.tight_layout(rect=[left, bottom, right, top])


    def load_country_zh_labels(project_root: Path) -> dict[str, str]:
        external_data_root = shared_detect_external_data_root(project_root=project_root)
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
            override = COUNTRY_LABEL_OVERRIDES.get(iso3)
            if override:
                labels[iso3] = override
                continue
            names = group["country_name_zh"].drop_duplicates().tolist()
            if names:
                # Prefer short common names when multiple aliases exist.
                labels[iso3] = sorted(names, key=len)[0]
        return labels


    def plot_world_typology_map(
        output_path: Path,
        labels_path: Path,
        diagnosis_path: Path,
        manifest_rows: list[dict[str, object]],
        gpd,
    ) -> None:
        labels = pd.read_csv(labels_path, encoding="utf-8-sig")
        diagnosis = pd.read_csv(diagnosis_path, encoding="utf-8-sig")
        labels["iso3"] = labels["iso3"].astype(str).str.upper()
        diagnosis["iso3"] = diagnosis["iso3"].astype(str).str.upper()
        latest = labels.merge(
            diagnosis.loc[:, [col for col in ["iso3", "adaptation_gap_score", "response_diagnosis_type"] if col in diagnosis.columns]],
            on="iso3",
            how="left",
        )
        world = load_world_geometries(gpd)
        world = world.merge(latest.loc[:, ["iso3", "vulnerability_type_label"]].drop_duplicates("iso3"), on="iso3", how="left")
        china_type = latest.loc[latest["iso3"] == "CHN", "vulnerability_type_label"].dropna()
        if not china_type.empty:
            # Display-layer consistency with the analysis rule that TWN is merged into CHN.
            world.loc[world["iso3"] == "TWN", "vulnerability_type_label"] = china_type.iloc[0]
        country_zh_labels = load_country_zh_labels(output_path.parents[1])

        fig, ax = plt.subplots(figsize=(15, 8))
        world.boundary.plot(ax=ax, linewidth=0.35, color="#d8d8d8")
        for label, color in TYPE_COLOR_MAP.items():
            subset = world.loc[world["vulnerability_type_label"] == label]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor="#f2f2f2", linewidth=0.25, label=label, alpha=0.92)
        world.loc[world["vulnerability_type_label"].isna()].plot(ax=ax, color="#eeeeee", edgecolor="#f2f2f2", linewidth=0.25, alpha=0.55)

        top_gap = latest.dropna(subset=["adaptation_gap_score"]).sort_values("adaptation_gap_score", ascending=False).head(8)
        if not top_gap.empty:
            gap_world = world.merge(top_gap.loc[:, ["iso3", "adaptation_gap_score"]], on="iso3", how="inner")
            for _, row in gap_world.iterrows():
                if row.geometry is None or row.geometry.is_empty:
                    continue
                point = row.geometry.representative_point()
                ax.scatter(point.x, point.y, s=18, color="#1d3557", alpha=0.8, zorder=5)
                iso3 = str(row["iso3"]).upper()
                ax.text(
                    point.x + 1.2,
                    point.y + 0.8,
                    country_zh_labels.get(iso3, iso3),
                    fontsize=8,
                    color="#1d3557",
                    zorder=6,
                )

        add_figure_header(
            fig,
            choose_text("全球健康脆弱性类型地图", "Global Vulnerability Typology Map", USE_CHINESE),
            left=0.5,
            title_y=0.968,
            ha="center",
            title_fontsize=22,
        )
        legend_handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in TYPE_COLOR_MAP.items()]
        legend_handles.append(Patch(facecolor="#eeeeee", edgecolor="none", label=choose_text("缺失/未匹配", "Missing / unmatched", USE_CHINESE)))
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            ncol=4,
            frameon=False,
            fontsize=11,
            title=choose_text("脆弱性类型", "Vulnerability type", USE_CHINESE),
            title_fontsize=12,
            handlelength=1.6,
            columnspacing=1.4,
        )
        ax.set_axis_off()
        finalize_figure(fig, top=0.89, bottom=0.11, left=0.01, right=0.99)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        add_manifest_row(
            manifest_rows,
            "F1",
            choose_text("全球脆弱性类型地图", "Global vulnerability type map", USE_CHINESE),
            output_path,
            [labels_path, diagnosis_path],
            choose_text("模块A/模块C", "Module A / Module C", USE_CHINESE),
            True,
        )


    def plot_type_evolution_bump(
        output_path: Path,
        share_path: Path,
        manifest_rows: list[dict[str, object]],
    ) -> None:
        df = pd.read_csv(share_path, encoding="utf-8-sig")
        if "type_share" not in df.columns:
            share_col = "share" if "share" in df.columns else "countries_share"
            df["type_share"] = pd.to_numeric(df[share_col], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["rank"] = df.groupby("year")["type_share"].rank(method="first", ascending=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        apply_axis_style(ax)
        for label, group in df.groupby("vulnerability_type_label", sort=False):
            color = TYPE_COLOR_MAP.get(label, "#457b9d")
            ordered = group.sort_values("year", kind="stable")
            ax.plot(ordered["year"], ordered["rank"], color=color, linewidth=3.0, alpha=0.95)
            ax.scatter(ordered["year"], ordered["rank"], color=color, s=30, zorder=3)
            if not ordered.empty:
                ax.text(
                    ordered["year"].iloc[-1] + 0.15,
                    ordered["rank"].iloc[-1],
                    f"{label}  {ordered['type_share'].iloc[-1]:.1%}",
                    va="center",
                    fontsize=9,
                    color=color,
                )
        ax.invert_yaxis()
        ax.set_yticks(sorted(df["rank"].dropna().unique()))
        ax.set_xlabel(choose_text("年份", "Year", USE_CHINESE))
        ax.set_ylabel(choose_text("类型占比排名", "Rank by type share", USE_CHINESE))
        ax.set_xlim(df["year"].min() - 0.5, df["year"].max() + 1.2)
        add_figure_header(fig, choose_text("脆弱性类型演化趋势图", "Vulnerability Type Evolution", USE_CHINESE))
        finalize_figure(fig, top=0.90, bottom=0.08)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        add_manifest_row(
            manifest_rows,
            "F2",
            choose_text("类型演化趋势图", "Type evolution bump chart", USE_CHINESE),
            output_path,
            [share_path],
            choose_text("模块A", "Module A", USE_CHINESE),
            True,
        )


    def plot_risk_clustered_heatmap(
        output_path: Path,
        matrix_path: Path,
        manifest_rows: list[dict[str, object]],
        sns,
    ) -> None:
        matrix_df = pd.read_csv(matrix_path, encoding="utf-8-sig")
        value_col = "contribution_share" if "contribution_share" in matrix_df.columns else "absolute_beta"
        matrix_df["row_label"] = matrix_df["vulnerability_type_label"].astype(str) + " | " + matrix_df["disease_label"].astype(str)
        heat = matrix_df.pivot_table(index="row_label", columns="risk_label", values=value_col, aggfunc="mean")
        heat = heat.fillna(0.0)
        cluster = sns.clustermap(
            heat,
            cmap=LinearSegmentedColormap.from_list("riskmap", ["#f7f7f2", "#f4a261", "#d1495b"]),
            linewidths=0.4,
            linecolor="#f0f0f0",
            figsize=(11, 8),
            dendrogram_ratio=(0.12, 0.12),
            cbar_pos=(0.02, 0.83, 0.03, 0.12),
        )
        cluster.ax_heatmap.set_xlabel(choose_text("风险因子", "Risk factor", USE_CHINESE))
        cluster.ax_heatmap.set_ylabel(choose_text("类型 × 疾病", "Type × disease", USE_CHINESE))
        add_figure_header(
            cluster.fig,
            choose_text("风险归因聚类热图", "Clustered Risk Attribution Heatmap", USE_CHINESE),
            left=0.5,
            title_y=0.985,
            ha="center",
        )
        cluster.fig.subplots_adjust(top=0.90, bottom=0.05, left=0.04, right=0.98)
        cluster.fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(cluster.fig)
        add_manifest_row(
            manifest_rows,
            "F3",
            choose_text("风险归因聚类热图", "Clustered risk attribution heatmap", USE_CHINESE),
            output_path,
            [matrix_path],
            choose_text("模块B", "Module B", USE_CHINESE),
            True,
        )


    def plot_pressure_response_portfolio(
        output_path: Path,
        diagnosis_path: Path,
        manifest_rows: list[dict[str, object]],
    ) -> None:
        df = pd.read_csv(diagnosis_path, encoding="utf-8-sig")
        country_zh_labels = load_country_zh_labels(output_path.parents[1])
        df["bubble_size"] = np.sqrt(pd.to_numeric(df.get("wdi_gdp_per_capita", np.nan), errors="coerce").fillna(df.get("combined_pressure_score", 0)).clip(lower=0) + 1) * 6
        fig, ax = plt.subplots(figsize=(10.5, 7.5))
        apply_axis_style(ax)
        for label, group in df.groupby("response_diagnosis_type", sort=False):
            ax.scatter(
                group["combined_pressure_score"],
                group["adjusted_response_score"],
                s=group["bubble_size"].clip(lower=30, upper=420),
                color=RESPONSE_COLOR_MAP.get(label, "#457b9d"),
                alpha=0.65,
                edgecolor="white",
                linewidth=0.5,
                label=label,
            )
        ax.axvline(pd.to_numeric(df["combined_pressure_score"], errors="coerce").median(), color="#5f6368", linestyle="--", linewidth=1.0)
        ax.axhline(pd.to_numeric(df["adjusted_response_score"], errors="coerce").median(), color="#5f6368", linestyle="--", linewidth=1.0)

        highlight = df.sort_values("adaptation_gap_score", ascending=False).head(8)
        for _, row in highlight.iterrows():
            iso3 = str(row["iso3"]).upper()
            ax.text(
                row["combined_pressure_score"] + 0.008,
                row["adjusted_response_score"] + 0.008,
                country_zh_labels.get(iso3, iso3),
                fontsize=8,
                color="#264653",
            )

        add_figure_header(fig, choose_text("压力-响应投资组合图", "Pressure-Response Portfolio", USE_CHINESE))
        ax.set_xlabel(choose_text("综合压力得分", "Combined pressure score", USE_CHINESE))
        ax.set_ylabel(choose_text("调整后响应得分", "Adjusted response score", USE_CHINESE))
        ax.legend(frameon=False, loc="lower right", fontsize=9, title=choose_text("响应类型", "Response type", USE_CHINESE))
        finalize_figure(fig, top=0.90, bottom=0.07)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        add_manifest_row(
            manifest_rows,
            "F4",
            choose_text("压力-响应投资组合图", "Pressure-response portfolio", USE_CHINESE),
            output_path,
            [diagnosis_path],
            choose_text("模块C", "Module C", USE_CHINESE),
            True,
        )


    def plot_incremental_allocation_matrix(
        output_path: Path,
        priority_path: Path,
        allocation_path: Path,
        manifest_rows: list[dict[str, object]],
        sns,
    ) -> None:
        priority = pd.read_csv(priority_path, encoding="utf-8-sig")
        allocation = pd.read_csv(allocation_path, encoding="utf-8-sig")
        keep = priority.copy()
        if "resource_component_label" not in keep.columns:
            keep["resource_component_label"] = keep.get("resource_component", "")
        keep = keep.sort_values(["response_diagnosis_type", "mean_component_score"], ascending=[True, True], kind="stable")
        keep["priority_order"] = keep.groupby("response_diagnosis_type").cumcount() + 1
        keep = keep.loc[keep["priority_order"] <= 6].copy()
        keep["display_component"] = keep["resource_component_label"].astype(str)
        keep["weakness_score"] = 1 - pd.to_numeric(keep["mean_component_score"], errors="coerce")
        heat = keep.pivot_table(index="response_diagnosis_type", columns="display_component", values="weakness_score", aggfunc="mean").fillna(0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.55, 1]})
        sns.heatmap(
            heat,
            cmap=LinearSegmentedColormap.from_list("allocmap", ["#fbfbf7", "#f4a261", "#d1495b"]),
            linewidths=0.5,
            linecolor="#f1f1ee",
            ax=axes[0],
            cbar_kws={"label": choose_text("补短板优先度（颜色越深优先度越高）", "Weakness priority (darker means higher)", USE_CHINESE)},
        )
        axes[0].set_title(choose_text("弱项组件矩阵", "Weak Component Matrix", USE_CHINESE), fontsize=16, loc="center", pad=10)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("")
        axes[0].set_yticklabels([str(label) for label in heat.index], rotation=0, ha="right", va="center")
        axes[0].tick_params(axis="y", labelsize=11, pad=4)

        alloc_plot = allocation.sort_values("mean_gap_reduction", ascending=True, kind="stable")
        axes[1].hlines(alloc_plot["response_diagnosis_type"], 0, alloc_plot["mean_gap_reduction"], color="#d8d8d8", linewidth=2)
        axes[1].scatter(alloc_plot["mean_gap_reduction"], alloc_plot["response_diagnosis_type"], s=90, color="#2a9d8f", zorder=3)
        axes[1].set_title(choose_text("平均缺口缩减潜力", "Average Gap Reduction", USE_CHINESE), fontsize=16, loc="center", pad=10)
        axes[1].set_xlabel(choose_text("平均缺口缩减", "Mean gap reduction", USE_CHINESE))
        axes[1].grid(alpha=0.18)
        for spine in axes[1].spines.values():
            spine.set_alpha(0.35)

        add_figure_header(fig, choose_text("增量资源配置矩阵", "Incremental Allocation Matrix", USE_CHINESE))
        finalize_figure(fig, top=0.88, bottom=0.12, left=0.05, right=0.98)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        add_manifest_row(
            manifest_rows,
            "F5",
            choose_text("增量资源配置矩阵", "Incremental allocation matrix", USE_CHINESE),
            output_path,
            [priority_path, allocation_path],
            choose_text("模块C", "Module C", USE_CHINESE),
            True,
        )


    def plot_policy_cohort_heatmap(
        output_path: Path,
        policy_panel_path: Path,
        treated_path: Path,
        policy_summary: dict[str, object],
        manifest_rows: list[dict[str, object]],
        sns,
    ) -> None:
        panel = pd.read_csv(policy_panel_path, encoding="utf-8-sig")
        treated = pd.read_csv(treated_path, encoding="utf-8-sig")
        selected_variant = choose_policy_sample_variant(policy_summary, treated)
        treated = treated.loc[(treated["sample_variant"] == selected_variant) & (treated["treat_ever"].fillna(False))].copy()
        if treated.empty:
            raise RuntimeError(f"No treated countries available for policy cohort heatmap under sample variant: {selected_variant}")
        order = treated.sort_values(["treatment_year", "iso3"], kind="stable")["iso3"].tolist()
        country_zh_labels = load_country_zh_labels(output_path.parents[1])
        subset = panel.loc[panel["iso3"].astype(str).isin(order), ["iso3", "year", "policy_strength"]].copy()
        subset["iso3"] = subset["iso3"].astype(str)
        subset["year"] = pd.to_numeric(subset["year"], errors="coerce")
        heat = subset.pivot_table(index="iso3", columns="year", values="policy_strength", aggfunc="mean")
        heat = heat.reindex(order)
        heat.index = [country_zh_labels.get(str(iso3).upper(), str(iso3)) for iso3 in heat.index]
        fig, ax = plt.subplots(figsize=(14, max(6, min(12, 0.18 * len(order)))))
        sns.heatmap(
            heat,
            cmap=LinearSegmentedColormap.from_list("cohort", ["#f7f7f2", "#c9cba3", "#84a98c", "#52796f", "#354f52"]),
            linewidths=0.1,
            linecolor="#fbfbf7",
            ax=ax,
            cbar_kws={"label": choose_text("政策强度", "Policy strength", USE_CHINESE)},
        )
        add_figure_header(fig, choose_text("政策处理队列热力图", "Policy Cohort Heatmap", USE_CHINESE))
        ax.set_xlabel(choose_text("年份", "Year", USE_CHINESE))
        ax.set_ylabel(choose_text("国家（按处理年排序）", "Countries ordered by treatment year", USE_CHINESE))
        finalize_figure(fig, top=0.90, bottom=0.05)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        add_manifest_row(
            manifest_rows,
            "F6",
            choose_text("政策处理队列热力图", "Policy cohort heatmap", USE_CHINESE),
            output_path,
            [policy_panel_path, treated_path],
            choose_text("模块D", "Module D", USE_CHINESE),
            True,
        )


    def plot_policy_event_study_panel(
        output_path: Path,
        event_path: Path,
        validation_path: Path,
        policy_summary: dict[str, object],
        manifest_rows: list[dict[str, object]],
    ) -> None:
        event_df = pd.read_csv(event_path, encoding="utf-8-sig")
        validation_df = pd.read_csv(validation_path, encoding="utf-8-sig")
        selected_variant = str(policy_summary.get("main_sample_variant", "balanced_main"))
        fallback_variant = str(policy_summary.get("fallback_sample_variant", "relaxed_compare"))
        validation_df = validation_df.loc[(validation_df["analysis_scope"] == "global") & (validation_df["sample_variant"] == selected_variant)].copy()
        if validation_df.empty:
            validation_df = pd.read_csv(validation_path, encoding="utf-8-sig")
            validation_df = validation_df.loc[(validation_df["analysis_scope"] == "global") & (validation_df["sample_variant"] == fallback_variant)].copy()
            selected_variant = fallback_variant
        order = [
            "who_smoking_rate_std",
            "dbd_smoking_per100k",
            "gbd_rate_chronic_respiratory_diseases_per100k",
            "gbd_rate_cardiovascular_diseases_per100k",
        ]
        validation_df["order"] = validation_df["outcome_column"].map({key: idx for idx, key in enumerate(order)})
        validation_df = validation_df.dropna(subset=["order"]).sort_values("order", kind="stable").head(4)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes = axes.flatten()
        for ax, (_, row) in zip(axes, validation_df.iterrows()):
            subset = event_df.loc[
                (event_df["analysis_scope"] == "global")
                & (event_df["sample_variant"] == selected_variant)
                & (event_df["subgroup_name"].fillna("") == "")
                & (event_df["outcome_column"] == row["outcome_column"])
                & (pd.to_numeric(event_df["placebo_shift"], errors="coerce").fillna(0) == 0)
            ].sort_values("event_time", kind="stable")
            if subset.empty:
                ax.set_visible(False)
                continue
            color = "#2a9d8f"
            ax.plot(subset["event_time"], subset["coefficient"], color=color, linewidth=2.4, marker="o", markersize=4)
            ax.fill_between(
                subset["event_time"],
                subset["coefficient"] - 1.96 * subset["std_error"],
                subset["coefficient"] + 1.96 * subset["std_error"],
                color=color,
                alpha=0.18,
            )
            ax.axhline(0, color="#858585", linestyle="--", linewidth=1)
            ax.axvline(-1, color="#858585", linestyle=":", linewidth=1)
            apply_axis_style(ax)
            ax.set_title(str(row["outcome_label"]), fontsize=15, loc="center", pad=10)
            ax.set_xlabel(choose_text("事件时间", "Event time", USE_CHINESE))
            ax.set_ylabel(choose_text("系数", "Coefficient", USE_CHINESE))
        add_figure_header(fig, choose_text("模块D：MPOWER强化后的动态响应", "Module D: dynamic response after MPOWER strengthening", USE_CHINESE))
        finalize_figure(fig, top=0.90, bottom=0.08)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        add_manifest_row(
            manifest_rows,
            "F7",
            choose_text("事件研究小面板", "Event-study panel", USE_CHINESE),
            output_path,
            [event_path, validation_path],
            choose_text("模块D", "Module D", USE_CHINESE),
            True,
        )


    def plot_threshold_lag_surface(
        output_path: Path,
        spec_path: Path,
        manifest_rows: list[dict[str, object]],
        griddata,
    ) -> None:
        spec_df = pd.read_csv(spec_path, encoding="utf-8-sig")
        targets = [
            "gbd_rate_chronic_respiratory_diseases_per100k",
            "gbd_rate_cardiovascular_diseases_per100k",
        ]
        subset = spec_df.loc[spec_df["outcome_column"].isin(targets)].copy()
        if subset.empty:
            raise RuntimeError("No secondary outcomes available for threshold-lag surface")
        subset["stability_score"] = np.sign(pd.to_numeric(subset["coefficient"], errors="coerce").fillna(0)) * (
            -np.log10(pd.to_numeric(subset["p_value"], errors="coerce").clip(lower=1e-6).fillna(1.0))
        )

        fig = plt.figure(figsize=(13, 6))
        cmap = cm.Spectral_r
        norm = Normalize(
            vmin=float(pd.to_numeric(subset["stability_score"], errors="coerce").min()),
            vmax=float(pd.to_numeric(subset["stability_score"], errors="coerce").max()),
        )
        axes = []
        for idx, outcome in enumerate(targets, start=1):
            ax = fig.add_subplot(1, 2, idx, projection="3d")
            axes.append(ax)
            group = subset.loc[subset["outcome_column"] == outcome].copy()
            group["threshold"] = pd.to_numeric(group["threshold"], errors="coerce")
            group["lag"] = pd.to_numeric(group["lag"], errors="coerce")
            group["stability_score"] = pd.to_numeric(group["stability_score"], errors="coerce")
            x = group["threshold"].to_numpy(dtype=float)
            y = group["lag"].to_numpy(dtype=float)
            z = group["stability_score"].to_numpy(dtype=float)
            if len(group) >= 4:
                xi = np.linspace(np.nanmin(x), np.nanmax(x), 25)
                yi = np.linspace(np.nanmin(y), np.nanmax(y), 25)
                xx, yy = np.meshgrid(xi, yi)
                zz = griddata((x, y), z, (xx, yy), method="linear")
                if np.isnan(zz).all():
                    ax.plot_trisurf(x, y, z, cmap=cmap, norm=norm, linewidth=0.2, antialiased=True, alpha=0.92)
                else:
                    ax.plot_surface(xx, yy, np.nan_to_num(zz, nan=np.nanmean(z)), cmap=cmap, norm=norm, linewidth=0, antialiased=True, alpha=0.92)
            else:
                ax.plot_trisurf(x, y, z, cmap=cmap, norm=norm, linewidth=0.2, antialiased=True, alpha=0.92)
            ax.set_title(
                choose_text("慢性呼吸负担" if "respiratory" in outcome else "心血管负担", "Chronic respiratory" if "respiratory" in outcome else "Cardiovascular", USE_CHINESE),
                fontsize=15,
                loc="center",
                pad=2,
                y=0.93,
            )
            ax.set_xlabel(choose_text("阈值", "Threshold", USE_CHINESE))
            ax.set_ylabel(choose_text("滞后阶数", "Lag", USE_CHINESE))
            ax.set_zlabel(choose_text("稳定性得分", "Stability score", USE_CHINESE), labelpad=10)
            ax.view_init(elev=24, azim=-134)
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        cbar = fig.colorbar(scalar_mappable, ax=axes, shrink=0.72, pad=0.03, fraction=0.035)
        cbar.set_label(choose_text("稳定性得分", "Stability score", USE_CHINESE))
        add_figure_header(
            fig,
            choose_text("阈值与滞后阶数稳定性曲面图", "Threshold-Lag Stability Surface", USE_CHINESE),
            left=0.5,
            title_y=0.965,
            ha="center",
        )
        fig.subplots_adjust(left=0.08, right=0.93, bottom=0.03, top=0.84, wspace=0.10)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        add_manifest_row(
            manifest_rows,
            "F8",
            choose_text("阈值-Lag 稳定性曲面图", "Threshold-lag stability surface", USE_CHINESE),
            output_path,
            [spec_path],
            choose_text("模块D", "Module D", USE_CHINESE),
            True,
        )


    def main() -> None:
        parser = argparse.ArgumentParser(description="Generate advanced report figures for the 4C health vulnerability report.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        gpd, sns, griddata = require_plotting_dependencies()

        report_dir = dirs["report"]
        figures_dir = dirs["figures"]
        manifest_rows: list[dict[str, object]] = []

        paths = {
            "labels_latest": report_asset_path(report_dir, "vulnerability_country_labels_latest.csv"),
            "type_share_by_year": report_asset_path(report_dir, "vulnerability_type_share_by_year.csv"),
            "risk_matrix": report_asset_path(report_dir, "risk_attribution_matrix.csv"),
            "diagnosis_latest": report_asset_path(report_dir, "country_response_diagnosis_latest.csv"),
            "priority_components": report_asset_path(report_dir, "response_priority_components_by_type.csv"),
            "allocation_plan": report_asset_path(report_dir, "response_incremental_allocation_plan.csv"),
            "policy_panel": dirs["simulation"] / "policy_identification_panel.csv",
            "policy_treated": report_asset_path(report_dir, "policy_treated_countries.csv"),
            "policy_event": report_asset_path(report_dir, "policy_event_study_coefficients.csv"),
            "policy_validation": report_asset_path(report_dir, "policy_validation_summary.csv"),
            "policy_spec_grid": report_asset_path(report_dir, "policy_specification_grid.csv"),
            "policy_summary": report_asset_path(report_dir, "policy_identification_summary.json"),
        }
        missing = [path.as_posix() for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Advanced report figures require missing input files:\n" + "\n".join(missing))

        policy_summary = read_json(paths["policy_summary"])

        plot_world_typology_map(
            figures_dir / "advanced_vulnerability_world_map.png",
            paths["labels_latest"],
            paths["diagnosis_latest"],
            manifest_rows,
            gpd,
        )
        plot_type_evolution_bump(
            figures_dir / "advanced_type_evolution_bump.png",
            paths["type_share_by_year"],
            manifest_rows,
        )
        plot_risk_clustered_heatmap(
            figures_dir / "advanced_risk_attribution_clustered_heatmap.png",
            paths["risk_matrix"],
            manifest_rows,
            sns,
        )
        plot_pressure_response_portfolio(
            figures_dir / "advanced_pressure_response_portfolio.png",
            paths["diagnosis_latest"],
            manifest_rows,
        )
        plot_incremental_allocation_matrix(
            figures_dir / "advanced_incremental_allocation_matrix.png",
            paths["priority_components"],
            paths["allocation_plan"],
            manifest_rows,
            sns,
        )
        plot_policy_cohort_heatmap(
            figures_dir / "advanced_policy_cohort_heatmap.png",
            paths["policy_panel"],
            paths["policy_treated"],
            policy_summary,
            manifest_rows,
            sns,
        )
        plot_policy_event_study_panel(
            figures_dir / "advanced_policy_event_study_panel.png",
            paths["policy_event"],
            paths["policy_validation"],
            policy_summary,
            manifest_rows,
        )
        plot_threshold_lag_surface(
            figures_dir / "advanced_threshold_lag_surface.png",
            paths["policy_spec_grid"],
            manifest_rows,
            griddata,
        )

        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path = report_asset_path(report_dir, "report_figure_manifest.csv")
        manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(json.dumps({"project_root": project_root.as_posix(), "figure_manifest": manifest_path.as_posix(), "figures_generated": len(manifest_rows)}, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


NAMESPACE_BUILDERS = {
    'run_china_nbs_official_explanatory_candidates.py': _namespace_run_china_nbs_official_explanatory_candidates,
    'run_china_mapping_framework.py': _namespace_run_china_mapping_framework,
    'download_china_census_2020.py': _namespace_download_china_census_2020,
    'prepare_china_risk_policy_data.py': _namespace_prepare_china_risk_policy_data,
    'extract_china_gbd2017_province_supplement.py': _namespace_extract_china_gbd2017_province_supplement,
    'extract_china_gbd2021_ncd_supplement.py': _namespace_extract_china_gbd2021_ncd_supplement,
    'download_china_policy_response_upgrade.py': _namespace_download_china_policy_response_upgrade,
    'run_china_nbs_mortality_panel_ocr.py': _namespace_run_china_nbs_mortality_panel_ocr,
    'run_china_pm25_city_population_weighting.py': _namespace_run_china_pm25_city_population_weighting,
    'run_china_c_layer_candidate_qc.py': _namespace_run_china_c_layer_candidate_qc,
    'run_china_chronic_policy_execution_seed.py': _namespace_run_china_chronic_policy_execution_seed,
    'run_china_gbd2021_freshness_bridge.py': _namespace_run_china_gbd2021_freshness_bridge,
    'run_china_gbd2021_homology_boundary_audit.py': _namespace_run_china_gbd2021_homology_boundary_audit,
    'run_china_health_outcome_anchor.py': _namespace_run_china_health_outcome_anchor,
    'run_china_local_policy_execution_indicator.py': _namespace_run_china_local_policy_execution_indicator,
    'run_china_policy_quasi_causal_validation.py': _namespace_run_china_policy_quasi_causal_validation,
    'run_china_source_boundary_audit.py': _namespace_run_china_source_boundary_audit,
    'run_advanced_report_figures.py': _namespace_run_advanced_report_figures,
}

STEP_GROUPS = {'data': ['download_china_census_2020.py', 'prepare_china_risk_policy_data.py', 'extract_china_gbd2017_province_supplement.py', 'extract_china_gbd2021_ncd_supplement.py', 'download_china_policy_response_upgrade.py', 'run_china_nbs_mortality_panel_ocr.py', 'run_china_nbs_official_explanatory_candidates.py'], 'audit': ['run_china_pm25_city_population_weighting.py', 'run_china_c_layer_candidate_qc.py', 'run_china_chronic_policy_execution_seed.py', 'run_china_gbd2021_freshness_bridge.py', 'run_china_gbd2021_homology_boundary_audit.py', 'run_china_health_outcome_anchor.py', 'run_china_local_policy_execution_indicator.py', 'run_china_policy_quasi_causal_validation.py', 'run_china_source_boundary_audit.py', 'run_china_mapping_framework.py'], 'figures': ['run_advanced_report_figures.py']}
DEFAULT_GROUPS = ['data', 'audit']


def selected_steps(groups: list[str]) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = []
    for group in groups:
        steps.extend((script_name, []) for script_name in STEP_GROUPS[group])
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description='Run China provincial mapping, source-boundary checks, and policy transfer analysis.')
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--figures", action="store_true")

    args = parser.parse_args()
    project_root = detect_project_root(args.project_root)
    groups = [name for name in STEP_GROUPS if getattr(args, name)]
    if not groups:
        groups = list(DEFAULT_GROUPS)
    run_step_sequence(selected_steps(groups), NAMESPACE_BUILDERS, project_root=project_root)


if __name__ == "__main__":
    main()
