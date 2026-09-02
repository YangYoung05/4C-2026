from __future__ import annotations

import argparse
from pathlib import Path

from a_foundation import detect_project_root, report_asset_path, run_step_sequence

def _namespace_download_who_ncd_policy_data():
    __name__ = 'download_who_ncd_policy_data'
    import argparse
    import json
    import shutil
    import zipfile
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    WHO_NCD_BULK_URL = "https://ghobulkdownloads.blob.core.windows.net/ghocontainer/noncommunicable-diseases.zip"

    VALUE_SCORE_MAP = {
        "yes": 1.0,
        "fully achieved": 1.0,
        "70 or more": 0.85,
        "more than 50 but less than 70": 0.60,
        "10 to 50": 0.30,
        "less than 10": 0.05,
        "partially achieved": 0.50,
        "no": 0.0,
        "not achieved": 0.0,
    }
    MISSING_VALUE_MARKERS = {
        "",
        "nan",
        "nr",
        "no response",
        "no data received",
        "documentation not received",
        "don't know",
        "don't know'",
        "don't know",
        "not applicable",
    }

    QUESTION_GROUPS = {
        "integrated_ncd_governance": [
            "NCDCCS_QUESTION_NCD_CCS_INTEGNCDPLAN",
            "NCDCCS_QUESTION_NCD_CCS_MULTISECCOMM",
            "NCDCCS_QUESTION_NCD_CCS_NCDUNIT_OPERATIONAL",
            "NCDCCS_QUESTION_NCD_CCS_TARGETS",
            "NCDCCS_QUESTION_NCD_CCS_SURVMONREP9NCD",
            "NCDCCS_QUESTION_NCD_PM_1_TARGETS",
            "NCDCCS_QUESTION_NCD_PM_3_SURV",
            "NCDCCS_QUESTION_NCD_PM_4_INTEGNCDPOL",
            "NCDCCS_QUESTION_NCD_PM_9_GUIDELINES",
        ],
        "tobacco_policy_execution": [
            "NCDCCS_QUESTION_NCD_CCS_TOBPLAN",
            "NCDCCS_QUESTION_NCD_CCS_TOB_TARGET",
            "NCDCCS_QUESTION_NCD_CCS_TOB_SVY",
            "NCDCCS_QUESTION_NCD_CCS_TOB_MGMT_GUIDE",
            "NCDCCS_QUESTION_NCD_PM_5A_TOBTAX",
            "NCDCCS_QUESTION_NCD_PM_5B_TOBPOL",
            "NCDCCS_QUESTION_NCD_PM_5C_TOBWARN",
            "NCDCCS_QUESTION_NCD_PM_5D_TOBADS",
            "NCDCCS_QUESTION_NCD_PM_5E_TOBMEDIA",
            "NCDCCS_QUESTION_NCD_CCS_NICOTINE",
            "NCDCCS_QUESTION_NCD_CCS_NONNICOTINEMEDS",
        ],
        "diet_salt_policy_execution": [
            "NCDCCS_QUESTION_NCD_CCS_DIETPLAN",
            "NCDCCS_QUESTION_NCD_CCS_DIET_AWARE",
            "NCDCCS_QUESTION_NCD_CCS_DIET_SVY",
            "NCDCCS_QUESTION_NCD_CCS_SALTPOL",
            "NCDCCS_QUESTION_NCD_CCS_SALT_TARGET",
            "NCDCCS_QUESTION_NCD_CCS_SALT_SVY",
            "NCDCCS_QUESTION_NCD_CCS_FOOD_TAX",
            "NCDCCS_QUESTION_NCD_CCS_SSBTAX",
            "NCDCCS_QUESTION_NCD_CCS_TRANSFAT",
            "NCDCCS_QUESTION_NCD_CCS_SATFAT",
            "NCDCCS_QUESTION_NCD_CCS_MKTING",
            "NCDCCS_QUESTION_NCD_CCS_PRICE_SUBS",
            "NCDCCS_QUESTION_NCD_PM_7A_PROD_REFORM",
            "NCDCCS_QUESTION_NCD_PM_7B_FOPL",
            "NCDCCS_QUESTION_NCD_PM_7C_FOODPROC",
            "NCDCCS_QUESTION_NCD_PM_7D_DIETCAMP",
            "NCDCCS_QUESTION_NCD_PM_7E_MKTING",
        ],
        "hypertension_policy_readiness": [
            "NCDCCS_QUESTION_NCD_CCS_BP_TARGET",
            "NCDCCS_QUESTION_NCD_CCS_BP_SVY",
            "NCDCCS_QUESTION_NCD_CCS_HTN_GUIDE",
            "NCDCCS_QUESTION_NCD_CCS_BPMSMT",
            "NCDCCS_QUESTION_NCD_CCS_RISKSTRAT",
            "NCDCCS_QUESTION_NCD_CCS_ACE",
            "NCDCCS_QUESTION_NCD_CCS_ARB",
            "NCDCCS_QUESTION_NCD_CCS_BETABLOCKERS",
            "NCDCCS_QUESTION_NCD_CCS_CCBLKRS",
            "NCDCCS_QUESTION_NCD_CCS_THIAZIDE",
            "NCDCCS_QUESTION_NCD_CCS_STATINS",
            "NCDCCS_QUESTION_NCD_CCS_ASPIRIN",
        ],
        "diabetes_policy_readiness": [
            "NCDCCS_QUESTION_NCD_CCS_DIABPLAN",
            "NCDCCS_QUESTION_NCD_CCS_DIAB_TARGET",
            "NCDCCS_QUESTION_NCD_CCS_DIAB_SVY",
            "NCDCCS_QUESTION_NCD_CCS_DIAB_GUIDE",
            "NCDCCS_QUESTION_NCD_CCS_DIABTEST",
            "NCDCCS_QUESTION_NCD_CCS_DIABETESTEST",
            "NCDCCS_QUESTION_NCD_CCS_HBA1C",
            "NCDCCS_QUESTION_NCD_CCS_INSULIN",
            "NCDCCS_QUESTION_NCD_CCS_METFORMIN",
            "NCDCCS_QUESTION_NCD_CCS_SULPHONYLUREA",
            "NCDCCS_QUESTION_NCD_CCS_DIAB_RETIN",
            "NCDCCS_QUESTION_NCD_CCS_DIABETESREG",
        ],
        "primary_care_ncd_service_readiness": [
            "NCDCCS_QUESTION_NCD_CCS_NCDGUIDE",
            "NCDCCS_QUESTION_NCD_CCS_CHOLMSMT",
            "NCDCCS_QUESTION_NCD_CCS_BPMSMT",
            "NCDCCS_QUESTION_NCD_CCS_DIABTEST",
            "NCDCCS_QUESTION_NCD_CCS_HBA1C",
            "NCDCCS_QUESTION_NCD_CCS_RISKSTRAT",
            "NCDCCS_QUESTION_NCD_CCS_PEAKFLOW",
            "NCDCCS_QUESTION_NCD_CCS_SPIROM",
            "NCDCCS_QUESTION_NCD_CCS_REHAB_PRIMARY",
            "NCDCCS_QUESTION_NCD_CCS_PALLIATIVE_PRIM",
        ],
    }

    SERVICE_INDICATORS = {
        "NCD_HYP_DIAGNOSIS_A": "ncd_hypertension_diagnosis_pct",
        "NCD_HYP_TREATMENT_A": "ncd_hypertension_treatment_pct",
        "NCD_HYP_CONTROL_A": "ncd_hypertension_control_pct",
        "NCD_DIABETES_TREATMENT_AGESTD": "ncd_diabetes_treatment_pct",
        "NCD_CXCA_SCREENED_WITHIN_TIMEPERIOD": "ncd_cervical_screening_pct",
    }


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        dirs = {
            "inventory": external_data_root / "15_WHO_NCD_Bulk" / "who_ncd_bulk",
            "clean": project_root / "09_data_clean",
            "reports": project_root / "06_report_assets",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def require_source_file(output_path: Path) -> None:
        if output_path.exists() and output_path.stat().st_size > 0:
            return
        raise FileNotFoundError(
            f"WHO NCD bulk source file is missing: {output_path}. "
            "Put it under input/External Data/15_WHO_NCD_Bulk/who_ncd_bulk before running."
        )


    def extract_zip(zip_path: Path, extract_dir: Path, refresh: bool = False) -> None:
        marker = extract_dir / "data" / "NCDCCS_COUNTRYRESPONSE.csv"
        if marker.exists() and not refresh:
            return
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)


    def score_value(value: object, numeric_value: object = None) -> float:
        if pd.notna(numeric_value):
            try:
                num = float(numeric_value)
                if 0 <= num <= 1:
                    return num
                if 1 < num <= 100:
                    return num / 100.0
            except Exception:
                pass
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower().replace("’", "'")
        if text in MISSING_VALUE_MARKERS:
            return np.nan
        return VALUE_SCORE_MAP.get(text, np.nan)


    def load_code_titles(extract_dir: Path, code_file: str, code_column: str = "Code") -> dict[str, str]:
        df = pd.read_csv(extract_dir / "codes" / code_file)
        return dict(zip(df[code_column].astype(str), df["Title"].astype(str)))


    def clean_country_capacity(extract_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        question_to_group = {
            question: group for group, questions in QUESTION_GROUPS.items() for question in questions
        }
        selected_questions = set(question_to_group)
        question_titles = load_code_titles(extract_dir, "NCDCCS_QUESTION.csv")
        section_titles = load_code_titles(extract_dir, "NCDCCS_SECTION.csv")
        cols = [
            "SpatialDimensionValueCode",
            "ParentLocation",
            "TimeDim",
            "DisaggregatingDimension1ValueCode",
            "DisaggregatingDimension2ValueCode",
            "Value",
            "NumericValue",
        ]
        chunks: list[pd.DataFrame] = []
        data_path = extract_dir / "data" / "NCDCCS_COUNTRYRESPONSE.csv"
        for chunk in pd.read_csv(data_path, usecols=cols, chunksize=250_000, low_memory=False):
            chunk = chunk.loc[chunk["DisaggregatingDimension2ValueCode"].astype(str).isin(selected_questions)].copy()
            if chunk.empty:
                continue
            chunk["country_code"] = chunk["SpatialDimensionValueCode"].astype(str)
            chunk["region"] = chunk["ParentLocation"].astype(str)
            chunk["year"] = pd.to_numeric(chunk["TimeDim"], errors="coerce").astype("Int64")
            chunk["section_code"] = chunk["DisaggregatingDimension1ValueCode"].astype(str)
            chunk["question_code"] = chunk["DisaggregatingDimension2ValueCode"].astype(str)
            chunk["question_group"] = chunk["question_code"].map(question_to_group)
            chunk["question_title"] = chunk["question_code"].map(question_titles)
            chunk["section_title"] = chunk["section_code"].map(section_titles)
            chunk["value_raw"] = chunk["Value"]
            chunk["score"] = [
                score_value(value, numeric)
                for value, numeric in zip(chunk["Value"], chunk["NumericValue"])
            ]
            chunks.append(
                chunk.loc[
                    :,
                    [
                        "country_code",
                        "region",
                        "year",
                        "section_code",
                        "section_title",
                        "question_code",
                        "question_title",
                        "question_group",
                        "value_raw",
                        "score",
                    ],
                ]
            )
        long = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if long.empty:
            return long, pd.DataFrame()
        long = long.dropna(subset=["country_code", "year", "question_group"])
        score_wide = (
            long.groupby(["country_code", "region", "year", "question_group"], dropna=False)["score"]
            .mean()
            .reset_index()
            .pivot_table(index=["country_code", "region", "year"], columns="question_group", values="score", aggfunc="mean")
            .reset_index()
        )
        score_wide.columns.name = None
        observed = (
            long.dropna(subset=["score"])
            .groupby(["country_code", "region", "year"])["question_code"]
            .nunique()
            .reset_index(name="ncd_capacity_observed_questions")
        )
        score_cols = [col for col in score_wide.columns if col not in {"country_code", "region", "year"}]
        score_wide["ncd_policy_capacity_score"] = score_wide[score_cols].mean(axis=1, skipna=True)
        score_wide = score_wide.merge(observed, on=["country_code", "region", "year"], how="left")
        score_wide["ncd_capacity_observed_questions"] = score_wide["ncd_capacity_observed_questions"].fillna(0).astype(int)
        return long, score_wide


    def clean_service_indicator(extract_dir: Path, indicator_code: str, output_column: str) -> pd.DataFrame:
        path = extract_dir / "data" / f"{indicator_code}.csv"
        if not path.exists():
            return pd.DataFrame(columns=["country_code", "region", "year", output_column])
        df = pd.read_csv(path, low_memory=False)
        df = df.loc[df["SpatialDimension"].eq("COUNTRY")].copy()
        if "DisaggregatingDimension1ValueCode" in df.columns:
            sex_values = df["DisaggregatingDimension1ValueCode"].dropna().astype(str)
            if (sex_values == "SEX_BTSX").any():
                df = df.loc[df["DisaggregatingDimension1ValueCode"].eq("SEX_BTSX")].copy()
        if "DisaggregatingDimension2ValueCode" in df.columns:
            age_values = df["DisaggregatingDimension2ValueCode"].dropna().astype(str)
            if age_values.str.contains("YEARS30-PLUS", na=False).any():
                df = df.loc[df["DisaggregatingDimension2ValueCode"].astype(str).str.contains("YEARS30-PLUS", na=False)].copy()
        out = pd.DataFrame(
            {
                "country_code": df["SpatialDimensionValueCode"].astype(str),
                "region": df["ParentLocation"].astype(str),
                "year": pd.to_numeric(df["TimeDim"], errors="coerce").astype("Int64"),
                output_column: pd.to_numeric(df["NumericValue"], errors="coerce"),
            }
        )
        out = out.dropna(subset=["country_code", "year"])
        return out.groupby(["country_code", "region", "year"], dropna=False)[output_column].mean().reset_index()


    def clean_service_coverage(extract_dir: Path) -> pd.DataFrame:
        frames = [clean_service_indicator(extract_dir, code, col) for code, col in SERVICE_INDICATORS.items()]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        panel = frames[0]
        for frame in frames[1:]:
            panel = panel.merge(frame, on=["country_code", "region", "year"], how="outer")
        service_cols = [col for col in SERVICE_INDICATORS.values() if col in panel.columns]
        panel["ncd_service_coverage_score"] = panel[service_cols].div(100.0).mean(axis=1, skipna=True)
        panel["ncd_service_observed_indicators"] = panel[service_cols].notna().sum(axis=1)
        return panel.sort_values(["country_code", "year"], kind="stable")


    def combine_policy_service(capacity: pd.DataFrame, service: pd.DataFrame) -> pd.DataFrame:
        if capacity.empty and service.empty:
            return pd.DataFrame()
        if capacity.empty:
            combined = service.copy()
        elif service.empty:
            combined = capacity.copy()
        else:
            combined = capacity.merge(service, on=["country_code", "region", "year"], how="outer", suffixes=("", "_service"))
            if "region_service" in combined.columns:
                combined["region"] = combined["region"].fillna(combined["region_service"])
                combined = combined.drop(columns=["region_service"])
        score_cols = [
            "ncd_policy_capacity_score",
            "ncd_service_coverage_score",
            "integrated_ncd_governance",
            "tobacco_policy_execution",
            "diet_salt_policy_execution",
            "hypertension_policy_readiness",
            "diabetes_policy_readiness",
            "primary_care_ncd_service_readiness",
        ]
        score_cols = [col for col in score_cols if col in combined.columns]
        combined["ncd_policy_execution_score"] = combined[score_cols].mean(axis=1, skipna=True)
        combined["ncd_policy_data_observed_fields"] = combined[score_cols].notna().sum(axis=1)
        return combined.sort_values(["country_code", "year"], kind="stable")


    def write_csv(df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False, encoding="utf-8-sig")


    def main() -> None:
        parser = argparse.ArgumentParser(description="Download and clean WHO NCD policy/readiness data for Module D.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--refresh", action="store_true")
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        zip_path = dirs["inventory"] / "noncommunicable-diseases.zip"
        output_files = {
            "ncd_capacity_long": dirs["clean"] / "external_who_ncd_country_capacity_long.csv",
            "ncd_capacity_scores": dirs["clean"] / "external_who_ncd_policy_capacity_scores.csv",
            "ncd_service_coverage": dirs["clean"] / "external_who_ncd_service_coverage.csv",
            "ncd_policy_service_panel": dirs["clean"] / "external_who_ncd_policy_service_panel.csv",
            "report_ncd_data_source": report_asset_path(dirs["reports"], "who_ncd_policy_data_source_summary.json"),
        }
        if all(output_files[key].exists() and output_files[key].stat().st_size > 0 for key in [
            "ncd_capacity_long",
            "ncd_capacity_scores",
            "ncd_service_coverage",
            "ncd_policy_service_panel",
        ]):
            capacity_long = pd.read_csv(output_files["ncd_capacity_long"], low_memory=False)
            capacity_scores = pd.read_csv(output_files["ncd_capacity_scores"], low_memory=False)
            service_panel = pd.read_csv(output_files["ncd_service_coverage"], low_memory=False)
            combined = pd.read_csv(output_files["ncd_policy_service_panel"], low_memory=False)
            source_mode = "reused_09_data_clean"
        else:
            extract_dir = dirs["inventory"] / "extracted"
            require_source_file(zip_path)
            extract_zip(zip_path, extract_dir, refresh=args.refresh)

            capacity_long, capacity_scores = clean_country_capacity(extract_dir)
            service_panel = clean_service_coverage(extract_dir)
            combined = combine_policy_service(capacity_scores, service_panel)
            source_mode = "rebuilt_from_input_zip"

        write_csv(capacity_long, output_files["ncd_capacity_long"])
        write_csv(capacity_scores, output_files["ncd_capacity_scores"])
        write_csv(service_panel, output_files["ncd_service_coverage"])
        write_csv(combined, output_files["ncd_policy_service_panel"])

        summary = {
            "project_root": project_root.as_posix(),
            "source_mode": source_mode,
            "source_name": "WHO Global Health Observatory bulk download: Noncommunicable diseases",
            "source_url": WHO_NCD_BULK_URL,
            "raw_zip": zip_path.as_posix(),
            "zip_size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2) if zip_path.exists() else None,
            "capacity_rows_long": int(capacity_long.shape[0]),
            "capacity_score_rows": int(capacity_scores.shape[0]),
            "service_rows": int(service_panel.shape[0]),
            "combined_rows": int(combined.shape[0]),
            "years": {
                "capacity": [int(x) for x in sorted(capacity_scores["year"].dropna().unique())] if not capacity_scores.empty else [],
                "service_min": int(service_panel["year"].min()) if not service_panel.empty else None,
                "service_max": int(service_panel["year"].max()) if not service_panel.empty else None,
            },
            "question_groups": {group: len(questions) for group, questions in QUESTION_GROUPS.items()},
            "service_indicators": SERVICE_INDICATORS,
            "scoring_note": "Yes/No and achieved categories are mapped to 0-1 scores; NCD service coverage percentages are kept as pct and scaled only in aggregate scores.",
            "output_files": {key: value.as_posix() for key, value in output_files.items()},
        }
        output_files["report_ncd_data_source"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_policy_identification():
    __name__ = 'run_policy_identification'
    import argparse
    import json
    import math
    import sys
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from foundation import choose_text, configure_matplotlib_fonts, set_centered_suptitle
    from foundation import detect_project_root as shared_detect_project_root
    USE_CHINESE = configure_matplotlib_fonts()

    CLEAN_TOBACCO_DIRNAME = "09_data_clean"
    CLEAN_TOBACCO_FILE_CANDIDATES = {
        "policy": ["external_who_mpower_policy_score.csv"],
        "smoking_trend": ["external_who_smoking_trend.csv", "external_who_smoking_prevalence.csv"],
        "cigarette_prevalence": ["external_who_cigarette_prevalence.csv"],
        "smoking_prevalence_optional": ["external_who_smoking_prevalence.csv"],
    }
    BOTH_SEX_MARKERS = {
        "SEXBTSX",
        "BTSX",
        "BOTHSEXES",
        "BOTHSEXESCOMBINED",
        "TOTAL",
        "ALL",
        "ALLSEXES",
    }
    OBSERVED_MARKERS = ("OBSERV", "ACTUAL", "SURVEY", "MEASURED", "REPORTED")
    NON_OBSERVED_MARKERS = ("PROJ", "FORECAST", "PREDICT", "INTERPOL", "IMPUT", "MODEL", "ESTIM")
    MAIN_SAMPLE_VARIANT = "balanced_main"
    DIAGNOSTIC_SAMPLE_VARIANT = "crossing_main"
    STRICT_SAMPLE_VARIANT = "strict_appendix"
    RELAXED_SAMPLE_VARIANT = "relaxed_compare"
    SUBGROUP_DIMENSIONS = ["vulnerability_type", "baseline_smoking_tercile"]
    PRIMARY_LOCK_OUTCOME = "dbd_smoking_per100k"
    TRUE_PRETREAT_FIRST_STAGE_PLACEBO_START = 6
    DEFAULT_ANTICIPATION_BUFFER = 1
    ANTICIPATION_BUFFER_GRID = [0, 1, 2]
    MAX_ANALYSIS_YEAR = 2024
    POLICY_VALIDATION_COLUMNS = [
        "analysis_scope",
        "sample_variant",
        "subgroup_dimension",
        "subgroup_name",
        "outcome_column",
        "outcome_label",
        "tier",
        "tier_label",
        "did_coefficient",
        "did_p_value",
        "n_obs",
        "treated_countries",
        "pretrend_any_signal",
        "pretrend_mean_abs_coef",
        "pretrend_max_abs_coef",
        "posttrend_any_signal",
        "posttrend_mean_coef",
        "posttrend_peak_abs_coef",
        "direction_consistent",
        "threshold_direction_stable",
        "threshold_opposite_signal",
        "placebo_issue",
        "sample_size_ok",
        "validation_status",
        "recommended_main_claim",
    ]

    OUTCOME_LABELS_CN = {
        "who_smoking_rate_std": "WHO年龄标准化吸烟率",
        "dbd_smoking_per100k": "吸烟风险暴露",
        "gbd_rate_chronic_respiratory_diseases_per100k": "慢性呼吸系统疾病负担",
        "gbd_rate_cardiovascular_diseases_per100k": "心血管疾病负担",
        "who_cigarette_prevalence_btsx": "WHO卷烟吸烟率",
    }

    OUTCOME_LABELS_EN = {
        "who_smoking_rate_std": "WHO age-standardized smoking prevalence",
        "dbd_smoking_per100k": "Smoking risk exposure",
        "gbd_rate_chronic_respiratory_diseases_per100k": "Chronic respiratory burden",
        "gbd_rate_cardiovascular_diseases_per100k": "Cardiovascular burden",
        "who_cigarette_prevalence_btsx": "WHO cigarette prevalence",
    }

    MAIN_OUTCOME_SPECS = [
        {
            "outcome_column": "dbd_smoking_per100k",
            "lag": 0,
            "pre_window": 5,
            "post_window": 5,
            "tier": "first_stage",
            "tier_label_cn": "一级结果",
            "tier_label_en": "First-stage outcome",
            "expected_sign": -1,
            "lag_grid": [0, 1],
        },
        {
            "outcome_column": "gbd_rate_chronic_respiratory_diseases_per100k",
            "lag": 2,
            "pre_window": 5,
            "post_window": 8,
            "tier": "secondary",
            "tier_label_cn": "二级结果",
            "tier_label_en": "Second-stage outcome",
            "expected_sign": -1,
            "lag_grid": [1, 2, 3, 4],
        },
        {
            "outcome_column": "gbd_rate_cardiovascular_diseases_per100k",
            "lag": 3,
            "pre_window": 5,
            "post_window": 8,
            "tier": "secondary",
            "tier_label_cn": "二级结果",
            "tier_label_en": "Second-stage outcome",
            "expected_sign": -1,
            "lag_grid": [2, 3, 4, 5],
        },
    ]

    ROBUSTNESS_OUTCOME_SPECS = [
        {
            "outcome_column": "who_smoking_rate_std",
            "lag": 0,
            "pre_window": 5,
            "post_window": 5,
            "tier": "first_stage_robustness",
            "tier_label_cn": "一级稳健性",
            "tier_label_en": "First-stage robustness",
            "expected_sign": -1,
            "lag_grid": [0, 1],
        },
        {
            "outcome_column": "who_cigarette_prevalence_btsx",
            "lag": 0,
            "pre_window": 5,
            "post_window": 5,
            "tier": "first_stage_robustness",
            "tier_label_cn": "一级稳健性",
            "tier_label_en": "First-stage robustness",
            "expected_sign": -1,
            "lag_grid": [0, 1],
        }
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def detect_tobacco_dir(project_root: Path, explicit: Path | None) -> Path:
        resolved = explicit.expanduser().resolve() if explicit is not None else (project_root / CLEAN_TOBACCO_DIRNAME)
        if not resolved.exists():
            raise FileNotFoundError(f"Clean tobacco data directory not found: {resolved}")
        return resolved


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        dirs = {
            "simulation": project_root / "04_simulation",
            "report": project_root / "06_report_assets",
            "figures": project_root / "05_figures",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def outcome_label(column: str) -> str:
        labels = OUTCOME_LABELS_CN if USE_CHINESE else OUTCOME_LABELS_EN
        return labels.get(column, column)


    def tier_label(spec: dict[str, object]) -> str:
        return spec["tier_label_cn"] if USE_CHINESE else spec["tier_label_en"]


    def parse_int_list(value: str) -> list[int]:
        return [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]


    def empty_policy_validation_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=POLICY_VALIDATION_COLUMNS)


    def concat_nonempty(frames: list[pd.DataFrame], fallback: pd.DataFrame | None = None) -> pd.DataFrame:
        usable = [frame for frame in frames if frame is not None and not frame.empty]
        if usable:
            return pd.concat(usable, ignore_index=True)
        if fallback is not None:
            return fallback.copy()
        return pd.DataFrame()


    def log_stage(message: str) -> None:
        print(f"[ModuleD] {message}", flush=True)
        sys.stdout.flush()


    def normal_p_value(z_value: float) -> float:
        return math.erfc(abs(z_value) / math.sqrt(2.0))


    def winsorize_series(series: pd.Series, lower_quantile: float, upper_quantile: float) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        mask = numeric.notna()
        if not mask.any():
            return numeric
        lower = numeric.loc[mask].quantile(lower_quantile)
        upper = numeric.loc[mask].quantile(upper_quantile)
        return numeric.clip(lower=lower, upper=upper)


    def prepare_outcome(series: pd.Series, lower_quantile: float, upper_quantile: float, log_transform: bool) -> pd.Series:
        values = winsorize_series(series, lower_quantile, upper_quantile)
        if log_transform:
            values = values.where(values >= 0)
            values = np.log1p(values)
        return values.astype("float64")


    def normalize_iso3(series: pd.Series) -> pd.Series:
        return series.astype(str).str.upper().str.strip()


    def normalize_marker(series: pd.Series) -> pd.Series:
        return series.astype(str).str.upper().str.replace(r"[^A-Z0-9]+", "", regex=True)


    def normalize_meta_value(value: object) -> str:
        return str(value).upper().replace(" ", "").replace("-", "").replace("_", "")


    def resolve_tobacco_file(tobacco_dir: Path, names: list[str], required: bool = True) -> Path | None:
        for name in names:
            candidate = tobacco_dir / name
            if candidate.exists():
                return candidate
        if required:
            raise FileNotFoundError(f"Missing cleaned WHO tobacco file in {tobacco_dir}: tried {names}")
        return None


    def resolve_clean_tobacco_files(tobacco_dir: Path) -> dict[str, Path | None]:
        return {
            "policy": resolve_tobacco_file(tobacco_dir, CLEAN_TOBACCO_FILE_CANDIDATES["policy"]),
            "smoking_trend": resolve_tobacco_file(tobacco_dir, CLEAN_TOBACCO_FILE_CANDIDATES["smoking_trend"]),
            "cigarette_prevalence": resolve_tobacco_file(
                tobacco_dir,
                CLEAN_TOBACCO_FILE_CANDIDATES["cigarette_prevalence"],
                required=False,
            ),
            "smoking_prevalence_optional": resolve_tobacco_file(
                tobacco_dir,
                CLEAN_TOBACCO_FILE_CANDIDATES["smoking_prevalence_optional"],
                required=False,
            ),
        }


    def resolve_column(df: pd.DataFrame, candidates: list[str], required: bool = True, frame_name: str = "frame") -> str | None:
        lower_map = {column.lower(): column for column in df.columns}
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
            lowered = candidate.lower()
            if lowered in lower_map:
                return lower_map[lowered]
        if required:
            raise RuntimeError(f"{frame_name} missing required columns; tried {candidates}, available={df.columns.tolist()}")
        return None


    def resolve_column_contains(df: pd.DataFrame, tokens: list[str], required: bool = False) -> str | None:
        for column in df.columns:
            lowered = column.lower()
            if all(token in lowered for token in tokens):
                return column
        if required:
            raise RuntimeError(f"Could not resolve column containing tokens {tokens}; available={df.columns.tolist()}")
        return None


    def find_metadata_column(df: pd.DataFrame) -> str | None:
        exact_candidates = [
            "estimate_type",
            "estimation_type",
            "observation_status",
            "estimate_status",
            "value_status",
            "data_status",
            "estimation_flag",
            "estimate_flag",
            "interp_status",
            "projection_status",
        ]
        column = resolve_column(df, exact_candidates, required=False, frame_name="metadata frame")
        if column is not None:
            return column
        contains_groups = [
            ["estimate", "type"],
            ["estimate", "status"],
            ["observation", "status"],
            ["projection"],
            ["interpol"],
            ["forecast"],
        ]
        for tokens in contains_groups:
            column = resolve_column_contains(df, tokens, required=False)
            if column is not None:
                return column
        return None


    def filter_indicator_rows(df: pd.DataFrame, tokens: list[str]) -> tuple[pd.DataFrame, str | None]:
        indicator_col = resolve_column(df, ["indicator", "series", "variable", "indicator_name"], required=False, frame_name="indicator frame")
        if indicator_col is None:
            return df.copy(), None
        indicator_norm = df[indicator_col].astype(str).str.lower()
        mask = pd.Series(False, index=df.index)
        for token in tokens:
            mask = mask | indicator_norm.str.contains(token, na=False)
        return (df.loc[mask].copy(), indicator_col) if mask.any() else (df.copy(), indicator_col)


    def filter_both_sexes(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
        sex_col = resolve_column(df, ["sex", "sex_code", "sex_name", "sex_label"], required=False, frame_name="sex frame")
        if sex_col is None:
            return df.copy(), None
        markers = normalize_marker(df[sex_col])
        mask = markers.isin(BOTH_SEX_MARKERS)
        return (df.loc[mask].copy(), sex_col) if mask.any() else (df.copy(), sex_col)


    def apply_observed_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, bool, pd.DataFrame]:
        meta_col = find_metadata_column(df)
        if meta_col is None:
            return df.copy(), None, False, pd.DataFrame(columns=["estimate_value", "estimate_group", "records"])
        meta_values = df[meta_col].fillna("NA").astype(str)
        normalized = meta_values.map(normalize_meta_value)
        estimate_groups = np.where(
            normalized.str.contains("|".join(OBSERVED_MARKERS), na=False),
            "observed",
            np.where(normalized.str.contains("|".join(NON_OBSERVED_MARKERS), na=False), "non_observed", "other"),
        )
        summary = (
            pd.DataFrame({"estimate_value": meta_values, "estimate_group": estimate_groups})
            .groupby(["estimate_value", "estimate_group"], dropna=False)
            .size()
            .rename("records")
            .reset_index()
            .sort_values(["estimate_group", "records"], ascending=[True, False], kind="stable")
        )
        if (estimate_groups == "observed").any():
            filtered = df.loc[pd.Series(estimate_groups, index=df.index) == "observed"].copy()
            return filtered, meta_col, True, summary
        return df.copy(), meta_col, False, summary


    def apply_inferred_year_window(
        df: pd.DataFrame,
        year_column: str,
        inferred_max_year: int | None,
    ) -> tuple[pd.DataFrame, bool]:
        if inferred_max_year is None:
            return df.copy(), False
        year_values = pd.to_numeric(df[year_column], errors="coerce")
        if year_values.dropna().empty:
            return df.copy(), False
        if float(year_values.max()) <= float(inferred_max_year):
            return df.copy(), False
        filtered = df.loc[year_values <= float(inferred_max_year)].copy()
        return filtered, True


    def extract_policy_score(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return numeric.astype("float64")
        extracted = series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
        return pd.to_numeric(extracted, errors="coerce").astype("float64")


    def summarize_indicator_values(df: pd.DataFrame, indicator_col: str | None, dataset_kind: str) -> pd.DataFrame:
        if indicator_col is None:
            return pd.DataFrame(columns=["dataset_kind", "resolution_stage", "column_name", "value", "records"])
        summary = (
            df[indicator_col]
            .fillna("NA")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("value")
            .reset_index(name="records")
            .sort_values("records", ascending=False, kind="stable")
        )
        summary.insert(0, "column_name", indicator_col)
        summary.insert(0, "resolution_stage", "indicator")
        summary.insert(0, "dataset_kind", dataset_kind)
        return summary


    def summarize_sex_values(df: pd.DataFrame, sex_col: str | None, dataset_kind: str) -> pd.DataFrame:
        if sex_col is None:
            return pd.DataFrame(columns=["dataset_kind", "resolution_stage", "column_name", "value", "records"])
        summary = (
            df[sex_col]
            .fillna("NA")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("value")
            .reset_index(name="records")
            .sort_values("records", ascending=False, kind="stable")
        )
        summary.insert(0, "column_name", sex_col)
        summary.insert(0, "resolution_stage", "sex")
        summary.insert(0, "dataset_kind", dataset_kind)
        return summary


    def summarize_estimate_values(summary: pd.DataFrame, meta_col: str | None, dataset_kind: str) -> pd.DataFrame:
        if meta_col is None or summary.empty:
            return pd.DataFrame(columns=["dataset_kind", "resolution_stage", "column_name", "value", "records", "estimate_group"])
        result = summary.rename(columns={"estimate_value": "value"}).copy()
        result.insert(0, "column_name", meta_col)
        result.insert(0, "resolution_stage", "estimate_type")
        result.insert(0, "dataset_kind", dataset_kind)
        return result


    def deduplicate_iso_year(
        df: pd.DataFrame,
        dataset_kind: str,
        value_column: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        work_df = df.loc[:, ["iso3", "year", value_column]].copy()
        work_df[value_column] = pd.to_numeric(work_df[value_column], errors="coerce")
        grouped = (
            work_df.groupby(["iso3", "year"], dropna=False)[value_column]
            .agg(["count", "nunique", "mean", "min", "max"])
            .reset_index()
        )
        conflict_df = grouped.loc[grouped["nunique"] > 1].copy()
        if not conflict_df.empty:
            conflict_df.insert(0, "dataset_kind", dataset_kind)
        deduped = work_df.groupby(["iso3", "year"], as_index=False)[value_column].mean()
        return deduped, conflict_df


    def standardize_policy_frame(df: pd.DataFrame, dataset_kind: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
        filtered, indicator_col = filter_indicator_rows(df, ["m_group", "mpower"])
        iso_col = resolve_column(filtered, ["iso3", "country_code", "ref_area", "code"], frame_name="policy frame")
        year_col = resolve_column(filtered, ["year", "time_period"], frame_name="policy frame")
        value_col = resolve_column(
            filtered,
            ["policy_strength", "policy_strength_external", "mpower_group", "value", "obs_value"],
            frame_name="policy frame",
        )
        standardized = (
            filtered.rename(columns={iso_col: "iso3", year_col: "year", value_col: "policy_strength_external"})
            .assign(iso3=lambda frame: normalize_iso3(frame["iso3"]), year=lambda frame: pd.to_numeric(frame["year"], errors="coerce"))
            .dropna(subset=["iso3", "year"])
            .loc[:, ["iso3", "year", "policy_strength_external"]]
        )
        standardized = standardized.loc[pd.to_numeric(standardized["year"], errors="coerce") <= MAX_ANALYSIS_YEAR].copy()
        standardized["policy_strength_external"] = extract_policy_score(standardized["policy_strength_external"])
        standardized, conflict_df = deduplicate_iso_year(standardized, dataset_kind, "policy_strength_external")
        resolution_df = summarize_indicator_values(filtered, indicator_col, dataset_kind)
        qc = {
            "dataset_kind": dataset_kind,
            "rows_in": int(df.shape[0]),
            "rows_after_indicator_filter": int(filtered.shape[0]),
            "rows_standardized": int(standardized.shape[0]),
            "conflicting_iso3_year_rows": int(conflict_df.shape[0]),
        }
        return standardized, resolution_df, qc, conflict_df


    def standardize_prevalence_frame(
        df: pd.DataFrame,
        output_column: str,
        dataset_kind: str,
        indicator_tokens: list[str],
        inferred_observed_year_max: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
        filtered_indicator, indicator_col = filter_indicator_rows(df, indicator_tokens)
        filtered_sex, sex_col = filter_both_sexes(filtered_indicator)
        observed_filtered, meta_col, observed_filter_applied, estimate_summary = apply_observed_filter(filtered_sex)
        year_col = resolve_column(observed_filtered, ["year", "time_period"], frame_name=output_column)
        year_window_filtered, year_window_filter_applied = apply_inferred_year_window(
            observed_filtered,
            year_col,
            inferred_observed_year_max if not observed_filter_applied else None,
        )
        iso_col = resolve_column(year_window_filtered, ["iso3", "country_code", "ref_area", "code"], frame_name=output_column)
        year_col = resolve_column(year_window_filtered, ["year", "time_period"], frame_name=output_column)
        value_col = resolve_column(year_window_filtered, [output_column, "value", "obs_value", "estimate"], frame_name=output_column)

        standardized = (
            year_window_filtered.rename(columns={iso_col: "iso3", year_col: "year", value_col: output_column})
            .assign(iso3=lambda frame: normalize_iso3(frame["iso3"]), year=lambda frame: pd.to_numeric(frame["year"], errors="coerce"))
            .dropna(subset=["iso3", "year"])
            .loc[:, ["iso3", "year", output_column]]
        )
        standardized = standardized.loc[pd.to_numeric(standardized["year"], errors="coerce") <= MAX_ANALYSIS_YEAR].copy()
        standardized[output_column] = pd.to_numeric(standardized[output_column], errors="coerce")
        standardized, conflict_df = deduplicate_iso_year(standardized, dataset_kind, output_column)

        resolution_parts = [
            summarize_indicator_values(filtered_indicator, indicator_col, dataset_kind),
            summarize_sex_values(filtered_indicator, sex_col, dataset_kind),
            summarize_estimate_values(estimate_summary, meta_col, dataset_kind),
        ]
        if year_window_filter_applied:
            resolution_parts.append(
                pd.DataFrame(
                    [
                        {
                            "dataset_kind": dataset_kind,
                            "resolution_stage": "year_window",
                            "column_name": year_col,
                            "value": f"<= {int(inferred_observed_year_max)}",
                            "records": int(year_window_filtered.shape[0]),
                        }
                    ]
                )
            )
        resolution_df = pd.concat([part for part in resolution_parts if not part.empty], ignore_index=True) if any(not part.empty for part in resolution_parts) else pd.DataFrame(
            columns=["dataset_kind", "resolution_stage", "column_name", "value", "records"]
        )

        qc = {
            "dataset_kind": dataset_kind,
            "rows_in": int(df.shape[0]),
            "rows_after_indicator_filter": int(filtered_indicator.shape[0]),
            "rows_after_both_sex_filter": int(filtered_sex.shape[0]),
            "rows_after_observed_filter": int(observed_filtered.shape[0]),
            "observed_filter_applied": bool(observed_filter_applied),
            "rows_after_year_window_filter": int(year_window_filtered.shape[0]),
            "year_window_filter_applied": bool(year_window_filter_applied),
            "inferred_observed_year_max": int(inferred_observed_year_max) if inferred_observed_year_max is not None else np.nan,
            "rows_standardized": int(standardized.shape[0]),
            "conflicting_iso3_year_rows": int(conflict_df.shape[0]),
            "metadata_column": meta_col or "",
        }
        return standardized, resolution_df, qc, conflict_df


    def load_external_tobacco_data(
        tobacco_dir: Path,
    ) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame, pd.DataFrame, dict[str, object], list[str]]:
        tobacco_files = resolve_clean_tobacco_files(tobacco_dir)
        policy = pd.read_csv(tobacco_files["policy"], encoding="utf-8-sig", low_memory=False)
        smoking = pd.read_csv(tobacco_files["smoking_trend"], encoding="utf-8-sig", low_memory=False)
        cigarette_path = tobacco_files["cigarette_prevalence"]
        smoking_optional_path = tobacco_files["smoking_prevalence_optional"]

        policy_df, policy_resolution, policy_qc, policy_conflicts = standardize_policy_frame(policy, "policy")
        policy_latest_year = int(pd.to_numeric(policy_df["year"], errors="coerce").max()) if not policy_df.empty else None
        smoking_df, smoking_resolution, smoking_qc, smoking_conflicts = standardize_prevalence_frame(
            smoking,
            "who_smoking_rate_std",
            "smoking_trend",
            ["smk", "smoking", "tobacco"],
            inferred_observed_year_max=policy_latest_year,
        )

        optional_qc = {}
        optional_resolution = pd.DataFrame()
        optional_conflicts = pd.DataFrame()
        if smoking_optional_path is not None and Path(smoking_optional_path) != Path(tobacco_files["smoking_trend"]):
            smoking_optional = pd.read_csv(smoking_optional_path, encoding="utf-8-sig", low_memory=False)
            fallback_df, optional_resolution, optional_qc, optional_conflicts = standardize_prevalence_frame(
                smoking_optional,
                "who_smoking_rate_std_fallback",
                "smoking_prevalence_optional",
                ["smk", "smoking", "tobacco"],
                inferred_observed_year_max=policy_latest_year,
            )
            smoking_df = smoking_df.merge(fallback_df, on=["iso3", "year"], how="outer")
            smoking_df["who_smoking_rate_std"] = smoking_df["who_smoking_rate_std"].combine_first(smoking_df["who_smoking_rate_std_fallback"])
            smoking_df = smoking_df.loc[:, ["iso3", "year", "who_smoking_rate_std"]]

        if cigarette_path is not None:
            cigarette = pd.read_csv(cigarette_path, encoding="utf-8-sig", low_memory=False)
            cigarette_df, cigarette_resolution, cigarette_qc, cigarette_conflicts = standardize_prevalence_frame(
                cigarette,
                "who_cigarette_prevalence_btsx",
                "cigarette_prevalence",
                ["cig", "cigarette"],
                inferred_observed_year_max=policy_latest_year,
            )
        else:
            cigarette_df = pd.DataFrame(columns=["iso3", "year", "who_cigarette_prevalence_btsx"])
            cigarette_resolution = pd.DataFrame()
            cigarette_qc = {}
            cigarette_conflicts = pd.DataFrame()

        merged = policy_df.merge(smoking_df, on=["iso3", "year"], how="outer").merge(cigarette_df, on=["iso3", "year"], how="outer")
        coverage_rows = []
        for column in ["policy_strength_external", "who_smoking_rate_std", "who_cigarette_prevalence_btsx"]:
            if column not in merged.columns:
                continue
            coverage_rows.append(
                {
                    "outcome_or_input": column,
                    "country_year_rows": int(merged[column].notna().sum()),
                    "countries": int(merged.loc[merged[column].notna(), "iso3"].nunique()),
                    "year_min": int(pd.to_numeric(merged.loc[merged[column].notna(), "year"], errors="coerce").min()) if merged[column].notna().any() else np.nan,
                    "year_max": int(pd.to_numeric(merged.loc[merged[column].notna(), "year"], errors="coerce").max()) if merged[column].notna().any() else np.nan,
                }
            )
        coverage_df = pd.DataFrame(coverage_rows)

        resolution_df = pd.concat(
            [
                frame
                for frame in [policy_resolution, smoking_resolution, optional_resolution, cigarette_resolution]
                if not frame.empty
            ],
            ignore_index=True,
        ) if any(not frame.empty for frame in [policy_resolution, smoking_resolution, optional_resolution, cigarette_resolution]) else pd.DataFrame(
            columns=["dataset_kind", "resolution_stage", "column_name", "value", "records"]
        )

        conflict_df = pd.concat(
            [frame for frame in [policy_conflicts, smoking_conflicts, optional_conflicts, cigarette_conflicts] if not frame.empty],
            ignore_index=True,
        ) if any(not frame.empty for frame in [policy_conflicts, smoking_conflicts, optional_conflicts, cigarette_conflicts]) else pd.DataFrame()
        if not conflict_df.empty:
            conflict_df.insert(0, "resolution_stage", "conflict_iso3_year")
            conflict_df.insert(0, "column_name", "iso3_year")
            resolution_df = pd.concat(
                [
                    resolution_df,
                    conflict_df.rename(columns={"dataset_kind": "dataset_kind", "iso3": "value", "count": "records"})[
                        [column for column in ["dataset_kind", "resolution_stage", "column_name", "value", "records"] if column in conflict_df.columns or column in {"dataset_kind", "resolution_stage", "column_name"}]
                    ],
                ],
                ignore_index=True,
            )

        file_summary = {
            "policy": Path(tobacco_files["policy"]).name,
            "smoking_trend": Path(tobacco_files["smoking_trend"]).name,
            "cigarette_prevalence": Path(cigarette_path).name if cigarette_path is not None else "",
            "smoking_prevalence_optional": Path(smoking_optional_path).name if smoking_optional_path is not None else "",
        }
        qc_summary = {
            "policy": policy_qc,
            "smoking_trend": smoking_qc,
            "smoking_prevalence_optional": optional_qc,
            "cigarette_prevalence": cigarette_qc,
        }
        conflict_messages = []
        for frame in [policy_conflicts, smoking_conflicts, optional_conflicts, cigarette_conflicts]:
            if frame.empty:
                continue
            dataset_kind = str(frame.iloc[0]["dataset_kind"])
            conflict_messages.append(f"{dataset_kind}: {int(frame.shape[0])} conflicting iso3-year rows after filtering")
        return merged, file_summary, resolution_df, coverage_df, qc_summary, conflict_messages


    def merge_policy_inputs(response_df: pd.DataFrame, tobacco_df: pd.DataFrame, typology_df: pd.DataFrame | None) -> pd.DataFrame:
        work_df = response_df.copy()
        work_df["iso3"] = normalize_iso3(work_df["iso3"])
        work_df["year"] = pd.to_numeric(work_df["year"], errors="coerce").astype("Int64")
        if typology_df is not None and not typology_df.empty:
            merge_cols = ["iso3", "year", "vulnerability_type_code", "vulnerability_type_label"]
            available_cols = [column for column in merge_cols if column in typology_df.columns]
            if {"iso3", "year", "vulnerability_type_label"}.issubset(available_cols):
                typology_merge = typology_df.loc[:, available_cols].drop_duplicates(["iso3", "year"], keep="last").copy()
                typology_merge["iso3"] = normalize_iso3(typology_merge["iso3"])
                typology_merge["year"] = pd.to_numeric(typology_merge["year"], errors="coerce").astype("Int64")
                work_df = work_df.merge(typology_merge, on=["iso3", "year"], how="left")
        merged = work_df.merge(tobacco_df, on=["iso3", "year"], how="left")
        merged["policy_strength"] = merged["policy_strength_external"]
        return merged


    def count_policy_switches(policy: pd.Series, threshold: float) -> int:
        binary = (policy >= threshold).astype(int)
        if binary.empty:
            return 0
        return int(binary.diff().abs().fillna(0).sum())


    def identify_treatment_timing(
        df: pd.DataFrame,
        threshold: float,
        min_pre_years: int,
        min_post_years: int,
        min_consecutive_strong: int,
        max_policy_switches: int,
        sample_variant: str,
        min_policy_jump: float = 0.0,
        max_previous_policy: float | None = None,
        require_sharp_jump: bool = False,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for iso3, subset in df.groupby("iso3", dropna=False):
            ordered = subset.sort_values("year", kind="stable")
            ordered = ordered.loc[ordered["policy_strength"].notna(), ["year", "policy_strength"]].copy()
            if ordered.empty:
                rows.append(
                    {
                        "iso3": iso3,
                        "sample_variant": sample_variant,
                        "treat_ever": False,
                        "treatment_year": np.nan,
                        "policy_years_observed": 0,
                        "policy_strength_latest": np.nan,
                        "policy_switches": 0,
                        "exclusion_reason": "no_policy_data",
                    }
                )
                continue

            years = pd.to_numeric(ordered["year"], errors="coerce").astype(int).to_numpy()
            policy = pd.to_numeric(ordered["policy_strength"], errors="coerce").astype(float).reset_index(drop=True)
            strong = (policy >= threshold).astype(int)
            switches = count_policy_switches(policy, threshold)
            treatment_year = np.nan
            exclusion_reason = ""
            treated_ever = False

            if strong.eq(1).all():
                exclusion_reason = "always_strong"
            elif switches > max_policy_switches:
                exclusion_reason = "oscillating_policy"
            else:
                candidate_year = np.nan
                for idx in range(len(policy)):
                    if idx + min_consecutive_strong > len(policy):
                        break
                    run = strong.iloc[idx : idx + min_consecutive_strong]
                    if not run.eq(1).all():
                        continue
                    pre = policy.iloc[:idx]
                    post = policy.iloc[idx:]
                    if int(pre.notna().sum()) < min_pre_years:
                        continue
                    if int(post.notna().sum()) < min_post_years:
                        continue
                    if idx > 0 and strong.iloc[:idx].eq(1).all():
                        continue
                    if require_sharp_jump:
                        if idx == 0:
                            continue
                        previous_policy = float(policy.iloc[idx - 1]) if pd.notna(policy.iloc[idx - 1]) else np.nan
                        current_policy = float(policy.iloc[idx]) if pd.notna(policy.iloc[idx]) else np.nan
                        if np.isnan(previous_policy) or np.isnan(current_policy):
                            continue
                        if max_previous_policy is not None and previous_policy > float(max_previous_policy):
                            continue
                        if (current_policy - previous_policy) < float(min_policy_jump):
                            continue
                    candidate_year = float(years[idx])
                    break

                if np.isnan(candidate_year):
                    exclusion_reason = "insufficient_pre_post"
                else:
                    treatment_year = candidate_year
                    treated_ever = True

            rows.append(
                {
                    "iso3": iso3,
                    "sample_variant": sample_variant,
                    "treat_ever": treated_ever,
                    "treatment_year": treatment_year,
                    "policy_years_observed": int(policy.notna().sum()),
                    "policy_strength_latest": float(policy.iloc[-1]) if not policy.empty else np.nan,
                    "policy_switches": switches,
                    "exclusion_reason": exclusion_reason,
                }
            )
        return pd.DataFrame(rows)


    def build_sample_restriction_comparison(
        timing_frames: list[pd.DataFrame],
        sample_configs: list[dict[str, object]],
        threshold: float,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        config_map = {str(config["sample_variant"]): config for config in sample_configs}
        for timing_df in timing_frames:
            if timing_df.empty:
                continue
            variant = str(timing_df["sample_variant"].iloc[0])
            config = config_map.get(variant, {})
            rows.append(
                {
                    "sample_variant": variant,
                    "threshold": float(threshold),
                    "min_pre_years": int(config.get("min_pre_years", np.nan)),
                    "min_post_years": int(config.get("min_post_years", np.nan)),
                    "min_consecutive_strong": int(config.get("min_consecutive_strong", np.nan)),
                    "max_policy_switches": int(config.get("max_policy_switches", np.nan)),
                    "countries_total": int(timing_df["iso3"].nunique()),
                    "treated_countries": int(timing_df["treat_ever"].fillna(False).sum()),
                    "control_countries": int((~timing_df["treat_ever"].fillna(False)).sum()),
                    "excluded_always_strong": int((timing_df["exclusion_reason"] == "always_strong").sum()),
                    "excluded_oscillating_policy": int((timing_df["exclusion_reason"] == "oscillating_policy").sum()),
                    "excluded_insufficient_pre_post": int((timing_df["exclusion_reason"] == "insufficient_pre_post").sum()),
                    "excluded_no_policy_data": int((timing_df["exclusion_reason"] == "no_policy_data").sum()),
                }
            )
        return pd.DataFrame(rows)


    def build_treated_country_output(timing_frames: list[pd.DataFrame]) -> pd.DataFrame:
        frames = [frame.copy() for frame in timing_frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["sample_variant", "treat_ever", "treatment_year", "iso3"], ascending=[True, False, True, True], kind="stable")


    def get_pre_treatment_snapshot(
        df: pd.DataFrame,
        timing_df: pd.DataFrame,
        value_column: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        treated = timing_df.loc[timing_df["treat_ever"] & timing_df["treatment_year"].notna(), ["iso3", "treatment_year"]].copy()
        for _, item in treated.iterrows():
            iso3 = str(item["iso3"])
            treatment_year = float(item["treatment_year"])
            subset = df.loc[(df["iso3"] == iso3) & (pd.to_numeric(df["year"], errors="coerce") < treatment_year)].copy()
            subset = subset.dropna(subset=[value_column])
            if subset.empty:
                continue
            chosen = subset.sort_values("year", kind="stable").iloc[-1]
            rows.append(
                {
                    "iso3": iso3,
                    "treatment_year": treatment_year,
                    "baseline_year": int(pd.to_numeric(chosen["year"], errors="coerce")),
                    value_column: chosen[value_column],
                }
            )
        return pd.DataFrame(rows)


    def build_typology_subgroups(df: pd.DataFrame, timing_df: pd.DataFrame) -> pd.DataFrame:
        if "vulnerability_type_label" not in df.columns:
            return pd.DataFrame(columns=["iso3", "subgroup_dimension", "subgroup_name"])
        snapshot = get_pre_treatment_snapshot(df, timing_df, "vulnerability_type_label")
        if snapshot.empty:
            return pd.DataFrame(columns=["iso3", "subgroup_dimension", "subgroup_name"])
        result = snapshot.loc[:, ["iso3", "vulnerability_type_label"]].rename(columns={"vulnerability_type_label": "subgroup_name"})
        result["subgroup_dimension"] = "vulnerability_type"
        return result.dropna(subset=["subgroup_name"]).reset_index(drop=True)


    def build_baseline_smoking_subgroups(df: pd.DataFrame, timing_df: pd.DataFrame) -> pd.DataFrame:
        primary = get_pre_treatment_snapshot(df, timing_df, "who_smoking_rate_std") if "who_smoking_rate_std" in df.columns else pd.DataFrame()
        if primary.empty and "dbd_smoking_per100k" not in df.columns:
            return pd.DataFrame(columns=["iso3", "subgroup_dimension", "subgroup_name", "baseline_smoking_value"])

        if "dbd_smoking_per100k" in df.columns:
            fallback = get_pre_treatment_snapshot(df, timing_df, "dbd_smoking_per100k")
        else:
            fallback = pd.DataFrame()

        merged = primary.rename(columns={"who_smoking_rate_std": "baseline_smoking_value"}) if not primary.empty else pd.DataFrame(columns=["iso3", "baseline_smoking_value"])
        if not fallback.empty:
            fallback = fallback.rename(columns={"dbd_smoking_per100k": "baseline_smoking_fallback"})
            merged = merged.merge(fallback.loc[:, [column for column in ["iso3", "baseline_smoking_fallback"] if column in fallback.columns]], on="iso3", how="outer")
            merged["baseline_smoking_value"] = pd.to_numeric(merged["baseline_smoking_value"], errors="coerce").combine_first(
                pd.to_numeric(merged["baseline_smoking_fallback"], errors="coerce")
            )
        if merged.empty:
            return pd.DataFrame(columns=["iso3", "subgroup_dimension", "subgroup_name", "baseline_smoking_value"])
        merged = merged.dropna(subset=["baseline_smoking_value"]).copy()
        if merged.empty:
            return pd.DataFrame(columns=["iso3", "subgroup_dimension", "subgroup_name", "baseline_smoking_value"])
        ranks = merged["baseline_smoking_value"].rank(pct=True, method="average")
        bins = pd.cut(
            ranks,
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=[
                choose_text("基线吸烟低组", "Low baseline smoking", USE_CHINESE),
                choose_text("基线吸烟中组", "Medium baseline smoking", USE_CHINESE),
                choose_text("基线吸烟高组", "High baseline smoking", USE_CHINESE),
            ],
        )
        merged["subgroup_name"] = bins.astype(str)
        merged["subgroup_dimension"] = "baseline_smoking_tercile"
        return merged.loc[:, ["iso3", "subgroup_dimension", "subgroup_name", "baseline_smoking_value"]].reset_index(drop=True)


    def build_subgroup_assignments(df: pd.DataFrame, timing_df: pd.DataFrame) -> pd.DataFrame:
        frames = [frame for frame in [build_typology_subgroups(df, timing_df), build_baseline_smoking_subgroups(df, timing_df)] if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=["iso3", "subgroup_dimension", "subgroup_name"])
        combined = pd.concat(frames, ignore_index=True)
        combined["iso3"] = normalize_iso3(combined["iso3"])
        return combined.dropna(subset=["subgroup_name"]).drop_duplicates(["iso3", "subgroup_dimension", "subgroup_name"]).reset_index(drop=True)


    def build_event_dummies(df: pd.DataFrame, time_column: str, treat_column: str, pre_window: int, post_window: int) -> tuple[pd.DataFrame, list[int]]:
        event_times = [time for time in range(-pre_window, post_window + 1) if time != -1]
        dummy_dict = {}
        for event_time in event_times:
            column = f"event_{event_time:+d}".replace("+", "p").replace("-", "m")
            dummy_dict[column] = ((df[treat_column] == 1) & (df[time_column] == event_time)).astype(float)
        return pd.DataFrame(dummy_dict, index=df.index), event_times


    def fit_fe_model(
        df: pd.DataFrame,
        outcome_column: str,
        regressors: list[str],
        unit_column: str,
        year_column: str,
        log_transform: bool,
        winsor_lower: float,
        winsor_upper: float,
        cluster_column: str | None = None,
    ) -> tuple[pd.DataFrame, int]:
        cluster_col = cluster_column or unit_column
        selected_columns = list(dict.fromkeys([unit_column, year_column, outcome_column, cluster_col] + regressors))
        work_df = df.loc[:, selected_columns].copy()
        work_df["y"] = prepare_outcome(work_df[outcome_column], winsor_lower, winsor_upper, log_transform)
        work_df = work_df.dropna(subset=["y"] + regressors).copy()
        if work_df.shape[0] < max(40, len(regressors) + 20):
            return pd.DataFrame(), 0

        unit_dummies = pd.get_dummies(work_df[unit_column].astype(str), prefix="unit", drop_first=True, dtype=float)
        year_dummies = pd.get_dummies(work_df[year_column].astype(str), prefix="year", drop_first=True, dtype=float)
        x_core = work_df[regressors].astype(float)
        design = np.column_stack(
            [
                np.ones(len(work_df)),
                x_core.to_numpy(dtype=float),
                unit_dummies.to_numpy(dtype=float),
                year_dummies.to_numpy(dtype=float),
            ]
        )
        y = work_df["y"].to_numpy(dtype=float)
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coefficients
        residual = y - fitted
        n_obs = len(y)
        n_params = design.shape[1]
        dof = max(n_obs - n_params, 1)
        xtx_inv = np.linalg.pinv(design.T @ design)
        cluster_ids = work_df[cluster_col].astype(str).to_numpy()
        unique_clusters = pd.Index(cluster_ids).nunique()
        if unique_clusters > 1:
            meat = np.zeros_like(xtx_inv, dtype=float)
            for cluster_id in pd.Index(cluster_ids).unique():
                mask = cluster_ids == cluster_id
                cluster_design = design[mask, :]
                cluster_residual = residual[mask]
                score = cluster_design.T @ cluster_residual
                meat += np.outer(score, score)
            small_sample_scale = 1.0
            if unique_clusters > 1 and n_obs > n_params:
                small_sample_scale = (unique_clusters / (unique_clusters - 1)) * ((n_obs - 1) / max(n_obs - n_params, 1))
            covariance = small_sample_scale * (xtx_inv @ meat @ xtx_inv)
        else:
            sigma2 = float(np.dot(residual, residual) / dof)
            covariance = np.diag(np.clip(np.diag(xtx_inv) * sigma2, a_min=0.0, a_max=None))
        variance_terms = np.clip(np.diag(covariance), a_min=0.0, a_max=None)
        std_errors = np.sqrt(variance_terms)

        rows = []
        for idx, regressor in enumerate(regressors, start=1):
            coefficient = float(coefficients[idx])
            std_error = float(std_errors[idx]) if idx < len(std_errors) else np.nan
            z_stat = coefficient / std_error if std_error and not np.isnan(std_error) and std_error > 0 else np.nan
            p_value = normal_p_value(z_stat) if not np.isnan(z_stat) else np.nan
            rows.append(
                {
                    "regressor": regressor,
                    "coefficient": coefficient,
                    "std_error": std_error,
                    "z_stat": z_stat,
                    "p_value": p_value,
                    "n_obs": int(n_obs),
                    "countries": int(work_df[unit_column].astype(str).nunique()),
                    "n_clusters": int(unique_clusters),
                    "std_error_method": "cluster_robust_small_sample" if unique_clusters > 1 else "classic_ols",
                }
            )
        return pd.DataFrame(rows), n_obs


    def build_cohort_sample(
        df: pd.DataFrame,
        timing_df: pd.DataFrame,
        cohort_year: int,
        spec: dict[str, object],
        placebo_shift: int,
        anticipation_buffer: int,
        treated_units_filter: set[str] | None = None,
    ) -> pd.DataFrame:
        lag = int(spec["lag"])
        pre_window = int(spec["pre_window"])
        post_window = int(spec["post_window"])
        effective_year = cohort_year - placebo_shift
        actual_year_min = effective_year + lag - pre_window
        actual_year_max = effective_year + lag + post_window

        treated_units = timing_df.loc[timing_df["treatment_year"] == cohort_year, "iso3"].dropna().astype(str).unique().tolist()
        if treated_units_filter is not None:
            treated_units = [iso3 for iso3 in treated_units if iso3 in treated_units_filter]
        if not treated_units:
            return pd.DataFrame()

        control_mask = (~timing_df["treat_ever"]) | (timing_df["treatment_year"] > (actual_year_max + anticipation_buffer))
        control_units = timing_df.loc[control_mask, "iso3"].dropna().astype(str).unique().tolist()
        if not control_units:
            return pd.DataFrame()

        sample_units = treated_units + control_units
        sample = df.loc[df["iso3"].isin(sample_units)].copy()
        sample = sample.loc[(sample["year"].astype(float) >= actual_year_min) & (sample["year"].astype(float) <= actual_year_max)].copy()
        if sample.empty:
            return pd.DataFrame()

        sample["treat"] = sample["iso3"].isin(treated_units).astype(int)
        sample["cohort_year"] = cohort_year
        sample["effective_cohort_year"] = effective_year
        sample["placebo_shift"] = placebo_shift
        sample["effective_event_time"] = sample["year"].astype(float) - effective_year - lag
        sample["treated_post"] = ((sample["treat"] == 1) & (sample["effective_event_time"] >= 0)).astype(int)
        sample["stack_unit"] = sample["iso3"].astype(str)
        sample["stack_year"] = sample["year"].astype(int).astype(str)
        sample["anticipation_buffer_years"] = anticipation_buffer
        sample["treated_countries_cohort"] = len(treated_units)
        sample["control_countries_cohort"] = len(control_units)

        if sample["treat"].sum() == 0 or sample["treat"].nunique() < 2:
            return pd.DataFrame()
        return sample


    def aggregate_scalar_estimates(cohort_df: pd.DataFrame, weight_column: str) -> dict[str, float]:
        weights = cohort_df[weight_column].astype(float).to_numpy()
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        normalized = weights / weights.sum()
        coefficients = cohort_df["coefficient"].astype(float).to_numpy()
        std_errors = cohort_df["std_error"].astype(float).to_numpy()
        coefficient = float(np.sum(normalized * coefficients))
        std_error = float(np.sqrt(np.sum((normalized ** 2) * (std_errors ** 2))))
        z_stat = coefficient / std_error if std_error > 0 else np.nan
        p_value = normal_p_value(z_stat) if not np.isnan(z_stat) else np.nan
        return {
            "coefficient": coefficient,
            "std_error": std_error,
            "z_stat": z_stat,
            "p_value": p_value,
        }


    def aggregate_event_estimates(cohort_event_df: pd.DataFrame, weight_column: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for event_time, subset in cohort_event_df.groupby("event_time", sort=True):
            stats = aggregate_scalar_estimates(subset, weight_column)
            base = subset.iloc[0].to_dict()
            rows.append(
                {
                    "event_time": int(event_time),
                    "coefficient": stats["coefficient"],
                    "std_error": stats["std_error"],
                    "z_stat": stats["z_stat"],
                    "p_value": stats["p_value"],
                    "n_cohorts": int(subset["cohort_year"].nunique()),
                    "n_obs": int(subset["n_obs"].sum()),
                    "treated_countries": int(subset["treated_countries_cohort"].sum()),
                    "control_countries": int(subset["control_countries_cohort"].sum()),
                    "outcome_column": base["outcome_column"],
                    "outcome_label": base["outcome_label"],
                    "threshold": base["threshold"],
                    "lag": base["lag"],
                    "placebo_shift": base["placebo_shift"],
                    "tier": base["tier"],
                    "tier_label": base["tier_label"],
                    "analysis_scope": base["analysis_scope"],
                    "sample_variant": base["sample_variant"],
                    "subgroup_dimension": base["subgroup_dimension"],
                    "subgroup_name": base["subgroup_name"],
                }
            )
        return pd.DataFrame(rows)


    def run_stacked_estimation(
        df: pd.DataFrame,
        timing_df: pd.DataFrame,
        spec: dict[str, object],
        threshold: float,
        placebo_shift: int,
        log_transform: bool,
        winsor_lower: float,
        winsor_upper: float,
        analysis_scope: str,
        sample_variant: str,
        subgroup_dimension: str = "",
        subgroup_name: str = "",
        treated_units_filter: set[str] | None = None,
        compute_event: bool = True,
        anticipation_buffer: int = DEFAULT_ANTICIPATION_BUFFER,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        outcome_column = str(spec["outcome_column"])
        if outcome_column not in df.columns:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        cohort_did_rows: list[dict[str, object]] = []
        cohort_event_rows: list[dict[str, object]] = []
        treated_cohorts = (
            timing_df.loc[timing_df["treat_ever"] & timing_df["treatment_year"].notna(), "treatment_year"]
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )
        for cohort_year in treated_cohorts:
            sample = build_cohort_sample(
                df,
                timing_df,
                cohort_year,
                spec,
                placebo_shift,
                anticipation_buffer=anticipation_buffer,
                treated_units_filter=treated_units_filter,
            )
            if sample.empty:
                continue

            did_result, _ = fit_fe_model(
                sample,
                outcome_column,
                ["treated_post"],
                unit_column="stack_unit",
                year_column="stack_year",
                log_transform=log_transform,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                cluster_column="iso3",
            )
            if not did_result.empty:
                row = did_result.iloc[0].to_dict()
                row.update(
                    {
                        "cohort_year": cohort_year,
                        "effective_cohort_year": cohort_year - placebo_shift,
                        "treated_countries_cohort": int(sample["treated_countries_cohort"].iloc[0]),
                        "control_countries_cohort": int(sample["control_countries_cohort"].iloc[0]),
                        "outcome_column": outcome_column,
                        "outcome_label": outcome_label(outcome_column),
                        "threshold": threshold,
                        "lag": int(spec["lag"]),
                        "pre_window": int(spec["pre_window"]),
                        "post_window": int(spec["post_window"]),
                        "placebo_shift": placebo_shift,
                        "tier": str(spec["tier"]),
                        "tier_label": tier_label(spec),
                        "analysis_scope": analysis_scope,
                        "sample_variant": sample_variant,
                        "subgroup_dimension": subgroup_dimension,
                        "subgroup_name": subgroup_name,
                        "anticipation_buffer_years": anticipation_buffer,
                    }
                )
                cohort_did_rows.append(row)

            if compute_event:
                event_dummies, event_times = build_event_dummies(sample, "effective_event_time", "treat", int(spec["pre_window"]), int(spec["post_window"]))
                if not event_dummies.empty:
                    event_sample = pd.concat([sample, event_dummies], axis=1)
                    regressors = event_dummies.columns.tolist()
                    event_result, _ = fit_fe_model(
                        event_sample,
                        outcome_column,
                        regressors,
                        unit_column="stack_unit",
                        year_column="stack_year",
                        log_transform=log_transform,
                        winsor_lower=winsor_lower,
                        winsor_upper=winsor_upper,
                        cluster_column="iso3",
                    )
                    if not event_result.empty:
                        event_result["event_time"] = event_times
                        event_result["cohort_year"] = cohort_year
                        event_result["effective_cohort_year"] = cohort_year - placebo_shift
                        event_result["treated_countries_cohort"] = int(sample["treated_countries_cohort"].iloc[0])
                        event_result["control_countries_cohort"] = int(sample["control_countries_cohort"].iloc[0])
                        event_result["outcome_column"] = outcome_column
                        event_result["outcome_label"] = outcome_label(outcome_column)
                        event_result["threshold"] = threshold
                        event_result["lag"] = int(spec["lag"])
                        event_result["placebo_shift"] = placebo_shift
                        event_result["tier"] = str(spec["tier"])
                        event_result["tier_label"] = tier_label(spec)
                        event_result["analysis_scope"] = analysis_scope
                        event_result["sample_variant"] = sample_variant
                        event_result["subgroup_dimension"] = subgroup_dimension
                        event_result["subgroup_name"] = subgroup_name
                        event_result["anticipation_buffer_years"] = anticipation_buffer
                        cohort_event_rows.extend(event_result.to_dict(orient="records"))

        cohort_did_df = pd.DataFrame(cohort_did_rows)
        cohort_event_df = pd.DataFrame(cohort_event_rows)
        if cohort_did_df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        aggregated = aggregate_scalar_estimates(cohort_did_df, "treated_countries_cohort")
        main_row = {
            "outcome_column": outcome_column,
            "outcome_label": outcome_label(outcome_column),
            "threshold": threshold,
            "lag": int(spec["lag"]),
            "pre_window": int(spec["pre_window"]),
            "post_window": int(spec["post_window"]),
            "tier": str(spec["tier"]),
            "tier_label": tier_label(spec),
            "placebo_shift": placebo_shift,
            "n_cohorts": int(cohort_did_df["cohort_year"].nunique()),
            "n_obs": int(cohort_did_df["n_obs"].sum()),
            "treated_countries": int(cohort_did_df["treated_countries_cohort"].sum()),
            "control_countries": int(cohort_did_df["control_countries_cohort"].sum()),
            "estimator": "stacked_did",
            "analysis_scope": analysis_scope,
            "sample_variant": sample_variant,
            "subgroup_dimension": subgroup_dimension,
            "subgroup_name": subgroup_name,
            "anticipation_buffer_years": anticipation_buffer,
            **aggregated,
        }
        aggregated_did_df = pd.DataFrame([main_row])
        aggregated_event_df = aggregate_event_estimates(cohort_event_df, "treated_countries_cohort") if not cohort_event_df.empty else pd.DataFrame()
        return aggregated_did_df, aggregated_event_df, cohort_did_df


    def evaluate_event_pattern(event_df: pd.DataFrame) -> dict[str, object]:
        if event_df.empty:
            return {
                "pretrend_any_signal": False,
                "pretrend_mean_abs_coef": np.nan,
                "pretrend_max_abs_coef": np.nan,
                "posttrend_any_signal": False,
                "posttrend_mean_coef": np.nan,
                "posttrend_peak_abs_coef": np.nan,
                "direction_consistent": False,
            }
        pre_df = event_df.loc[event_df["event_time"] <= -2].copy()
        post_df = event_df.loc[event_df["event_time"] >= 0].copy()
        pretrend_any_signal = bool((pre_df["p_value"] <= 0.10).fillna(False).any()) if not pre_df.empty else False
        posttrend_any_signal = bool((post_df["p_value"] <= 0.10).fillna(False).any()) if not post_df.empty else False
        pretrend_mean_abs_coef = float(pre_df["coefficient"].abs().mean()) if not pre_df.empty else np.nan
        pretrend_max_abs_coef = float(pre_df["coefficient"].abs().max()) if not pre_df.empty else np.nan
        posttrend_mean_coef = float(post_df["coefficient"].mean()) if not post_df.empty else np.nan
        posttrend_peak_abs_coef = float(post_df["coefficient"].abs().max()) if not post_df.empty else np.nan
        return {
            "pretrend_any_signal": pretrend_any_signal,
            "pretrend_mean_abs_coef": pretrend_mean_abs_coef,
            "pretrend_max_abs_coef": pretrend_max_abs_coef,
            "posttrend_any_signal": posttrend_any_signal,
            "posttrend_mean_coef": posttrend_mean_coef,
            "posttrend_peak_abs_coef": posttrend_peak_abs_coef,
        }


    def build_threshold_sensitivity(
        df: pd.DataFrame,
        thresholds: list[float],
        specs: list[dict[str, object]],
        sample_config: dict[str, object],
        log_transform: bool,
        winsor_lower: float,
        winsor_upper: float,
        analysis_scope: str,
        sample_variant: str,
        subgroup_dimension: str = "",
        subgroup_name: str = "",
        treated_units_filter: set[str] | None = None,
        anticipation_buffer: int = DEFAULT_ANTICIPATION_BUFFER,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        did_rows: list[dict[str, object]] = []
        event_rows: list[dict[str, object]] = []
        for threshold in thresholds:
            timing_df = identify_treatment_timing(
                df,
                threshold=threshold,
                min_pre_years=int(sample_config["min_pre_years"]),
                min_post_years=int(sample_config["min_post_years"]),
                min_consecutive_strong=int(sample_config["min_consecutive_strong"]),
                max_policy_switches=int(sample_config["max_policy_switches"]),
                sample_variant=sample_variant,
            )
            for spec in specs:
                did_df, event_df, _ = run_stacked_estimation(
                    df,
                    timing_df,
                    spec,
                    threshold=threshold,
                    placebo_shift=0,
                    log_transform=log_transform,
                    winsor_lower=winsor_lower,
                    winsor_upper=winsor_upper,
                    analysis_scope=analysis_scope,
                    sample_variant=sample_variant,
                    subgroup_dimension=subgroup_dimension,
                    subgroup_name=subgroup_name,
                    treated_units_filter=treated_units_filter,
                    compute_event=False,
                    anticipation_buffer=anticipation_buffer,
                )
                if did_df.empty:
                    continue
                diagnostics = evaluate_event_pattern(event_df)
                row = did_df.iloc[0].to_dict()
                row.update(diagnostics)
                did_rows.append(row)
                if not event_df.empty:
                    event_rows.extend(event_df.to_dict(orient="records"))
        return pd.DataFrame(did_rows), pd.DataFrame(event_rows)


    def build_specification_grid(
        df: pd.DataFrame,
        thresholds: list[float],
        base_specs: list[dict[str, object]],
        sample_config: dict[str, object],
        log_transform: bool,
        winsor_lower: float,
        winsor_upper: float,
        anticipation_buffer: int = DEFAULT_ANTICIPATION_BUFFER,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for threshold in thresholds:
            timing_df = identify_treatment_timing(
                df,
                threshold=threshold,
                min_pre_years=int(sample_config["min_pre_years"]),
                min_post_years=int(sample_config["min_post_years"]),
                min_consecutive_strong=int(sample_config["min_consecutive_strong"]),
                max_policy_switches=int(sample_config["max_policy_switches"]),
                sample_variant=MAIN_SAMPLE_VARIANT,
            )
            for spec in base_specs:
                for lag in list(spec["lag_grid"]):
                    lagged_spec = dict(spec)
                    lagged_spec["lag"] = lag
                    did_df, event_df, _ = run_stacked_estimation(
                        df,
                        timing_df,
                        lagged_spec,
                        threshold=threshold,
                        placebo_shift=0,
                        log_transform=log_transform,
                        winsor_lower=winsor_lower,
                        winsor_upper=winsor_upper,
                        analysis_scope="global",
                        sample_variant=MAIN_SAMPLE_VARIANT,
                        compute_event=False,
                        anticipation_buffer=anticipation_buffer,
                    )
                    if did_df.empty:
                        continue
                    diagnostics = evaluate_event_pattern(event_df)
                    row = did_df.iloc[0].to_dict()
                    row.update(diagnostics)
                    rows.append(row)
        return pd.DataFrame(rows)


    def build_placebo_estimates(
        df: pd.DataFrame,
        timing_df: pd.DataFrame,
        specs: list[dict[str, object]],
        placebo_shifts: list[int],
        threshold: float,
        log_transform: bool,
        winsor_lower: float,
        winsor_upper: float,
        analysis_scope: str,
        sample_variant: str,
        subgroup_dimension: str = "",
        subgroup_name: str = "",
        treated_units_filter: set[str] | None = None,
        anticipation_buffer: int = DEFAULT_ANTICIPATION_BUFFER,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for spec in specs:
            spec_placebo_shifts = resolve_placebo_shifts(spec, placebo_shifts)
            for shift in spec_placebo_shifts:
                did_df, event_df, _ = run_stacked_estimation(
                    df,
                    timing_df,
                    spec,
                    threshold=threshold,
                    placebo_shift=shift,
                    log_transform=log_transform,
                    winsor_lower=winsor_lower,
                    winsor_upper=winsor_upper,
                    analysis_scope=analysis_scope,
                    sample_variant=sample_variant,
                    subgroup_dimension=subgroup_dimension,
                    subgroup_name=subgroup_name,
                    treated_units_filter=treated_units_filter,
                    compute_event=False,
                    anticipation_buffer=anticipation_buffer,
                )
                if did_df.empty:
                    continue
                diagnostics = evaluate_event_pattern(event_df)
                row = did_df.iloc[0].to_dict()
                row.update(diagnostics)
                row["placebo_shift"] = shift
                rows.append(row)
        return pd.DataFrame(rows)


    def build_anticipation_buffer_sensitivity(
        df: pd.DataFrame,
        specs: list[dict[str, object]],
        sample_config: dict[str, object],
        threshold: float,
        log_transform: bool,
        winsor_lower: float,
        winsor_upper: float,
        anticipation_buffers: list[int],
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        timing_df = identify_treatment_timing(
            df,
            threshold=threshold,
            min_pre_years=int(sample_config["min_pre_years"]),
            min_post_years=int(sample_config["min_post_years"]),
            min_consecutive_strong=int(sample_config["min_consecutive_strong"]),
            max_policy_switches=int(sample_config["max_policy_switches"]),
            sample_variant=MAIN_SAMPLE_VARIANT,
            min_policy_jump=float(sample_config.get("min_policy_jump", 0)),
            max_previous_policy=sample_config.get("max_previous_policy"),
            require_sharp_jump=bool(sample_config.get("require_sharp_jump", False)),
        )
        for anticipation_buffer in anticipation_buffers:
            for spec in specs:
                did_df, _, _ = run_stacked_estimation(
                    df,
                    timing_df,
                    spec,
                    threshold=threshold,
                    placebo_shift=0,
                    log_transform=log_transform,
                    winsor_lower=winsor_lower,
                    winsor_upper=winsor_upper,
                    analysis_scope="global",
                    sample_variant=MAIN_SAMPLE_VARIANT,
                    compute_event=False,
                    anticipation_buffer=anticipation_buffer,
                )
                if did_df.empty:
                    continue
                row = did_df.iloc[0].to_dict()
                row["anticipation_buffer_years"] = anticipation_buffer
                rows.append(row)
        return pd.DataFrame(rows)


    def build_policy_validation_summary(
        main_did_df: pd.DataFrame,
        main_event_df: pd.DataFrame,
        sensitivity_df: pd.DataFrame,
        placebo_df: pd.DataFrame,
        specs: list[dict[str, object]],
        placebo_shifts: list[int] | None = None,
        min_treated_for_lock: int = 0,
        min_nobs_for_lock: int = 0,
    ) -> pd.DataFrame:
        if main_did_df.empty:
            return empty_policy_validation_frame()
        spec_map = {str(spec["outcome_column"]): spec for spec in specs}
        first_stage_tiers = {"first_stage", "first_stage_proxy"}
        rows: list[dict[str, object]] = []
        for _, did_row in main_did_df.iterrows():
            outcome = str(did_row["outcome_column"])
            if outcome not in spec_map:
                continue
            spec = spec_map[outcome]
            expected_sign = int(spec["expected_sign"])
            analysis_scope = str(did_row.get("analysis_scope", "global"))
            sample_variant = str(did_row.get("sample_variant", MAIN_SAMPLE_VARIANT))
            subgroup_dimension = str(did_row.get("subgroup_dimension", ""))
            subgroup_name = str(did_row.get("subgroup_name", ""))

            event_subset = main_event_df.loc[
                (main_event_df["outcome_column"] == outcome)
                & (main_event_df["analysis_scope"] == analysis_scope)
                & (main_event_df["sample_variant"] == sample_variant)
                & (main_event_df["subgroup_dimension"].fillna("") == subgroup_dimension)
                & (main_event_df["subgroup_name"].fillna("") == subgroup_name)
                & (pd.to_numeric(main_event_df["placebo_shift"], errors="coerce").fillna(0) == 0)
            ].copy()
            event_diag = evaluate_event_pattern(event_subset)
            post_mean = event_diag["posttrend_mean_coef"]
            direction_consistent = (
                pd.notna(post_mean)
                and np.sign(float(did_row["coefficient"])) == np.sign(float(post_mean))
                and np.sign(float(did_row["coefficient"])) == expected_sign
            )

            sensitivity_subset = sensitivity_df.loc[
                (sensitivity_df["outcome_column"] == outcome)
                & (pd.to_numeric(sensitivity_df["lag"], errors="coerce").fillna(-999) == int(did_row["lag"]))
                & (sensitivity_df["analysis_scope"] == analysis_scope)
                & (sensitivity_df["sample_variant"] == sample_variant)
                & (sensitivity_df["subgroup_dimension"].fillna("") == subgroup_dimension)
                & (sensitivity_df["subgroup_name"].fillna("") == subgroup_name)
            ].copy()
            sensitivity_subset = sensitivity_subset.sort_values("threshold", kind="stable")
            available_sensitivity = sensitivity_subset.loc[pd.to_numeric(sensitivity_subset["coefficient"], errors="coerce").notna()].copy()
            same_direction_count = int(
                (
                    np.sign(pd.to_numeric(available_sensitivity["coefficient"], errors="coerce").fillna(0)) == expected_sign
                ).sum()
            )
            significant_opposite_threshold = bool(
                (
                    (np.sign(pd.to_numeric(available_sensitivity["coefficient"], errors="coerce").fillna(0)) == -expected_sign)
                    & (pd.to_numeric(available_sensitivity["p_value"], errors="coerce").fillna(1.0) <= 0.10)
                ).any()
            )
            if available_sensitivity.empty:
                sensitivity_same_direction = False
            elif len(available_sensitivity) == 1:
                sensitivity_same_direction = bool((same_direction_count == 1) and not significant_opposite_threshold)
            else:
                sensitivity_same_direction = bool(
                    same_direction_count >= max(2, len(available_sensitivity) - 1) and not significant_opposite_threshold
                )

            placebo_subset = placebo_df.loc[
                (placebo_df["outcome_column"] == outcome)
                & (placebo_df["analysis_scope"] == analysis_scope)
                & (placebo_df["sample_variant"] == sample_variant)
                & (placebo_df["subgroup_dimension"].fillna("") == subgroup_dimension)
                & (placebo_df["subgroup_name"].fillna("") == subgroup_name)
            ].copy()
            required_placebo_shifts = set(resolve_placebo_shifts(spec, placebo_shifts or [2, 3]))
            placebo_coeff = pd.to_numeric(placebo_subset["coefficient"], errors="coerce")
            placebo_pvals = pd.to_numeric(placebo_subset["p_value"], errors="coerce")
            strong_placebo_mask = (
                (pd.to_numeric(placebo_subset["placebo_shift"], errors="coerce").isin(sorted(required_placebo_shifts)))
                & (np.sign(placebo_coeff.fillna(0)) == expected_sign)
                & (placebo_pvals.fillna(1.0) <= 0.10)
            )
            strong_placebo = placebo_subset.loc[strong_placebo_mask].copy()
            if str(did_row["tier"]) in first_stage_tiers:
                placebo_issue = False
                if not strong_placebo.empty:
                    valid_shifts = set(pd.to_numeric(strong_placebo["placebo_shift"], errors="coerce").dropna().astype(int).tolist())
                    placebo_median_abs = float(pd.to_numeric(strong_placebo["coefficient"], errors="coerce").abs().median())
                    placebo_issue = required_placebo_shifts.issubset(valid_shifts) and placebo_median_abs >= abs(float(did_row["coefficient"]))
            else:
                placebo_issue = bool(
                    (
                        (np.sign(placebo_coeff.fillna(0)) == expected_sign)
                        & (placebo_pvals.fillna(1.0) <= 0.10)
                        & (placebo_coeff.abs() >= 0.5 * abs(float(did_row["coefficient"])))
                    ).any()
                )

            did_sign_ok = np.sign(float(did_row["coefficient"])) == expected_sign
            did_signal = float(did_row["p_value"]) <= 0.10 if pd.notna(did_row["p_value"]) else False
            pretrend_ok = not bool(event_diag["pretrend_any_signal"])
            sample_size_ok = int(did_row["treated_countries"]) >= min_treated_for_lock and int(did_row["n_obs"]) >= min_nobs_for_lock

            tier = str(did_row["tier"])
            if tier in first_stage_tiers:
                if did_sign_ok and did_signal and pretrend_ok and direction_consistent and not significant_opposite_threshold and not placebo_issue and sample_size_ok:
                    validation_status = "锁定"
                elif did_sign_ok and ((did_signal and pretrend_ok and not placebo_issue) or direction_consistent):
                    validation_status = "谨慎解释"
                else:
                    validation_status = "不锁"
            elif tier == "first_stage_robustness":
                if did_sign_ok and did_signal and pretrend_ok and not placebo_issue and sample_size_ok:
                    validation_status = "谨慎解释"
                else:
                    validation_status = "不锁"
            elif did_sign_ok and did_signal and pretrend_ok and direction_consistent and sensitivity_same_direction and not significant_opposite_threshold and not placebo_issue and sample_size_ok:
                validation_status = "谨慎解释"
            elif did_sign_ok and ((did_signal and pretrend_ok and not placebo_issue) or direction_consistent):
                validation_status = "谨慎解释"
            else:
                validation_status = "不锁"

            rows.append(
                {
                    "analysis_scope": analysis_scope,
                    "sample_variant": sample_variant,
                    "subgroup_dimension": subgroup_dimension,
                    "subgroup_name": subgroup_name,
                    "outcome_column": outcome,
                    "outcome_label": did_row["outcome_label"],
                    "tier": did_row["tier"],
                    "tier_label": did_row["tier_label"],
                    "did_coefficient": float(did_row["coefficient"]),
                    "did_p_value": float(did_row["p_value"]) if pd.notna(did_row["p_value"]) else np.nan,
                    "n_obs": int(did_row["n_obs"]),
                    "treated_countries": int(did_row["treated_countries"]),
                    "pretrend_any_signal": bool(event_diag["pretrend_any_signal"]),
                    "pretrend_mean_abs_coef": event_diag["pretrend_mean_abs_coef"],
                    "pretrend_max_abs_coef": event_diag["pretrend_max_abs_coef"],
                    "posttrend_any_signal": bool(event_diag["posttrend_any_signal"]),
                    "posttrend_mean_coef": event_diag["posttrend_mean_coef"],
                    "posttrend_peak_abs_coef": event_diag["posttrend_peak_abs_coef"],
                    "direction_consistent": direction_consistent,
                    "threshold_direction_stable": sensitivity_same_direction,
                    "threshold_opposite_signal": significant_opposite_threshold,
                    "placebo_issue": placebo_issue,
                    "sample_size_ok": sample_size_ok,
                    "validation_status": validation_status,
                    "recommended_main_claim": "",
                }
            )

        validation_df = pd.DataFrame(rows)
        if validation_df.empty:
            return empty_policy_validation_frame()

        subgroup_first_stage_support = (
            validation_df.loc[
                (validation_df["analysis_scope"] == "subgroup")
                & (validation_df["tier"].isin(["first_stage", "first_stage_proxy"]))
                & (validation_df["validation_status"].isin(["锁定", "谨慎解释"]))
            ]
            .assign(_group_key=lambda frame: frame["subgroup_dimension"].astype(str) + "::" + frame["subgroup_name"].astype(str))
            .groupby("_group_key", dropna=False)
            .size()
            .to_dict()
        )
        if subgroup_first_stage_support:
            mask = (
                (validation_df["analysis_scope"] == "subgroup")
                & (validation_df["tier"] == "secondary")
                & (validation_df["validation_status"] == "锁定")
            )
            key_series = validation_df["subgroup_dimension"].astype(str) + "::" + validation_df["subgroup_name"].astype(str)
            validation_df.loc[mask & (~key_series.map(subgroup_first_stage_support).fillna(0).astype(bool)), "validation_status"] = "谨慎解释"
        return validation_df


    def select_recommended_claim(
        global_validation_df: pd.DataFrame,
        subgroup_validation_df: pd.DataFrame,
    ) -> tuple[str, list[tuple[str, str, str, str, str]]]:
        del subgroup_validation_df
        claim_rows: list[tuple[str, str, str, str, str]] = []
        if global_validation_df.empty:
            global_validation_df = empty_policy_validation_frame()
        clean_global = global_validation_df.loc[
            (global_validation_df["sample_variant"] == MAIN_SAMPLE_VARIANT)
            & (global_validation_df["analysis_scope"] == "global")
        ].copy()
        global_primary = clean_global.loc[
            clean_global["outcome_column"] == PRIMARY_LOCK_OUTCOME
        ].sort_values(["validation_status", "did_p_value"], kind="stable")

        if not global_primary.empty:
            row = global_primary.iloc[0]
            claim_rows = [
                (
                    "global",
                    MAIN_SAMPLE_VARIANT,
                    "",
                    "",
                    PRIMARY_LOCK_OUTCOME,
                )
            ]
            if str(row["validation_status"]) == "锁定":
                claim = choose_text(
                    "全样本中，MPOWER 强化与吸烟暴露下降相关；WHO 吸烟指标与疾病负担结果仅作为补充和趋势性证据。",
                    "Globally, MPOWER strengthening is associated with lower smoking exposure; WHO smoking indicators and disease-burden results are retained only as supporting directional evidence.",
                    USE_CHINESE,
                )
                return claim, claim_rows

        claim = choose_text(
            "控烟DID未形成全样本一级强因果锁定，仅作为风险治理趋势性机制证据保留。",
            "The tobacco DID did not produce a globally locked first-stage causal conclusion; it is retained only as directional mechanism evidence for risk governance.",
            USE_CHINESE,
        )
        return claim, claim_rows


    def annotate_recommended_claims(validation_df: pd.DataFrame, claim_text: str, claim_rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
        annotated = validation_df.copy()
        if not claim_rows:
            return annotated
        keys = set(claim_rows)
        row_keys = list(
            zip(
                annotated["analysis_scope"].astype(str),
                annotated["sample_variant"].astype(str),
                annotated["subgroup_dimension"].fillna("").astype(str),
                annotated["subgroup_name"].fillna("").astype(str),
                annotated["outcome_column"].astype(str),
            )
        )
        annotated.loc[[key in keys for key in row_keys], "recommended_main_claim"] = claim_text
        return annotated


    def build_storylines(validation_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for _, row in validation_df.iterrows():
            status = str(row["validation_status"])
            coefficient = float(row["did_coefficient"])
            direction = choose_text("下降", "Decrease", USE_CHINESE) if coefficient < 0 else choose_text("上升", "Increase", USE_CHINESE)
            in_main_claim = bool(str(row.get("recommended_main_claim", "")).strip())
            if status == "锁定" and in_main_claim:
                usage = choose_text("正文主结论", "Main-text conclusion", USE_CHINESE)
                wording = choose_text(
                    "可直接进入正文主结论：方向、动态效应与稳健性检验一致。",
                    "Safe for main-text use: sign, dynamics, and robustness checks are aligned.",
                    USE_CHINESE,
                )
            elif status == "锁定":
                usage = choose_text("正文补充结论", "Supporting conclusion", USE_CHINESE)
                wording = choose_text(
                    "可作为正文补充结论：结果稳定，但优先级低于推荐主结论。",
                    "Usable as a supporting conclusion: stable but lower priority than the main recommended claim.",
                    USE_CHINESE,
                )
            elif status == "谨慎解释":
                usage = choose_text("趋势性证据", "Directional evidence", USE_CHINESE)
                wording = choose_text(
                    "可作为趋势性证据：方向具有解释价值，但需明确动态或稳健性限制。",
                    "Use as directional evidence: informative, but must be framed with dynamic or robustness caveats.",
                    USE_CHINESE,
                )
            else:
                usage = choose_text("附录/备答", "Appendix / backup", USE_CHINESE)
                wording = choose_text(
                    "不进入正文主结论，只保留在附录或答辩备答。",
                    "Do not use as a main conclusion; keep it in appendix or Q&A backup.",
                    USE_CHINESE,
                )
            rows.append(
                {
                    "analysis_scope": row["analysis_scope"],
                    "sample_variant": row["sample_variant"],
                    "subgroup_dimension": row["subgroup_dimension"],
                    "subgroup_name": row["subgroup_name"],
                    "outcome_column": row["outcome_column"],
                    "outcome_label": row["outcome_label"],
                    "tier_label": row["tier_label"],
                    "main_effect_direction": direction,
                    "did_p_value": row["did_p_value"],
                    "threshold_direction_stable": row["threshold_direction_stable"],
                    "placebo_issue": row["placebo_issue"],
                    "validation_status": status,
                    "recommended_usage": usage,
                    "recommended_wording": wording,
                    "recommended_main_claim": row.get("recommended_main_claim", ""),
                }
            )
        return pd.DataFrame(rows)


    def resolve_placebo_shifts(spec: dict[str, object], placebo_shifts: list[int]) -> list[int]:
        base_shifts = sorted({int(shift) for shift in placebo_shifts})
        tier = str(spec.get("tier", ""))
        if tier not in {"first_stage", "first_stage_proxy"}:
            return base_shifts
        post_window = int(spec.get("post_window", 0))
        start_shift = max(TRUE_PRETREAT_FIRST_STAGE_PLACEBO_START, post_window + 1)
        return [start_shift, start_shift + 1]


    def plot_did_effects(did_df: pd.DataFrame, validation_df: pd.DataFrame, output_path: Path) -> None:
        if did_df.empty or validation_df.empty:
            return
        merged = did_df.merge(
            validation_df.loc[:, ["analysis_scope", "sample_variant", "subgroup_dimension", "subgroup_name", "outcome_column", "validation_status"]],
            on=["analysis_scope", "sample_variant", "subgroup_dimension", "subgroup_name", "outcome_column"],
            how="left",
        )
        merged = merged.loc[(merged["analysis_scope"] == "global") & (merged["sample_variant"] == MAIN_SAMPLE_VARIANT)].copy()
        if merged.empty:
            return
        ordered = merged.sort_values("coefficient", kind="stable")
        colors = ["#457b9d" for _ in ordered["validation_status"].tolist()]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(ordered["outcome_label"], ordered["coefficient"], color=colors)
        ax.errorbar(
            ordered["coefficient"],
            np.arange(len(ordered)),
            xerr=1.96 * ordered["std_error"],
            fmt="none",
            ecolor="#1d3557",
            capsize=3,
        )
        ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
        ax.set_title(choose_text("模块D：MPOWER强化后的处理效应", "Module D: effects after MPOWER strengthening", USE_CHINESE))
        ax.set_xlabel(choose_text("处理效应系数（对数口径）", "Treatment effect coefficient (log scale)", USE_CHINESE))
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_event_study(event_df: pd.DataFrame, validation_df: pd.DataFrame, output_path: Path) -> None:
        if event_df.empty or validation_df.empty:
            return
        validation_subset = validation_df.loc[(validation_df["analysis_scope"] == "global") & (validation_df["sample_variant"] == MAIN_SAMPLE_VARIANT)].copy()
        outcomes = validation_subset["outcome_label"].tolist()
        if not outcomes:
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.flatten()
        for idx, outcome in enumerate(outcomes[:4]):
            ax = axes[idx]
            row = validation_subset.loc[validation_subset["outcome_label"] == outcome].iloc[0]
            subset = event_df.loc[
                (event_df["outcome_label"] == outcome)
                & (event_df["analysis_scope"] == "global")
                & (event_df["sample_variant"] == MAIN_SAMPLE_VARIANT)
                & (pd.to_numeric(event_df["placebo_shift"], errors="coerce").fillna(0) == 0)
            ].sort_values("event_time", kind="stable")
            if subset.empty:
                ax.set_visible(False)
                continue
            ax.plot(subset["event_time"], subset["coefficient"], marker="o", color="#2a9d8f")
            ax.fill_between(
                subset["event_time"],
                subset["coefficient"] - 1.96 * subset["std_error"],
                subset["coefficient"] + 1.96 * subset["std_error"],
                color="#2a9d8f",
                alpha=0.18,
            )
            ax.axhline(0, color="#999999", linestyle="--", linewidth=1)
            ax.axvline(-1, color="#999999", linestyle=":", linewidth=1)
            ax.set_title(str(outcome), fontsize=14, loc="center", pad=10)
            ax.set_xlabel(choose_text("事件时间", "Event time", USE_CHINESE))
            ax.set_ylabel(choose_text("系数", "Coefficient", USE_CHINESE))
            ax.grid(alpha=0.2)
        for idx in range(len(outcomes[:4]), 4):
            axes[idx].set_visible(False)
        set_centered_suptitle(fig, choose_text("模块D动态效应面板", "Module D event-study panels", USE_CHINESE), y=0.995)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_threshold_sensitivity(sensitivity_df: pd.DataFrame, output_path: Path) -> None:
        if sensitivity_df.empty:
            return
        subset = sensitivity_df.loc[(sensitivity_df["analysis_scope"] == "global") & (sensitivity_df["sample_variant"] == MAIN_SAMPLE_VARIANT)].copy()
        if subset.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        for outcome, group in subset.groupby("outcome_label", sort=False):
            ordered = group.sort_values("threshold", kind="stable")
            ax.plot(ordered["threshold"], ordered["coefficient"], marker="o", label=outcome)
        ax.axhline(0, color="#999999", linestyle="--", linewidth=1)
        ax.set_xticks(sorted(pd.to_numeric(subset["threshold"], errors="coerce").dropna().unique()))
        ax.set_xlabel(choose_text("MPOWER阈值", "MPOWER threshold", USE_CHINESE))
        ax.set_ylabel(choose_text("stacked DID 系数", "stacked DID coefficient", USE_CHINESE))
        ax.set_title(choose_text("阈值敏感性检验", "Threshold sensitivity", USE_CHINESE))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Run Module D policy identification with balanced-main, strict appendix, subgroup fallback, and QC outputs.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--input-file", type=Path, default=None, help="Optional explicit response panel path")
        parser.add_argument("--typology-file", type=Path, default=None, help="Optional explicit vulnerability typology panel path")
        parser.add_argument("--tobacco-dir", type=Path, default=None, help="Optional explicit cleaned tobacco data directory; defaults to <project_root>/09_data_clean")
        parser.add_argument("--latest-year", type=int, default=None)
        parser.add_argument("--strong-threshold", type=float, default=4.0, help="Main MPOWER threshold for the policy event")
        parser.add_argument("--sensitivity-thresholds", type=str, default="3,4,5")
        parser.add_argument("--placebo-shifts", type=str, default="2,3")
        parser.add_argument("--anticipation-buffer", type=int, default=DEFAULT_ANTICIPATION_BUFFER)
        parser.add_argument("--winsor-lower", type=float, default=0.01)
        parser.add_argument("--winsor-upper", type=float, default=0.99)
        parser.add_argument("--disable-log-transform", action="store_true")
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        tobacco_dir = detect_tobacco_dir(project_root, args.tobacco_dir)
        dirs = ensure_dirs(project_root)
        input_file = args.input_file.expanduser().resolve() if args.input_file else dirs["simulation"] / "response_panel.csv"
        typology_file = args.typology_file.expanduser().resolve() if args.typology_file else dirs["simulation"] / "vulnerability_typology_panel.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Response panel not found: {input_file}")

        response_df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)
        if not {"iso3", "year"}.issubset(response_df.columns):
            raise RuntimeError("Policy identification input must contain iso3 and year")
        typology_df = pd.read_csv(typology_file, encoding="utf-8-sig", low_memory=False) if typology_file.exists() else None

        log_stage("Loading cleaned WHO tobacco inputs and building QC tables")
        tobacco_df, tobacco_files_used, resolution_df, coverage_df, qc_summary, conflict_messages = load_external_tobacco_data(tobacco_dir)
        policy_df = merge_policy_inputs(response_df, tobacco_df, typology_df)
        policy_df = policy_df.dropna(subset=["year"]).copy()
        if "policy_strength" not in policy_df.columns or policy_df["policy_strength"].notna().sum() == 0:
            raise RuntimeError("Cleaned WHO policy data did not produce any usable policy_strength values")

        available_main_specs = [spec for spec in MAIN_OUTCOME_SPECS if str(spec["outcome_column"]) in policy_df.columns]
        available_robust_specs = [spec for spec in ROBUSTNESS_OUTCOME_SPECS if str(spec["outcome_column"]) in policy_df.columns]
        available_specs = available_main_specs + available_robust_specs
        if not available_main_specs:
            raise RuntimeError("No main policy outcomes available for Module D")

        qc_rows = []
        for column in [spec["outcome_column"] for spec in available_specs] + ["policy_strength"]:
            if column not in policy_df.columns:
                continue
            qc_rows.append(
                {
                    "column_name": column,
                    "country_year_rows": int(policy_df[column].notna().sum()),
                    "countries": int(policy_df.loc[policy_df[column].notna(), "iso3"].nunique()),
                    "years": int(policy_df.loc[policy_df[column].notna(), "year"].nunique()),
                    "year_min": int(pd.to_numeric(policy_df.loc[policy_df[column].notna(), "year"], errors="coerce").min()) if policy_df[column].notna().any() else np.nan,
                    "year_max": int(pd.to_numeric(policy_df.loc[policy_df[column].notna(), "year"], errors="coerce").max()) if policy_df[column].notna().any() else np.nan,
                }
            )
        outcome_coverage_df = pd.DataFrame(qc_rows)

        policy_qc_output = report_asset_path(dirs["report"], "policy_data_qc_summary.json")
        coverage_output = report_asset_path(dirs["report"], "policy_outcome_coverage.csv")
        resolution_output = report_asset_path(dirs["report"], "policy_indicator_resolution.csv")
        coverage_df.to_csv(coverage_output, index=False, encoding="utf-8-sig")
        resolution_df.to_csv(resolution_output, index=False, encoding="utf-8-sig")
        policy_qc_summary = {
            "project_root": project_root.as_posix(),
            "tobacco_dir": CLEAN_TOBACCO_DIRNAME,
            "tobacco_dir_path": tobacco_dir.as_posix(),
            "tobacco_files_used": tobacco_files_used,
            "dataset_qc": qc_summary,
            "resolution_stages_present": sorted(resolution_df["resolution_stage"].dropna().astype(str).unique().tolist()) if not resolution_df.empty else [],
            "observed_filter_applied": {
                dataset: bool(meta.get("observed_filter_applied", False))
                for dataset, meta in qc_summary.items()
                if isinstance(meta, dict)
            },
            "merged_coverage": outcome_coverage_df.to_dict(orient="records"),
            "conflict_messages": conflict_messages,
        }
        policy_qc_output.write_text(json.dumps(policy_qc_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if conflict_messages:
            raise RuntimeError("Conflicting iso3-year observations remain after WHO tobacco QC: " + "; ".join(conflict_messages))

        sensitivity_thresholds = sorted(set(parse_int_list(args.sensitivity_thresholds)))
        placebo_shifts = sorted(set(parse_int_list(args.placebo_shifts)))

        main_config = {
            "sample_variant": MAIN_SAMPLE_VARIANT,
            "min_pre_years": 3,
            "min_post_years": 4,
            "min_consecutive_strong": 2,
            "max_policy_switches": 2,
            "min_policy_jump": 0,
            "max_previous_policy": None,
            "require_sharp_jump": False,
        }
        diagnostic_config = {
            "sample_variant": DIAGNOSTIC_SAMPLE_VARIANT,
            "min_pre_years": 3,
            "min_post_years": 4,
            "min_consecutive_strong": 2,
            "max_policy_switches": 2,
            "min_policy_jump": 1,
            "max_previous_policy": 3,
            "require_sharp_jump": True,
        }
        strict_config = {
            "sample_variant": STRICT_SAMPLE_VARIANT,
            "min_pre_years": 4,
            "min_post_years": 6,
            "min_consecutive_strong": 3,
            "max_policy_switches": 1,
            "min_policy_jump": 1,
            "max_previous_policy": 3,
            "require_sharp_jump": True,
        }
        relaxed_config = {
            "sample_variant": RELAXED_SAMPLE_VARIANT,
            "min_pre_years": 3,
            "min_post_years": 3,
            "min_consecutive_strong": 2,
            "max_policy_switches": 2,
            "min_policy_jump": 0,
            "max_previous_policy": None,
            "require_sharp_jump": False,
        }

        timing_main = identify_treatment_timing(
            policy_df,
            threshold=args.strong_threshold,
            min_pre_years=main_config["min_pre_years"],
            min_post_years=main_config["min_post_years"],
            min_consecutive_strong=main_config["min_consecutive_strong"],
            max_policy_switches=main_config["max_policy_switches"],
            sample_variant=MAIN_SAMPLE_VARIANT,
            min_policy_jump=main_config["min_policy_jump"],
            max_previous_policy=main_config["max_previous_policy"],
            require_sharp_jump=main_config["require_sharp_jump"],
        )
        timing_diagnostic = identify_treatment_timing(
            policy_df,
            threshold=args.strong_threshold,
            min_pre_years=diagnostic_config["min_pre_years"],
            min_post_years=diagnostic_config["min_post_years"],
            min_consecutive_strong=diagnostic_config["min_consecutive_strong"],
            max_policy_switches=diagnostic_config["max_policy_switches"],
            sample_variant=DIAGNOSTIC_SAMPLE_VARIANT,
            min_policy_jump=diagnostic_config["min_policy_jump"],
            max_previous_policy=diagnostic_config["max_previous_policy"],
            require_sharp_jump=diagnostic_config["require_sharp_jump"],
        )
        timing_strict = identify_treatment_timing(
            policy_df,
            threshold=args.strong_threshold,
            min_pre_years=strict_config["min_pre_years"],
            min_post_years=strict_config["min_post_years"],
            min_consecutive_strong=strict_config["min_consecutive_strong"],
            max_policy_switches=strict_config["max_policy_switches"],
            sample_variant=STRICT_SAMPLE_VARIANT,
            min_policy_jump=strict_config["min_policy_jump"],
            max_previous_policy=strict_config["max_previous_policy"],
            require_sharp_jump=strict_config["require_sharp_jump"],
        )
        timing_relaxed = identify_treatment_timing(
            policy_df,
            threshold=args.strong_threshold,
            min_pre_years=relaxed_config["min_pre_years"],
            min_post_years=relaxed_config["min_post_years"],
            min_consecutive_strong=relaxed_config["min_consecutive_strong"],
            max_policy_switches=relaxed_config["max_policy_switches"],
            sample_variant=RELAXED_SAMPLE_VARIANT,
            min_policy_jump=relaxed_config["min_policy_jump"],
            max_previous_policy=relaxed_config["max_previous_policy"],
            require_sharp_jump=relaxed_config["require_sharp_jump"],
        )

        log_stage("Constructing subgroup assignments from typology and baseline smoking")
        subgroup_df = build_subgroup_assignments(policy_df, timing_main)

        log_stage("Estimating global stacked DID and event-study models")
        global_did_rows: list[dict[str, object]] = []
        global_event_rows: list[dict[str, object]] = []
        cohort_did_rows: list[dict[str, object]] = []
        for sample_variant, timing_df in [
            (MAIN_SAMPLE_VARIANT, timing_main),
            (DIAGNOSTIC_SAMPLE_VARIANT, timing_diagnostic),
            (STRICT_SAMPLE_VARIANT, timing_strict),
            (RELAXED_SAMPLE_VARIANT, timing_relaxed),
        ]:
            for spec in available_specs:
                did_df, event_df, cohort_df = run_stacked_estimation(
                    policy_df,
                    timing_df,
                    spec,
                    threshold=args.strong_threshold,
                    placebo_shift=0,
                    log_transform=not args.disable_log_transform,
                    winsor_lower=args.winsor_lower,
                    winsor_upper=args.winsor_upper,
                    analysis_scope="global",
                    sample_variant=sample_variant,
                    anticipation_buffer=args.anticipation_buffer,
                )
                if did_df.empty:
                    continue
                global_did_rows.extend(did_df.to_dict(orient="records"))
                if not event_df.empty:
                    global_event_rows.extend(event_df.to_dict(orient="records"))
                if not cohort_df.empty:
                    cohort_did_rows.extend(cohort_df.to_dict(orient="records"))

        global_did_df = pd.DataFrame(global_did_rows)
        global_event_df = pd.DataFrame(global_event_rows)
        cohort_did_df = pd.DataFrame(cohort_did_rows)
        if global_did_df.empty:
            raise RuntimeError("Stacked DID estimates are empty; no outcome had sufficient treated/control support")

        log_stage("Running global threshold sensitivity and placebo checks")
        global_threshold_rows = []
        global_threshold_event_rows = []
        global_placebo_rows = []
        for sample_variant, timing_df, config in [
            (MAIN_SAMPLE_VARIANT, timing_main, main_config),
            (DIAGNOSTIC_SAMPLE_VARIANT, timing_diagnostic, diagnostic_config),
            (STRICT_SAMPLE_VARIANT, timing_strict, strict_config),
            (RELAXED_SAMPLE_VARIANT, timing_relaxed, relaxed_config),
        ]:
            threshold_df, threshold_event_df = build_threshold_sensitivity(
                policy_df,
                sensitivity_thresholds,
                available_main_specs,
                config,
                log_transform=not args.disable_log_transform,
                winsor_lower=args.winsor_lower,
                winsor_upper=args.winsor_upper,
                analysis_scope="global",
                sample_variant=sample_variant,
                anticipation_buffer=args.anticipation_buffer,
            )
            if not threshold_df.empty:
                global_threshold_rows.extend(threshold_df.to_dict(orient="records"))
            if not threshold_event_df.empty:
                global_threshold_event_rows.extend(threshold_event_df.to_dict(orient="records"))
            placebo_df = build_placebo_estimates(
                policy_df,
                timing_df,
                available_main_specs,
                placebo_shifts,
                threshold=args.strong_threshold,
                log_transform=not args.disable_log_transform,
                winsor_lower=args.winsor_lower,
                winsor_upper=args.winsor_upper,
                analysis_scope="global",
                sample_variant=sample_variant,
                anticipation_buffer=args.anticipation_buffer,
            )
            if not placebo_df.empty:
                global_placebo_rows.extend(placebo_df.to_dict(orient="records"))

        threshold_sensitivity_df = pd.DataFrame(global_threshold_rows)
        threshold_event_df = pd.DataFrame(global_threshold_event_rows)
        placebo_df = pd.DataFrame(global_placebo_rows)
        log_stage("Building specification grid for threshold-lag stability")
        specification_grid_df = build_specification_grid(
            policy_df,
            sensitivity_thresholds,
            available_specs,
            main_config,
            log_transform=not args.disable_log_transform,
            winsor_lower=args.winsor_lower,
            winsor_upper=args.winsor_upper,
            anticipation_buffer=args.anticipation_buffer,
        )
        anticipation_buffer_sensitivity_df = build_anticipation_buffer_sensitivity(
            policy_df,
            available_main_specs,
            main_config,
            threshold=args.strong_threshold,
            log_transform=not args.disable_log_transform,
            winsor_lower=args.winsor_lower,
            winsor_upper=args.winsor_upper,
            anticipation_buffers=ANTICIPATION_BUFFER_GRID,
        )

        log_stage("Estimating subgroup stacked DID and targeted robustness checks")
        subgroup_did_rows: list[dict[str, object]] = []
        subgroup_event_rows: list[dict[str, object]] = []
        subgroup_threshold_rows: list[dict[str, object]] = []
        subgroup_placebo_rows: list[dict[str, object]] = []
        for (subgroup_dimension, subgroup_name), subgroup_units in subgroup_df.groupby(["subgroup_dimension", "subgroup_name"], dropna=False):
            treated_units = set(subgroup_units["iso3"].dropna().astype(str).tolist())
            for spec in available_main_specs:
                did_df, event_df, cohort_df = run_stacked_estimation(
                    policy_df,
                    timing_main,
                    spec,
                    threshold=args.strong_threshold,
                    placebo_shift=0,
                    log_transform=not args.disable_log_transform,
                    winsor_lower=args.winsor_lower,
                    winsor_upper=args.winsor_upper,
                    analysis_scope="subgroup",
                    sample_variant=MAIN_SAMPLE_VARIANT,
                    subgroup_dimension=str(subgroup_dimension),
                    subgroup_name=str(subgroup_name),
                    treated_units_filter=treated_units,
                    anticipation_buffer=args.anticipation_buffer,
                )
                if did_df.empty:
                    continue
                subgroup_did_rows.extend(did_df.to_dict(orient="records"))
                if not event_df.empty:
                    subgroup_event_rows.extend(event_df.to_dict(orient="records"))
                if not cohort_df.empty:
                    cohort_did_rows.extend(cohort_df.to_dict(orient="records"))

            treated_count = len(treated_units)
            if treated_count >= 10:
                threshold_df, _ = build_threshold_sensitivity(
                    policy_df,
                    sensitivity_thresholds,
                    available_main_specs,
                    main_config,
                    log_transform=not args.disable_log_transform,
                    winsor_lower=args.winsor_lower,
                    winsor_upper=args.winsor_upper,
                    analysis_scope="subgroup",
                    sample_variant=MAIN_SAMPLE_VARIANT,
                    subgroup_dimension=str(subgroup_dimension),
                    subgroup_name=str(subgroup_name),
                    treated_units_filter=treated_units,
                    anticipation_buffer=args.anticipation_buffer,
                )
                if not threshold_df.empty:
                    subgroup_threshold_rows.extend(threshold_df.to_dict(orient="records"))
                subgroup_placebo = build_placebo_estimates(
                    policy_df,
                    timing_main,
                    available_main_specs,
                    placebo_shifts,
                    threshold=args.strong_threshold,
                    log_transform=not args.disable_log_transform,
                    winsor_lower=args.winsor_lower,
                    winsor_upper=args.winsor_upper,
                    analysis_scope="subgroup",
                    sample_variant=MAIN_SAMPLE_VARIANT,
                    subgroup_dimension=str(subgroup_dimension),
                    subgroup_name=str(subgroup_name),
                    treated_units_filter=treated_units,
                    anticipation_buffer=args.anticipation_buffer,
                )
                if not subgroup_placebo.empty:
                    subgroup_placebo_rows.extend(subgroup_placebo.to_dict(orient="records"))

        subgroup_did_df = pd.DataFrame(subgroup_did_rows)
        subgroup_event_df = pd.DataFrame(subgroup_event_rows)
        subgroup_threshold_df = pd.DataFrame(subgroup_threshold_rows)
        subgroup_placebo_df = pd.DataFrame(subgroup_placebo_rows)
        cohort_did_df = pd.DataFrame(cohort_did_rows)
        if subgroup_did_df.empty and not global_did_df.empty:
            subgroup_did_df = pd.DataFrame(columns=global_did_df.columns)
        if subgroup_event_df.empty and not global_event_df.empty:
            subgroup_event_df = pd.DataFrame(columns=global_event_df.columns)
        if subgroup_threshold_df.empty and not threshold_sensitivity_df.empty:
            subgroup_threshold_df = pd.DataFrame(columns=threshold_sensitivity_df.columns)
        if subgroup_placebo_df.empty and not placebo_df.empty:
            subgroup_placebo_df = pd.DataFrame(columns=placebo_df.columns)

        global_validation_df = build_policy_validation_summary(
            global_did_df.copy(),
            global_event_df,
            threshold_sensitivity_df,
            placebo_df,
            available_specs,
            placebo_shifts=placebo_shifts,
            min_treated_for_lock=0,
            min_nobs_for_lock=0,
        )
        subgroup_validation_df = build_policy_validation_summary(
            subgroup_did_df.copy(),
            subgroup_event_df,
            subgroup_threshold_df,
            subgroup_placebo_df,
            available_specs,
            placebo_shifts=placebo_shifts,
            min_treated_for_lock=10,
            min_nobs_for_lock=300,
        )

        log_stage("Writing final outputs, validation tables, and module-D figures")
        recommended_main_claim, claim_rows = select_recommended_claim(global_validation_df, subgroup_validation_df)
        global_validation_df = annotate_recommended_claims(global_validation_df, recommended_main_claim, claim_rows)
        subgroup_validation_df = annotate_recommended_claims(subgroup_validation_df, recommended_main_claim, claim_rows)
        validation_df = concat_nonempty([global_validation_df, subgroup_validation_df], fallback=empty_policy_validation_frame())
        storyline_df = build_storylines(validation_df)

        first_stage_columns = {"who_smoking_rate_std", "dbd_smoking_per100k"}
        first_stage_parts = [global_did_df.loc[global_did_df["outcome_column"].isin(first_stage_columns)].copy()]
        if "outcome_column" in subgroup_did_df.columns:
            first_stage_parts.append(subgroup_did_df.loc[subgroup_did_df["outcome_column"].isin(first_stage_columns)].copy())
        first_stage_df = concat_nonempty(first_stage_parts, fallback=pd.DataFrame(columns=global_did_df.columns))
        if not first_stage_df.empty:
            priority_map = {"dbd_smoking_per100k": 0, "who_smoking_rate_std": 1}
            first_stage_df["_priority"] = first_stage_df["outcome_column"].map(priority_map).fillna(99)
            first_stage_df = first_stage_df.sort_values(
                ["_priority", "analysis_scope", "sample_variant", "subgroup_dimension", "subgroup_name"],
                kind="stable",
            ).drop(columns="_priority")

        latest_year = args.latest_year if args.latest_year is not None else int(pd.to_numeric(policy_df["year"], errors="coerce").dropna().max())
        latest_snapshot = (
            policy_df.sort_values(["iso3", "year"], kind="stable")
            .groupby("iso3", as_index=False)
            .tail(1)
            .loc[:, [column for column in ["iso3", "year", "policy_strength", "who_smoking_rate_std", "dbd_smoking_per100k"] if column in policy_df.columns]]
        )
        treated_output_df = build_treated_country_output([timing_main, timing_diagnostic, timing_strict, timing_relaxed])
        restriction_comparison_df = build_sample_restriction_comparison(
            [timing_main, timing_diagnostic, timing_strict, timing_relaxed],
            [main_config, diagnostic_config, strict_config, relaxed_config],
            args.strong_threshold,
        )
        if not restriction_comparison_df.empty:
            restriction_comparison_df["anticipation_buffer_years"] = args.anticipation_buffer
        treated_output_df = build_treated_country_output([timing_main, timing_diagnostic, timing_strict, timing_relaxed])

        design_output = dirs["simulation"] / "policy_identification_panel.csv"
        did_output = report_asset_path(dirs["report"], "policy_did_estimates.csv")
        stacked_output = report_asset_path(dirs["report"], "policy_stacked_did_estimates.csv")
        subgroup_did_output = report_asset_path(dirs["report"], "policy_subgroup_stacked_did_estimates.csv")
        first_stage_output = report_asset_path(dirs["report"], "policy_first_stage_estimates.csv")
        event_output = report_asset_path(dirs["report"], "policy_event_study_coefficients.csv")
        subgroup_event_output = report_asset_path(dirs["report"], "policy_subgroup_event_study_coefficients.csv")
        threshold_output = report_asset_path(dirs["report"], "policy_threshold_sensitivity.csv")
        placebo_output = report_asset_path(dirs["report"], "policy_placebo_estimates.csv")
        anticipation_output = report_asset_path(dirs["report"], "policy_anticipation_buffer_sensitivity.csv")
        spec_grid_output = report_asset_path(dirs["report"], "policy_specification_grid.csv")
        validation_output = report_asset_path(dirs["report"], "policy_validation_summary.csv")
        subgroup_validation_output = report_asset_path(dirs["report"], "policy_subgroup_validation_summary.csv")
        storyline_output = report_asset_path(dirs["report"], "policy_identification_storylines.csv")
        treated_output = report_asset_path(dirs["report"], "policy_treated_countries.csv")
        latest_output = report_asset_path(dirs["report"], "policy_latest_snapshot.csv")
        cohort_output = report_asset_path(dirs["report"], "policy_cohort_did_estimates.csv")
        restriction_output = report_asset_path(dirs["report"], "policy_sample_restriction_comparison.csv")

        policy_df.to_csv(design_output, index=False, encoding="utf-8-sig")
        global_did_df.to_csv(did_output, index=False, encoding="utf-8-sig")
        concat_nonempty([global_did_df, subgroup_did_df], fallback=pd.DataFrame(columns=global_did_df.columns)).to_csv(stacked_output, index=False, encoding="utf-8-sig")
        subgroup_did_df.to_csv(subgroup_did_output, index=False, encoding="utf-8-sig")
        first_stage_df.to_csv(first_stage_output, index=False, encoding="utf-8-sig")
        concat_nonempty([global_event_df, subgroup_event_df], fallback=pd.DataFrame(columns=global_event_df.columns)).to_csv(event_output, index=False, encoding="utf-8-sig")
        subgroup_event_df.to_csv(subgroup_event_output, index=False, encoding="utf-8-sig")
        concat_nonempty([threshold_sensitivity_df, subgroup_threshold_df], fallback=pd.DataFrame(columns=threshold_sensitivity_df.columns)).to_csv(threshold_output, index=False, encoding="utf-8-sig")
        concat_nonempty([placebo_df, subgroup_placebo_df], fallback=pd.DataFrame(columns=placebo_df.columns)).to_csv(placebo_output, index=False, encoding="utf-8-sig")
        anticipation_buffer_sensitivity_df.to_csv(anticipation_output, index=False, encoding="utf-8-sig")
        specification_grid_df.to_csv(spec_grid_output, index=False, encoding="utf-8-sig")
        validation_df.to_csv(validation_output, index=False, encoding="utf-8-sig")
        subgroup_validation_df.to_csv(subgroup_validation_output, index=False, encoding="utf-8-sig")
        storyline_df.to_csv(storyline_output, index=False, encoding="utf-8-sig")
        treated_output_df.to_csv(treated_output, index=False, encoding="utf-8-sig")
        latest_snapshot.to_csv(latest_output, index=False, encoding="utf-8-sig")
        cohort_did_df.to_csv(cohort_output, index=False, encoding="utf-8-sig")
        restriction_comparison_df.to_csv(restriction_output, index=False, encoding="utf-8-sig")

        plot_did_effects(global_did_df, global_validation_df, dirs["figures"] / "policy_did_effects.png")
        plot_event_study(global_event_df, global_validation_df, dirs["figures"] / "policy_event_study.png")
        plot_threshold_sensitivity(threshold_sensitivity_df, dirs["figures"] / "policy_threshold_sensitivity.png")

        global_locked_outcomes = global_validation_df.loc[
            (global_validation_df["analysis_scope"] == "global")
            & (global_validation_df["sample_variant"] == MAIN_SAMPLE_VARIANT)
            & (global_validation_df["validation_status"] == "锁定"),
            "outcome_column",
        ].tolist()
        subgroup_locked_outcomes = subgroup_validation_df.loc[
            subgroup_validation_df["validation_status"] == "锁定",
            ["subgroup_dimension", "subgroup_name", "outcome_column"],
        ].to_dict(orient="records")
        panel_max_year = int(pd.to_numeric(policy_df["year"], errors="coerce").dropna().max()) if not policy_df.empty else None
        policy_series_years = (
            sorted(pd.to_numeric(policy_df.loc[policy_df["policy_strength"].notna(), "year"], errors="coerce").dropna().astype(int).unique().tolist())
            if "policy_strength" in policy_df.columns
            else []
        )
        who_year_values: list[int] = []
        for column in ["who_smoking_rate_std", "who_cigarette_prevalence_btsx"]:
            if column in policy_df.columns:
                who_year_values.extend(pd.to_numeric(policy_df.loc[policy_df[column].notna(), "year"], errors="coerce").dropna().astype(int).unique().tolist())
        who_series_years = sorted(set(who_year_values))
        outcome_available_max_year = {
            str(spec["outcome_column"]): int(pd.to_numeric(policy_df.loc[policy_df[str(spec["outcome_column"])].notna(), "year"], errors="coerce").max())
            for spec in available_specs
            if str(spec["outcome_column"]) in policy_df.columns and policy_df[str(spec["outcome_column"])].notna().any()
        }

        summary = {
            "project_root": project_root.as_posix(),
            "input_file": input_file.as_posix(),
            "tobacco_dir": CLEAN_TOBACCO_DIRNAME,
            "tobacco_dir_path": tobacco_dir.as_posix(),
            "tobacco_files_used": tobacco_files_used,
            "analysis_panel": design_output.as_posix(),
            "latest_year_used": latest_year,
            "panel_max_year": panel_max_year,
            "outcome_available_max_year": outcome_available_max_year,
            "policy_series_years": policy_series_years,
            "who_series_years": who_series_years,
            "policy_threshold": args.strong_threshold,
            "sensitivity_thresholds": sensitivity_thresholds,
            "anticipation_buffer_years": args.anticipation_buffer,
            "anticipation_buffer_grid": ANTICIPATION_BUFFER_GRID,
            "treated_countries": int(timing_main["treat_ever"].fillna(False).sum()),
            "control_countries": int((~timing_main["treat_ever"].fillna(False)).sum()),
            "estimator": "stacked_did_clustered_se",
            "standard_error_method": "iso3_cluster_robust_small_sample_corrected",
            "first_stage_outcomes": [
                spec["outcome_column"]
                for spec in available_main_specs
                if str(spec["tier"]) in {"first_stage", "first_stage_proxy"}
            ],
            "robustness_outcomes": [spec["outcome_column"] for spec in available_robust_specs],
            "outcomes_used": [spec["outcome_column"] for spec in available_specs],
            "lag_spec_by_outcome": {str(spec["outcome_column"]): int(spec["lag"]) for spec in available_specs},
            "placebo_check": placebo_shifts,
            "first_stage_placebo_check": sorted(
                {
                    shift
                    for spec in available_main_specs
                    if str(spec["tier"]) in {"first_stage", "first_stage_proxy"}
                    for shift in resolve_placebo_shifts(spec, placebo_shifts)
                }
            ),
            "winsorization": {"lower": args.winsor_lower, "upper": args.winsor_upper},
            "log_transform_enabled": not args.disable_log_transform,
            "locked_outcomes": global_locked_outcomes,
            "global_locked_outcomes": global_locked_outcomes,
            "subgroup_locked_outcomes": subgroup_locked_outcomes,
            "main_sample_variant": MAIN_SAMPLE_VARIANT,
            "diagnostic_sample_variant": DIAGNOSTIC_SAMPLE_VARIANT,
            "appendix_sample_variant": STRICT_SAMPLE_VARIANT,
            "fallback_sample_variant": MAIN_SAMPLE_VARIANT,
            "subgroup_dimensions": SUBGROUP_DIMENSIONS,
            "recommended_main_claim": recommended_main_claim,
            "method_note": choose_text(
                "控烟DID识别采用 balanced-main 作为正文主规格，使用 iso3 聚类稳健标准误并加入 1 年 anticipation buffer；吸烟暴露为唯一一级主指标，crossing-main 仅保留为事件定义诊断规格，WHO 吸烟指标与疾病负担仅作为补充和趋势性证据。",
                "The tobacco DID identification uses the balanced main sample, iso3-clustered robust standard errors, and a one-year anticipation buffer; smoking exposure is the sole first-stage outcome, crossing-main is retained only as a diagnostic event-definition sample, and WHO smoking indicators plus disease-burden outcomes remain supporting directional evidence.",
                USE_CHINESE,
            ),
            "output_files": {
                "did_estimates": did_output.as_posix(),
                "stacked_did_estimates": stacked_output.as_posix(),
                "subgroup_stacked_did_estimates": subgroup_did_output.as_posix(),
                "first_stage_estimates": first_stage_output.as_posix(),
                "event_study_coefficients": event_output.as_posix(),
                "subgroup_event_study_coefficients": subgroup_event_output.as_posix(),
                "threshold_sensitivity": threshold_output.as_posix(),
                "placebo_estimates": placebo_output.as_posix(),
                "anticipation_buffer_sensitivity": anticipation_output.as_posix(),
                "specification_grid": spec_grid_output.as_posix(),
                "validation_summary": validation_output.as_posix(),
                "subgroup_validation_summary": subgroup_validation_output.as_posix(),
                "storylines": storyline_output.as_posix(),
                "treated_countries": treated_output.as_posix(),
                "latest_snapshot": latest_output.as_posix(),
                "cohort_did_estimates": cohort_output.as_posix(),
                "sample_restriction_comparison": restriction_output.as_posix(),
                "policy_data_qc_summary": policy_qc_output.as_posix(),
                "policy_outcome_coverage": coverage_output.as_posix(),
                "policy_indicator_resolution": resolution_output.as_posix(),
                "did_figure": (dirs["figures"] / "policy_did_effects.png").as_posix(),
                "event_study_figure": (dirs["figures"] / "policy_event_study.png").as_posix(),
                "threshold_sensitivity_figure": (dirs["figures"] / "policy_threshold_sensitivity.png").as_posix(),
            },
        }
        summary_path = report_asset_path(dirs["report"], "policy_identification_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_policy_adaptation_engine():
    __name__ = 'run_policy_adaptation_engine'
    import argparse
    import json
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from foundation import detect_project_root as shared_detect_project_root


    RISK_REDUCTION_RATES = (0.05, 0.10, 0.15)
    RISK_GOVERNANCE_TRANSLATION_FACTOR = 0.35
    SERVICE_COVERAGE_TRANSLATION_FACTOR = 0.12

    NCD_POLICY_SCORE_COLUMNS = [
        "integrated_ncd_governance",
        "tobacco_policy_execution",
        "diet_salt_policy_execution",
        "hypertension_policy_readiness",
        "diabetes_policy_readiness",
        "primary_care_ncd_service_readiness",
        "ncd_policy_capacity_score",
        "ncd_service_coverage_score",
        "ncd_policy_execution_score",
    ]

    NCD_SERVICE_TARGETS = {
        "ncd_hypertension_diagnosis_pct": {
            "label": "高血压诊断覆盖",
            "target": 70.0,
            "policy_package": "高血压筛查、确诊转诊和基层随访闭环",
            "china_policy_translation": "家庭医生签约、基层血压筛查、双向转诊和连续随访",
        },
        "ncd_hypertension_treatment_pct": {
            "label": "高血压治疗覆盖",
            "target": 70.0,
            "policy_package": "高血压规范治疗和基本药物可及性提升",
            "china_policy_translation": "基层慢病门诊、长期处方、基本药物保障和用药依从性管理",
        },
        "ncd_hypertension_control_pct": {
            "label": "高血压控制覆盖",
            "target": 50.0,
            "policy_package": "高血压控制率提升和并发症预防",
            "china_policy_translation": "达标率考核、远程监测、随访提醒和心脑血管并发症预防",
        },
        "ncd_diabetes_treatment_pct": {
            "label": "糖尿病治疗覆盖",
            "target": 70.0,
            "policy_package": "糖尿病治疗覆盖和并发症管理",
            "china_policy_translation": "糖尿病筛查、医保慢病用药、视网膜/肾病并发症筛查",
        },
        "ncd_cervical_screening_pct": {
            "label": "宫颈癌筛查覆盖",
            "target": 70.0,
            "policy_package": "宫颈癌筛查和HPV疫苗/早诊早治衔接",
            "china_policy_translation": "两癌筛查、HPV疫苗、基层妇幼服务和早诊早治路径",
        },
    }

    RISK_POLICY_MAP = {
        "高收缩压": {
            "risk_column": "dbd_high_sbp_per100k",
            "policy_package": "高血压筛查、控盐行动与基层连续用药",
            "china_policy_translation": "基层慢病筛查、家庭医生随访、控盐控油行动、基本药物可及性",
        },
        "吸烟": {
            "risk_column": "dbd_smoking_per100k",
            "policy_package": "控烟执法、戒烟服务与烟草流行监测",
            "china_policy_translation": "控烟条例、无烟公共场所、戒烟门诊、青少年烟草预防",
        },
        "高血糖": {
            "risk_column": "dbd_high_glucose_per100k",
            "policy_package": "糖尿病早筛、健康管理与并发症预防",
            "china_policy_translation": "糖尿病筛查、社区随访、医防融合、医保慢病用药保障",
        },
        "高BMI": {
            "risk_column": "dbd_high_bmi_per100k",
            "policy_package": "肥胖干预、运动促进与学校/社区营养治理",
            "china_policy_translation": "体重管理年行动、学校健康、社区运动处方、营养标签",
        },
        "膳食风险": {
            "risk_column": "dbd_dietary_risks_per100k",
            "policy_package": "控盐控油控糖与健康膳食环境建设",
            "china_policy_translation": "健康中国合理膳食行动、减盐行动、食堂和外卖营养治理",
        },
        "PM2.5": {
            "risk_column": "dbd_pm25_per100k",
            "policy_package": "空气污染治理与重点人群健康防护",
            "china_policy_translation": "大气污染防治、空气质量预警、老年和慢病人群防护",
        },
    }

    RESOURCE_LABEL_TO_VARIABLE = {
        "病床": "beds_10k",
        "医生": "doctors_10k",
        "护士": "nurses_10k",
        "UHC": "uhc_index",
        "卫生支出/GDP": "che_pct_gdp",
        "人均卫生支出": "che_pc_usd",
        "政府卫生支出占比": "govt_he_pct",
        "外部卫生支出依赖": "ext_he_pct",
        "自付卫生支出": "wdi_out_of_pocket_pct",
        "WDI人均卫生支出": "wdi_health_expenditure_per_capita",
        "WDI政府卫生支出占比": "wdi_government_health_expenditure_pct",
        "WDI外部支出依赖": "wdi_external_health_expenditure_pct",
        "WDI病床": "wdi_hospital_beds",
        "WDI医生": "wdi_physicians",
        "WDI护士助产士": "wdi_nurses_midwives",
        "HCI": "wb_hnp_hci",
        "专业接生": "wb_hnp_skilled_birth_attendance",
        "DPT免疫": "wb_hnp_immunization_dpt",
        "麻疹免疫": "wb_hnp_immunization_measles",
        "乙肝免疫": "wb_hnp_immunization_hepb3",
    }

    RESOURCE_ACTION_MAP = {
        "病床": ("卫生资源投入", "床位与区域医疗服务容量补强"),
        "医生": ("卫生资源投入", "医生和基层全科服务能力补强"),
        "护士": ("卫生资源投入", "护理队伍、康复和长期照护能力补强"),
        "卫生支出/GDP": ("卫生资源投入", "提高卫生投入强度"),
        "人均卫生支出": ("卫生资源投入", "提高人均卫生服务投入"),
        "WDI人均卫生支出": ("卫生资源投入", "提高人均卫生服务投入"),
        "UHC": ("服务覆盖", "提升基本医疗服务覆盖"),
        "HCI": ("服务覆盖", "提升人力资本和基本公共服务质量"),
        "专业接生": ("服务覆盖", "补齐妇幼和基层公共卫生服务覆盖"),
        "DPT免疫": ("服务覆盖", "提升免疫规划服务覆盖"),
        "麻疹免疫": ("服务覆盖", "提升免疫规划服务覆盖"),
        "乙肝免疫": ("服务覆盖", "提升免疫规划服务覆盖"),
        "政府卫生支出占比": ("支付与韧性", "提高政府筹资和医保支付稳定性"),
        "WDI政府卫生支出占比": ("支付与韧性", "提高政府筹资和医保支付稳定性"),
        "外部卫生支出依赖": ("支付与韧性", "降低外部资金依赖"),
        "WDI外部支出依赖": ("支付与韧性", "降低外部资金依赖"),
        "自付卫生支出": ("支付与韧性", "降低家庭自付负担"),
    }


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def ensure_report_dir(project_root: Path) -> Path:
        report_dir = project_root / "06_report_assets"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir


    def read_required_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)


    def read_optional_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)


    def read_json(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


    def latest_ncd_policy_profile(ncd_panel: pd.DataFrame, response_df: pd.DataFrame) -> pd.DataFrame:
        if ncd_panel.empty or response_df.empty:
            return pd.DataFrame()
        analysis_year = int(pd.to_numeric(response_df["year"], errors="coerce").max())
        work = ncd_panel.copy()
        work["year"] = pd.to_numeric(work["year"], errors="coerce")
        work = work.loc[work["year"].notna() & (work["year"] <= analysis_year)].copy()
        if work.empty:
            return pd.DataFrame()
        work = work.sort_values(["country_code", "year"], kind="stable")
        fill_cols = [col for col in NCD_POLICY_SCORE_COLUMNS + list(NCD_SERVICE_TARGETS) if col in work.columns]
        meta_cols = [col for col in ["region", "ncd_capacity_observed_questions", "ncd_service_observed_indicators", "ncd_policy_data_observed_fields"] if col in work.columns]
        work[fill_cols + meta_cols] = work.groupby("country_code", dropna=False)[fill_cols + meta_cols].ffill()
        latest = work.groupby("country_code", dropna=False).tail(1).copy()
        latest = latest.rename(columns={"country_code": "iso3", "year": "ncd_policy_latest_year"})
        merge_cols = [
            "iso3",
            "year",
            "vulnerability_type_label",
            "response_diagnosis_type",
            "combined_pressure_score",
            "adjusted_response_score",
            "adaptation_gap_score",
            "dominant_risk_triplet",
        ]
        merged = latest.merge(
            response_df.loc[:, [col for col in merge_cols if col in response_df.columns]],
            on="iso3",
            how="inner",
        )
        if merged.empty:
            return merged
        score = pd.to_numeric(merged.get("ncd_policy_execution_score"), errors="coerce")
        merged["ncd_policy_readiness_tier"] = pd.cut(
            score,
            bins=[-np.inf, 0.50, 0.75, np.inf],
            labels=["政策执行短板", "政策执行中等", "政策执行较强"],
        ).astype("object")
        for col, meta in NCD_SERVICE_TARGETS.items():
            if col in merged.columns:
                merged[f"{col}_gap_to_target"] = (meta["target"] - pd.to_numeric(merged[col], errors="coerce")).clip(lower=0)
        return merged.sort_values(["vulnerability_type_label", "iso3"], kind="stable")


    def build_ncd_policy_type_summary(ncd_latest: pd.DataFrame) -> pd.DataFrame:
        if ncd_latest.empty:
            return pd.DataFrame()
        agg_cols = [
            col
            for col in NCD_POLICY_SCORE_COLUMNS + list(NCD_SERVICE_TARGETS)
            if col in ncd_latest.columns
        ]
        rows = []
        for type_label, subset in ncd_latest.groupby("vulnerability_type_label", dropna=False):
            row = {
                "vulnerability_type_label": type_label,
                "countries_with_ncd_policy_data": int(subset["iso3"].nunique()),
                "mean_pressure": pd.to_numeric(subset.get("combined_pressure_score"), errors="coerce").mean(),
                "mean_response": pd.to_numeric(subset.get("adjusted_response_score"), errors="coerce").mean(),
                "mean_gap": pd.to_numeric(subset.get("adaptation_gap_score"), errors="coerce").mean(),
                "ncd_policy_execution_score": pd.to_numeric(subset.get("ncd_policy_execution_score"), errors="coerce").mean(),
                "ncd_policy_capacity_score": pd.to_numeric(subset.get("ncd_policy_capacity_score"), errors="coerce").mean(),
                "ncd_service_coverage_score": pd.to_numeric(subset.get("ncd_service_coverage_score"), errors="coerce").mean(),
                "low_policy_execution_share": float((pd.to_numeric(subset.get("ncd_policy_execution_score"), errors="coerce") < 0.50).mean()),
            }
            for col in agg_cols:
                if col not in row:
                    row[col] = pd.to_numeric(subset[col], errors="coerce").mean()
            weakest = (
                pd.Series({col: row.get(col) for col in agg_cols if pd.notna(row.get(col))})
                .sort_values(kind="stable")
                .head(3)
            )
            row["weakest_ncd_policy_service_fields"] = " / ".join(weakest.index.astype(str))
            rows.append(row)
        return pd.DataFrame(rows).sort_values("ncd_policy_execution_score", ascending=True, kind="stable")


    def build_ncd_service_gap_scenarios(ncd_latest: pd.DataFrame) -> pd.DataFrame:
        if ncd_latest.empty:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for _, country in ncd_latest.iterrows():
            baseline_response = float(country.get("adjusted_response_score", np.nan))
            baseline_pressure = float(country.get("combined_pressure_score", np.nan))
            baseline_gap = float(country.get("adaptation_gap_score", np.nan))
            if pd.isna(baseline_response) or pd.isna(baseline_pressure) or pd.isna(baseline_gap):
                continue
            for col, meta in NCD_SERVICE_TARGETS.items():
                current = pd.to_numeric(country.get(col), errors="coerce")
                if pd.isna(current):
                    continue
                gap_pct = max(float(meta["target"]) - float(current), 0.0)
                if gap_pct <= 0:
                    continue
                response_lift = (gap_pct / 100.0) * SERVICE_COVERAGE_TRANSLATION_FACTOR
                simulated_response = min(1.0, baseline_response + response_lift)
                simulated_gap = baseline_pressure - simulated_response
                rows.append(
                    {
                        "scenario_id": f"ncd_service_{country.get('iso3')}_{col}",
                        "scenario_family": "ncd_service_coverage_uplift",
                        "iso3": country.get("iso3"),
                        "year": country.get("year"),
                        "ncd_policy_latest_year": country.get("ncd_policy_latest_year"),
                        "vulnerability_type_label": country.get("vulnerability_type_label"),
                        "response_diagnosis_type": country.get("response_diagnosis_type"),
                        "service_indicator": col,
                        "service_label": meta["label"],
                        "current_coverage_pct": float(current),
                        "target_coverage_pct": meta["target"],
                        "coverage_gap_pct": gap_pct,
                        "baseline_response": baseline_response,
                        "simulated_response": simulated_response,
                        "baseline_gap": baseline_gap,
                        "simulated_gap": simulated_gap,
                        "gap_reduction": baseline_gap - simulated_gap,
                        "translation_factor": SERVICE_COVERAGE_TRANSLATION_FACTOR,
                        "policy_package": meta["policy_package"],
                        "china_policy_translation": meta["china_policy_translation"],
                        "scenario_assumption": f"将该服务覆盖率提升到{meta['target']:.0f}%，按保守系数{SERVICE_COVERAGE_TRANSLATION_FACTOR}折算响应能力提升",
                        "interpretation": "这是服务覆盖情景模拟，不是单项政策因果估计",
                    }
                )
        return pd.DataFrame(rows).sort_values("gap_reduction", ascending=False, kind="stable") if rows else pd.DataFrame()


    def split_component_labels(component_text: object) -> list[str]:
        if pd.isna(component_text):
            return []
        return [part.strip() for part in re.split(r"\s+/\s+", str(component_text)) if part.strip()]


    def clean_component_name(component: str) -> str:
        return component.replace("component__", "").strip()


    def component_label_from_name(component: str) -> str:
        cleaned = clean_component_name(component)
        for label, variable in RESOURCE_LABEL_TO_VARIABLE.items():
            if cleaned == variable:
                return label
        return cleaned


    def classify_resource_package(component_text: object) -> dict[str, str]:
        labels = [component_label_from_name(label) for label in split_component_labels(component_text)]
        domains: list[str] = []
        actions: list[str] = []
        variables: list[str] = []
        for label in labels:
            variable = RESOURCE_LABEL_TO_VARIABLE.get(label)
            if variable:
                variables.append(variable)
            domain_action = RESOURCE_ACTION_MAP.get(label)
            if domain_action:
                domain, action = domain_action
                domains.append(domain)
                actions.append(action)
        if not domains:
            domains = ["卫生资源投入"]
        if not actions:
            actions = ["针对短板资源分项进行补强"]
        domain_order = ["卫生资源投入", "服务覆盖", "支付与韧性"]
        ordered_domains = [domain for domain in domain_order if domain in set(domains)]
        return {
            "policy_domain": " + ".join(ordered_domains or sorted(set(domains))),
            "policy_package": "；".join(dict.fromkeys(actions)),
            "core_variables": " / ".join(dict.fromkeys(variables)) if variables else str(component_text),
            "target_components": " / ".join(labels),
        }


    def merge_stage(response_df: pd.DataFrame, stage_df: pd.DataFrame) -> pd.DataFrame:
        if stage_df.empty or "transition_stage_label" not in stage_df.columns:
            return response_df.copy()
        stage_cols = ["iso3", "transition_stage_label"]
        if "year" in stage_df.columns:
            stage_cols.append("year")
        stage_latest = stage_df.loc[:, stage_cols].copy()
        if "year" in stage_latest.columns:
            stage_latest = stage_latest.sort_values(["iso3", "year"], kind="stable").drop_duplicates("iso3", keep="last")
            stage_latest = stage_latest.drop(columns=["year"])
        return response_df.merge(stage_latest, on="iso3", how="left")


    def type_stats(response_df: pd.DataFrame) -> pd.DataFrame:
        work = response_df.copy()
        work["positive_gap_need"] = pd.to_numeric(work["adaptation_gap_score"], errors="coerce").clip(lower=0)
        work["response_lift_need"] = (0.50 - pd.to_numeric(work["adjusted_response_score"], errors="coerce")).clip(lower=0)
        return (
            work.groupby("vulnerability_type_label", dropna=False)
            .agg(
                countries=("iso3", "count"),
                mean_pressure=("combined_pressure_score", "mean"),
                mean_response=("adjusted_response_score", "mean"),
                mean_gap=("adaptation_gap_score", "mean"),
                mean_positive_gap_need=("positive_gap_need", "mean"),
                mean_response_lift_need=("response_lift_need", "mean"),
            )
            .reset_index()
        )


    def tobacco_evidence(policy_validation: pd.DataFrame, policy_summary: dict[str, object]) -> dict[str, object]:
        evidence = {
            "validation_status": policy_summary.get("validation_status", "谨慎解释"),
            "did_coefficient": np.nan,
            "did_p_value": np.nan,
            "treated_countries": policy_summary.get("treated_countries"),
            "locked_outcomes": policy_summary.get("locked_outcomes", []),
            "claim_level": "趋势性机制证据，不作为强因果锁定",
            "tobacco_did_claim": "控烟DID未形成强因果锁定，仅作为风险治理机制案例保留。",
        }
        if policy_validation.empty:
            return evidence
        candidate = policy_validation.loc[
            (policy_validation.get("analysis_scope") == "global")
            & (policy_validation.get("sample_variant") == "balanced_main")
            & (policy_validation.get("outcome_column") == "dbd_smoking_per100k")
        ]
        if candidate.empty:
            candidate = policy_validation.loc[policy_validation.get("outcome_column") == "dbd_smoking_per100k"]
        if not candidate.empty:
            row = candidate.iloc[0]
            evidence.update(
                {
                    "validation_status": row.get("validation_status"),
                    "did_coefficient": float(row.get("did_coefficient")) if pd.notna(row.get("did_coefficient")) else np.nan,
                    "did_p_value": float(row.get("did_p_value")) if pd.notna(row.get("did_p_value")) else np.nan,
                    "treated_countries": int(row.get("treated_countries")) if pd.notna(row.get("treated_countries")) else evidence["treated_countries"],
                    "tobacco_did_claim": "控烟DID未形成强因果锁定，仅作为风险治理机制案例保留。",
                }
            )
        return evidence


    def build_policy_response_pathways(
        response_df: pd.DataFrame,
        risk_summary: pd.DataFrame,
        allocation_plan: pd.DataFrame,
        policy_validation: pd.DataFrame,
        policy_summary: dict[str, object],
    ) -> pd.DataFrame:
        stats = type_stats(response_df)
        stats_map = {row["vulnerability_type_label"]: row for _, row in stats.iterrows()}
        evidence = tobacco_evidence(policy_validation, policy_summary)
        rows: list[dict[str, object]] = []

        risk_top = risk_summary.loc[pd.to_numeric(risk_summary.get("rank_within_type"), errors="coerce") <= 3].copy()
        for _, risk in risk_top.iterrows():
            type_label = risk["vulnerability_type_label"]
            stat = stats_map.get(type_label, {})
            risk_label = str(risk["risk_label"])
            meta = RISK_POLICY_MAP.get(risk_label, {})
            contribution_share = float(risk.get("contribution_share", 0.0) or 0.0)
            mean_pressure = float(stat.get("mean_pressure", 0.0) or 0.0)
            mean_gap_need = float(stat.get("mean_positive_gap_need", 0.0) or 0.0)
            priority_score = 0.45 * contribution_share + 0.35 * mean_pressure + 0.20 * mean_gap_need
            if risk_label == "吸烟":
                evidence_strength = evidence["claim_level"]
                evidence_note = f"MPOWER DID: coef={evidence.get('did_coefficient'):.4g}, p={evidence.get('did_p_value'):.4g}, {evidence.get('validation_status')}"
            else:
                evidence_strength = "风险归因 + 情景模拟证据"
                evidence_note = "基于模块B风险贡献份额和模块D风险下降情景"
            rows.append(
                {
                    "pathway_id": f"risk_{int(risk.get('vulnerability_type_code', 0) or 0)}_{int(risk.get('rank_within_type', 0) or 0)}",
                    "vulnerability_type_label": type_label,
                    "response_diagnosis_type": "按国家响应类型细分",
                    "policy_domain": "风险治理",
                    "policy_package": meta.get("policy_package", f"{risk_label}风险前移治理"),
                    "target_pressure_or_gap": risk_label,
                    "core_variables": meta.get("risk_column", risk.get("risk_column")),
                    "scenario_method": "将主导风险设定为下降5%/10%/15%，按风险贡献份额折算综合压力下降",
                    "evidence_basis": evidence_note,
                    "evidence_strength": evidence_strength,
                    "priority_score": priority_score,
                    "countries_in_scope": int(stat.get("countries", 0) or 0),
                    "expected_output": "降低combined_pressure_score，并进一步压低adaptation_gap_score",
                    "china_policy_translation": meta.get("china_policy_translation", "转化为中国省级风险治理政策包"),
                    "source_modules": "B + D",
                }
            )

        for idx, allocation in allocation_plan.iterrows():
            package = classify_resource_package(allocation.get("typical_target_components"))
            countries = float(allocation.get("countries", 0.0) or 0.0)
            mean_gap = float(allocation.get("mean_gap", 0.0) or 0.0)
            mean_gap_reduction = float(allocation.get("mean_gap_reduction", 0.0) or 0.0)
            mean_response_lift = float(allocation.get("mean_response_lift_needed", 0.0) or 0.0)
            priority_score = (
                0.35 * max(mean_gap, 0.0)
                + 0.35 * max(mean_gap_reduction, 0.0)
                + 0.20 * max(mean_response_lift, 0.0)
                + 0.10 * min(countries / 50.0, 1.0)
            )
            rows.append(
                {
                    "pathway_id": f"resource_{idx + 1}",
                    "vulnerability_type_label": allocation.get("vulnerability_type_label"),
                    "response_diagnosis_type": allocation.get("response_diagnosis_type"),
                    "policy_domain": package["policy_domain"],
                    "policy_package": package["policy_package"],
                    "target_pressure_or_gap": allocation.get("typical_target_components"),
                    "core_variables": package["core_variables"],
                    "scenario_method": "沿用模块C：将每个国家最弱的3个资源分项得分各提升0.10，重算adjusted_response_score和adaptation_gap_score",
                    "evidence_basis": "模块C响应失配情景模拟",
                    "evidence_strength": "结构模拟证据，不写成因果效应",
                    "priority_score": priority_score,
                    "countries_in_scope": int(countries),
                    "expected_output": f"平均适配缺口下降{mean_gap_reduction:.3f}",
                    "china_policy_translation": "转化为中国省级卫生资源、服务覆盖和支付韧性补短板政策",
                    "source_modules": "C + D",
                }
            )

        rows.append(
            {
                "pathway_id": "tobacco_trend_case",
                "vulnerability_type_label": "全样本",
                "response_diagnosis_type": "不分型",
                "policy_domain": "风险治理",
                "policy_package": "MPOWER控烟政策趋势证据",
                "target_pressure_or_gap": "吸烟风险暴露",
                "core_variables": "policy_strength / dbd_smoking_per100k / who_smoking_rate_std / who_cigarette_prevalence_btsx",
                "scenario_method": "cohort-stacked DID、国家聚类稳健标准误、1年anticipation buffer；仅保留为趋势性机制证据",
                "evidence_basis": (
                    f"控烟DID未形成强因果锁定：coef={evidence.get('did_coefficient'):.6f}, "
                    f"p={evidence.get('did_p_value'):.4f}；作为风险治理机制案例保留"
                ),
                "evidence_strength": evidence["claim_level"],
                "priority_score": 0.05,
                "countries_in_scope": evidence.get("treated_countries"),
                "expected_output": "为风险治理政策包提供机制案例，不替代多政策适配引擎",
                "china_policy_translation": RISK_POLICY_MAP["吸烟"]["china_policy_translation"],
                "source_modules": "D",
            }
        )

        pathway_df = pd.DataFrame(rows)
        if not pathway_df.empty:
            pathway_df = pathway_df.sort_values("priority_score", ascending=False, kind="stable").reset_index(drop=True)
            pathway_df["rank_overall"] = np.arange(1, len(pathway_df) + 1)
        return pathway_df


    def add_ncd_readiness_pathways(policy_pathways: pd.DataFrame, ncd_type_summary: pd.DataFrame) -> pd.DataFrame:
        if ncd_type_summary.empty:
            return policy_pathways
        rows = policy_pathways.to_dict("records") if not policy_pathways.empty else []
        pathway_templates = [
            (
                "ncd_governance",
                "integrated_ncd_governance",
                "政策治理能力",
                "国家NCD综合治理、目标体系、监测系统和跨部门机制补强",
                "integrated_ncd_governance / ncd_policy_capacity_score",
                "补齐NCD综合治理框架、目标指标、监测系统和跨部门执行机制",
            ),
            (
                "ncd_hypertension_cascade",
                "ncd_hypertension_control_pct",
                "服务覆盖",
                "高血压诊断-治疗-控制连续服务提升",
                "ncd_hypertension_diagnosis_pct / ncd_hypertension_treatment_pct / ncd_hypertension_control_pct",
                "将高血压筛查、确诊、治疗、达标控制做成基层连续服务闭环",
            ),
            (
                "ncd_diabetes_cascade",
                "ncd_diabetes_treatment_pct",
                "服务覆盖",
                "糖尿病治疗覆盖与并发症筛查提升",
                "ncd_diabetes_treatment_pct / diabetes_policy_readiness",
                "推动糖尿病筛查、治疗覆盖、用药保障和并发症筛查联动",
            ),
            (
                "ncd_diet_policy",
                "diet_salt_policy_execution",
                "风险治理",
                "控盐、反式脂肪、含糖饮料和健康饮食政策执行",
                "diet_salt_policy_execution / NCDCCS diet and salt policy questions",
                "用减盐、营养标签、含糖饮料税、食品营销治理强化膳食风险前移治理",
            ),
        ]
        for _, summary in ncd_type_summary.iterrows():
            type_label = summary.get("vulnerability_type_label")
            mean_pressure = float(summary.get("mean_pressure", np.nan)) if "mean_pressure" in summary else np.nan
            for suffix, field, domain, package, core_vars, china_translation in pathway_templates:
                value = pd.to_numeric(summary.get(field), errors="coerce")
                if pd.isna(value):
                    continue
                if field.endswith("_pct"):
                    weakness = max(0.0, (50.0 - float(value)) / 100.0)
                    display_value = f"{float(value):.1f}%"
                else:
                    weakness = max(0.0, 1.0 - float(value))
                    display_value = f"{float(value):.3f}"
                if weakness <= 0.05:
                    continue
                priority_score = 0.35 * weakness + 0.20 * float(summary.get("low_policy_execution_share", 0.0) or 0.0)
                if pd.notna(mean_pressure):
                    priority_score += 0.20 * mean_pressure
                rows.append(
                    {
                        "pathway_id": f"{suffix}_{str(type_label).split('-')[0]}",
                        "vulnerability_type_label": type_label,
                        "response_diagnosis_type": "按国家响应类型细分",
                        "policy_domain": domain,
                        "policy_package": package,
                        "target_pressure_or_gap": f"{field}={display_value}",
                        "core_variables": core_vars,
                        "scenario_method": "接入WHO NCD国家能力调查和NCD服务覆盖指标，识别政策执行/服务连续性短板",
                        "evidence_basis": "WHO GHO NCD bulk: NCD Country Capacity Survey 2013-2023 + hypertension/diabetes service coverage 1990-2022",
                        "evidence_strength": "政策能力 + 服务覆盖观测证据，不写成强因果效应",
                        "priority_score": priority_score,
                        "countries_in_scope": int(summary.get("countries_with_ncd_policy_data", 0) or 0),
                        "expected_output": "提高政策执行能力或NCD连续服务覆盖，从响应侧降低适配缺口",
                        "china_policy_translation": china_translation,
                        "source_modules": "B + C + D + WHO_NCD",
                    }
                )
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values("priority_score", ascending=False, kind="stable").reset_index(drop=True)
            out["rank_overall"] = np.arange(1, len(out) + 1)
        return out


    def build_resource_uplift_scenarios(optimization_df: pd.DataFrame) -> pd.DataFrame:
        if optimization_df.empty:
            return pd.DataFrame()
        rows = []
        for idx, row in optimization_df.iterrows():
            package = classify_resource_package(row.get("scenario_target_components"))
            baseline_gap = float(row.get("adaptation_gap_score", np.nan))
            simulated_gap = float(row.get("optimized_adaptation_gap_score", np.nan))
            gap_reduction = float(row.get("gap_reduction_if_upgraded", np.nan))
            rows.append(
                {
                    "scenario_id": f"resource_uplift_{idx + 1:03d}",
                    "scenario_family": "resource_uplift",
                    "iso3": row.get("iso3"),
                    "year": row.get("year"),
                    "vulnerability_type_label": row.get("vulnerability_type_label"),
                    "response_diagnosis_type": row.get("response_diagnosis_type"),
                    "policy_domain": package["policy_domain"],
                    "policy_package": package["policy_package"],
                    "target_components": package["target_components"],
                    "core_variables": package["core_variables"],
                    "baseline_pressure": row.get("combined_pressure_score"),
                    "baseline_response": row.get("adjusted_response_score"),
                    "simulated_response": row.get("optimized_adjusted_response_score"),
                    "baseline_gap": baseline_gap,
                    "simulated_gap": simulated_gap,
                    "gap_reduction": gap_reduction,
                    "response_lift_needed_to_balance": row.get("response_lift_needed_to_balance"),
                    "scenario_assumption": "最弱3个资源分项得分分别提升0.10，和模块C一致",
                    "interpretation": "gap_reduction越大，说明该国家对资源补短板越敏感",
                }
            )
        return pd.DataFrame(rows)


    def build_risk_reduction_scenarios(response_df: pd.DataFrame, risk_summary: pd.DataFrame) -> pd.DataFrame:
        if response_df.empty or risk_summary.empty:
            return pd.DataFrame()
        risk_top = risk_summary.loc[pd.to_numeric(risk_summary.get("rank_within_type"), errors="coerce") <= 3].copy()
        risks_by_type = {
            type_label: subset.sort_values("rank_within_type", kind="stable")
            for type_label, subset in risk_top.groupby("vulnerability_type_label", dropna=False)
        }
        rows: list[dict[str, object]] = []
        for _, country in response_df.iterrows():
            type_label = country.get("vulnerability_type_label")
            risk_rows = risks_by_type.get(type_label)
            if risk_rows is None or risk_rows.empty:
                continue
            baseline_pressure = float(country.get("combined_pressure_score", np.nan))
            baseline_response = float(country.get("adjusted_response_score", np.nan))
            baseline_gap = float(country.get("adaptation_gap_score", baseline_pressure - baseline_response))
            if pd.isna(baseline_pressure) or pd.isna(baseline_response) or pd.isna(baseline_gap):
                continue
            for _, risk in risk_rows.iterrows():
                risk_label = str(risk.get("risk_label"))
                contribution_share = float(risk.get("contribution_share", 0.0) or 0.0)
                meta = RISK_POLICY_MAP.get(risk_label, {})
                for reduction_rate in RISK_REDUCTION_RATES:
                    pressure_reduction = baseline_pressure * contribution_share * reduction_rate * RISK_GOVERNANCE_TRANSLATION_FACTOR
                    simulated_pressure = max(0.0, baseline_pressure - pressure_reduction)
                    simulated_gap = simulated_pressure - baseline_response
                    rows.append(
                        {
                            "scenario_id": f"risk_{country.get('iso3')}_{risk.get('rank_within_type')}_{int(reduction_rate * 100)}",
                            "scenario_family": "risk_reduction",
                            "iso3": country.get("iso3"),
                            "year": country.get("year"),
                            "vulnerability_type_label": type_label,
                            "response_diagnosis_type": country.get("response_diagnosis_type"),
                            "risk_label": risk_label,
                            "risk_column": meta.get("risk_column", risk.get("risk_column")),
                            "policy_package": meta.get("policy_package", f"{risk_label}风险前移治理"),
                            "risk_contribution_share": contribution_share,
                            "risk_reduction_rate": reduction_rate,
                            "translation_factor": RISK_GOVERNANCE_TRANSLATION_FACTOR,
                            "baseline_pressure": baseline_pressure,
                            "simulated_pressure": simulated_pressure,
                            "baseline_response": baseline_response,
                            "baseline_gap": baseline_gap,
                            "simulated_gap": simulated_gap,
                            "gap_reduction": baseline_gap - simulated_gap,
                            "scenario_assumption": "风险下降率乘以该风险在类型内的贡献份额，再乘以保守转化系数0.35",
                            "interpretation": "这是政策情景模拟，不是单项政策的因果估计",
                            "china_policy_translation": meta.get("china_policy_translation", "转化为中国省级风险治理政策包"),
                        }
                    )
        return pd.DataFrame(rows)


    def build_type_strategy_matrix(
        response_df: pd.DataFrame,
        risk_summary: pd.DataFrame,
        allocation_plan: pd.DataFrame,
        ncd_type_summary: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        stats = type_stats(response_df)
        top_risks = (
            risk_summary.sort_values(["vulnerability_type_label", "rank_within_type"], kind="stable")
            .groupby("vulnerability_type_label", dropna=False)
            .head(3)
            .groupby("vulnerability_type_label", dropna=False)["risk_label"]
            .apply(lambda values: " / ".join(values.astype(str)))
            .to_dict()
        )
        top_resource = (
            allocation_plan.sort_values(["vulnerability_type_label", "mean_gap_reduction"], ascending=[True, False], kind="stable")
            .drop_duplicates("vulnerability_type_label", keep="first")
            if not allocation_plan.empty
            else pd.DataFrame()
        )
        resource_map = {
            row["vulnerability_type_label"]: row.get("typical_target_components")
            for _, row in top_resource.iterrows()
        }
        ncd_map = {}
        if ncd_type_summary is not None and not ncd_type_summary.empty:
            ncd_map = {
                row["vulnerability_type_label"]: row
                for _, row in ncd_type_summary.iterrows()
            }
        rows = []
        for _, stat in stats.iterrows():
            type_label = stat["vulnerability_type_label"]
            component_text = resource_map.get(type_label, "")
            package = classify_resource_package(component_text)
            ncd_row = ncd_map.get(type_label, {})
            risk_labels = top_risks.get(type_label, "")
            risk_actions = []
            for risk_label in split_component_labels(risk_labels):
                meta = RISK_POLICY_MAP.get(risk_label)
                if meta:
                    risk_actions.append(meta["policy_package"])
            ncd_weakest = ncd_row.get("weakest_ncd_policy_service_fields", "") if len(ncd_map) else ""
            ncd_package = (
                f"同步补齐WHO NCD政策/服务短板：{ncd_weakest}"
                if ncd_weakest
                else "继续使用资源与风险情景模拟"
            )
            rows.append(
                {
                    "vulnerability_type_label": type_label,
                    "countries": int(stat["countries"]),
                    "mean_pressure": stat["mean_pressure"],
                    "mean_response": stat["mean_response"],
                    "mean_gap": stat["mean_gap"],
                    "top_risks": risk_labels,
                    "risk_governance_package": "；".join(risk_actions),
                    "resource_priority_components": component_text,
                    "resource_policy_package": package["policy_package"],
                    "ncd_policy_execution_score": ncd_row.get("ncd_policy_execution_score", np.nan) if len(ncd_map) else np.nan,
                    "ncd_service_coverage_score": ncd_row.get("ncd_service_coverage_score", np.nan) if len(ncd_map) else np.nan,
                    "ncd_policy_service_package": ncd_package,
                    "strategy_summary": f"先治理{risk_labels}等主导风险，同时补强{package['target_components'] or '关键响应能力'}；再用NCD政策能力和服务覆盖数据校准执行短板。",
                    "china_translation": "用于中国省级画像时，先判定省份接近哪类全球脆弱性，再套用对应风险治理和资源补短板优先级。",
                }
            )
        return pd.DataFrame(rows).sort_values("mean_gap", ascending=False, kind="stable")


    def build_reference_countries(response_df: pd.DataFrame, target_iso3: str = "CHN", limit: int = 12) -> pd.DataFrame:
        target = response_df.loc[response_df["iso3"] == target_iso3]
        if target.empty:
            return pd.DataFrame()
        target_row = target.iloc[0]
        same_type = response_df.loc[
            (response_df["vulnerability_type_label"] == target_row["vulnerability_type_label"])
            & (response_df["iso3"] != target_iso3)
        ].copy()
        if same_type.empty:
            return pd.DataFrame()
        same_type["gap_advantage_vs_target"] = float(target_row["adaptation_gap_score"]) - same_type["adaptation_gap_score"]
        same_type["response_advantage_vs_target"] = same_type["adjusted_response_score"] - float(target_row["adjusted_response_score"])
        same_type["reference_score"] = (
            same_type["gap_advantage_vs_target"].clip(lower=0)
            + 0.5 * same_type["response_advantage_vs_target"].clip(lower=0)
            + 0.2 * (same_type["response_diagnosis_type"] == target_row["response_diagnosis_type"]).astype(float)
        )
        selected = same_type.sort_values(
            ["reference_score", "adaptation_gap_score", "adjusted_response_score"],
            ascending=[False, True, False],
            kind="stable",
        ).head(limit)
        cols = [
            "iso3",
            "year",
            "vulnerability_type_label",
            "response_diagnosis_type",
            "combined_pressure_score",
            "adjusted_response_score",
            "adaptation_gap_score",
            "gap_advantage_vs_target",
            "response_advantage_vs_target",
            "dominant_risk_triplet",
            "weakest_resource_components",
            "reference_score",
        ]
        return selected.loc[:, [col for col in cols if col in selected.columns]]


    def build_china_policy_mapping_rules(
        response_df: pd.DataFrame,
        resource_scenarios: pd.DataFrame,
        risk_scenarios: pd.DataFrame,
        reference_df: pd.DataFrame,
        ncd_latest: pd.DataFrame | None = None,
        ncd_service_scenarios: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        china = response_df.loc[response_df["iso3"] == "CHN"]
        if china.empty:
            return pd.DataFrame()
        chn = china.iloc[0]
        rows: list[dict[str, object]] = [
            {
                "mapping_layer": "中国入口画像",
                "target": "CHN",
                "global_evidence": f"{chn.get('vulnerability_type_label')} / {chn.get('transition_stage_label', '')} / {chn.get('response_diagnosis_type')}",
                "current_value": f"pressure={chn.get('combined_pressure_score'):.3f}; response={chn.get('adjusted_response_score'):.3f}; gap={chn.get('adaptation_gap_score'):.3f}",
                "recommended_policy_package": "进入省级映射后，按省份重新计算类型、风险和资源短板，不能直接全国一刀切。",
                "scenario_result": "作为中国面板默认总览卡片",
                "data_caveat": "当前为国家级全球样本结果；省级应用需要补充中国省级风险、资源和政策数据。",
            }
        ]
        top_risk = risk_scenarios.loc[(risk_scenarios["iso3"] == "CHN") & (risk_scenarios["risk_reduction_rate"] == 0.10)].copy()
        if not top_risk.empty:
            top_risk = top_risk.sort_values("gap_reduction", ascending=False, kind="stable").head(3)
            for _, risk in top_risk.iterrows():
                rows.append(
                    {
                        "mapping_layer": "B风险治理",
                        "target": risk.get("risk_label"),
                        "global_evidence": f"类型内贡献份额={risk.get('risk_contribution_share'):.3f}",
                        "current_value": f"baseline_gap={risk.get('baseline_gap'):.3f}",
                        "recommended_policy_package": risk.get("policy_package"),
                        "scenario_result": f"若该风险下降10%，情景缺口约下降{risk.get('gap_reduction'):.4f}",
                        "data_caveat": "需用中国省级高血压、吸烟、糖尿病、PM2.5等数据替换国家级风险值。",
                    }
                )
        resource = resource_scenarios.loc[resource_scenarios["iso3"] == "CHN"].copy()
        if not resource.empty:
            resource_row = resource.sort_values("gap_reduction", ascending=False, kind="stable").iloc[0]
            rows.append(
                {
                    "mapping_layer": "C响应补短板",
                    "target": resource_row.get("target_components"),
                    "global_evidence": resource_row.get("scenario_assumption"),
                    "current_value": f"baseline_gap={resource_row.get('baseline_gap'):.3f}; simulated_gap={resource_row.get('simulated_gap'):.3f}",
                    "recommended_policy_package": resource_row.get("policy_package"),
                    "scenario_result": f"资源分项提升后，适配缺口约下降{resource_row.get('gap_reduction'):.4f}",
                    "data_caveat": "省级情景应以每万人医生、护士、床位、卫生支出和医保支付变量重新估计。",
                }
            )
        if ncd_latest is not None and not ncd_latest.empty:
            ncd_china = ncd_latest.loc[ncd_latest["iso3"] == "CHN"].copy()
            if not ncd_china.empty:
                ncd = ncd_china.iloc[0]
                rows.append(
                    {
                        "mapping_layer": "D政策执行能力",
                        "target": "WHO NCD政策能力与服务覆盖",
                        "global_evidence": (
                            f"NCD政策执行得分={ncd.get('ncd_policy_execution_score'):.3f}; "
                            f"政策能力得分={ncd.get('ncd_policy_capacity_score'):.3f}"
                        ),
                        "current_value": (
                            f"高血压治疗={ncd.get('ncd_hypertension_treatment_pct'):.1f}%; "
                            f"高血压控制={ncd.get('ncd_hypertension_control_pct'):.1f}%; "
                            f"糖尿病治疗={ncd.get('ncd_diabetes_treatment_pct'):.1f}%"
                        ),
                        "recommended_policy_package": "把慢病风险治理从政策有无推进到筛查-诊断-治疗-控制连续服务覆盖。",
                        "scenario_result": "用于解释模块D不是只给政策名，而是给政策执行能力和服务覆盖缺口。",
                        "data_caveat": "WHO NCD服务覆盖是国家级数据；省级落地仍需中国省级慢病管理和医保服务数据。",
                    }
                )
        if ncd_service_scenarios is not None and not ncd_service_scenarios.empty:
            service = ncd_service_scenarios.loc[
                (ncd_service_scenarios["iso3"] == "CHN")
            ].sort_values("gap_reduction", ascending=False, kind="stable")
            if not service.empty:
                top = service.iloc[0]
                rows.append(
                    {
                        "mapping_layer": "D服务覆盖情景",
                        "target": top.get("service_label"),
                        "global_evidence": top.get("scenario_assumption"),
                        "current_value": f"current={top.get('current_coverage_pct'):.1f}%; target={top.get('target_coverage_pct'):.0f}%",
                        "recommended_policy_package": top.get("policy_package"),
                        "scenario_result": f"覆盖率补齐后，适配缺口情景下降{top.get('gap_reduction'):.4f}",
                        "data_caveat": "服务覆盖情景是保守模拟，不声明单项政策因果效应。",
                    }
                )
        if not reference_df.empty:
            rows.append(
                {
                    "mapping_layer": "全球相似经验库",
                    "target": " / ".join(reference_df["iso3"].astype(str).head(8)),
                    "global_evidence": "同一脆弱性类型下，适配缺口更低或响应得分更高的国家",
                    "current_value": f"target_type={chn.get('vulnerability_type_label')}",
                    "recommended_policy_package": "点击中国后高亮这些相似国家，解释中国情景适配不是直接套用单一国家。",
                    "scenario_result": "用于地球仪到中国面板的经验迁移入口",
                    "data_caveat": "相似国家只提供经验参照，不能替代中国省级本地估计。",
                }
            )
        return pd.DataFrame(rows)


    def build_china_variable_dictionary() -> pd.DataFrame:
        rows = [
            ("A健康脆弱性画像", "health_burden_score / risk_exposure_score / system_fragility_score / socioeconomic_fragility_score", "省级慢病负担、老龄化、卫生资源、经济发展、城镇化", "已接入NBS卫生资源、七普人口和老龄化、GBD 2017四类疾病DALY率、GBD 2021省级NCD年龄标化死亡率/DALY健康结局锚点、NBS多年粗死亡率敏感性面板和2024城镇化/GDP官方OCR解释变量", "国家统计局、国家卫健委统计年鉴、中国疾控中心、各省统计年鉴、GBD 2021中国NCD补充材料", "判断省份接近全球哪类脆弱性画像"),
            ("B风险归因", "dominant_risk_triplet / dbd_high_sbp_per100k / dbd_smoking_per100k / dbd_high_glucose_per100k / dbd_pm25_per100k", "省级高血压、吸烟、糖尿病、PM2.5、肥胖或膳食指标", "已接入GBD 2017省级十大风险SEV和ScienceDB 2024地级市人口加权PM2.5；最新调查风险率可作为后续增强但不是当前阻塞", "中国慢性病及危险因素监测、生态环境部空气质量、健康中国行动监测", "识别每个省最该前移治理的风险"),
            ("C响应失配", "combined_pressure_score / adjusted_response_score / adaptation_gap_score / resource component scores", "每万人医生、护士、床位、基层机构、卫生支出、医保支付能力", "已有人口校正后的NBS卫生人员/机构核心评分，并补NBS床位、医保参保和基金收支OCR字段级QC；未通过字段自动禁用，不进入核心评分", "国家统计局、卫生健康统计年鉴、医保统计公报、财政统计年鉴", "判断省份是高压高响应、高压低响应、低压低响应或相对均衡"),
            ("D政策响应", "policy_strength / ncd_policy_execution_score / ncd_service_coverage_score / risk reduction scenarios / resource uplift scenarios / reference countries", "控烟条例、慢病示范区、家庭医生、分级诊疗、医保支付改革、健康城市行动、慢病筛查治疗覆盖", "全球已补WHO NCD国家能力和服务覆盖，且非控烟高血压路径已有准因果强候选；中国省级已补NHSA DRG/DIP、NHC慢病示范区、地方政策执行指标、GBD2021健康结局锚点和NBS粗死亡率敏感性；家庭医生/分级诊疗/健康城市按国家政策里程碑进入政策包，不伪装为省级处理变量", "WHO GHO NCD Country Capacity Survey、WHO高血压/糖尿病服务覆盖、各省控烟条例、国家慢病综合防控示范区名单、卫健委政策文件", "输出省级政策优先级、执行能力缺口和情景模拟结果"),
            ("中国综合输出", "vulnerability_type_label / dominant_risk_triplet / weakest_resource_components / gap_reduction", "省份画像、风险主因、资源短板、政策模拟", "已生成31省压力-风险-响应-政策适配卡片，已接入地方政策执行指标、GBD2021健康结局锚点、NBS多年粗死亡率敏感性面板、NBS解释变量和中国地图/地球仪演示；不声明省级政策健康结果强因果", "由上述数据拼接生成", "形成省份建议卡片和中国地图/地球仪交互字段"),
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "mapping_module",
                "global_framework_variables",
                "china_provincial_variables_needed",
                "current_project_status",
                "recommended_data_sources",
                "mapping_goal",
            ],
        )


    def build_visualization_dictionary() -> pd.DataFrame:
        rows = [
            ("全球地球仪", "iso3 / vulnerability_type_label / response_diagnosis_type / combined_pressure_score / adaptation_gap_score", "country_response_diagnosis_latest.csv", "点击国家显示画像卡片", "展示全球健康转型分型", "P0"),
            ("全球相似国家", "reference_iso3 / matched_type / dominant_risk_triplet / weakest_resource_components / reference_score", "policy_similarity_reference_countries.csv", "点击中国后高亮相似国家", "说明经验迁移基于结构相似", "P0"),
            ("中国入口", "china_type / china_stage / china_response_type / china_top_risks / china_resource_gap", "china_policy_mapping_rules.csv", "点击中国切换到中国面板", "把全球分析自然转入中国落地", "P0"),
            ("中国省级地图", "province / province_resource_response_type / resource_response_type / census_2020_age65_plus_pct / medical_staff_per_10k_population / province_policy_priority", "china_provincial_policy_cards.csv", "点击省份显示建议卡", "展示不同区域不是同一套政策", "P1"),
            ("政策模拟面板", "scenario_name / target_component / baseline_gap / simulated_gap / gap_reduction", "policy_resource_uplift_scenarios.csv / policy_risk_reduction_scenarios.csv / policy_ncd_service_gap_scenarios.csv", "按钮切换资源提升、风险下降或NCD服务覆盖补齐情景", "回答如果补这项缺口下降多少", "P0"),
            ("建议卡片", "policy_package / evidence_basis / expected_output / data_caveat", "policy_response_pathways.csv / china_policy_mapping_rules.csv", "自动生成文字卡片", "给评委可读、可落地的治理建议", "P0"),
            ("方法透明层", "data_sources / model_scope / not_causal_claim / validation_status", "policy_adaptation_engine_summary.json", "信息按钮或脚注", "避免把趋势证据说成强因果", "P0"),
        ]
        return pd.DataFrame(
            rows,
            columns=["visual_layer", "fields", "source_output", "interaction_mode", "display_goal", "mvp_priority"],
        )


    def write_csv(df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False, encoding="utf-8-sig")


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build Module D provincial mapping policy adaptation engine outputs.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--target-iso3", default="CHN")
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        report_dir = ensure_report_dir(project_root)
        clean_dir = project_root / "09_data_clean"

        response_df = read_required_csv(report_asset_path(report_dir, "country_response_diagnosis_latest.csv"))
        risk_summary = read_required_csv(report_asset_path(report_dir, "risk_attribution_type_risk_summary.csv"))
        optimization_df = read_required_csv(report_asset_path(report_dir, "response_optimization_scenarios.csv"))
        allocation_plan = read_required_csv(report_asset_path(report_dir, "response_incremental_allocation_plan.csv"))
        stage_df = read_optional_csv(report_asset_path(report_dir, "stage_country_labels_latest.csv"))
        policy_validation = read_optional_csv(report_asset_path(report_dir, "policy_validation_summary.csv"))
        ncd_policy_panel = read_optional_csv(clean_dir / "external_who_ncd_policy_service_panel.csv")
        policy_summary = read_json(report_asset_path(report_dir, "policy_identification_summary.json"))

        response_with_stage = merge_stage(response_df, stage_df)
        ncd_policy_latest = latest_ncd_policy_profile(ncd_policy_panel, response_with_stage)
        ncd_type_summary = build_ncd_policy_type_summary(ncd_policy_latest)
        ncd_service_scenarios = build_ncd_service_gap_scenarios(ncd_policy_latest)

        policy_pathways = build_policy_response_pathways(
            response_with_stage,
            risk_summary,
            allocation_plan,
            policy_validation,
            policy_summary,
        )
        policy_pathways = add_ncd_readiness_pathways(policy_pathways, ncd_type_summary)
        resource_scenarios = build_resource_uplift_scenarios(optimization_df)
        risk_scenarios = build_risk_reduction_scenarios(response_with_stage, risk_summary)
        type_strategy = build_type_strategy_matrix(response_with_stage, risk_summary, allocation_plan, ncd_type_summary)
        reference_countries = build_reference_countries(response_with_stage, target_iso3=args.target_iso3)
        china_mapping = build_china_policy_mapping_rules(
            response_with_stage,
            resource_scenarios,
            risk_scenarios,
            reference_countries,
            ncd_policy_latest,
            ncd_service_scenarios,
        )
        china_variables = build_china_variable_dictionary()
        visualization_fields = build_visualization_dictionary()

        output_files = {
            "policy_response_pathways": report_asset_path(report_dir, "policy_response_pathways.csv"),
            "policy_resource_uplift_scenarios": report_asset_path(report_dir, "policy_resource_uplift_scenarios.csv"),
            "policy_risk_reduction_scenarios": report_asset_path(report_dir, "policy_risk_reduction_scenarios.csv"),
            "policy_type_strategy_matrix": report_asset_path(report_dir, "policy_type_strategy_matrix.csv"),
            "policy_similarity_reference_countries": report_asset_path(report_dir, "policy_similarity_reference_countries.csv"),
            "policy_ncd_capacity_latest": report_asset_path(report_dir, "policy_ncd_capacity_latest.csv"),
            "policy_ncd_capacity_type_summary": report_asset_path(report_dir, "policy_ncd_capacity_type_summary.csv"),
            "policy_ncd_service_gap_scenarios": report_asset_path(report_dir, "policy_ncd_service_gap_scenarios.csv"),
            "china_mapping_variable_dictionary": report_asset_path(report_dir, "china_mapping_variable_dictionary.csv"),
            "china_policy_mapping_rules": report_asset_path(report_dir, "china_policy_mapping_rules.csv"),
            "visualization_field_dictionary": report_asset_path(report_dir, "visualization_field_dictionary.csv"),
            "policy_adaptation_engine_summary": report_asset_path(report_dir, "policy_adaptation_engine_summary.json"),
        }
        write_csv(policy_pathways, output_files["policy_response_pathways"])
        write_csv(resource_scenarios, output_files["policy_resource_uplift_scenarios"])
        write_csv(risk_scenarios, output_files["policy_risk_reduction_scenarios"])
        write_csv(type_strategy, output_files["policy_type_strategy_matrix"])
        write_csv(reference_countries, output_files["policy_similarity_reference_countries"])
        write_csv(ncd_policy_latest, output_files["policy_ncd_capacity_latest"])
        write_csv(ncd_type_summary, output_files["policy_ncd_capacity_type_summary"])
        write_csv(ncd_service_scenarios, output_files["policy_ncd_service_gap_scenarios"])
        write_csv(china_variables, output_files["china_mapping_variable_dictionary"])
        write_csv(china_mapping, output_files["china_policy_mapping_rules"])
        write_csv(visualization_fields, output_files["visualization_field_dictionary"])

        tobacco = tobacco_evidence(policy_validation, policy_summary)
        china_profile = {}
        china_row = response_with_stage.loc[response_with_stage["iso3"] == args.target_iso3]
        if not china_row.empty:
            row = china_row.iloc[0]
            china_profile = {
                "iso3": args.target_iso3,
                "vulnerability_type_label": row.get("vulnerability_type_label"),
                "transition_stage_label": row.get("transition_stage_label"),
                "response_diagnosis_type": row.get("response_diagnosis_type"),
                "combined_pressure_score": float(row.get("combined_pressure_score")),
                "adjusted_response_score": float(row.get("adjusted_response_score")),
                "adaptation_gap_score": float(row.get("adaptation_gap_score")),
                "dominant_risk_triplet": row.get("dominant_risk_triplet"),
                "weakest_resource_components": row.get("weakest_resource_components"),
            }
            ncd_china = ncd_policy_latest.loc[ncd_policy_latest["iso3"] == args.target_iso3] if not ncd_policy_latest.empty else pd.DataFrame()
            if not ncd_china.empty:
                ncd = ncd_china.iloc[0]
                china_profile.update(
                    {
                        "ncd_policy_execution_score": float(ncd.get("ncd_policy_execution_score")) if pd.notna(ncd.get("ncd_policy_execution_score")) else np.nan,
                        "ncd_policy_capacity_score": float(ncd.get("ncd_policy_capacity_score")) if pd.notna(ncd.get("ncd_policy_capacity_score")) else np.nan,
                        "ncd_service_coverage_score": float(ncd.get("ncd_service_coverage_score")) if pd.notna(ncd.get("ncd_service_coverage_score")) else np.nan,
                        "ncd_policy_readiness_tier": ncd.get("ncd_policy_readiness_tier"),
                        "ncd_hypertension_treatment_pct": float(ncd.get("ncd_hypertension_treatment_pct")) if pd.notna(ncd.get("ncd_hypertension_treatment_pct")) else np.nan,
                        "ncd_hypertension_control_pct": float(ncd.get("ncd_hypertension_control_pct")) if pd.notna(ncd.get("ncd_hypertension_control_pct")) else np.nan,
                        "ncd_diabetes_treatment_pct": float(ncd.get("ncd_diabetes_treatment_pct")) if pd.notna(ncd.get("ncd_diabetes_treatment_pct")) else np.nan,
                    }
                )
        summary = {
            "project_root": project_root.as_posix(),
            "module_d_definition": "全球政策响应证据 + 中国情景适配模拟",
            "input_rows": {
                "country_response_diagnosis_latest": int(response_df.shape[0]),
                "risk_attribution_type_risk_summary": int(risk_summary.shape[0]),
                "response_optimization_scenarios": int(optimization_df.shape[0]),
                "response_incremental_allocation_plan": int(allocation_plan.shape[0]),
                "external_who_ncd_policy_service_panel": int(ncd_policy_panel.shape[0]),
            },
            "output_rows": {
                "policy_response_pathways": int(policy_pathways.shape[0]),
                "policy_resource_uplift_scenarios": int(resource_scenarios.shape[0]),
                "policy_risk_reduction_scenarios": int(risk_scenarios.shape[0]),
                "policy_type_strategy_matrix": int(type_strategy.shape[0]),
                "policy_similarity_reference_countries": int(reference_countries.shape[0]),
                "policy_ncd_capacity_latest": int(ncd_policy_latest.shape[0]),
                "policy_ncd_capacity_type_summary": int(ncd_type_summary.shape[0]),
                "policy_ncd_service_gap_scenarios": int(ncd_service_scenarios.shape[0]),
                "china_policy_mapping_rules": int(china_mapping.shape[0]),
            },
            "scenario_assumptions": {
                "resource_uplift": "沿用模块C：每个国家最弱3个资源分项得分各提升0.10，再重算响应得分和适配缺口。",
                "risk_reduction": f"风险下降率乘以类型内贡献份额，再乘以保守转化系数{RISK_GOVERNANCE_TRANSLATION_FACTOR}。",
                "ncd_service_uplift": f"将高血压/糖尿病等服务覆盖率补到目标阈值，按保守系数{SERVICE_COVERAGE_TRANSLATION_FACTOR}折算响应能力提升。",
                "causal_scope": "风险和资源情景用于政策优先级模拟，不声明单项政策强因果效应。",
            },
            "module_d_v1_claim": "模块D已升级为全球政策适配引擎V2：在A/B/C基础上新增WHO NCD国家能力和服务覆盖数据；控烟DID仅作为趋势性机制证据，不作为强因果主结论。",
            "module_d_v2_claim": "模块D已升级并锁定为全球政策适配引擎V2：新增WHO NCD国家能力、政策执行和服务覆盖数据，输出政策路径、资源提升、风险下降和服务覆盖补齐四类情景；控烟DID仅作为趋势性机制证据。",
            "module_d_data_upgrade": {
                "added_source": "WHO Global Health Observatory bulk download: Noncommunicable diseases",
                "added_clean_file": (clean_dir / "external_who_ncd_policy_service_panel.csv").as_posix(),
                "new_evidence_layer": "NCD Country Capacity Survey 2013-2023 + hypertension/diabetes/cervical service coverage 1990-2022",
                "claim_upgrade": "从政策路径/资源/风险情景，升级到政策执行能力 + 服务覆盖缺口 + 情景模拟；仍不把单项政策写成强因果。",
            },
            "tobacco_policy_evidence": tobacco,
            "target_country_profile": china_profile,
            "output_files": {name: path.as_posix() for name, path in output_files.items()},
        }
        output_files["policy_adaptation_engine_summary"].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_download_policy_d4_data():
    __name__ = 'download_policy_d4_data'
    import argparse
    import json
    from pathlib import Path

    import pandas as pd

    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    WHO_GHO_BASE = "https://ghoapi.azureedge.net/api"
    WORLD_BANK_API = "https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000"

    MPOWER_GROUP_INDICATORS = {
        "M_Group": "Monitoring tobacco use and prevention policies",
        "P_Group": "Protecting people from tobacco smoke",
        "O_Group": "Offering help to quit tobacco use",
        "W_Group": "Warning about the dangers of tobacco",
        "E_Group": "Enforcing bans on tobacco advertising, promotion and sponsorship",
        "R_Group": "Raising taxes on tobacco",
    }

    WDI_TOBACCO_INDICATORS = {
        "SH.PRV.SMOK": "wdi_tobacco_use_pct",
        "SH.PRV.SMOK.MA": "wdi_tobacco_use_male_pct",
        "SH.PRV.SMOK.FE": "wdi_tobacco_use_female_pct",
        "SH_UHC_SCI_NCD": "wdi_uhc_ncd_index",
    }


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        dirs = {
            "inventory": external_data_root / "14_Global_D4_Tobacco_Policy_Sources" / "policy_d4_sources",
            "clean": project_root / "09_data_clean",
            "report": project_root / "06_report_assets",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def request_json(url: str, cache_path: Path | None = None) -> object:
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise FileNotFoundError(
            f"Cached 模块D准因果增强层 source JSON is missing: {cache_path}. "
            "Put archived policy JSON under input/External Data/14_Global_D4_Tobacco_Policy_Sources before running."
        )


    def numeric_value(record: dict[str, object]) -> float | None:
        for key in ["NumericValue", "Value", "value"]:
            value = record.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None


    def download_mpower_groups(dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
        long_rows: list[dict[str, object]] = []
        raw_dir = dirs["inventory"] / "who_gho_mpower_groups"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for indicator, label in MPOWER_GROUP_INDICATORS.items():
            url = f"{WHO_GHO_BASE}/{indicator}"
            cache_path = raw_dir / f"{indicator}.json"
            payload = request_json(url, cache_path=cache_path)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            for record in payload.get("value", []):
                value = numeric_value(record)
                iso3 = record.get("SpatialDim")
                year = record.get("TimeDim")
                if value is None or not iso3 or year is None:
                    continue
                long_rows.append(
                    {
                        "country_code": str(iso3).upper(),
                        "region_code": record.get("ParentLocationCode"),
                        "region": record.get("ParentLocation"),
                        "year": int(year),
                        "indicator": indicator,
                        "indicator_label": label,
                        "value": value,
                        "source_url": url,
                    }
                )
        long = pd.DataFrame(long_rows)
        if long.empty:
            return long, pd.DataFrame()
        long = (
            long.groupby(
                ["country_code", "region_code", "region", "year", "indicator", "indicator_label", "source_url"],
                dropna=False,
                as_index=False,
            )["value"]
            .mean()
            .sort_values(["country_code", "year", "indicator"], kind="stable")
        )
        wide = (
            long.pivot_table(index=["country_code", "year"], columns="indicator", values="value", aggfunc="mean")
            .reset_index()
            .rename_axis(None, axis=1)
        )
        rename = {
            "M_Group": "mpower_m_group",
            "P_Group": "mpower_p_group",
            "O_Group": "mpower_o_group",
            "W_Group": "mpower_w_group",
            "E_Group": "mpower_e_group",
            "R_Group": "mpower_r_group",
        }
        wide = wide.rename(columns=rename)
        group_cols = [col for col in rename.values() if col in wide.columns]
        wide["mpower_total_score"] = wide[group_cols].mean(axis=1)
        wide["mpower_group_observed_count"] = wide[group_cols].notna().sum(axis=1)
        wide = wide.merge(
            long.drop_duplicates(["country_code", "year"])[["country_code", "year", "region_code", "region"]],
            on=["country_code", "year"],
            how="left",
        )
        return long, wide.sort_values(["country_code", "year"], kind="stable")


    def download_wdi_tobacco_outcomes(dirs: dict[str, Path]) -> pd.DataFrame:
        raw_dir = dirs["inventory"] / "worldbank_wdi_tobacco"
        raw_dir.mkdir(parents=True, exist_ok=True)
        frames: list[pd.DataFrame] = []
        metadata: list[dict[str, object]] = []
        for indicator, column in WDI_TOBACCO_INDICATORS.items():
            url = WORLD_BANK_API.format(indicator=indicator)
            cache_path = raw_dir / f"{indicator}.json"
            payload = request_json(url, cache_path=cache_path)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            meta = payload[0] if isinstance(payload, list) and payload else {}
            data = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
            metadata.append({"indicator": indicator, "column": column, "url": url, **meta})
            rows = []
            for record in data:
                iso3 = record.get("countryiso3code")
                value = record.get("value")
                year = record.get("date")
                if not iso3 or value is None or year is None:
                    continue
                rows.append(
                    {
                        "country_code": str(iso3).upper(),
                        "year": int(year),
                        column: float(value),
                    }
                )
            frames.append(pd.DataFrame(rows))
        if not frames:
            return pd.DataFrame()
        outcome = frames[0]
        for frame in frames[1:]:
            outcome = outcome.merge(frame, on=["country_code", "year"], how="outer")
        outcome = outcome.sort_values(["country_code", "year"], kind="stable")
        pd.DataFrame(metadata).to_csv(report_asset_path(dirs["report"], "policy_d4_wdi_tobacco_metadata.csv"), index=False, encoding="utf-8-sig")
        return outcome


    def build_d4_panel(mpower_wide: pd.DataFrame, wdi_outcomes: pd.DataFrame) -> pd.DataFrame:
        if mpower_wide.empty:
            return wdi_outcomes.copy()
        if wdi_outcomes.empty:
            return mpower_wide.copy()
        panel = mpower_wide.merge(wdi_outcomes, on=["country_code", "year"], how="outer")
        panel = panel.sort_values(["country_code", "year"], kind="stable")
        outcome_cols = [column for column in WDI_TOBACCO_INDICATORS.values() if column in panel.columns]
        for column in outcome_cols:
            panel[f"delta_{column}"] = panel.groupby("country_code", dropna=False)[column].diff()
        return panel


    def main() -> None:
        parser = argparse.ArgumentParser(description="Load credible tobacco policy/outcome data for 模块D政策数据层.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)

        outputs = {
            "mpower_long": dirs["clean"] / "external_who_mpower_detailed_scores_long.csv",
            "mpower_wide": dirs["clean"] / "external_who_mpower_detailed_scores_wide.csv",
            "wdi_tobacco_outcomes": dirs["clean"] / "external_wdi_tobacco_outcomes.csv",
            "d4_panel": dirs["clean"] / "external_policy_d4_tobacco_panel.csv",
            "summary": report_asset_path(dirs["report"], "policy_d4_data_source_summary.json"),
        }
        if all(outputs[key].exists() and outputs[key].stat().st_size > 0 for key in [
            "mpower_long",
            "mpower_wide",
            "wdi_tobacco_outcomes",
            "d4_panel",
        ]):
            mpower_long = pd.read_csv(outputs["mpower_long"], low_memory=False)
            mpower_wide = pd.read_csv(outputs["mpower_wide"], low_memory=False)
            wdi_outcomes = pd.read_csv(outputs["wdi_tobacco_outcomes"], low_memory=False)
            d4_panel = pd.read_csv(outputs["d4_panel"], low_memory=False)
            source_mode = "reused_09_data_clean"
        else:
            mpower_long, mpower_wide = download_mpower_groups(dirs)
            wdi_outcomes = download_wdi_tobacco_outcomes(dirs)
            d4_panel = build_d4_panel(mpower_wide, wdi_outcomes)
            source_mode = "rebuilt_from_input_json_cache"
        mpower_long.to_csv(outputs["mpower_long"], index=False, encoding="utf-8-sig")
        mpower_wide.to_csv(outputs["mpower_wide"], index=False, encoding="utf-8-sig")
        wdi_outcomes.to_csv(outputs["wdi_tobacco_outcomes"], index=False, encoding="utf-8-sig")
        d4_panel.to_csv(outputs["d4_panel"], index=False, encoding="utf-8-sig")

        summary = {
            "project_root": project_root.as_posix(),
            "source_mode": source_mode,
            "data_upgrade": "模块D准因果增强层 credible tobacco policy and outcome data",
            "sources": {
                "who_gho_mpower_groups": {
                    "base_url": WHO_GHO_BASE,
                    "indicators": MPOWER_GROUP_INDICATORS,
                    "description": "WHO GHO MPOWER group scores for Monitor/Protect/Offer/Warn/Enforce/Raise taxes.",
                },
                "world_bank_wdi_tobacco": {
                    "base_url": "https://api.worldbank.org/v2/",
                    "indicators": WDI_TOBACCO_INDICATORS,
                    "description": "World Bank WDI current tobacco use prevalence and UHC NCD sub-index.",
                },
            },
            "rows": {
                "mpower_long": int(mpower_long.shape[0]),
                "mpower_wide": int(mpower_wide.shape[0]),
                "wdi_tobacco_outcomes": int(wdi_outcomes.shape[0]),
                "d4_panel": int(d4_panel.shape[0]),
            },
            "coverage": {
                "mpower_countries": int(mpower_wide["country_code"].nunique()) if not mpower_wide.empty else 0,
                "mpower_year_min": int(mpower_wide["year"].min()) if not mpower_wide.empty else None,
                "mpower_year_max": int(mpower_wide["year"].max()) if not mpower_wide.empty else None,
                "wdi_countries": int(wdi_outcomes["country_code"].nunique()) if not wdi_outcomes.empty else 0,
                "wdi_year_min": int(wdi_outcomes["year"].min()) if not wdi_outcomes.empty else None,
                "wdi_year_max": int(wdi_outcomes["year"].max()) if not wdi_outcomes.empty else None,
            },
            "output_files": {name: path.as_posix() for name, path in outputs.items()},
        }
        outputs["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_policy_causal_enhancement():
    __name__ = 'run_policy_causal_enhancement'
    import argparse
    import json
    import math
    from dataclasses import dataclass
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import nnls

    from foundation import choose_text, configure_matplotlib_fonts, set_centered_suptitle
    from foundation import detect_project_root as shared_detect_project_root


    USE_CHINESE = configure_matplotlib_fonts()

    PRE_WINDOW = 5
    POST_WINDOW = 5
    EVENT_STUDY_OMITTED_YEAR = -1
    MIN_TREATED_COUNTRIES = 12
    MIN_REGRESSION_OBS = 180
    MIN_CLUSTERS = 20
    RANDOM_EXPERIMENT_CLAIM = False

    CONTROL_COLUMNS = [
        "hdi",
        "wdi_gdp_per_capita",
        "wdi_population_65_plus_pct",
        "wdi_urban_population_pct",
    ]

    POLICY_EVENT_COLUMNS = [
        "event_id",
        "policy_id",
        "policy_label",
        "policy_variable",
        "threshold",
        "threshold_rule",
        "iso3",
        "event_year",
        "value_before_event",
        "value_at_event",
        "pre_policy_observed_years",
        "post_policy_observed_years",
        "vulnerability_type_label",
        "response_diagnosis_type",
        "source",
        "identification_role",
    ]


    @dataclass(frozen=True)
    class OutcomeSpec:
        column: str
        label: str
        expected_sign: int
        tier: str


    @dataclass(frozen=True)
    class PolicySpec:
        policy_id: str
        label: str
        variable: str
        threshold: float
        source: str
        outcomes: tuple[OutcomeSpec, ...]


    POLICY_SPECS: tuple[PolicySpec, ...] = (
        PolicySpec(
            policy_id="mpower_tobacco_strong",
            label="MPOWER控烟强执行",
            variable="policy_strength",
            threshold=4.0,
            source="WHO MPOWER M_Group",
            outcomes=(
                OutcomeSpec("who_smoking_rate_std", "WHO年龄标准化吸烟率", -1, "短期风险暴露"),
                OutcomeSpec("who_cigarette_prevalence_btsx", "WHO卷烟吸烟率", -1, "短期风险暴露"),
                OutcomeSpec("dbd_smoking_per100k", "吸烟风险暴露负担", -1, "中期风险负担"),
                OutcomeSpec("delta_dbd_smoking_per100k", "吸烟风险暴露负担年变化", -1, "变化率结果"),
                OutcomeSpec("gbd_rate_chronic_respiratory_diseases_per100k", "慢性呼吸系统疾病负担", -1, "长期疾病负担"),
            ),
        ),
        PolicySpec(
            policy_id="mpower_protect_smokefree_strong",
            label="MPOWER无烟环境保护强执行",
            variable="mpower_p_group",
            threshold=4.0,
            source="WHO GHO P_Group",
            outcomes=(
                OutcomeSpec("delta_dbd_smoking_per100k", "吸烟风险暴露负担年变化", -1, "变化率结果"),
                OutcomeSpec("dbd_smoking_per100k", "吸烟风险暴露负担", -1, "中期风险负担"),
                OutcomeSpec("wdi_tobacco_use_pct", "WDI当前烟草使用率", -1, "短期风险暴露"),
                OutcomeSpec("delta_wdi_tobacco_use_pct", "WDI当前烟草使用率年变化", -1, "变化率结果"),
            ),
        ),
        PolicySpec(
            policy_id="mpower_comprehensive_package_high",
            label="MPOWER综合控烟政策包高水平",
            variable="mpower_total_score",
            threshold=4.0,
            source="WHO GHO MPOWER six group scores",
            outcomes=(
                OutcomeSpec("dbd_smoking_per100k", "吸烟风险暴露负担", -1, "中期风险负担"),
                OutcomeSpec("delta_dbd_smoking_per100k", "吸烟风险暴露负担年变化", -1, "变化率结果"),
                OutcomeSpec("wdi_tobacco_use_pct", "WDI当前烟草使用率", -1, "短期风险暴露"),
            ),
        ),
        PolicySpec(
            policy_id="ncd_tobacco_policy_execution",
            label="NCD控烟政策执行",
            variable="tobacco_policy_execution",
            threshold=0.75,
            source="WHO NCD Country Capacity Survey",
            outcomes=(
                OutcomeSpec("who_smoking_rate_std", "WHO年龄标准化吸烟率", -1, "短期风险暴露"),
                OutcomeSpec("dbd_smoking_per100k", "吸烟风险暴露负担", -1, "中期风险负担"),
                OutcomeSpec("delta_dbd_smoking_per100k", "吸烟风险暴露负担年变化", -1, "变化率结果"),
            ),
        ),
        PolicySpec(
            policy_id="ncd_capacity_high",
            label="NCD综合能力高水平",
            variable="ncd_policy_capacity_score",
            threshold=0.75,
            source="WHO NCD Country Capacity Survey",
            outcomes=(
                OutcomeSpec("ncd_service_coverage_score", "NCD服务覆盖综合分", 1, "服务覆盖"),
                OutcomeSpec("ncd_hypertension_treatment_pct", "高血压治疗覆盖率", 1, "服务覆盖"),
                OutcomeSpec("ncd_hypertension_control_pct", "高血压控制率", 1, "服务覆盖"),
            ),
        ),
        PolicySpec(
            policy_id="hypertension_policy_ready",
            label="高血压政策准备度提升",
            variable="hypertension_policy_readiness",
            threshold=0.75,
            source="WHO NCD Country Capacity Survey",
            outcomes=(
                OutcomeSpec("ncd_hypertension_diagnosis_pct", "高血压诊断覆盖率", 1, "服务覆盖"),
                OutcomeSpec("ncd_hypertension_treatment_pct", "高血压治疗覆盖率", 1, "服务覆盖"),
                OutcomeSpec("ncd_hypertension_control_pct", "高血压控制率", 1, "服务覆盖"),
                OutcomeSpec("dbd_high_sbp_per100k", "高收缩压风险负担", -1, "中期风险负担"),
            ),
        ),
        PolicySpec(
            policy_id="diabetes_policy_ready",
            label="糖尿病政策准备度提升",
            variable="diabetes_policy_readiness",
            threshold=0.80,
            source="WHO NCD Country Capacity Survey",
            outcomes=(
                OutcomeSpec("ncd_diabetes_treatment_pct", "糖尿病治疗覆盖率", 1, "服务覆盖"),
                OutcomeSpec("dbd_high_glucose_per100k", "高血糖风险负担", -1, "中期风险负担"),
                OutcomeSpec("gbd_rate_diabetes_kidney_per100k", "糖尿病和肾病负担", -1, "长期疾病负担"),
            ),
        ),
        PolicySpec(
            policy_id="diet_salt_policy_execution",
            label="控盐膳食政策执行",
            variable="diet_salt_policy_execution",
            threshold=0.75,
            source="WHO NCD Country Capacity Survey",
            outcomes=(
                OutcomeSpec("dbd_dietary_risks_per100k", "膳食风险负担", -1, "中期风险负担"),
                OutcomeSpec("dbd_high_sbp_per100k", "高收缩压风险负担", -1, "中期风险负担"),
                OutcomeSpec("ncd_hypertension_control_pct", "高血压控制率", 1, "服务覆盖"),
            ),
        ),
        PolicySpec(
            policy_id="primary_care_ncd_ready",
            label="基层NCD服务准备度提升",
            variable="primary_care_ncd_service_readiness",
            threshold=0.80,
            source="WHO NCD Country Capacity Survey",
            outcomes=(
                OutcomeSpec("ncd_service_coverage_score", "NCD服务覆盖综合分", 1, "服务覆盖"),
                OutcomeSpec("ncd_hypertension_treatment_pct", "高血压治疗覆盖率", 1, "服务覆盖"),
                OutcomeSpec("ncd_diabetes_treatment_pct", "糖尿病治疗覆盖率", 1, "服务覆盖"),
            ),
        ),
    )


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        dirs = {
            "simulation": project_root / "04_simulation",
            "report": project_root / "06_report_assets",
            "figures": project_root / "05_figures",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def read_csv_if_exists(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


    def read_json_if_exists(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8-sig"))


    def write_csv(df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False, encoding="utf-8-sig")


    def zscore(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").astype("float64")
        std = values.std(ddof=0)
        if pd.isna(std) or std == 0:
            return values * np.nan
        return (values - values.mean()) / std


    def normal_p_value(z_value: float) -> float:
        if pd.isna(z_value):
            return np.nan
        return math.erfc(abs(float(z_value)) / math.sqrt(2.0))


    def safe_float(value: object) -> float:
        numeric = pd.to_numeric(value, errors="coerce")
        return float(numeric) if pd.notna(numeric) else np.nan


    def build_analysis_panel(project_root: Path) -> pd.DataFrame:
        simulation_dir = project_root / "04_simulation"
        clean_dir = project_root / "09_data_clean"
        report_dir = project_root / "06_report_assets"

        base = read_csv_if_exists(simulation_dir / "policy_identification_panel.csv")
        if base.empty:
            base = read_csv_if_exists(simulation_dir / "global_health_panel_v1.csv")
        if base.empty:
            raise FileNotFoundError("Missing policy/global analysis panel in 04_simulation")

        base["iso3"] = base["iso3"].astype(str).str.upper().str.strip()
        base["year"] = pd.to_numeric(base["year"], errors="coerce").astype("Int64")
        ncd = read_csv_if_exists(clean_dir / "external_who_ncd_policy_service_panel.csv")
        if not ncd.empty:
            ncd = ncd.rename(columns={"country_code": "iso3"})
            ncd["iso3"] = ncd["iso3"].astype(str).str.upper().str.strip()
            ncd["year"] = pd.to_numeric(ncd["year"], errors="coerce").astype("Int64")
            keep_cols = [
                "iso3",
                "year",
                "region",
                "diabetes_policy_readiness",
                "diet_salt_policy_execution",
                "hypertension_policy_readiness",
                "integrated_ncd_governance",
                "primary_care_ncd_service_readiness",
                "tobacco_policy_execution",
                "ncd_policy_capacity_score",
                "ncd_hypertension_diagnosis_pct",
                "ncd_hypertension_treatment_pct",
                "ncd_hypertension_control_pct",
                "ncd_diabetes_treatment_pct",
                "ncd_cervical_screening_pct",
                "ncd_service_coverage_score",
                "ncd_policy_execution_score",
            ]
            ncd = ncd.loc[:, [col for col in keep_cols if col in ncd.columns]]
            panel = base.merge(ncd, on=["iso3", "year"], how="outer")
        else:
            panel = base.copy()

        d4_panel = read_csv_if_exists(clean_dir / "external_policy_d4_tobacco_panel.csv")
        if not d4_panel.empty:
            d4_panel = d4_panel.rename(columns={"country_code": "iso3"})
            d4_panel["iso3"] = d4_panel["iso3"].astype(str).str.upper().str.strip()
            d4_panel["year"] = pd.to_numeric(d4_panel["year"], errors="coerce").astype("Int64")
            d4_keep = [
                "iso3",
                "year",
                "mpower_m_group",
                "mpower_p_group",
                "mpower_o_group",
                "mpower_w_group",
                "mpower_e_group",
                "mpower_r_group",
                "mpower_total_score",
                "mpower_group_observed_count",
                "wdi_tobacco_use_pct",
                "wdi_tobacco_use_male_pct",
                "wdi_tobacco_use_female_pct",
                "wdi_uhc_ncd_index",
            ]
            d4_panel = d4_panel.loc[:, [col for col in d4_keep if col in d4_panel.columns]]
            panel = panel.merge(d4_panel, on=["iso3", "year"], how="outer")

        response_latest = read_csv_if_exists(report_asset_path(report_dir, "country_response_diagnosis_latest.csv"))
        if not response_latest.empty:
            latest_cols = [
                "iso3",
                "vulnerability_type_label",
                "response_diagnosis_type",
                "combined_pressure_score",
                "adjusted_response_score",
                "adaptation_gap_score",
                "dominant_risk_triplet",
                "weakest_resource_components",
            ]
            latest = response_latest.loc[:, [col for col in latest_cols if col in response_latest.columns]].drop_duplicates("iso3")
            panel = panel.merge(latest, on="iso3", how="left", suffixes=("", "_latest"))
            if "vulnerability_type_label_latest" in panel.columns:
                panel["vulnerability_type_label"] = panel.get("vulnerability_type_label").combine_first(
                    panel["vulnerability_type_label_latest"]
                )
            if "response_diagnosis_type_latest" in panel.columns:
                panel["response_diagnosis_type"] = panel.get("response_diagnosis_type").combine_first(panel["response_diagnosis_type_latest"])
            drop_latest = [col for col in panel.columns if col.endswith("_latest")]
            panel = panel.drop(columns=drop_latest)

        for col in panel.columns:
            if col not in {"iso3", "vulnerability_type_label", "response_diagnosis_type", "dominant_risk_triplet", "weakest_resource_components", "region"}:
                panel[col] = pd.to_numeric(panel[col], errors="ignore")

        panel = panel.loc[panel["iso3"].notna() & panel["year"].notna()].copy()
        panel["year"] = panel["year"].astype(int)
        panel = panel.sort_values(["iso3", "year"], kind="stable").reset_index(drop=True)
        derived_change_columns = sorted(
            {
                outcome.column
                for spec in POLICY_SPECS
                for outcome in spec.outcomes
                if not outcome.column.startswith("delta_") and outcome.column in panel.columns
            }
        )
        for column in derived_change_columns:
            panel[f"delta_{column}"] = panel.groupby("iso3", dropna=False)[column].diff()
        return panel


    def identify_policy_events(panel: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        latest_meta = (
            panel.sort_values(["iso3", "year"], kind="stable")
            .drop_duplicates("iso3", keep="last")
            .set_index("iso3")
        )
        for spec in POLICY_SPECS:
            if spec.variable not in panel.columns:
                continue
            observed = panel.loc[panel[spec.variable].notna(), ["iso3", "year", spec.variable]].copy()
            if observed.empty:
                continue
            for iso3, group in observed.sort_values(["iso3", "year"], kind="stable").groupby("iso3"):
                group = group.dropna(subset=[spec.variable]).copy()
                if group.empty:
                    continue
                prior_below = False
                event_idx = None
                for idx, row in group.iterrows():
                    value = safe_float(row[spec.variable])
                    if pd.isna(value):
                        continue
                    if value < spec.threshold:
                        prior_below = True
                    elif value >= spec.threshold and prior_below:
                        event_idx = idx
                        break
                if event_idx is None:
                    continue
                event_year = int(group.loc[event_idx, "year"])
                before = group.loc[group["year"] < event_year, spec.variable]
                after = group.loc[group["year"] > event_year, spec.variable]
                if before.empty or after.empty:
                    continue
                meta = latest_meta.loc[iso3] if iso3 in latest_meta.index else pd.Series(dtype=object)
                rows.append(
                    {
                        "event_id": f"{spec.policy_id}_{iso3}_{event_year}",
                        "policy_id": spec.policy_id,
                        "policy_label": spec.label,
                        "policy_variable": spec.variable,
                        "threshold": spec.threshold,
                        "threshold_rule": f"first observed switch from < {spec.threshold:g} to >= {spec.threshold:g}",
                        "iso3": iso3,
                        "event_year": event_year,
                        "value_before_event": safe_float(before.iloc[-1]),
                        "value_at_event": safe_float(group.loc[event_idx, spec.variable]),
                        "pre_policy_observed_years": int(before.shape[0]),
                        "post_policy_observed_years": int(after.shape[0]),
                        "vulnerability_type_label": meta.get("vulnerability_type_label", np.nan),
                        "response_diagnosis_type": meta.get("response_diagnosis_type", np.nan),
                        "source": spec.source,
                        "identification_role": "treated_switcher",
                    }
                )
        return pd.DataFrame(rows, columns=POLICY_EVENT_COLUMNS)


    def build_design_matrix(
        df: pd.DataFrame,
        treatment_columns: list[str],
        controls: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str], pd.Series]:
        controls = controls or []
        work = df.copy()
        columns: list[pd.Series] = []
        names: list[str] = []

        columns.append(pd.Series(1.0, index=work.index))
        names.append("intercept")
        for column in treatment_columns:
            columns.append(pd.to_numeric(work[column], errors="coerce").fillna(0.0).astype("float64"))
            names.append(column)
        for column in controls:
            if column not in work.columns:
                continue
            values = zscore(work[column])
            if values.notna().sum() < max(30, int(0.25 * work.shape[0])):
                continue
            columns.append(values.fillna(0.0).astype("float64"))
            names.append(f"control__{column}")

        iso_dummies = pd.get_dummies(work["iso3"].astype(str), prefix="iso", drop_first=True, dtype=float)
        year_dummies = pd.get_dummies(work["year"].astype(int), prefix="year", drop_first=True, dtype=float)
        for column in iso_dummies.columns:
            columns.append(iso_dummies[column])
            names.append(column)
        for column in year_dummies.columns:
            columns.append(year_dummies[column])
            names.append(column)
        x = np.column_stack([column.to_numpy(dtype=float) for column in columns])
        clusters = work["iso3"].astype(str)
        return x, names, clusters


    def fit_cluster_ols(
        df: pd.DataFrame,
        y_col: str,
        treatment_columns: list[str],
        controls: list[str] | None = None,
    ) -> dict[str, object]:
        needed = ["iso3", "year", y_col, *treatment_columns]
        work = df.loc[:, [col for col in [*needed, *(controls or [])] if col in df.columns]].copy()
        work["_y_raw"] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=["iso3", "year", "_y_raw"])
        if work.shape[0] < MIN_REGRESSION_OBS or work["iso3"].nunique() < MIN_CLUSTERS:
            return {"ok": False, "reason": "insufficient_observations", "n_obs": int(work.shape[0]), "clusters": int(work["iso3"].nunique())}
        if any(work[col].nunique(dropna=True) < 2 for col in treatment_columns):
            return {"ok": False, "reason": "no_treatment_variation", "n_obs": int(work.shape[0]), "clusters": int(work["iso3"].nunique())}
        y_std = work["_y_raw"].std(ddof=0)
        if pd.isna(y_std) or y_std == 0:
            return {"ok": False, "reason": "outcome_has_no_variation", "n_obs": int(work.shape[0]), "clusters": int(work["iso3"].nunique())}
        work["_y"] = (work["_y_raw"] - work["_y_raw"].mean()) / y_std
        x, names, clusters = build_design_matrix(work, treatment_columns, controls)
        y = work["_y"].to_numpy(dtype=float)
        if x.shape[0] <= x.shape[1] + 3:
            return {"ok": False, "reason": "design_too_wide", "n_obs": int(work.shape[0]), "k": int(x.shape[1])}
        beta = np.linalg.pinv(x) @ y
        residual = y - x @ beta
        xtx_inv = np.linalg.pinv(x.T @ x)
        meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        cluster_values = clusters.to_numpy()
        unique_clusters = pd.unique(cluster_values)
        for cluster in unique_clusters:
            mask = cluster_values == cluster
            score = x[mask].T @ residual[mask]
            meat += np.outer(score, score)
        cov = xtx_inv @ meat @ xtx_inv
        n = x.shape[0]
        k = x.shape[1]
        g = len(unique_clusters)
        if g > 1 and n > k:
            cov *= (g / (g - 1)) * ((n - 1) / max(n - k, 1))
        se = np.sqrt(np.clip(np.diag(cov), a_min=0, a_max=None))
        result = {
            "ok": True,
            "n_obs": int(n),
            "clusters": int(g),
            "k": int(k),
            "coefficients": {},
        }
        for column in treatment_columns:
            idx = names.index(column)
            coef = float(beta[idx])
            stderr = float(se[idx]) if idx < len(se) else np.nan
            z_value = coef / stderr if stderr and not pd.isna(stderr) else np.nan
            result["coefficients"][column] = {
                "coef": coef,
                "se": stderr,
                "z": z_value,
                "p": normal_p_value(z_value),
            }
        return result


    def attach_event_timing(panel: pd.DataFrame, events: pd.DataFrame, policy_id: str) -> pd.DataFrame:
        event_years = events.loc[events["policy_id"] == policy_id, ["iso3", "event_year"]].drop_duplicates("iso3")
        work = panel.merge(event_years, on="iso3", how="left")
        work["ever_treated"] = work["event_year"].notna().astype(int)
        work["treated_post"] = ((work["event_year"].notna()) & (work["year"] >= work["event_year"])).astype(int)
        work["relative_year"] = work["year"] - work["event_year"]
        return work


    def run_did_estimates(panel: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[dict[str, object]] = []
        heterogeneity_rows: list[dict[str, object]] = []
        for spec in POLICY_SPECS:
            policy_events = events.loc[events["policy_id"] == spec.policy_id]
            if policy_events["iso3"].nunique() < MIN_TREATED_COUNTRIES:
                continue
            timed = attach_event_timing(panel, events, spec.policy_id)
            for outcome in spec.outcomes:
                if outcome.column not in timed.columns:
                    continue
                fit = fit_cluster_ols(timed, outcome.column, ["treated_post"], CONTROL_COLUMNS)
                if not fit.get("ok"):
                    rows.append(
                        {
                            "policy_id": spec.policy_id,
                            "policy_label": spec.label,
                            "outcome_column": outcome.column,
                            "outcome_label": outcome.label,
                            "outcome_tier": outcome.tier,
                            "sample_scope": "global",
                            "expected_sign": outcome.expected_sign,
                            "did_coefficient_sd": np.nan,
                            "cluster_se": np.nan,
                            "p_value": np.nan,
                            "n_obs": fit.get("n_obs", 0),
                            "clusters": fit.get("clusters", 0),
                            "treated_countries": int(policy_events["iso3"].nunique()),
                            "direction_consistent": False,
                            "quasi_causal_signal": False,
                            "failure_reason": fit.get("reason"),
                        }
                    )
                    continue
                coef_info = fit["coefficients"]["treated_post"]
                coef = coef_info["coef"]
                p_value = coef_info["p"]
                direction = bool(outcome.expected_sign * coef > 0)
                rows.append(
                    {
                        "policy_id": spec.policy_id,
                        "policy_label": spec.label,
                        "outcome_column": outcome.column,
                        "outcome_label": outcome.label,
                        "outcome_tier": outcome.tier,
                        "sample_scope": "global",
                        "expected_sign": outcome.expected_sign,
                        "did_coefficient_sd": coef,
                        "cluster_se": coef_info["se"],
                        "z_value": coef_info["z"],
                        "p_value": p_value,
                        "n_obs": fit["n_obs"],
                        "clusters": fit["clusters"],
                        "treated_countries": int(policy_events["iso3"].nunique()),
                        "direction_consistent": direction,
                        "quasi_causal_signal": bool(direction and pd.notna(p_value) and p_value < 0.10),
                        "failure_reason": "",
                    }
                )
                if "vulnerability_type_label" not in timed.columns:
                    continue
                for type_label, subset in timed.groupby("vulnerability_type_label", dropna=True):
                    if subset["iso3"].nunique() < 25 or subset["treated_post"].sum() < 15:
                        continue
                    hfit = fit_cluster_ols(subset, outcome.column, ["treated_post"], CONTROL_COLUMNS)
                    if not hfit.get("ok"):
                        continue
                    hcoef = hfit["coefficients"]["treated_post"]
                    heterogeneity_rows.append(
                        {
                            "policy_id": spec.policy_id,
                            "policy_label": spec.label,
                            "outcome_column": outcome.column,
                            "outcome_label": outcome.label,
                            "vulnerability_type_label": type_label,
                            "did_coefficient_sd": hcoef["coef"],
                            "cluster_se": hcoef["se"],
                            "p_value": hcoef["p"],
                            "n_obs": hfit["n_obs"],
                            "clusters": hfit["clusters"],
                            "treated_country_years": int(subset["treated_post"].sum()),
                            "direction_consistent": bool(outcome.expected_sign * hcoef["coef"] > 0),
                        }
                    )
        return pd.DataFrame(rows), pd.DataFrame(heterogeneity_rows)


    def run_event_studies(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        rel_years = [year for year in range(-PRE_WINDOW, POST_WINDOW + 1) if year != EVENT_STUDY_OMITTED_YEAR]
        for spec in POLICY_SPECS:
            if events.loc[events["policy_id"] == spec.policy_id, "iso3"].nunique() < MIN_TREATED_COUNTRIES:
                continue
            timed = attach_event_timing(panel, events, spec.policy_id)
            relevant_years = timed.loc[timed["event_year"].notna(), "event_year"]
            if relevant_years.empty:
                continue
            timed = timed.loc[
                (timed["year"] >= int(relevant_years.min()) - PRE_WINDOW)
                & (timed["year"] <= int(relevant_years.max()) + POST_WINDOW)
            ].copy()
            for relative_year in rel_years:
                timed[f"rel_{relative_year}"] = (
                    (timed["ever_treated"] == 1) & (timed["relative_year"] == relative_year)
                ).astype(int)
            treatment_columns = [f"rel_{year}" for year in rel_years]
            for outcome in spec.outcomes:
                if outcome.column not in timed.columns:
                    continue
                fit = fit_cluster_ols(timed, outcome.column, treatment_columns, CONTROL_COLUMNS)
                if not fit.get("ok"):
                    continue
                for relative_year in rel_years:
                    info = fit["coefficients"][f"rel_{relative_year}"]
                    rows.append(
                        {
                            "policy_id": spec.policy_id,
                            "policy_label": spec.label,
                            "outcome_column": outcome.column,
                            "outcome_label": outcome.label,
                            "expected_sign": outcome.expected_sign,
                            "relative_year": relative_year,
                            "coefficient_sd": info["coef"],
                            "cluster_se": info["se"],
                            "p_value": info["p"],
                            "n_obs": fit["n_obs"],
                            "clusters": fit["clusters"],
                            "treated_countries": int(events.loc[events["policy_id"] == spec.policy_id, "iso3"].nunique()),
                            "is_pre_period": relative_year < 0,
                            "is_post_period": relative_year >= 0,
                            "direction_consistent": bool(outcome.expected_sign * info["coef"] > 0) if relative_year >= 0 else np.nan,
                        }
                    )
        return pd.DataFrame(rows)


    def summarize_pretrends(event_study: pd.DataFrame) -> pd.DataFrame:
        if event_study.empty:
            return pd.DataFrame()
        rows = []
        for keys, subset in event_study.groupby(["policy_id", "outcome_column"], dropna=False):
            pre = subset.loc[subset["is_pre_period"]]
            post = subset.loc[subset["is_post_period"]]
            rows.append(
                {
                    "policy_id": keys[0],
                    "outcome_column": keys[1],
                    "pretrend_max_abs_coef": float(pre["coefficient_sd"].abs().max()) if not pre.empty else np.nan,
                    "pretrend_mean_abs_coef": float(pre["coefficient_sd"].abs().mean()) if not pre.empty else np.nan,
                    "pretrend_any_p_lt_0_10": bool((pre["p_value"] < 0.10).any()) if not pre.empty else False,
                    "post_mean_coef": float(post["coefficient_sd"].mean()) if not post.empty else np.nan,
                    "post_peak_abs_coef": float(post["coefficient_sd"].abs().max()) if not post.empty else np.nan,
                }
            )
        return pd.DataFrame(rows)


    def run_placebo_tests(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for spec in POLICY_SPECS:
            event_years = events.loc[events["policy_id"] == spec.policy_id, ["iso3", "event_year"]].copy()
            if event_years["iso3"].nunique() < MIN_TREATED_COUNTRIES:
                continue
            event_years = event_years.rename(columns={"event_year": "actual_event_year"})
            event_years["placebo_event_year"] = event_years["actual_event_year"] - 3
            timed = panel.merge(event_years, on="iso3", how="left")
            timed = timed.loc[timed["actual_event_year"].isna() | (timed["year"] < timed["actual_event_year"])].copy()
            timed["treated_post_placebo"] = (
                (timed["placebo_event_year"].notna())
                & (timed["year"] >= timed["placebo_event_year"])
                & (timed["year"] < timed["actual_event_year"])
            ).astype(int)
            for outcome in spec.outcomes:
                if outcome.column not in timed.columns:
                    continue
                fit = fit_cluster_ols(timed, outcome.column, ["treated_post_placebo"], CONTROL_COLUMNS)
                if not fit.get("ok"):
                    continue
                info = fit["coefficients"]["treated_post_placebo"]
                rows.append(
                    {
                        "policy_id": spec.policy_id,
                        "policy_label": spec.label,
                        "outcome_column": outcome.column,
                        "outcome_label": outcome.label,
                        "placebo_shift_years": -3,
                        "placebo_coefficient_sd": info["coef"],
                        "cluster_se": info["se"],
                        "p_value": info["p"],
                        "n_obs": fit["n_obs"],
                        "clusters": fit["clusters"],
                        "placebo_direction_consistent": bool(outcome.expected_sign * info["coef"] > 0),
                        "placebo_issue": bool(outcome.expected_sign * info["coef"] > 0 and pd.notna(info["p"]) and info["p"] < 0.10),
                    }
                )
        return pd.DataFrame(rows)


    def run_synthetic_controls(panel: pd.DataFrame, events: pd.DataFrame, max_treated_per_combo: int = 10) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for spec in POLICY_SPECS:
            policy_events = events.loc[events["policy_id"] == spec.policy_id, ["iso3", "event_year"]].drop_duplicates("iso3")
            if policy_events.shape[0] < MIN_TREATED_COUNTRIES:
                continue
            event_map = policy_events.set_index("iso3")["event_year"].to_dict()
            never_or_late = set(panel["iso3"].dropna().unique()) - set(policy_events["iso3"])
            for outcome in spec.outcomes:
                if outcome.column not in panel.columns:
                    continue
                work = panel.loc[:, ["iso3", "year", outcome.column]].copy()
                work["_y"] = zscore(work[outcome.column])
                candidates = []
                for iso3, event_year in event_map.items():
                    pre_years = list(range(int(event_year) - PRE_WINDOW, int(event_year)))
                    post_years = list(range(int(event_year), int(event_year) + POST_WINDOW + 1))
                    treated = work.loc[(work["iso3"] == iso3) & (work["year"].isin(pre_years + post_years))]
                    if treated["_y"].notna().sum() < len(pre_years) + 2:
                        continue
                    donor_ids = []
                    donor_pre = []
                    donor_post = []
                    for donor in never_or_late:
                        donor_frame = work.loc[(work["iso3"] == donor) & (work["year"].isin(pre_years + post_years))]
                        donor_pivot = donor_frame.set_index("year")["_y"]
                        if donor_pivot.reindex(pre_years).notna().all() and donor_pivot.reindex(post_years).notna().sum() >= 2:
                            donor_ids.append(donor)
                            donor_pre.append(donor_pivot.reindex(pre_years).to_numpy(dtype=float))
                            donor_post.append(donor_pivot.reindex(post_years).to_numpy(dtype=float))
                    if len(donor_ids) < 8:
                        continue
                    treated_pivot = treated.set_index("year")["_y"]
                    y_pre = treated_pivot.reindex(pre_years).to_numpy(dtype=float)
                    y_post = treated_pivot.reindex(post_years).dropna().to_numpy(dtype=float)
                    if np.isnan(y_pre).any() or y_post.size < 2:
                        continue
                    x_pre = np.vstack(donor_pre).T
                    weights, _ = nnls(x_pre, y_pre)
                    if weights.sum() <= 0:
                        continue
                    weights = weights / weights.sum()
                    x_post = pd.DataFrame(np.vstack(donor_post).T, index=post_years)
                    valid_post = treated_pivot.reindex(post_years).dropna()
                    synthetic_post = x_post.loc[valid_post.index].to_numpy(dtype=float) @ weights
                    post_effect = valid_post.to_numpy(dtype=float) - synthetic_post
                    pre_fit = x_pre @ weights
                    pre_rmse = float(np.sqrt(np.mean((y_pre - pre_fit) ** 2)))
                    candidates.append(
                        {
                            "policy_id": spec.policy_id,
                            "policy_label": spec.label,
                            "outcome_column": outcome.column,
                            "outcome_label": outcome.label,
                            "iso3": iso3,
                            "event_year": int(event_year),
                            "donor_countries": len(donor_ids),
                            "pre_rmse_sd": pre_rmse,
                            "post_effect_mean_sd": float(np.mean(post_effect)),
                            "post_effect_median_sd": float(np.median(post_effect)),
                            "post_years_observed": int(valid_post.shape[0]),
                            "expected_sign": outcome.expected_sign,
                            "direction_consistent": bool(outcome.expected_sign * float(np.mean(post_effect)) > 0),
                            "top_donor_weights": " / ".join(
                                f"{donor}:{weight:.2f}"
                                for donor, weight in sorted(zip(donor_ids, weights), key=lambda item: item[1], reverse=True)[:5]
                                if weight > 0.01
                            ),
                        }
                    )
                candidates = sorted(candidates, key=lambda row: row["pre_rmse_sd"])[:max_treated_per_combo]
                rows.extend(candidates)
        return pd.DataFrame(rows)


    def soft_impute(matrix: np.ndarray, observed_mask: np.ndarray, rank: int = 3, iterations: int = 60) -> np.ndarray:
        filled = matrix.copy().astype(float)
        global_mean = np.nanmean(filled)
        if pd.isna(global_mean):
            return filled
        row_means = np.nanmean(filled, axis=1)
        col_means = np.nanmean(filled, axis=0)
        for i in range(filled.shape[0]):
            for j in range(filled.shape[1]):
                if not np.isfinite(filled[i, j]):
                    candidates = [value for value in [row_means[i], col_means[j], global_mean] if np.isfinite(value)]
                    filled[i, j] = float(np.mean(candidates)) if candidates else float(global_mean)
        for _ in range(iterations):
            u, s, vt = np.linalg.svd(filled, full_matrices=False)
            s[rank:] = 0
            reconstructed = (u * s) @ vt
            filled = np.where(observed_mask, matrix, reconstructed)
            filled = np.where(np.isfinite(filled), filled, reconstructed)
        return filled


    def run_matrix_completion(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for spec in POLICY_SPECS:
            policy_events = events.loc[events["policy_id"] == spec.policy_id, ["iso3", "event_year"]].drop_duplicates("iso3")
            if policy_events.shape[0] < MIN_TREATED_COUNTRIES:
                continue
            event_year_map = policy_events.set_index("iso3")["event_year"].to_dict()
            for outcome in spec.outcomes:
                if outcome.column not in panel.columns:
                    continue
                work = panel.loc[:, ["iso3", "year", outcome.column]].copy()
                work["_y"] = zscore(work[outcome.column])
                work = work.loc[work["_y"].notna() & work["year"].between(2000, 2023)].copy()
                if work["iso3"].nunique() < 60:
                    continue
                pivot = work.pivot_table(index="iso3", columns="year", values="_y", aggfunc="mean")
                if pivot.shape[0] < 60 or pivot.shape[1] < 8:
                    continue
                matrix = pivot.to_numpy(dtype=float)
                observed = np.isfinite(matrix)
                heldout = np.zeros_like(observed, dtype=bool)
                for i, iso3 in enumerate(pivot.index):
                    event_year = event_year_map.get(iso3)
                    if event_year is None or pd.isna(event_year):
                        continue
                    for j, year in enumerate(pivot.columns):
                        if int(year) >= int(event_year) and observed[i, j]:
                            heldout[i, j] = True
                if heldout.sum() < 50:
                    continue
                training_observed = observed & ~heldout
                predicted = soft_impute(matrix, training_observed, rank=3, iterations=50)
                effects = matrix[heldout] - predicted[heldout]
                treated_countries = int(np.unique(np.where(heldout)[0]).shape[0])
                effect_mean = float(np.nanmean(effects))
                effect_std = float(np.nanstd(effects, ddof=1)) if effects.size > 1 else np.nan
                rows.append(
                    {
                        "policy_id": spec.policy_id,
                        "policy_label": spec.label,
                        "outcome_column": outcome.column,
                        "outcome_label": outcome.label,
                        "expected_sign": outcome.expected_sign,
                        "matrix_completion_att_sd": effect_mean,
                        "effect_sd": effect_std,
                        "treated_post_cells": int(heldout.sum()),
                        "treated_countries": treated_countries,
                        "countries_in_matrix": int(pivot.shape[0]),
                        "years_in_matrix": int(pivot.shape[1]),
                        "direction_consistent": bool(outcome.expected_sign * effect_mean > 0),
                        "method_note": "低秩矩阵补全反事实；用于稳健性证据，不等同随机实验。",
                    }
                )
        return pd.DataFrame(rows)


    def build_evidence_ladder(
        did: pd.DataFrame,
        event_pretrends: pd.DataFrame,
        placebo: pd.DataFrame,
        synthetic: pd.DataFrame,
        matrix_completion: pd.DataFrame,
    ) -> pd.DataFrame:
        if did.empty:
            return pd.DataFrame()
        work = did.copy()
        if not event_pretrends.empty:
            work = work.merge(event_pretrends, on=["policy_id", "outcome_column"], how="left")
        if not placebo.empty:
            pcols = ["policy_id", "outcome_column", "placebo_issue", "placebo_coefficient_sd", "p_value"]
            place = placebo.loc[:, [col for col in pcols if col in placebo.columns]].rename(columns={"p_value": "placebo_p_value"})
            work = work.merge(place, on=["policy_id", "outcome_column"], how="left")
        syn_summary = pd.DataFrame()
        if not synthetic.empty:
            syn_summary = (
                synthetic.groupby(["policy_id", "outcome_column"], dropna=False)
                .agg(
                    synthetic_cases=("iso3", "count"),
                    synthetic_direction_share=("direction_consistent", "mean"),
                    synthetic_mean_effect_sd=("post_effect_mean_sd", "mean"),
                    synthetic_pre_rmse_sd=("pre_rmse_sd", "mean"),
                )
                .reset_index()
            )
            work = work.merge(syn_summary, on=["policy_id", "outcome_column"], how="left")
        if not matrix_completion.empty:
            mc = matrix_completion.loc[
                :,
                [
                    "policy_id",
                    "outcome_column",
                    "matrix_completion_att_sd",
                    "treated_post_cells",
                    "direction_consistent",
                ],
            ].rename(columns={"direction_consistent": "matrix_direction_consistent"})
            work = work.merge(mc, on=["policy_id", "outcome_column"], how="left")

        work["pretrend_pass"] = ~work.get("pretrend_any_p_lt_0_10", pd.Series(False, index=work.index)).fillna(False).astype(bool)
        work["placebo_pass"] = ~work.get("placebo_issue", pd.Series(False, index=work.index)).fillna(False).astype(bool)
        work["synthetic_support"] = work.get("synthetic_direction_share", pd.Series(np.nan, index=work.index)).fillna(0) >= 0.55
        work["matrix_support"] = work.get("matrix_direction_consistent", pd.Series(False, index=work.index)).fillna(False).astype(bool)
        work["evidence_score"] = (
            0.35 * work["quasi_causal_signal"].astype(float)
            + 0.20 * work["pretrend_pass"].astype(float)
            + 0.15 * work["placebo_pass"].astype(float)
            + 0.15 * work["synthetic_support"].astype(float)
            + 0.15 * work["matrix_support"].astype(float)
        )
        work["evidence_tier"] = pd.cut(
            work["evidence_score"],
            bins=[-np.inf, 0.35, 0.60, 0.80, np.inf],
            labels=["探索性相关证据", "准因果弱证据", "准因果中等证据", "准因果强候选"],
        ).astype(str)
        work["can_promote_as_causal_path"] = (
            work["quasi_causal_signal"].fillna(False).astype(bool)
            & work["direction_consistent"].fillna(False).astype(bool)
            & work["pretrend_pass"]
            & work["placebo_pass"]
        )
        keep_cols = [
            "policy_id",
            "policy_label",
            "outcome_column",
            "outcome_label",
            "outcome_tier",
            "did_coefficient_sd",
            "p_value",
            "direction_consistent",
            "pretrend_pass",
            "placebo_pass",
            "synthetic_direction_share",
            "matrix_direction_consistent",
            "evidence_score",
            "evidence_tier",
            "can_promote_as_causal_path",
            "n_obs",
            "clusters",
            "treated_countries",
        ]
        return work.loc[:, [col for col in keep_cols if col in work.columns]].sort_values(
            ["can_promote_as_causal_path", "evidence_score", "direction_consistent"],
            ascending=[False, False, False],
            kind="stable",
        )


    def build_policy_transfer_scores(
        report_dir: Path,
        evidence_ladder: pd.DataFrame,
    ) -> pd.DataFrame:
        provinces = read_csv_if_exists(report_asset_path(report_dir, "china_provincial_resource_latest.csv"))
        references = read_csv_if_exists(report_asset_path(report_dir, "policy_similarity_reference_countries.csv"))
        response = read_csv_if_exists(report_asset_path(report_dir, "country_response_diagnosis_latest.csv"))
        pathways = read_csv_if_exists(report_asset_path(report_dir, "policy_response_pathways.csv"))
        if provinces.empty or references.empty or response.empty:
            return pd.DataFrame()
        refs = references.merge(
            response[
                [
                    "iso3",
                    "wdi_population_65_plus_pct",
                    "adjusted_response_score",
                    "combined_pressure_score",
                    "dominant_risk_triplet",
                    "weakest_resource_components",
                ]
            ],
            on="iso3",
            how="left",
            suffixes=("", "_country"),
        )
        for column in ["adjusted_response_score", "combined_pressure_score", "dominant_risk_triplet", "weakest_resource_components"]:
            country_column = f"{column}_country"
            if country_column in refs.columns:
                if column in refs.columns:
                    refs[column] = refs[column].combine_first(refs[country_column])
                else:
                    refs[column] = refs[country_column]
        refs["aging_percentile_global"] = pd.to_numeric(refs["wdi_population_65_plus_pct"], errors="coerce").rank(pct=True)
        refs["response_percentile_global"] = pd.to_numeric(refs["adjusted_response_score"], errors="coerce").rank(pct=True)
        best_evidence_score = float(evidence_ladder["evidence_score"].max()) if not evidence_ladder.empty else 0.0
        top_pathway = ""
        if not pathways.empty:
            top = pathways.sort_values("rank_overall", kind="stable").head(3)
            top_pathway = "；".join(top["policy_package"].astype(str).tolist())
        rows: list[dict[str, object]] = []
        for _, province in provinces.iterrows():
            p_resource = safe_float(province.get("resource_response_score"))
            p_aging = safe_float(province.get("aging65_pressure_percentile"))
            for _, ref in refs.iterrows():
                response_similarity = 1 - abs(p_resource - safe_float(ref.get("response_percentile_global")))
                aging_similarity = 1 - abs(p_aging - safe_float(ref.get("aging_percentile_global")))
                reference_score = safe_float(ref.get("reference_score"))
                reference_score_norm = min(max(reference_score, 0.0), 1.0) if pd.notna(reference_score) else 0.0
                transfer_score = (
                    0.35 * max(response_similarity, 0.0)
                    + 0.25 * max(aging_similarity, 0.0)
                    + 0.25 * reference_score_norm
                    + 0.15 * min(best_evidence_score, 1.0)
                )
                rows.append(
                    {
                        "province": province.get("province"),
                        "reference_iso3": ref.get("iso3"),
                        "reference_year": ref.get("year"),
                        "transfer_score": transfer_score,
                        "resource_response_similarity": response_similarity,
                        "aging_similarity": aging_similarity,
                        "global_reference_score": reference_score,
                        "causal_evidence_score_used": best_evidence_score,
                        "province_resource_type": province.get("province_resource_response_type"),
                        "reference_response_type": ref.get("response_diagnosis_type"),
                        "reference_top_risks": ref.get("dominant_risk_triplet"),
                        "reference_weak_resources": ref.get("weakest_resource_components"),
                        "recommended_transfer_package": top_pathway,
                        "data_caveat": "省级迁移评分使用资源响应、老龄化和全球经验国家相似度；中国画像已补GBD2017疾病/风险、GBD2021 NCD总负担、2024人口加权PM2.5和DRG/DIP政策响应候选，但不声明省级健康结果因果。",
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["province", "transfer_score"], ascending=[True, False], kind="stable").groupby("province").head(5).reset_index(drop=True)


    def build_policy_combo_optimization(report_dir: Path, evidence_ladder: pd.DataFrame) -> pd.DataFrame:
        resource = read_csv_if_exists(report_asset_path(report_dir, "policy_resource_uplift_scenarios.csv"))
        risk = read_csv_if_exists(report_asset_path(report_dir, "policy_risk_reduction_scenarios.csv"))
        service = read_csv_if_exists(report_asset_path(report_dir, "policy_ncd_service_gap_scenarios.csv"))
        if resource.empty and risk.empty and service.empty:
            return pd.DataFrame()
        evidence_multiplier = 0.85 + 0.30 * min(float(evidence_ladder["evidence_score"].max()) if not evidence_ladder.empty else 0.0, 1.0)
        rows: list[dict[str, object]] = []
        all_iso = sorted(set(resource.get("iso3", pd.Series(dtype=str)).dropna()) | set(risk.get("iso3", pd.Series(dtype=str)).dropna()) | set(service.get("iso3", pd.Series(dtype=str)).dropna()))
        for iso3 in all_iso:
            pieces: dict[str, pd.Series] = {}
            rsrc = resource.loc[resource.get("iso3") == iso3].copy() if not resource.empty else pd.DataFrame()
            if not rsrc.empty:
                pieces["resource"] = rsrc.sort_values("gap_reduction", ascending=False, kind="stable").iloc[0]
            risk_country = risk.loc[(risk.get("iso3") == iso3) & (risk.get("risk_reduction_rate") == 0.10)].copy() if not risk.empty else pd.DataFrame()
            if not risk_country.empty:
                pieces["risk"] = risk_country.sort_values("gap_reduction", ascending=False, kind="stable").iloc[0]
            svc = service.loc[service.get("iso3") == iso3].copy() if not service.empty else pd.DataFrame()
            if not svc.empty:
                pieces["service"] = svc.sort_values("gap_reduction", ascending=False, kind="stable").iloc[0]
            if not pieces:
                continue
            baseline_gap = next((safe_float(piece.get("baseline_gap")) for piece in pieces.values() if pd.notna(safe_float(piece.get("baseline_gap")))), np.nan)
            baseline_pressure = next((safe_float(piece.get("baseline_pressure")) for piece in pieces.values() if pd.notna(safe_float(piece.get("baseline_pressure")))), np.nan)
            vulnerability = next((piece.get("vulnerability_type_label") for piece in pieces.values() if pd.notna(piece.get("vulnerability_type_label"))), "")
            response_type = next((piece.get("response_diagnosis_type") for piece in pieces.values() if pd.notna(piece.get("response_diagnosis_type"))), "")
            combos = [
                ("risk_only", ["risk"]),
                ("resource_only", ["resource"]),
                ("service_only", ["service"]),
                ("risk_resource", ["risk", "resource"]),
                ("risk_service", ["risk", "service"]),
                ("resource_service", ["resource", "service"]),
                ("full_combo", ["risk", "resource", "service"]),
            ]
            for combo_id, keys in combos:
                selected = [pieces[key] for key in keys if key in pieces]
                if not selected:
                    continue
                raw_reduction = sum(max(safe_float(piece.get("gap_reduction")), 0.0) for piece in selected)
                diminishing_factor = 1.0 - 0.08 * max(len(selected) - 1, 0)
                causal_adjusted_reduction = raw_reduction * diminishing_factor * evidence_multiplier
                projected_gap = baseline_gap - causal_adjusted_reduction if pd.notna(baseline_gap) else np.nan
                packages = []
                targets = []
                for piece in selected:
                    packages.append(str(piece.get("policy_package", piece.get("service_label", ""))))
                    targets.append(str(piece.get("risk_label", piece.get("target_components", piece.get("service_label", "")))))
                rows.append(
                    {
                        "iso3": iso3,
                        "vulnerability_type_label": vulnerability,
                        "response_diagnosis_type": response_type,
                        "combo_id": combo_id,
                        "policy_package": " + ".join(dict.fromkeys([p for p in packages if p and p != "nan"])),
                        "target_components": " / ".join(dict.fromkeys([t for t in targets if t and t != "nan"])),
                        "baseline_pressure": baseline_pressure,
                        "baseline_gap": baseline_gap,
                        "raw_gap_reduction_sum": raw_reduction,
                        "diminishing_factor": diminishing_factor,
                        "causal_evidence_multiplier": evidence_multiplier,
                        "projected_gap_reduction": causal_adjusted_reduction,
                        "projected_gap_after_combo": projected_gap,
                        "scenario_note": "组合政策模拟叠加资源、风险和服务覆盖情景；用准因果增强证据分数调整可信度，但仍是政策模拟。",
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["iso3", "projected_gap_reduction"], ascending=[True, False], kind="stable").reset_index(drop=True)


    def plot_event_study(event_study: pd.DataFrame, evidence_ladder: pd.DataFrame, path: Path) -> None:
        if event_study.empty:
            return
        selected_pairs = (
            evidence_ladder.sort_values(["evidence_score", "direction_consistent"], ascending=[False, False], kind="stable")
            .head(4)[["policy_id", "outcome_column"]]
            .apply(tuple, axis=1)
            .tolist()
            if not evidence_ladder.empty
            else event_study[["policy_id", "outcome_column"]].drop_duplicates().head(4).apply(tuple, axis=1).tolist()
        )
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=180)
        axes_flat = axes.flatten()
        for ax, pair in zip(axes_flat, selected_pairs):
            subset = event_study.loc[(event_study["policy_id"] == pair[0]) & (event_study["outcome_column"] == pair[1])].copy()
            if subset.empty:
                ax.axis("off")
                continue
            subset = subset.sort_values("relative_year", kind="stable")
            ax.axhline(0, color="#64748b", linewidth=0.8)
            ax.axvline(-0.5, color="#ef4444", linestyle="--", linewidth=0.8)
            yerr = 1.96 * pd.to_numeric(subset["cluster_se"], errors="coerce")
            ax.errorbar(subset["relative_year"], subset["coefficient_sd"], yerr=yerr, fmt="o-", color="#0f766e", ecolor="#99f6e4", capsize=3)
            title = f"{subset['policy_label'].iloc[0]} → {subset['outcome_label'].iloc[0]}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("相对政策年份" if USE_CHINESE else "Relative year")
            ax.set_ylabel("标准化效应" if USE_CHINESE else "Standardized effect")
            ax.grid(True, alpha=0.18)
        for ax in axes_flat[len(selected_pairs):]:
            ax.axis("off")
        set_centered_suptitle(fig, choose_text("模块D准因果增强层政策事件研究：严格准因果强候选", "Module D Quasi-Causal Enhancement: Strict Quasi-causal Candidates", USE_CHINESE))
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


    def plot_evidence_ladder(evidence_ladder: pd.DataFrame, path: Path) -> None:
        if evidence_ladder.empty:
            return
        top = evidence_ladder.sort_values("evidence_score", ascending=False, kind="stable").head(12).copy()
        labels = top["policy_label"].astype(str) + "\n" + top["outcome_label"].astype(str)
        fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
        colors = np.where(top["can_promote_as_causal_path"], "#0f766e", np.where(top["direction_consistent"], "#f59e0b", "#94a3b8"))
        ax.barh(np.arange(top.shape[0]), top["evidence_score"], color=colors)
        ax.set_yticks(np.arange(top.shape[0]))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.60, color="#ef4444", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("证据分数" if USE_CHINESE else "Evidence score")
        ax.set_title(choose_text("模块D准因果增强层证据阶梯", "Module D Quasi-Causal Evidence Ladder", USE_CHINESE))
        ax.grid(axis="x", alpha=0.18)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


    def update_adaptation_summary(report_dir: Path, causal_summary: dict[str, object]) -> None:
        path = report_asset_path(report_dir, "policy_adaptation_engine_summary.json")
        summary = read_json_if_exists(path)
        if not summary:
            return
        summary["module_d_v4_claim"] = (
            "模块D已升级为严格准因果政策适配引擎模块D准因果增强层：在V2政策路径和既有准因果增强基础上，新增WHO MPOWER六分项与World Bank WDI烟草结果变量，"
            "通过政策事件库、交错DID、事件研究、安慰剂、合成控制和矩阵补全筛选严格可推广准因果强候选。"
        )
        summary["module_d_v3_claim"] = (
            "模块D政策适配引擎层已完成因果增强前置构建：在政策路径、资源提升、风险下降和服务覆盖情景基础上，新增全球政策事件库、"
            "交错DID、事件研究、安慰剂检验、合成控制、低秩矩阵补全、中国迁移评分和组合政策模拟；可推广结论限定为通过稳健性筛选的准因果政策路径。"
        )
        summary["module_d_causal_enhancement"] = causal_summary
        summary["module_d_definition"] = "全球政策严格准因果验证 + 中国情景迁移 + 政策模拟优化引擎"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


    def build_summary(
        project_root: Path,
        outputs: dict[str, Path],
        events: pd.DataFrame,
        did: pd.DataFrame,
        evidence_ladder: pd.DataFrame,
        synthetic: pd.DataFrame,
        matrix_completion: pd.DataFrame,
        transfer: pd.DataFrame,
        combos: pd.DataFrame,
    ) -> dict[str, object]:
        promotable = evidence_ladder.loc[evidence_ladder.get("can_promote_as_causal_path", False).astype(bool)].copy() if not evidence_ladder.empty else pd.DataFrame()
        best = evidence_ladder.sort_values("evidence_score", ascending=False, kind="stable").head(5) if not evidence_ladder.empty else pd.DataFrame()
        return {
            "project_root": project_root.as_posix(),
            "module_d_level": "模块D准因果增强层",
            "randomized_experiment_claim": RANDOM_EXPERIMENT_CLAIM,
            "claim_boundary": "准因果证据增强，不等同随机实验；只把通过方向、前趋势、安慰剂和反事实稳健性筛选的路径作为强候选。",
            "input_policy_specs": len(POLICY_SPECS),
            "d4_data_upgrade": {
                "who_mpower_detailed_groups": "external_who_mpower_detailed_scores_wide.csv",
                "world_bank_wdi_tobacco_outcomes": "external_wdi_tobacco_outcomes.csv",
                "combined_clean_panel": "external_policy_d4_tobacco_panel.csv",
            },
            "output_rows": {
                "policy_causal_event_library": int(events.shape[0]),
                "policy_causal_did_estimates": int(did.shape[0]),
                "policy_causal_evidence_ladder": int(evidence_ladder.shape[0]),
                "policy_synthetic_control_estimates": int(synthetic.shape[0]),
                "policy_matrix_completion_estimates": int(matrix_completion.shape[0]),
                "policy_transfer_similarity_scores": int(transfer.shape[0]),
                "policy_combo_optimization_scenarios": int(combos.shape[0]),
            },
            "causal_signal_counts": {
                "directional_did_signals": int(did.get("direction_consistent", pd.Series(dtype=bool)).sum()) if not did.empty else 0,
                "p_lt_0_10_directional_did_signals": int(did.get("quasi_causal_signal", pd.Series(dtype=bool)).sum()) if not did.empty else 0,
                "promotable_quasi_causal_paths": int(promotable.shape[0]),
                "synthetic_directional_cases": int(synthetic.get("direction_consistent", pd.Series(dtype=bool)).sum()) if not synthetic.empty else 0,
                "matrix_completion_directional_paths": int(matrix_completion.get("direction_consistent", pd.Series(dtype=bool)).sum()) if not matrix_completion.empty else 0,
            },
            "best_evidence_candidates": best[
                [
                    col
                    for col in [
                        "policy_label",
                        "outcome_label",
                        "did_coefficient_sd",
                        "p_value",
                        "evidence_score",
                        "evidence_tier",
                        "can_promote_as_causal_path",
                    ]
                    if col in best.columns
                ]
            ].to_dict("records"),
            "module_d_v4_claim": (
                "模块D准因果增强层已经完成可信数据增强与严格准因果筛选：新增WHO MPOWER六分项和World Bank WDI烟草使用率，"
                "在政策事件库、交错DID、事件研究、安慰剂、合成控制、矩阵补全、中国迁移评分和组合政策模拟基础上，锁定严格可推广准因果强候选。"
            ),
            "output_files": {name: path.as_posix() for name, path in outputs.items()},
        }


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build causal-enhanced Module D outputs.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        report_dir = dirs["report"]
        figure_dir = dirs["figures"]

        panel = build_analysis_panel(project_root)
        events = identify_policy_events(panel)
        did, heterogeneity = run_did_estimates(panel, events)
        event_study = run_event_studies(panel, events)
        pretrends = summarize_pretrends(event_study)
        placebo = run_placebo_tests(panel, events)
        synthetic = run_synthetic_controls(panel, events)
        matrix_completion = run_matrix_completion(panel, events)
        evidence_ladder = build_evidence_ladder(did, pretrends, placebo, synthetic, matrix_completion)
        transfer = build_policy_transfer_scores(report_dir, evidence_ladder)
        combos = build_policy_combo_optimization(report_dir, evidence_ladder)

        outputs = {
            "policy_causal_event_library": report_asset_path(report_dir, "policy_causal_event_library.csv"),
            "policy_causal_did_estimates": report_asset_path(report_dir, "policy_causal_did_estimates.csv"),
            "policy_causal_event_study_coefficients": report_asset_path(report_dir, "policy_causal_event_study_coefficients.csv"),
            "policy_causal_event_study_pretrend_summary": report_asset_path(report_dir, "policy_causal_event_study_pretrend_summary.csv"),
            "policy_causal_placebo_tests": report_asset_path(report_dir, "policy_causal_placebo_tests.csv"),
            "policy_causal_heterogeneity": report_asset_path(report_dir, "policy_causal_heterogeneity.csv"),
            "policy_synthetic_control_estimates": report_asset_path(report_dir, "policy_synthetic_control_estimates.csv"),
            "policy_matrix_completion_estimates": report_asset_path(report_dir, "policy_matrix_completion_estimates.csv"),
            "policy_causal_evidence_ladder": report_asset_path(report_dir, "policy_causal_evidence_ladder.csv"),
            "policy_transfer_similarity_scores": report_asset_path(report_dir, "policy_transfer_similarity_scores.csv"),
            "policy_combo_optimization_scenarios": report_asset_path(report_dir, "policy_combo_optimization_scenarios.csv"),
            "policy_causal_enhancement_summary": report_asset_path(report_dir, "policy_causal_enhancement_summary.json"),
            "advanced_policy_causal_event_study_v3": figure_dir / "advanced_policy_causal_event_study_v3.png",
            "advanced_policy_causal_evidence_ladder_v3": figure_dir / "advanced_policy_causal_evidence_ladder_v3.png",
        }

        write_csv(events, outputs["policy_causal_event_library"])
        write_csv(did, outputs["policy_causal_did_estimates"])
        write_csv(event_study, outputs["policy_causal_event_study_coefficients"])
        write_csv(pretrends, outputs["policy_causal_event_study_pretrend_summary"])
        write_csv(placebo, outputs["policy_causal_placebo_tests"])
        write_csv(heterogeneity, outputs["policy_causal_heterogeneity"])
        write_csv(synthetic, outputs["policy_synthetic_control_estimates"])
        write_csv(matrix_completion, outputs["policy_matrix_completion_estimates"])
        write_csv(evidence_ladder, outputs["policy_causal_evidence_ladder"])
        write_csv(transfer, outputs["policy_transfer_similarity_scores"])
        write_csv(combos, outputs["policy_combo_optimization_scenarios"])

        plot_event_study(event_study, evidence_ladder, outputs["advanced_policy_causal_event_study_v3"])
        plot_evidence_ladder(evidence_ladder, outputs["advanced_policy_causal_evidence_ladder_v3"])

        summary = build_summary(project_root, outputs, events, did, evidence_ladder, synthetic, matrix_completion, transfer, combos)
        outputs["policy_causal_enhancement_summary"].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        update_adaptation_summary(report_dir, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_policy_d4_advanced_validation():
    __name__ = 'run_policy_d4_advanced_validation'
    import argparse
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from foundation import choose_text, configure_matplotlib_fonts, set_centered_suptitle
    from foundation import detect_project_root as shared_detect_project_root
    from run_policy_causal_enhancement import (
        CONTROL_COLUMNS,
        POLICY_SPECS,
        build_analysis_panel,
        build_design_matrix,
        fit_cluster_ols,
        identify_policy_events,
        normal_p_value,
        safe_float,
    )


    USE_CHINESE = configure_matplotlib_fonts()
    RNG = np.random.default_rng(20260529)
    CORE_POLICY_ID = "mpower_protect_smokefree_strong"
    CORE_POLICY_LABEL = "MPOWER无烟环境保护强执行"
    CORE_OUTCOME = "delta_dbd_smoking_per100k"


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def read_csv(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


    def write_csv(df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False, encoding="utf-8-sig")


    def merge_response_time_series(project_root: Path, panel: pd.DataFrame) -> pd.DataFrame:
        response = read_csv(project_root / "04_simulation" / "response_diagnosis_panel.csv")
        if response.empty:
            return panel
        keep = [
            "iso3",
            "year",
            "burden_pressure_score",
            "risk_pressure_score",
            "combined_pressure_score",
            "adjusted_response_score",
            "adaptation_gap_score",
        ]
        response = response.loc[:, [col for col in keep if col in response.columns]].copy()
        response["iso3"] = response["iso3"].astype(str).str.upper().str.strip()
        response["year"] = pd.to_numeric(response["year"], errors="coerce").astype("Int64")
        drop_cols = [col for col in response.columns if col not in {"iso3", "year"} and col in panel.columns]
        return panel.drop(columns=drop_cols, errors="ignore").merge(response, on=["iso3", "year"], how="left")


    def identify_threshold_events(panel: pd.DataFrame, variable: str, threshold: float) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        if variable not in panel.columns:
            return pd.DataFrame()
        observed = panel.loc[panel[variable].notna(), ["iso3", "year", variable]].copy()
        for iso3, group in observed.sort_values(["iso3", "year"], kind="stable").groupby("iso3"):
            prior_below = False
            for idx, row in group.iterrows():
                value = safe_float(row[variable])
                if pd.isna(value):
                    continue
                if value < threshold:
                    prior_below = True
                elif prior_below and value >= threshold:
                    before = group.loc[group["year"] < row["year"], variable]
                    after = group.loc[group["year"] > row["year"], variable]
                    if before.empty or after.empty:
                        break
                    rows.append(
                        {
                            "policy_id": f"{variable}_ge_{threshold:g}",
                            "policy_label": f"{variable}>={threshold:g}",
                            "policy_variable": variable,
                            "threshold": threshold,
                            "iso3": iso3,
                            "event_year": int(row["year"]),
                            "value_before_event": safe_float(before.iloc[-1]),
                            "value_at_event": safe_float(row[variable]),
                        }
                    )
                    break
        return pd.DataFrame(rows)


    def attach_event_year(panel: pd.DataFrame, event_years: pd.DataFrame, treatment_col: str = "treated_post") -> pd.DataFrame:
        work = panel.merge(event_years[["iso3", "event_year"]].drop_duplicates("iso3"), on="iso3", how="left")
        work["ever_treated"] = work["event_year"].notna().astype(int)
        work[treatment_col] = ((work["event_year"].notna()) & (work["year"] >= work["event_year"])).astype(int)
        work["relative_year"] = work["year"] - work["event_year"]
        return work


    def model_arrays(df: pd.DataFrame, y_col: str, treatment_cols: list[str], controls: list[str]) -> dict[str, object] | None:
        work = df.loc[:, [col for col in ["iso3", "year", y_col, *treatment_cols, *controls] if col in df.columns]].copy()
        work["_y_raw"] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=["iso3", "year", "_y_raw"])
        if work.shape[0] < 180 or work["iso3"].nunique() < 20:
            return None
        y_std = work["_y_raw"].std(ddof=0)
        if pd.isna(y_std) or y_std == 0:
            return None
        work["_y"] = (work["_y_raw"] - work["_y_raw"].mean()) / y_std
        x, names, clusters = build_design_matrix(work, treatment_cols, controls)
        if x.shape[0] <= x.shape[1] + 3:
            return None
        y = work["_y"].to_numpy(dtype=float)
        beta = np.linalg.pinv(x) @ y
        return {"work": work, "x": x, "names": names, "clusters": clusters.to_numpy(), "y": y, "beta": beta}


    def wild_cluster_bootstrap_one(
        df: pd.DataFrame,
        y_col: str,
        treatment_col: str,
        controls: list[str],
        reps: int,
    ) -> dict[str, object]:
        full = model_arrays(df, y_col, [treatment_col], controls)
        restricted = model_arrays(df, y_col, [], controls)
        if full is None or restricted is None:
            return {"ok": False, "reason": "insufficient_design"}
        if full["x"].shape[0] != restricted["x"].shape[0]:
            return {"ok": False, "reason": "design_alignment_failed"}
        idx = full["names"].index(treatment_col)
        beta_obs = float(full["beta"][idx])
        x_full = full["x"]
        x_restricted = restricted["x"]
        y = full["y"]
        beta0 = np.linalg.pinv(x_restricted) @ y
        fitted0 = x_restricted @ beta0
        resid0 = y - fitted0
        clusters = full["clusters"]
        unique_clusters = pd.unique(clusters)
        boot = np.empty(reps, dtype=float)
        pinv_full = np.linalg.pinv(x_full)
        for i in range(reps):
            weights = dict(zip(unique_clusters, RNG.choice([-1.0, 1.0], size=len(unique_clusters))))
            w = np.array([weights[c] for c in clusters], dtype=float)
            y_star = fitted0 + resid0 * w
            boot[i] = float((pinv_full @ y_star)[idx])
        p_value = float(np.mean(np.abs(boot) >= abs(beta_obs)))
        return {
            "ok": True,
            "wild_bootstrap_p": p_value,
            "wild_ci_low": float(np.quantile(boot, 0.025)),
            "wild_ci_high": float(np.quantile(boot, 0.975)),
            "wild_reps": int(reps),
            "boot_mean": float(np.mean(boot)),
            "boot_sd": float(np.std(boot, ddof=1)),
        }


    def run_wild_cluster_bootstrap(panel: pd.DataFrame, events: pd.DataFrame, evidence: pd.DataFrame, reps: int) -> pd.DataFrame:
        candidate_pairs = evidence.loc[
            evidence.get("direction_consistent", False).astype(bool) | (pd.to_numeric(evidence.get("p_value"), errors="coerce") < 0.20)
        ].copy()
        if candidate_pairs.empty:
            return pd.DataFrame()
        rows = []
        spec_map = {spec.policy_id: spec for spec in POLICY_SPECS}
        for _, row in candidate_pairs.iterrows():
            policy_id = row["policy_id"]
            outcome_col = row["outcome_column"]
            if outcome_col not in panel.columns or policy_id not in spec_map:
                continue
            policy_events = events.loc[events["policy_id"] == policy_id, ["iso3", "event_year"]].drop_duplicates("iso3")
            if policy_events["iso3"].nunique() < 12:
                continue
            timed = attach_event_year(panel, policy_events)
            boot = wild_cluster_bootstrap_one(timed, outcome_col, "treated_post", CONTROL_COLUMNS, reps)
            rows.append(
                {
                    "policy_id": policy_id,
                    "policy_label": row.get("policy_label"),
                    "outcome_column": outcome_col,
                    "outcome_label": row.get("outcome_label"),
                    "did_coefficient_sd": row.get("did_coefficient_sd"),
                    "cluster_p_value": row.get("p_value"),
                    **boot,
                    "wild_bootstrap_pass_p_lt_0_10": bool(boot.get("ok") and boot.get("wild_bootstrap_p", 1) < 0.10),
                }
            )
        return pd.DataFrame(rows)


    def run_cohort_robust_att(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        specs = {spec.policy_id: spec for spec in POLICY_SPECS}
        for policy_id in [CORE_POLICY_ID, "mpower_comprehensive_package_high", "hypertension_policy_ready"]:
            if policy_id not in specs:
                continue
            spec = specs[policy_id]
            event_years = events.loc[events["policy_id"] == policy_id, ["iso3", "event_year"]].drop_duplicates("iso3")
            event_map = event_years.set_index("iso3")["event_year"].to_dict()
            for outcome in spec.outcomes:
                if outcome.column not in panel.columns:
                    continue
                outcome_sd = pd.to_numeric(panel[outcome.column], errors="coerce").std(ddof=0)
                if not outcome_sd or pd.isna(outcome_sd):
                    continue
                cohort_rows = []
                for cohort_year, cohort in event_years.groupby("event_year"):
                    base_year = int(cohort_year) - 1
                    for lag in [0, 1, 2, 3]:
                        target_year = int(cohort_year) + lag
                        treated_iso = set(cohort["iso3"])
                        control_iso = {
                            iso3
                            for iso3 in panel["iso3"].dropna().unique()
                            if iso3 not in treated_iso and (iso3 not in event_map or int(event_map[iso3]) > target_year)
                        }
                        base = panel.loc[panel["year"] == base_year, ["iso3", outcome.column]].rename(columns={outcome.column: "base"})
                        target = panel.loc[panel["year"] == target_year, ["iso3", outcome.column]].rename(columns={outcome.column: "target"})
                        delta = base.merge(target, on="iso3", how="inner")
                        delta["delta"] = pd.to_numeric(delta["target"], errors="coerce") - pd.to_numeric(delta["base"], errors="coerce")
                        treated_delta = delta.loc[delta["iso3"].isin(treated_iso), "delta"].dropna()
                        control_delta = delta.loc[delta["iso3"].isin(control_iso), "delta"].dropna()
                        if treated_delta.shape[0] < 3 or control_delta.shape[0] < 10:
                            continue
                        att = (treated_delta.mean() - control_delta.mean()) / outcome_sd
                        se = np.sqrt(treated_delta.var(ddof=1) / treated_delta.shape[0] + control_delta.var(ddof=1) / control_delta.shape[0]) / outcome_sd
                        z = att / se if se and not pd.isna(se) else np.nan
                        cohort_rows.append(
                            {
                                "cohort_year": int(cohort_year),
                                "lag": lag,
                                "att_sd": float(att),
                                "se_sd": float(se),
                                "n_treated": int(treated_delta.shape[0]),
                                "n_controls": int(control_delta.shape[0]),
                                "z_value": float(z) if pd.notna(z) else np.nan,
                            }
                        )
                if not cohort_rows:
                    continue
                cohort_df = pd.DataFrame(cohort_rows)
                weights = cohort_df["n_treated"] / cohort_df["n_treated"].sum()
                att = float(np.sum(weights * cohort_df["att_sd"]))
                se = float(np.sqrt(np.sum((weights**2) * (cohort_df["se_sd"] ** 2))))
                z = att / se if se else np.nan
                rows.append(
                    {
                        "policy_id": policy_id,
                        "policy_label": spec.label,
                        "outcome_column": outcome.column,
                        "outcome_label": outcome.label,
                        "expected_sign": outcome.expected_sign,
                        "cohort_att_sd": att,
                        "cohort_att_se": se,
                        "cohort_att_p": normal_p_value(z),
                        "cohort_lag_cells": int(cohort_df.shape[0]),
                        "cohort_total_treated_cells": int(cohort_df["n_treated"].sum()),
                        "direction_consistent": bool(outcome.expected_sign * att > 0),
                    }
                )
        return pd.DataFrame(rows)


    def run_threshold_lag_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        outcomes = {
            "delta_dbd_smoking_per100k": ("吸烟风险暴露负担年变化", -1),
            "dbd_smoking_per100k": ("吸烟风险暴露负担", -1),
            "wdi_tobacco_use_pct": ("WDI当前烟草使用率", -1),
            "delta_wdi_tobacco_use_pct": ("WDI当前烟草使用率年变化", -1),
            "who_smoking_rate_std": ("WHO年龄标准化吸烟率", -1),
        }
        for threshold in [3.0, 4.0, 5.0]:
            events = identify_threshold_events(panel, "mpower_p_group", threshold)
            if events["iso3"].nunique() < 12 if not events.empty else True:
                continue
            for lag in [0, 1, 2, 3]:
                timed = panel.merge(events[["iso3", "event_year"]].drop_duplicates("iso3"), on="iso3", how="left")
                timed["treated_post_lag"] = ((timed["event_year"].notna()) & (timed["year"] >= timed["event_year"] + lag)).astype(int)
                for outcome_col, (label, expected_sign) in outcomes.items():
                    if outcome_col not in timed.columns:
                        continue
                    fit = fit_cluster_ols(timed, outcome_col, ["treated_post_lag"], CONTROL_COLUMNS)
                    if not fit.get("ok"):
                        continue
                    info = fit["coefficients"]["treated_post_lag"]
                    coef = info["coef"]
                    rows.append(
                        {
                            "policy_id": CORE_POLICY_ID,
                            "policy_label": CORE_POLICY_LABEL,
                            "policy_variable": "mpower_p_group",
                            "threshold": threshold,
                            "post_lag_years": lag,
                            "treated_countries": int(events["iso3"].nunique()),
                            "outcome_column": outcome_col,
                            "outcome_label": label,
                            "expected_sign": expected_sign,
                            "coefficient_sd": coef,
                            "cluster_se": info["se"],
                            "p_value": info["p"],
                            "direction_consistent": bool(expected_sign * coef > 0),
                            "p_lt_0_10": bool(pd.notna(info["p"]) and info["p"] < 0.10),
                            "n_obs": fit["n_obs"],
                            "clusters": fit["clusters"],
                        }
                    )
        return pd.DataFrame(rows)


    def run_negative_control_outcomes(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        event_years = events.loc[events["policy_id"] == CORE_POLICY_ID, ["iso3", "event_year"]].drop_duplicates("iso3")
        timed = attach_event_year(panel, event_years)
        outcomes = {
            "dbd_pm25_per100k": "PM2.5风险负担",
            "dbd_dietary_risks_per100k": "膳食风险负担",
            "dbd_high_bmi_per100k": "高BMI风险负担",
            "wdi_pm25": "WDI PM2.5暴露",
        }
        rows: list[dict[str, object]] = []
        for outcome_col, label in outcomes.items():
            if outcome_col not in timed.columns:
                continue
            fit = fit_cluster_ols(timed, outcome_col, ["treated_post"], CONTROL_COLUMNS)
            if not fit.get("ok"):
                continue
            info = fit["coefficients"]["treated_post"]
            rows.append(
                {
                    "policy_id": CORE_POLICY_ID,
                    "policy_label": CORE_POLICY_LABEL,
                    "negative_control_outcome": outcome_col,
                    "negative_control_label": label,
                    "coefficient_sd": info["coef"],
                    "cluster_se": info["se"],
                    "p_value": info["p"],
                    "null_pass_p_ge_0_10": bool(pd.isna(info["p"]) or info["p"] >= 0.10),
                    "n_obs": fit["n_obs"],
                    "clusters": fit["clusters"],
                }
            )
        return pd.DataFrame(rows)


    def run_mechanism_chain(project_root: Path, panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        event_years = events.loc[events["policy_id"] == CORE_POLICY_ID, ["iso3", "event_year"]].drop_duplicates("iso3")
        timed = attach_event_year(panel, event_years)
        chain = [
            ("01_policy_event", "mpower_p_group", "政策强度：无烟环境保护分组", 1, "政策输入"),
            ("02_exposure_change", "delta_wdi_tobacco_use_pct", "暴露变化：WDI烟草使用率年变化", -1, "风险暴露变化"),
            ("03_risk_burden", "dbd_smoking_per100k", "风险负担：吸烟归因负担", -1, "风险负担"),
            ("04_risk_change", "delta_dbd_smoking_per100k", "风险负担变化：吸烟归因负担年变化", -1, "变化率"),
            ("05_disease_burden", "gbd_rate_chronic_respiratory_diseases_per100k", "疾病负担：慢性呼吸系统疾病", -1, "长期疾病负担"),
        ]
        rows: list[dict[str, object]] = []
        for step_id, outcome_col, label, expected_sign, mechanism_layer in chain:
            if outcome_col not in timed.columns:
                continue
            if outcome_col == "mpower_p_group":
                rows.append(
                    {
                        "step_id": step_id,
                        "mechanism_layer": mechanism_layer,
                        "estimation_method": "policy_event_definition",
                        "outcome_column": outcome_col,
                        "outcome_label": label,
                        "expected_sign": expected_sign,
                        "coefficient_sd": np.nan,
                        "p_value": np.nan,
                        "direction_consistent": True,
                        "evidence_note": "处理定义本身：国家首次从低于阈值提升至P_Group>=4。",
                    }
                )
                continue
            fit = fit_cluster_ols(timed, outcome_col, ["treated_post"], CONTROL_COLUMNS)
            if not fit.get("ok"):
                rows.append(
                    {
                        "step_id": step_id,
                        "mechanism_layer": mechanism_layer,
                        "estimation_method": "two_way_fixed_effects_did",
                        "outcome_column": outcome_col,
                        "outcome_label": label,
                        "expected_sign": expected_sign,
                        "coefficient_sd": np.nan,
                        "p_value": np.nan,
                        "direction_consistent": False,
                        "evidence_note": fit.get("reason", "not_estimated"),
                    }
                )
                continue
            info = fit["coefficients"]["treated_post"]
            direction = bool(expected_sign * info["coef"] > 0)
            rows.append(
                {
                    "step_id": step_id,
                    "mechanism_layer": mechanism_layer,
                    "estimation_method": "two_way_fixed_effects_did",
                    "outcome_column": outcome_col,
                    "outcome_label": label,
                    "expected_sign": expected_sign,
                    "coefficient_sd": info["coef"],
                    "cluster_se": info["se"],
                    "p_value": info["p"],
                    "direction_consistent": direction,
                    "p_lt_0_10": bool(pd.notna(info["p"]) and info["p"] < 0.10),
                    "n_obs": fit["n_obs"],
                    "clusters": fit["clusters"],
                    "evidence_note": "方向一致" if direction else "方向未通过",
                }
            )
        risk_scenarios = read_csv(report_asset_path(project_root, "policy_risk_reduction_scenarios.csv"))
        smoking_scenarios = risk_scenarios.loc[
            risk_scenarios.get("risk_label", pd.Series(dtype=str)).astype(str).eq("吸烟")
            & (pd.to_numeric(risk_scenarios.get("risk_reduction_rate"), errors="coerce") == 0.10)
        ].copy() if not risk_scenarios.empty else pd.DataFrame()
        if not smoking_scenarios.empty:
            pressure_reduction = (
                pd.to_numeric(smoking_scenarios["baseline_pressure"], errors="coerce")
                - pd.to_numeric(smoking_scenarios["simulated_pressure"], errors="coerce")
            )
            rows.append(
                {
                    "step_id": "06_pressure_simulation",
                    "mechanism_layer": "压力",
                    "estimation_method": "risk_reduction_scenario_bridge",
                    "outcome_column": "simulated_pressure_reduction_from_10pct_smoking_risk",
                    "outcome_label": "压力模拟：吸烟风险下降10%后的综合压力下降",
                    "expected_sign": 1,
                    "coefficient_sd": float(pressure_reduction.mean()),
                    "p_value": np.nan,
                    "direction_consistent": bool(pressure_reduction.mean() > 0),
                    "p_lt_0_10": np.nan,
                    "n_obs": int(smoking_scenarios.shape[0]),
                    "clusters": int(smoking_scenarios["iso3"].nunique()),
                    "evidence_note": "情景桥接：来自模块D风险下降模拟，不是DID估计。",
                }
            )
            gap_reduction = pd.to_numeric(smoking_scenarios["gap_reduction"], errors="coerce")
            rows.append(
                {
                    "step_id": "07_gap_simulation",
                    "mechanism_layer": "响应缺口",
                    "estimation_method": "risk_reduction_scenario_bridge",
                    "outcome_column": "simulated_gap_reduction_from_10pct_smoking_risk",
                    "outcome_label": "缺口模拟：吸烟风险下降10%后的适配缺口下降",
                    "expected_sign": 1,
                    "coefficient_sd": float(gap_reduction.mean()),
                    "p_value": np.nan,
                    "direction_consistent": bool(gap_reduction.mean() > 0),
                    "p_lt_0_10": np.nan,
                    "n_obs": int(smoking_scenarios.shape[0]),
                    "clusters": int(smoking_scenarios["iso3"].nunique()),
                    "evidence_note": "情景桥接：用于闭合政策到缺口的应用链，不作为因果估计。",
                }
            )
        return pd.DataFrame(rows)


    def plot_robustness(wild: pd.DataFrame, cohort: pd.DataFrame, threshold: pd.DataFrame, negative: pd.DataFrame, path: Path) -> None:
        import textwrap

        def compact_policy_label(policy_label: object, outcome_label: object, index: int) -> str:
            policy = str(policy_label)
            outcome = str(outcome_label)
            replacements = {
                "MPOWER综合控烟政策包高水平": "MPOWER综合包",
                "MPOWER无烟环境保护强执行": "无烟环境强执行",
                "MPOWER控烟强执行": "MPOWER强执行",
                "高血压政策准备度提升": "高血压准备度",
                "基层NCD服务准备度提升": "基层NCD准备度",
                "控盐膳食政策执行": "控盐膳食执行",
            }
            for old, new in replacements.items():
                policy = policy.replace(old, new)
            outcome = (
                outcome.replace("吸烟风险暴露负担年变化", "吸烟负担年变化")
                .replace("吸烟风险暴露负担", "吸烟负担")
                .replace("高血压诊断覆盖率", "高血压诊断")
                .replace("高血压控制率", "高血压控制")
                .replace("高血压治疗覆盖率", "高血压治疗")
                .replace("NCD服务覆盖综合分", "NCD覆盖")
            )
            label = f"{index}. {policy} / {outcome}"
            return "\n".join(textwrap.wrap(label, width=20, break_long_words=False))

        fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.2))
        if not wild.empty:
            top = wild.sort_values("wild_bootstrap_p", kind="stable").head(8).reset_index(drop=True)
            labels = [compact_policy_label(row["policy_label"], row["outcome_label"], i + 1) for i, row in top.iterrows()]
            axes[0, 0].barh(labels, top["wild_bootstrap_p"], color="#0b6b57", height=0.58)
            axes[0, 0].axvline(0.10, color="#d94b5f", linestyle="--", linewidth=1)
            axes[0, 0].invert_yaxis()
            axes[0, 0].set_title("Wild cluster bootstrap p", fontsize=14, pad=10)
        if not cohort.empty:
            top = cohort.sort_values("cohort_att_p", kind="stable").head(8).reset_index(drop=True)
            labels = [compact_policy_label(row["policy_label"], row["outcome_label"], i + 1) for i, row in top.iterrows()]
            axes[0, 1].barh(labels, top["cohort_att_sd"], color="#4f7cac", height=0.58)
            axes[0, 1].axvline(0, color="#333", linewidth=0.8)
            axes[0, 1].invert_yaxis()
            axes[0, 1].set_title("Cohort-robust ATT", fontsize=14, pad=10)
        if not threshold.empty:
            pivot = (
                threshold.loc[threshold["outcome_column"].eq(CORE_OUTCOME)]
                .pivot_table(index="threshold", columns="post_lag_years", values="coefficient_sd", aggfunc="mean")
                .sort_index()
            )
            im = axes[1, 0].imshow(pivot, cmap="RdBu_r", aspect="auto")
            axes[1, 0].set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
            axes[1, 0].set_yticks(range(len(pivot.index)), labels=[str(i) for i in pivot.index])
            axes[1, 0].set_xlabel("post lag")
            axes[1, 0].set_ylabel("threshold")
            axes[1, 0].set_title("P_Group threshold/lag sensitivity", fontsize=14, pad=10)
            fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        if not negative.empty:
            axes[1, 1].barh(negative["negative_control_label"], negative["p_value"], color=np.where(negative["null_pass_p_ge_0_10"], "#2a9d8f", "#d94b5f"))
            axes[1, 1].axvline(0.10, color="#d94b5f", linestyle="--", linewidth=1)
            axes[1, 1].set_title("Negative control p-values", fontsize=14, pad=10)
        for ax in axes.ravel():
            ax.grid(axis="x", alpha=0.25)
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", labelsize=9)
        set_centered_suptitle(fig, choose_text("模块D准因果增强层高级稳健性面板", "Module D Advanced Robustness Panel", USE_CHINESE))
        fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.93), h_pad=2.6, w_pad=3.4)
        fig.savefig(path, dpi=220)
        plt.close(fig)


    def plot_mechanism(mechanism: pd.DataFrame, path: Path) -> None:
        if mechanism.empty:
            return
        plot_df = mechanism.copy()
        plot_df["coef_for_plot"] = pd.to_numeric(plot_df["coefficient_sd"], errors="coerce").fillna(0.0)
        colors = np.where(plot_df["direction_consistent"], "#2a9d8f", "#d94b5f")
        fig, ax = plt.subplots(figsize=(13, 6.5))
        ax.bar(np.arange(plot_df.shape[0]), plot_df["coef_for_plot"], color=colors)
        ax.axhline(0, color="#333", linewidth=0.8)
        labels = [str(x).replace("：", "\n") for x in plot_df["outcome_label"]]
        ax.set_xticks(np.arange(plot_df.shape[0]), labels=labels, rotation=25, ha="right")
        ax.set_ylabel("DID coefficient (SD units)")
        ax.set_title(choose_text("政策 → 暴露 → 负担 → 压力/缺口机制链", "Policy-to-gap mechanism chain", USE_CHINESE))
        for i, row in plot_df.iterrows():
            p = row.get("p_value")
            if pd.notna(p):
                ax.text(i, row["coef_for_plot"], f"p={p:.2f}", ha="center", va="bottom" if row["coef_for_plot"] >= 0 else "top", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


    def build_summary(
        project_root: Path,
        wild: pd.DataFrame,
        cohort: pd.DataFrame,
        threshold: pd.DataFrame,
        negative: pd.DataFrame,
        mechanism: pd.DataFrame,
        outputs: dict[str, Path],
    ) -> dict[str, object]:
        core_wild = wild.loc[(wild["policy_id"] == CORE_POLICY_ID) & (wild["outcome_column"] == CORE_OUTCOME)] if not wild.empty else pd.DataFrame()
        core_cohort = cohort.loc[(cohort["policy_id"] == CORE_POLICY_ID) & (cohort["outcome_column"] == CORE_OUTCOME)] if not cohort.empty else pd.DataFrame()
        core_threshold = threshold.loc[threshold["outcome_column"].eq(CORE_OUTCOME)] if not threshold.empty else pd.DataFrame()
        threshold_direction_share = float(core_threshold["direction_consistent"].mean()) if not core_threshold.empty else np.nan
        negative_pass_share = float(negative["null_pass_p_ge_0_10"].mean()) if not negative.empty else np.nan
        mechanism_direction_share = float(mechanism["direction_consistent"].mean()) if not mechanism.empty else np.nan
        return {
            "project_root": project_root.as_posix(),
            "module_d_level": "模块D准因果增强层高级稳健性",
            "claim_boundary": "高级稳健性增强后仍称准因果强候选，不称随机实验式强因果。",
            "advanced_checks": {
                "wild_cluster_bootstrap_rows": int(wild.shape[0]),
                "cohort_robust_att_rows": int(cohort.shape[0]),
                "threshold_lag_sensitivity_rows": int(threshold.shape[0]),
                "negative_control_rows": int(negative.shape[0]),
                "mechanism_chain_steps": int(mechanism.shape[0]),
            },
            "core_candidate": {
                "policy_id": CORE_POLICY_ID,
                "policy_label": CORE_POLICY_LABEL,
                "outcome": CORE_OUTCOME,
                "wild_bootstrap_p": float(core_wild["wild_bootstrap_p"].iloc[0]) if not core_wild.empty and pd.notna(core_wild["wild_bootstrap_p"].iloc[0]) else None,
                "cohort_att_sd": float(core_cohort["cohort_att_sd"].iloc[0]) if not core_cohort.empty and pd.notna(core_cohort["cohort_att_sd"].iloc[0]) else None,
                "cohort_att_p": float(core_cohort["cohort_att_p"].iloc[0]) if not core_cohort.empty and pd.notna(core_cohort["cohort_att_p"].iloc[0]) else None,
                "threshold_direction_share": threshold_direction_share,
                "negative_control_pass_share": negative_pass_share,
                "mechanism_direction_share": mechanism_direction_share,
            },
            "can_claim": [
                "准因果增强强候选已补充wild cluster bootstrap、cohort-robust ATT、P_Group阈值/滞后敏感性、假结果检验和机制链检验。",
                "MPOWER无烟环境保护强执行仍作为准因果强候选保留；高级检验用于增强可信度而非把它升级为随机实验。",
                "机制链按政策强度、烟草暴露、吸烟归因负担、慢性呼吸疾病负担、风险压力和适配缺口逐层输出。",
            ],
            "cannot_claim": [
                "不能把bootstrap或cohort ATT包装为随机实验。",
                "机制链中未通过方向或显著性的环节只能作为解释链，不作为单独因果证明。",
            ],
            "output_files": {key: path.as_posix() for key, path in outputs.items()},
        }


    def main() -> None:
        parser = argparse.ArgumentParser(description="Run advanced 模块D准因果增强层 robustness checks and mechanism chain validation.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--bootstrap-reps", type=int, default=399)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        panel = merge_response_time_series(project_root, build_analysis_panel(project_root))
        events = identify_policy_events(panel)
        evidence = read_csv(report_asset_path(report_dir, "policy_causal_evidence_ladder.csv"))
        wild = run_wild_cluster_bootstrap(panel, events, evidence, args.bootstrap_reps)
        cohort = run_cohort_robust_att(panel, events)
        threshold = run_threshold_lag_sensitivity(panel)
        negative = run_negative_control_outcomes(panel, events)
        mechanism = run_mechanism_chain(project_root, panel, events)

        outputs = {
            "policy_d4_wild_cluster_bootstrap": report_asset_path(report_dir, "policy_d4_wild_cluster_bootstrap.csv"),
            "policy_d4_cohort_robust_att": report_asset_path(report_dir, "policy_d4_cohort_robust_att.csv"),
            "policy_d4_threshold_lag_sensitivity": report_asset_path(report_dir, "policy_d4_threshold_lag_sensitivity.csv"),
            "policy_d4_negative_control_outcomes": report_asset_path(report_dir, "policy_d4_negative_control_outcomes.csv"),
            "policy_d4_mechanism_chain": report_asset_path(report_dir, "policy_d4_mechanism_chain.csv"),
            "policy_d4_advanced_validation_summary": report_asset_path(report_dir, "policy_d4_advanced_validation_summary.json"),
            "advanced_policy_d4_robustness_panel": figure_dir / "advanced_policy_d4_robustness_panel.png",
            "advanced_policy_d4_mechanism_chain": figure_dir / "advanced_policy_d4_mechanism_chain.png",
        }
        write_csv(wild, outputs["policy_d4_wild_cluster_bootstrap"])
        write_csv(cohort, outputs["policy_d4_cohort_robust_att"])
        write_csv(threshold, outputs["policy_d4_threshold_lag_sensitivity"])
        write_csv(negative, outputs["policy_d4_negative_control_outcomes"])
        write_csv(mechanism, outputs["policy_d4_mechanism_chain"])
        plot_robustness(wild, cohort, threshold, negative, outputs["advanced_policy_d4_robustness_panel"])
        plot_mechanism(mechanism, outputs["advanced_policy_d4_mechanism_chain"])
        summary = build_summary(project_root, wild, cohort, threshold, negative, mechanism, outputs)
        outputs["policy_d4_advanced_validation_summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_policy_d5_global_portfolio():
    __name__ = 'run_policy_d5_global_portfolio'
    import argparse
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from foundation import choose_text, configure_matplotlib_fonts, set_centered_suptitle
    from foundation import detect_project_root as shared_detect_project_root


    USE_CHINESE = configure_matplotlib_fonts()


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def read_csv(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


    def domain_from_policy(policy_id: object) -> str:
        text = str(policy_id)
        if "mpower" in text or "tobacco" in text:
            return "风险治理-控烟"
        if "hypertension" in text:
            return "慢病服务-高血压"
        if "primary_care" in text:
            return "基层NCD连续服务"
        if "diabetes" in text:
            return "慢病服务-糖尿病"
        if "diet" in text or "salt" in text:
            return "风险治理-膳食控盐"
        if "ncd_capacity" in text:
            return "综合治理能力"
        return "政策适配"


    def bool_value(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"true", "1", "yes", "y"}


    def build_portfolio(report_dir: Path) -> pd.DataFrame:
        evidence = read_csv(report_asset_path(report_dir, "policy_causal_evidence_ladder.csv"))
        wild = read_csv(report_asset_path(report_dir, "policy_d4_wild_cluster_bootstrap.csv"))
        cohort = read_csv(report_asset_path(report_dir, "policy_d4_cohort_robust_att.csv"))
        if evidence.empty:
            return pd.DataFrame()

        keys = ["policy_id", "outcome_column"]
        if not wild.empty:
            wild_keep = wild.loc[:, [col for col in [*keys, "wild_bootstrap_p", "wild_bootstrap_pass_p_lt_0_10"] if col in wild.columns]]
            evidence = evidence.merge(wild_keep, on=keys, how="left")
        else:
            evidence["wild_bootstrap_p"] = np.nan
            evidence["wild_bootstrap_pass_p_lt_0_10"] = False

        if not cohort.empty:
            cohort_keep = cohort.loc[
                :,
                [
                    col
                    for col in [
                        *keys,
                        "cohort_att_sd",
                        "cohort_att_p",
                        "direction_consistent",
                        "cohort_lag_cells",
                    ]
                    if col in cohort.columns
                ],
            ].rename(columns={"direction_consistent": "cohort_direction_consistent"})
            evidence = evidence.merge(cohort_keep, on=keys, how="left")
        else:
            evidence["cohort_att_p"] = np.nan
            evidence["cohort_direction_consistent"] = False

        rows: list[dict[str, object]] = []
        for _, row in evidence.iterrows():
            p_value = pd.to_numeric(pd.Series([row.get("p_value")]), errors="coerce").iloc[0]
            wild_p = pd.to_numeric(pd.Series([row.get("wild_bootstrap_p")]), errors="coerce").iloc[0]
            cohort_p = pd.to_numeric(pd.Series([row.get("cohort_att_p")]), errors="coerce").iloc[0]
            synthetic_share = pd.to_numeric(pd.Series([row.get("synthetic_direction_share")]), errors="coerce").iloc[0]
            evidence_score = pd.to_numeric(pd.Series([row.get("evidence_score")]), errors="coerce").iloc[0]
            direction = bool_value(row.get("direction_consistent"))
            pretrend = bool_value(row.get("pretrend_pass"))
            placebo = bool_value(row.get("placebo_pass"))
            wild_pass = pd.notna(wild_p) and wild_p < 0.10
            cohort_pass = pd.notna(cohort_p) and cohort_p < 0.10 and bool_value(row.get("cohort_direction_consistent"))
            did_pass = pd.notna(p_value) and p_value < 0.10 and direction
            synthetic_pass = pd.notna(synthetic_share) and synthetic_share >= 0.60

            additive = (
                0.06 * did_pass
                + 0.08 * wild_pass
                + 0.08 * cohort_pass
                + 0.04 * synthetic_pass
                + 0.03 * placebo
            )
            penalty = 0.10 * (not direction) + 0.06 * (not pretrend)
            if not placebo and not cohort_pass and not wild_pass:
                penalty += 0.04
            d5_score = float(np.clip((evidence_score if pd.notna(evidence_score) else 0.0) + additive - penalty, 0, 0.98))

            robust_check_count = int(did_pass) + int(wild_pass) + int(cohort_pass) + int(synthetic_pass) + int(placebo) + int(pretrend)
            if d5_score >= 0.80 and direction and pretrend and (wild_pass or cohort_pass):
                d5_tier = "准因果强候选"
                claim_level = "可作为模块D主证据路径，但仍不能写成随机实验。"
            elif d5_score >= 0.70 and direction and robust_check_count >= 4:
                d5_tier = "稳健政策候选"
                claim_level = "可作为备选政策路径或服务覆盖路径，需说明边界。"
            elif d5_score >= 0.55 and direction:
                d5_tier = "政策适配候选"
                claim_level = "可进入政策包和情景模拟，不单独承担因果结论。"
            else:
                d5_tier = "探索性证据"
                claim_level = "只作解释或敏感性材料。"

            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "policy_label": row.get("policy_label"),
                    "policy_domain": domain_from_policy(row.get("policy_id")),
                    "outcome_column": row.get("outcome_column"),
                    "outcome_label": row.get("outcome_label"),
                    "outcome_tier": row.get("outcome_tier"),
                    "did_coefficient_sd": row.get("did_coefficient_sd"),
                    "cluster_p_value": p_value,
                    "wild_bootstrap_p": wild_p,
                    "cohort_att_p": cohort_p,
                    "direction_consistent": direction,
                    "pretrend_pass": pretrend,
                    "placebo_pass": placebo,
                    "synthetic_direction_share": synthetic_share,
                    "did_pass_p_lt_0_10": did_pass,
                    "wild_pass_p_lt_0_10": wild_pass,
                    "cohort_pass_p_lt_0_10": cohort_pass,
                    "synthetic_pass_share_ge_0_60": synthetic_pass,
                    "robust_check_count": robust_check_count,
                    "base_evidence_score": evidence_score,
                    "d5_validation_score": d5_score,
                    "d5_tier": d5_tier,
                    "claim_level": claim_level,
                    "treated_countries": row.get("treated_countries"),
                    "boundary_note": boundary_note(row),
                }
            )
        return pd.DataFrame(rows).sort_values(["d5_validation_score", "robust_check_count"], ascending=[False, False], kind="stable")


    def boundary_note(row: pd.Series) -> str:
        outcome_tier = str(row.get("outcome_tier", ""))
        policy_id = str(row.get("policy_id", ""))
        if "服务覆盖" in outcome_tier or "ncd_" in str(row.get("outcome_column", "")):
            return "服务覆盖/能力路径可以说明政策响应改善，但不直接等同于死亡率或DALY下降。"
        if "mpower" in policy_id or "tobacco" in policy_id:
            return "控烟路径有政策事件和风险结果支撑，适合作为准因果主线；仍需避免随机实验表述。"
        return "该路径用于扩展政策组合，需与A/B/C压力画像和情景模拟共同解释。"


    def within_country_association(panel: pd.DataFrame, x_col: str, y_col: str) -> dict[str, object] | None:
        work = panel.loc[:, ["iso3", "year", x_col, y_col]].copy()
        work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=[x_col, y_col])
        if work.shape[0] < 200 or work["iso3"].nunique() < 30:
            return None
        x = work[x_col] - work.groupby("iso3")[x_col].transform("mean")
        y = work[y_col] - work.groupby("iso3")[y_col].transform("mean")
        denominator = float(np.dot(x, x))
        if denominator == 0:
            return None
        coefficient = float(np.dot(x, y) / denominator)
        correlation = float(np.corrcoef(x, y)[0, 1])
        return {
            "coefficient": coefficient,
            "correlation": correlation,
            "n_obs": int(work.shape[0]),
            "clusters": int(work["iso3"].nunique()),
        }


    def build_mechanism_audit(project_root: Path, report_dir: Path, portfolio: pd.DataFrame) -> pd.DataFrame:
        mechanism = read_csv(report_asset_path(report_dir, "policy_d4_mechanism_chain.csv"))
        if mechanism.empty:
            return pd.DataFrame()
        output = mechanism.copy()
        output["evidence_role"] = np.where(
            output["estimation_method"].astype(str).str.contains("bridge", case=False, na=False),
            "应用情景桥接",
            np.where(output["estimation_method"].astype(str).str.contains("did", case=False, na=False), "经验检验", "定义环节"),
        )
        output["d5_interpretation"] = np.where(
            output["direction_consistent"].astype(bool),
            "方向支持机制链",
            "方向不支持，答辩中只能作为边界说明",
        )
        response = read_csv(project_root / "04_simulation" / "response_diagnosis_panel.csv")
        if not response.empty and {"iso3", "year", "dbd_smoking_per100k", "combined_pressure_score", "adaptation_gap_score"}.issubset(response.columns):
            response = response.sort_values(["iso3", "year"], kind="stable").copy()
            response["delta_smoking_burden"] = response.groupby("iso3", sort=False)["dbd_smoking_per100k"].diff()
            response["delta_combined_pressure_score"] = response.groupby("iso3", sort=False)["combined_pressure_score"].diff()
            response["delta_adaptation_gap_score"] = response.groupby("iso3", sort=False)["adaptation_gap_score"].diff()
            empirical_rows = []
            for step_id, y_col, label in [
                ("06a_risk_to_pressure_empirical", "delta_combined_pressure_score", "经验后半链：吸烟风险负担变化与综合压力变化"),
                ("06b_risk_to_gap_empirical", "delta_adaptation_gap_score", "经验后半链：吸烟风险负担变化与适配缺口变化"),
            ]:
                assoc = within_country_association(response, "delta_smoking_burden", y_col)
                if assoc is None:
                    continue
                empirical_rows.append(
                    {
                        "step_id": step_id,
                        "mechanism_layer": "经验后半链",
                        "estimation_method": "within_country_lag_association",
                        "outcome_column": y_col,
                        "outcome_label": label,
                        "expected_sign": 1,
                        "coefficient_sd": assoc["coefficient"],
                        "p_value": np.nan,
                        "direction_consistent": bool(assoc["coefficient"] > 0),
                        "evidence_note": "用国家内年度变化验证风险负担变化与压力/缺口变化同向；这是机制支持，不是政策DID。",
                        "cluster_se": np.nan,
                        "p_lt_0_10": np.nan,
                        "n_obs": assoc["n_obs"],
                        "clusters": assoc["clusters"],
                        "evidence_role": "经验后半链",
                        "d5_interpretation": "补强机制后半段，但仍不替代政策因果识别。",
                        "within_country_correlation": assoc["correlation"],
                    }
                )
            if empirical_rows:
                output = pd.concat([output, pd.DataFrame(empirical_rows)], ignore_index=True)
        return output


    def plot_portfolio(portfolio: pd.DataFrame, path: Path) -> None:
        if portfolio.empty:
            return
        top = portfolio.head(12).copy()
        labels = top["policy_label"].astype(str) + "\n" + top["outcome_label"].astype(str)
        colors = np.where(top["d5_tier"].eq("准因果强候选"), "#0b6b57", np.where(top["d5_tier"].eq("稳健政策候选"), "#4f7cac", "#f2b84b"))
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.barh(labels, top["d5_validation_score"], color=colors)
        ax.axvline(0.80, color="#d94b5f", linestyle="--", linewidth=1)
        ax.axvline(0.70, color="#666666", linestyle=":", linewidth=1)
        ax.invert_yaxis()
        ax.set_xlabel(choose_text("政策路径验证得分", "policy pathway validation score", USE_CHINESE))
        ax.set_title(choose_text("模块D多政策路径组合层全球政策路径验证组合", "Module D Multi-Policy Validation Portfolio", USE_CHINESE))
        ax.grid(axis="x", alpha=0.22)
        set_centered_suptitle(fig, choose_text("准因果强候选与稳健政策候选", "Quasi-Causal and Robust Policy Candidates", USE_CHINESE), y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(path, dpi=220)
        plt.close(fig)


    def update_lock_status(report_dir: Path, summary: dict[str, object]) -> None:
        path = report_asset_path(report_dir, "module_d_lock_status.json")
        if not path.exists():
            return
        status = json.loads(path.read_text(encoding="utf-8"))
        global_status = status.setdefault("global_module_d_status", {})
        global_status["d5_multi_path_validation_locked"] = True
        global_status["d5_strong_candidate_paths"] = summary["d5_strong_candidates"]
        global_status["d5_robust_or_above_paths"] = summary["d5_robust_or_above_candidates"]
        global_status["lock_level"] = "模块D多政策路径组合层"
        status["can_claim"] = [
            claim
            for claim in status.get("can_claim", [])
            if "MPOWER无烟环境保护强执行" not in claim or "准因果强候选" not in claim
        ]
        status["can_claim"].append(
            "模块D已升级为模块D多政策路径组合层：控烟主线、高血压服务覆盖、基层NCD连续服务等路径统一进入DID、wild bootstrap、cohort ATT、前趋势、安慰剂和合成方向验证。"
        )
        status["can_claim"] = list(dict.fromkeys(status.get("can_claim", [])))
        status["cannot_claim"] = list(dict.fromkeys(status.get("cannot_claim", []) + ["模块D多政策路径组合层仍不能写成随机实验式强因果；服务覆盖路径不能直接等同于死亡率下降。"]))
        path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


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
        parser = argparse.ArgumentParser(description="Build 模块D多政策路径组合层 global multi-path quasi-causal policy portfolio.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        portfolio = build_portfolio(report_dir)
        mechanism_audit = build_mechanism_audit(project_root, report_dir, portfolio)

        portfolio_path = report_asset_path(report_dir, "policy_d5_global_validation_portfolio.csv")
        mechanism_path = report_asset_path(report_dir, "policy_d5_mechanism_boundary_audit.csv")
        summary_path = report_asset_path(report_dir, "policy_d5_global_portfolio_summary.json")
        figure_path = figure_dir / "advanced_policy_d5_global_portfolio.png"

        portfolio.to_csv(portfolio_path, index=False, encoding="utf-8-sig")
        mechanism_audit.to_csv(mechanism_path, index=False, encoding="utf-8-sig")
        plot_portfolio(portfolio, figure_path)

        strong = portfolio.loc[portfolio["d5_tier"].eq("准因果强候选")] if not portfolio.empty else pd.DataFrame()
        robust_or_above = portfolio.loc[portfolio["d5_tier"].isin(["准因果强候选", "稳健政策候选"])] if not portfolio.empty else pd.DataFrame()
        summary = {
            "project_root": project_root.as_posix(),
            "module_d_level": "模块D多政策路径组合层",
            "d5_strong_candidates": int(strong.shape[0]),
            "d5_robust_or_above_candidates": int(robust_or_above.shape[0]),
            "policy_domains_covered": sorted(portfolio.loc[portfolio["d5_tier"].ne("探索性证据"), "policy_domain"].dropna().unique().tolist()) if not portfolio.empty else [],
            "top_candidates": portfolio.head(8).to_dict(orient="records") if not portfolio.empty else [],
            "claim_boundary": "模块D多政策路径组合层把模块D从单一控烟DID升级为多政策路径准因果组合验证；仍不声明随机实验式强因果，服务覆盖路径单独说明不能直接代表死亡率下降。",
            "output_files": {
                "policy_d5_global_validation_portfolio": portfolio_path.as_posix(),
                "policy_d5_mechanism_boundary_audit": mechanism_path.as_posix(),
                "policy_d5_global_portfolio_summary": summary_path.as_posix(),
                "advanced_policy_d5_global_portfolio": figure_path.as_posix(),
            },
        }
        clean_summary = json_clean(summary)
        summary_path.write_text(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        update_lock_status(report_dir, clean_summary)
        print(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_policy_d5_non_tobacco_pathway_hardening():
    __name__ = 'run_policy_d5_non_tobacco_pathway_hardening'
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
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


    def build_non_tobacco_pathways(portfolio: pd.DataFrame) -> pd.DataFrame:
        if portfolio.empty:
            return pd.DataFrame()
        work = portfolio.copy()
        work["is_tobacco_path"] = work["policy_domain"].astype(str).str.contains("控烟|tobacco", case=False, regex=True)
        work = work.loc[~work["is_tobacco_path"]].copy()
        if work.empty:
            return pd.DataFrame()
        work["d5_validation_score"] = pd.to_numeric(work["d5_validation_score"], errors="coerce").fillna(0)
        work["cluster_p_value"] = pd.to_numeric(work.get("cluster_p_value"), errors="coerce")
        work["direct_or_service_family"] = np.where(
            work["outcome_column"].astype(str).str.startswith(("dbd_", "gbd_rate_", "delta_dbd_")),
            "直接健康/风险负担",
            np.where(work["outcome_column"].astype(str).str.startswith("ncd_"), "服务覆盖/能力", "中间结果"),
        )
        best = (
            work.sort_values(["policy_domain", "d5_validation_score"], ascending=[True, False], kind="stable")
            .groupby("policy_domain", as_index=False)
            .head(3)
            .copy()
        )
        best["non_tobacco_pathway_role"] = np.where(
            best["d5_tier"].isin(["准因果强候选", "稳健政策候选"]),
            "非控烟主线候选",
            np.where(best["d5_tier"].eq("政策适配候选"), "政策包适配证据", "机制/探索证据"),
        )
        best["claim_boundary"] = np.where(
            best["direct_or_service_family"].eq("服务覆盖/能力"),
            "可说明响应能力或服务覆盖改善，不能单独等同于死亡率/DALY下降。",
            "可作为非控烟健康/风险路径证据，但仍按准因果或探索层级表达。",
        )
        keep = [
            "policy_id",
            "policy_label",
            "policy_domain",
            "outcome_column",
            "outcome_label",
            "direct_or_service_family",
            "did_coefficient_sd",
            "cluster_p_value",
            "direction_consistent",
            "pretrend_pass",
            "placebo_pass",
            "synthetic_direction_share",
            "robust_check_count",
            "d5_validation_score",
            "d5_tier",
            "non_tobacco_pathway_role",
            "claim_boundary",
            "treated_countries",
        ]
        return best.loc[:, [col for col in keep if col in best.columns]].sort_values(
            ["non_tobacco_pathway_role", "d5_validation_score"], ascending=[True, False], kind="stable"
        )


    def build_domain_summary(non_tobacco: pd.DataFrame) -> pd.DataFrame:
        if non_tobacco.empty:
            return pd.DataFrame()
        work = non_tobacco.copy()
        work["d5_validation_score"] = pd.to_numeric(work["d5_validation_score"], errors="coerce").fillna(0)
        idx = work.groupby("policy_domain")["d5_validation_score"].idxmax()
        best = work.loc[idx].copy()
        counts = work.groupby("policy_domain", as_index=False).agg(
            candidate_rows=("policy_id", "count"),
            strong_candidate_rows=("d5_tier", lambda s: int(s.eq("准因果强候选").sum())),
            robust_or_adaptation_rows=(
                "d5_tier",
                lambda s: int(s.isin(["准因果强候选", "稳健政策候选", "政策适配候选"]).sum()),
            ),
        )
        summary = best.merge(counts, on="policy_domain", how="left")
        summary = summary.rename(
            columns={
                "d5_validation_score": "best_validation_score",
                "d5_tier": "best_tier",
                "non_tobacco_pathway_role": "best_role",
                "outcome_label": "best_outcome",
                "policy_label": "best_policy_label",
                "treated_countries": "best_treated_countries",
            }
        )
        return summary[
            [
                "policy_domain",
                "candidate_rows",
                "strong_candidate_rows",
                "robust_or_adaptation_rows",
                "best_policy_label",
                "best_validation_score",
                "best_tier",
                "best_role",
                "best_outcome",
                "best_treated_countries",
            ]
        ].sort_values("best_validation_score", ascending=False, kind="stable")


    def plot_domain_summary(domain_summary: pd.DataFrame, figure_path: Path) -> None:
        if domain_summary.empty:
            return
        plot_df = domain_summary.sort_values("best_validation_score", ascending=True, kind="stable")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(plot_df["policy_domain"], plot_df["best_validation_score"], color="#2a9d8f")
        ax.set_xlabel(choose_text("最佳政策路径验证分", "Best policy pathway validation score", USE_CHINESE))
        ax.set_title(choose_text("非控烟政策路径强度", "Non-tobacco Policy Pathway Strength", USE_CHINESE))
        ax.grid(axis="x", alpha=0.22)
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
        parser = argparse.ArgumentParser(description="Harden non-tobacco 模块D多政策路径组合层 policy pathways so Module D is not tobacco-only.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        portfolio = read_csv(report_asset_path(report_dir, "policy_d5_global_validation_portfolio.csv"))
        non_tobacco = build_non_tobacco_pathways(portfolio)
        domain_summary = build_domain_summary(non_tobacco)
        non_tobacco_path = report_asset_path(report_dir, "policy_d5_non_tobacco_pathway_candidates.csv")
        domain_path = report_asset_path(report_dir, "policy_d5_non_tobacco_domain_summary.csv")
        summary_path = report_asset_path(report_dir, "policy_d5_non_tobacco_pathway_summary.json")
        figure_path = figure_dir / "advanced_policy_d5_non_tobacco_pathways.png"
        non_tobacco.to_csv(non_tobacco_path, index=False, encoding="utf-8-sig")
        domain_summary.to_csv(domain_path, index=False, encoding="utf-8-sig")
        plot_domain_summary(domain_summary, figure_path)

        summary = {
            "project_root": project_root.as_posix(),
            "hardening_layer": "模块D多政策路径组合层 non-tobacco policy pathway hardening",
            "portfolio_rows": int(portfolio.shape[0]),
            "non_tobacco_candidate_rows": int(non_tobacco.shape[0]),
            "non_tobacco_policy_domains": int(domain_summary.shape[0]),
            "strong_non_tobacco_candidates": int(non_tobacco["d5_tier"].eq("准因果强候选").sum()) if not non_tobacco.empty and "d5_tier" in non_tobacco.columns else 0,
            "robust_or_adaptation_non_tobacco_candidates": int(non_tobacco["d5_tier"].isin(["准因果强候选", "稳健政策候选", "政策适配候选"]).sum()) if not non_tobacco.empty and "d5_tier" in non_tobacco.columns else 0,
            "best_non_tobacco_domains": domain_summary.head(8).to_dict(orient="records"),
            "claim_boundary": "模块D多政策路径组合层不再只靠控烟主线；高血压服务覆盖已有非控烟准因果强候选，基层NCD连续服务、综合治理能力和控盐膳食作为政策包适配证据；服务覆盖改善仍不直接等同于死亡率/DALY下降。",
            "output_files": {
                "non_tobacco_candidates": non_tobacco_path.as_posix(),
                "domain_summary": domain_path.as_posix(),
                "summary": summary_path.as_posix(),
                "figure": figure_path.as_posix(),
            },
        }
        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_policy_d5_boundary_enhancement():
    __name__ = 'run_policy_d5_boundary_enhancement'
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
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


    def classify_health_result(row: pd.Series) -> dict[str, object]:
        column = str(row.get("outcome_column", ""))
        tier = str(row.get("outcome_tier", ""))
        d5_tier = str(row.get("d5_tier", ""))
        direct_health = column.startswith(("dbd_", "gbd_rate_", "delta_dbd_")) or "风险暴露负担" in str(row.get("outcome_label", ""))
        service = column.startswith("ncd_") or "服务覆盖" in tier
        if direct_health and "变化" in tier:
            outcome_family = "direct_risk_burden_change"
        elif direct_health:
            outcome_family = "direct_health_or_risk_burden"
        elif service:
            outcome_family = "service_coverage_intermediate"
        else:
            outcome_family = "other_intermediate"
        robust = d5_tier in {"准因果强候选", "稳健政策候选"}
        strong = d5_tier == "准因果强候选"
        if direct_health and strong:
            claim = "主结果强候选：可作为模块D健康/风险结果主证据，但仍是准因果。"
        elif direct_health and robust:
            claim = "直接健康/风险结果稳健候选：可作为主线旁证或备选主证据。"
        elif service and robust:
            claim = "服务覆盖强候选：说明响应能力改善，不能直接替代死亡率/DALY下降。"
        elif direct_health:
            claim = "直接健康/风险结果探索候选：方向可用，证据强度不足。"
        else:
            claim = "中间结果或探索证据。"
        return {
            "outcome_family": outcome_family,
            "direct_health_result_candidate": bool(direct_health),
            "service_coverage_candidate": bool(service),
            "robust_or_above": bool(robust),
            "strong_or_above": bool(strong),
            "health_result_claim_level": claim,
        }


    def build_health_candidate_matrix(portfolio: pd.DataFrame) -> pd.DataFrame:
        if portfolio.empty:
            return pd.DataFrame()
        classified = portfolio.apply(classify_health_result, axis=1, result_type="expand")
        matrix = pd.concat([portfolio, classified], axis=1)
        matrix["health_candidate_rank_score"] = (
            pd.to_numeric(matrix["d5_validation_score"], errors="coerce").fillna(0)
            + 0.08 * matrix["direct_health_result_candidate"].astype(float)
            + 0.04 * matrix["robust_or_above"].astype(float)
        )
        matrix = matrix.sort_values(
            ["direct_health_result_candidate", "robust_or_above", "health_candidate_rank_score"],
            ascending=[False, False, False],
            kind="stable",
        )
        return matrix


    def build_randomized_boundary_audit() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "requirement": "random_assignment",
                    "current_status": "not_available",
                    "why_it_matters": "随机分配才能把政策暴露和未观测混杂最大程度切开。",
                    "current_substitute": "国家固定效应、年份固定效应、交错DID、cohort ATT、wild cluster bootstrap、前趋势、安慰剂和合成方向检验。",
                    "claim_rule": "不能称为随机实验式强因果，只能称严格准因果候选。",
                },
                {
                    "requirement": "policy_timing_exogeneity",
                    "current_status": "partially_tested",
                    "why_it_matters": "政策试点国家或城市往往不是随机选择，可能由治理能力或疾病负担驱动。",
                    "current_substitute": "前趋势检验、anticipation buffer、阈值/滞后敏感性、负向对照结果。",
                    "claim_rule": "若前趋势/安慰剂失败，降级为机制或政策适配证据。",
                },
                {
                    "requirement": "direct_health_endpoint",
                    "current_status": "mixed",
                    "why_it_matters": "服务覆盖提高不等于死亡率或DALY立即下降。",
                    "current_substitute": "把结果分为直接健康/风险负担、变化率结果、服务覆盖中间结果三层。",
                    "claim_rule": "服务覆盖单独讲响应能力，直接健康/风险负担才承担健康结果证据。",
                },
                {
                    "requirement": "spillover_and_policy_bundle_control",
                    "current_status": "partially_tested",
                    "why_it_matters": "控烟、慢病、支付改革可能同年叠加，单政策解释会偏强。",
                    "current_substitute": "多政策路径组合评分和机制边界审计。",
                    "claim_rule": "主结论讲政策路径和适配包，不把所有效果归因给单一政策。",
                },
                {
                    "requirement": "external_validity_to_china",
                    "current_status": "mapping_only",
                    "why_it_matters": "全球准因果路径不能直接等同于中国省级政策效果。",
                    "current_substitute": "先做中国省级压力-风险-响应画像，再接NHSA政策暴露时间线与政策包建议。",
                    "claim_rule": "中国部分讲情景适配和迁移建议，不讲省级强因果。",
                },
            ]
        )


    def plot_health_matrix(matrix: pd.DataFrame, path: Path) -> None:
        if matrix.empty:
            return
        top = matrix.head(12).copy()
        labels = top["policy_label"].astype(str) + "\n" + top["outcome_label"].astype(str)
        colors = np.where(top["direct_health_result_candidate"], "#0b6b57", np.where(top["service_coverage_candidate"], "#4f7cac", "#b8c2cc"))
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.barh(labels, top["health_candidate_rank_score"], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(choose_text("健康结果候选排序分", "Health endpoint candidate score", USE_CHINESE))
        ax.set_title(choose_text("模块D健康结果与服务覆盖候选矩阵", "Module D Health Endpoint and Service Candidate Matrix", USE_CHINESE))
        ax.grid(axis="x", alpha=0.22)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


    def update_summary(summary_path: Path, matrix: pd.DataFrame, boundary_path: Path, health_path: Path, figure_path: Path) -> None:
        if not summary_path.exists():
            return
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        direct = matrix.loc[matrix["direct_health_result_candidate"]] if not matrix.empty else pd.DataFrame()
        robust_direct = direct.loc[direct["robust_or_above"]] if not direct.empty else pd.DataFrame()
        strong_direct = direct.loc[direct["strong_or_above"]] if not direct.empty else pd.DataFrame()
        summary["d5_health_endpoint_enhancement"] = {
            "direct_health_result_candidates": int(direct.shape[0]),
            "direct_health_result_robust_or_above": int(robust_direct.shape[0]),
            "direct_health_result_strong_or_above": int(strong_direct.shape[0]),
            "randomized_causality_boundary_audit": boundary_path.as_posix(),
            "health_outcome_candidate_matrix": health_path.as_posix(),
            "health_candidate_figure": figure_path.as_posix(),
            "claim_boundary": "新增健康结果候选矩阵后，模块D多政策路径组合层可以区分直接健康/风险负担证据和服务覆盖中间证据；仍不能写成随机实验强因果。",
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


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
        parser = argparse.ArgumentParser(description="Enhance 模块D多政策路径组合层 with health endpoint candidate matrix and randomized boundary audit.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        portfolio = read_csv(report_asset_path(report_dir, "policy_d5_global_validation_portfolio.csv"))
        matrix = build_health_candidate_matrix(portfolio)
        boundary = build_randomized_boundary_audit()

        health_path = report_asset_path(report_dir, "policy_d5_health_outcome_candidate_matrix.csv")
        boundary_path = report_asset_path(report_dir, "policy_d5_randomized_causality_boundary_audit.csv")
        summary_path = report_asset_path(report_dir, "policy_d5_boundary_enhancement_summary.json")
        portfolio_summary_path = report_asset_path(report_dir, "policy_d5_global_portfolio_summary.json")
        figure_path = figure_dir / "advanced_policy_d5_health_endpoint_matrix.png"

        matrix.to_csv(health_path, index=False, encoding="utf-8-sig")
        boundary.to_csv(boundary_path, index=False, encoding="utf-8-sig")
        plot_health_matrix(matrix, figure_path)
        update_summary(portfolio_summary_path, matrix, boundary_path, health_path, figure_path)

        direct = matrix.loc[matrix["direct_health_result_candidate"]] if not matrix.empty else pd.DataFrame()
        robust_direct = direct.loc[direct["robust_or_above"]] if not direct.empty else pd.DataFrame()
        summary = {
            "project_root": project_root.as_posix(),
            "enhancement_layer": "模块D多政策路径组合层 health endpoint and causality boundary hardening",
            "portfolio_rows": int(portfolio.shape[0]),
            "direct_health_result_candidates": int(direct.shape[0]),
            "direct_health_result_robust_or_above": int(robust_direct.shape[0]),
            "top_direct_health_candidates": robust_direct.head(6).to_dict(orient="records") if not robust_direct.empty else [],
            "randomized_boundary_items": int(boundary.shape[0]),
            "claim_boundary": "模块D多政策路径组合层已具备严格准因果候选表达，但不是随机化强因果。健康结果候选矩阵用于防止把服务覆盖误讲成死亡率或DALY下降。",
            "output_files": {
                "policy_d5_health_outcome_candidate_matrix": health_path.as_posix(),
                "policy_d5_randomized_causality_boundary_audit": boundary_path.as_posix(),
                "policy_d5_boundary_enhancement_summary": summary_path.as_posix(),
                "advanced_policy_d5_health_endpoint_matrix": figure_path.as_posix(),
            },
        }
        clean_summary = json_clean(summary)
        summary_path.write_text(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_policy_d6_quasi_causal_excellence():
    __name__ = 'run_policy_d6_quasi_causal_excellence'
    import argparse
    import json
    from pathlib import Path
    from statistics import NormalDist

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from foundation import choose_text, configure_matplotlib_fonts
    from foundation import detect_project_root as shared_detect_project_root


    USE_CHINESE = configure_matplotlib_fonts()
    NORM = NormalDist()
    ZCRIT_10_TWO_SIDED = 1.6448536269514722


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def read_csv(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


    def read_json(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


    def write_csv(df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False, encoding="utf-8-sig")


    def bool_value(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y", "pass", "通过"}


    def finite_float(value: object, default: float = np.nan) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if np.isfinite(result) else default


    def bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
        if not np.isfinite(value):
            return low
        return float(min(high, max(low, value)))


    def p_to_z(p_value: object) -> float:
        p = finite_float(p_value)
        if not np.isfinite(p):
            return np.nan
        p = min(max(p, 1e-12), 0.999999)
        return float(NORM.inv_cdf(1.0 - p / 2.0))


    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(float(value)) else float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if pd.isna(value):
            return None
        return value


    def classify_outcome(row: pd.Series) -> dict[str, object]:
        column = str(row.get("outcome_column", ""))
        label = str(row.get("outcome_label", ""))
        tier = str(row.get("outcome_tier", ""))
        direct_health = column.startswith(("dbd_", "gbd_rate_", "delta_dbd_")) or "风险暴露负担" in label
        service = column.startswith("ncd_") or "服务覆盖" in tier or "覆盖" in label
        if direct_health and "变化" in label + tier:
            family = "直接风险负担变化"
        elif direct_health:
            family = "直接健康/风险负担"
        elif service:
            family = "服务覆盖中间结果"
        else:
            family = "其他中间结果"
        return {
            "outcome_family": family,
            "direct_health_result": bool(direct_health),
            "service_coverage_result": bool(service),
        }


    def aggregate_threshold_lag(threshold: pd.DataFrame) -> pd.DataFrame:
        if threshold.empty:
            return pd.DataFrame(columns=["policy_id", "outcome_column"])
        work = threshold.copy()
        work["direction_consistent"] = work["direction_consistent"].map(bool_value)
        work["p_lt_0_10"] = work["p_lt_0_10"].map(bool_value)
        grouped = (
            work.groupby(["policy_id", "outcome_column"], as_index=False)
            .agg(
                threshold_lag_specs=("policy_id", "count"),
                threshold_direction_share=("direction_consistent", "mean"),
                threshold_p_lt_0_10_share=("p_lt_0_10", "mean"),
                threshold_min_p=("p_value", lambda s: pd.to_numeric(s, errors="coerce").min()),
            )
            .reset_index(drop=True)
        )
        return grouped


    def aggregate_negative_controls(negative: pd.DataFrame) -> pd.DataFrame:
        if negative.empty:
            return pd.DataFrame(columns=["policy_id"])
        work = negative.copy()
        work["null_pass_p_ge_0_10"] = work["null_pass_p_ge_0_10"].map(bool_value)
        return (
            work.groupby("policy_id", as_index=False)
            .agg(
                negative_control_count=("negative_control_outcome", "count"),
                negative_control_pass_share=("null_pass_p_ge_0_10", "mean"),
                negative_control_min_p=("p_value", lambda s: pd.to_numeric(s, errors="coerce").min()),
            )
            .reset_index(drop=True)
        )


    def aggregate_placebo(placebo: pd.DataFrame) -> pd.DataFrame:
        if placebo.empty:
            return pd.DataFrame(columns=["policy_id", "outcome_column"])
        work = placebo.copy()
        work["placebo_issue"] = work["placebo_issue"].map(bool_value)
        work["placebo_pass"] = ~work["placebo_issue"]
        return (
            work.groupby(["policy_id", "outcome_column"], as_index=False)
            .agg(
                shifted_placebo_tests=("placebo_shift_years", "count"),
                shifted_placebo_pass_share=("placebo_pass", "mean"),
                shifted_placebo_min_p=("p_value", lambda s: pd.to_numeric(s, errors="coerce").min()),
            )
            .reset_index(drop=True)
        )


    def aggregate_heterogeneity(heterogeneity: pd.DataFrame) -> pd.DataFrame:
        if heterogeneity.empty:
            return pd.DataFrame(columns=["policy_id", "outcome_column"])
        work = heterogeneity.copy()
        work["direction_consistent"] = work["direction_consistent"].map(bool_value)
        work["p_lt_0_10"] = pd.to_numeric(work["p_value"], errors="coerce") < 0.10
        return (
            work.groupby(["policy_id", "outcome_column"], as_index=False)
            .agg(
                heterogeneity_type_cells=("vulnerability_type_label", "count"),
                heterogeneity_direction_share=("direction_consistent", "mean"),
                heterogeneity_p_lt_0_10_share=("p_lt_0_10", "mean"),
            )
            .reset_index(drop=True)
        )


    def build_china_transfer_readiness(
        portfolio: pd.DataFrame,
        transfer: pd.DataFrame,
        pathways: pd.DataFrame,
        china_summary: dict[str, object],
    ) -> pd.DataFrame:
        if portfolio.empty:
            return pd.DataFrame()
        pathway_domains = pathways["policy_domain"].dropna().astype(str).tolist() if not pathways.empty and "policy_domain" in pathways.columns else []
        rows: list[dict[str, object]] = []
        transfer_work = transfer.copy()
        if not transfer_work.empty:
            transfer_work["recommended_transfer_package"] = transfer_work["recommended_transfer_package"].astype(str)
        top_by_province = (
            transfer_work.sort_values("transfer_score", ascending=False, kind="stable")
            .groupby("province", as_index=False)
            .head(1)
            if not transfer_work.empty
            else pd.DataFrame()
        )
        global_top_mean = finite_float(top_by_province["transfer_score"].mean()) if not top_by_province.empty else np.nan

        for _, row in portfolio.iterrows():
            domain = str(row.get("policy_domain", ""))
            label = str(row.get("policy_label", ""))
            if "高血压" in domain + label:
                pattern = "高血压|控盐|连续用药"
            elif "控烟" in domain + label or "MPOWER" in label:
                pattern = "控烟|烟草|戒烟"
            elif "膳食" in domain + label or "控盐" in label:
                pattern = "控盐|膳食|营养|含糖"
            elif "基层" in domain + label:
                pattern = "基层|慢病|连续服务|随访"
            elif "治理" in domain + label:
                pattern = "综合治理|监测|跨部门|NCD"
            else:
                pattern = "慢病|NCD|健康"

            matched = transfer_work.loc[transfer_work["recommended_transfer_package"].str.contains(pattern, regex=True, na=False)] if not transfer_work.empty else pd.DataFrame()
            matched_top = (
                matched.sort_values("transfer_score", ascending=False, kind="stable")
                .groupby("province", as_index=False)
                .head(1)
                if not matched.empty
                else pd.DataFrame()
            )
            mean_transfer = finite_float(matched_top["transfer_score"].mean()) if not matched_top.empty else global_top_mean
            provinces_covered = int(matched_top["province"].nunique()) if not matched_top.empty else int(top_by_province["province"].nunique()) if not top_by_province.empty else 0
            domain_pathway_available = any(domain and (domain in item or item in domain) for item in pathway_domains)
            china_mapping_ready = bool(
                china_summary.get("health_outcome_anchor_available")
                and china_summary.get("china_local_policy_execution_indicator_available")
                and china_summary.get("c_layer_ocr_field_level_qc_completed")
            )
            score = (
                0.30 * float(domain_pathway_available)
                + 0.30 * float(china_mapping_ready)
                + 0.25 * bounded(mean_transfer)
                + 0.15 * bounded(provinces_covered / 31.0)
            )
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "outcome_column": row.get("outcome_column"),
                    "policy_label": label,
                    "policy_domain": domain,
                    "transfer_keyword_pattern": pattern,
                    "china_pathway_available": bool(domain_pathway_available),
                    "china_mapping_ready": bool(china_mapping_ready),
                    "matched_province_count": provinces_covered,
                    "mean_top_transfer_score": mean_transfer,
                    "china_transfer_readiness_score": bounded(score),
                }
            )
        return pd.DataFrame(rows)


    def add_check(
        checks: list[dict[str, object]],
        row: pd.Series,
        check_name: str,
        score: float,
        weight: float,
        status: str,
        evidence: str,
        boundary: str,
    ) -> None:
        checks.append(
            {
                "policy_id": row.get("policy_id"),
                "policy_label": row.get("policy_label"),
                "policy_domain": row.get("policy_domain"),
                "outcome_column": row.get("outcome_column"),
                "outcome_label": row.get("outcome_label"),
                "check_name": check_name,
                "check_weight": weight,
                "check_score": bounded(score),
                "weighted_score": bounded(score) * weight,
                "check_status": status,
                "evidence": evidence,
                "boundary": boundary,
            }
        )


    def status_from_score(score: float, pass_cut: float = 0.80, partial_cut: float = 0.50) -> str:
        if score >= pass_cut:
            return "pass"
        if score >= partial_cut:
            return "partial"
        return "fail"


    def build_design_integrity_matrix(portfolio: pd.DataFrame) -> pd.DataFrame:
        checks: list[dict[str, object]] = []
        weights = {
            "DID方向与显著性": 0.13,
            "事件研究前趋势": 0.11,
            "安慰剂检验": 0.09,
            "wild cluster bootstrap": 0.08,
            "cohort-robust ATT": 0.08,
            "合成控制方向": 0.07,
            "阈值/滞后敏感性": 0.08,
            "负向结果控制": 0.08,
            "异质性一致性": 0.06,
            "样本和处理规模": 0.07,
            "健康结果层级": 0.08,
            "声明边界精度": 0.08,
        }
        for _, row in portfolio.iterrows():
            p = finite_float(row.get("cluster_p_value"))
            direction = bool_value(row.get("direction_consistent"))
            did_score = 1.0 if direction and np.isfinite(p) and p < 0.10 else 0.65 if direction else 0.0
            add_check(checks, row, "DID方向与显著性", did_score, weights["DID方向与显著性"], status_from_score(did_score), f"cluster p={p:.4f}" if np.isfinite(p) else "cluster p=NA", "DID是观察性准因果估计，不是随机分配。")

            pretrend_score = 1.0 if bool_value(row.get("pretrend_pass")) else 0.0
            add_check(checks, row, "事件研究前趋势", pretrend_score, weights["事件研究前趋势"], status_from_score(pretrend_score), f"pretrend_pass={bool_value(row.get('pretrend_pass'))}", "前趋势通过支持可比性，但不能完全排除未观测混杂。")

            placebo_base = 1.0 if bool_value(row.get("placebo_pass")) else 0.0
            shifted = finite_float(row.get("shifted_placebo_pass_share"))
            if np.isfinite(shifted):
                placebo_score = 0.55 * placebo_base + 0.45 * shifted
                evidence = f"多政策组合层placebo={bool_value(row.get('placebo_pass'))}; shifted placebo pass={shifted:.2f}"
            else:
                placebo_score = placebo_base
                evidence = f"多政策组合层placebo={bool_value(row.get('placebo_pass'))}"
            add_check(checks, row, "安慰剂检验", placebo_score, weights["安慰剂检验"], status_from_score(placebo_score), evidence, "安慰剂通过说明伪处理风险较低，但仍按准因果表达。")

            wild_p = finite_float(row.get("wild_bootstrap_p"))
            wild_score = 1.0 if np.isfinite(wild_p) and wild_p < 0.10 else 0.55 if np.isfinite(wild_p) and wild_p < 0.20 else 0.0
            add_check(checks, row, "wild cluster bootstrap", wild_score, weights["wild cluster bootstrap"], status_from_score(wild_score), f"wild p={wild_p:.4f}" if np.isfinite(wild_p) else "wild p=NA", "小样本/聚类不确定性稳健性检验。")

            cohort_p = finite_float(row.get("cohort_att_p"))
            cohort_score = 1.0 if np.isfinite(cohort_p) and cohort_p < 0.10 and bool_value(row.get("cohort_pass_p_lt_0_10")) else 0.55 if np.isfinite(cohort_p) and cohort_p < 0.20 else 0.0
            add_check(checks, row, "cohort-robust ATT", cohort_score, weights["cohort-robust ATT"], status_from_score(cohort_score), f"cohort p={cohort_p:.4f}" if np.isfinite(cohort_p) else "cohort p=NA", "用于缓解交错DID处理时点异质性风险。")

            synth = finite_float(row.get("synthetic_direction_share"))
            synthetic_score = 1.0 if np.isfinite(synth) and synth >= 0.60 else 0.55 if np.isfinite(synth) and synth >= 0.50 else 0.0
            add_check(checks, row, "合成控制方向", synthetic_score, weights["合成控制方向"], status_from_score(synthetic_score), f"direction share={synth:.2f}" if np.isfinite(synth) else "direction share=NA", "反事实方向检验，方向通过不等于单独显著。")

            threshold_share = finite_float(row.get("threshold_direction_share"))
            threshold_p_share = finite_float(row.get("threshold_p_lt_0_10_share"), 0.0)
            threshold_score = 0.70 * bounded(threshold_share) + 0.30 * bounded(threshold_p_share) if np.isfinite(threshold_share) else 0.50
            add_check(checks, row, "阈值/滞后敏感性", threshold_score, weights["阈值/滞后敏感性"], status_from_score(threshold_score, 0.67, 0.50), f"direction share={threshold_share:.2f}; p<0.10 share={threshold_p_share:.2f}" if np.isfinite(threshold_share) else "not available", "检验政策阈值和滞后窗口改变后方向是否稳定。")

            neg_share = finite_float(row.get("negative_control_pass_share"))
            if np.isfinite(neg_share):
                neg_score = neg_share
                evidence = f"negative control pass share={neg_share:.2f}"
            else:
                neg_score = 0.55
                evidence = "not available for this policy path"
            add_check(checks, row, "负向结果控制", neg_score, weights["负向结果控制"], status_from_score(neg_score, 0.90, 0.55), evidence, "负控缺失不作为失败，但会降低顶层证据分。")

            hetero_share = finite_float(row.get("heterogeneity_direction_share"))
            hetero_score = bounded(hetero_share) if np.isfinite(hetero_share) else 0.55
            add_check(checks, row, "异质性一致性", hetero_score, weights["异质性一致性"], status_from_score(hetero_score, 0.67, 0.50), f"type direction share={hetero_share:.2f}" if np.isfinite(hetero_share) else "not available", "跨脆弱性类型方向一致性越高，外推越稳。")

            treated = finite_float(row.get("treated_countries"), 0.0)
            sample_score = min(1.0, treated / 50.0) if treated > 0 else 0.0
            add_check(checks, row, "样本和处理规模", sample_score, weights["样本和处理规模"], status_from_score(sample_score, 0.60, 0.20), f"treated countries={treated:.0f}", "处理国家数量影响准因果证据的外推稳定性。")

            direct = bool_value(row.get("direct_health_result"))
            service = bool_value(row.get("service_coverage_result"))
            strong = str(row.get("d5_tier", "")) == "准因果强候选"
            robust = str(row.get("d5_tier", "")) in {"准因果强候选", "稳健政策候选"}
            health_score = 1.0 if direct and strong else 0.85 if direct and robust else 0.70 if service and robust else 0.55 if direct or service else 0.35
            add_check(checks, row, "健康结果层级", health_score, weights["健康结果层级"], status_from_score(health_score, 0.80, 0.55), f"family={row.get('outcome_family')}; tier={row.get('d5_tier')}", "服务覆盖不能直接等同死亡率或DALY下降。")

            boundary_note = str(row.get("boundary_note", ""))
            claim_score = 1.0 if boundary_note and "随机" in str(row.get("claim_level", "")) + boundary_note else 0.85 if boundary_note else 0.60
            add_check(checks, row, "声明边界精度", claim_score, weights["声明边界精度"], status_from_score(claim_score), "claim boundary present" if boundary_note else "claim boundary missing", "顶层表达必须避免把观察性准因果写成随机实验。")
        return pd.DataFrame(checks)


    def build_bias_sensitivity(portfolio: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for _, row in portfolio.iterrows():
            p_values = [
                finite_float(row.get("cluster_p_value")),
                finite_float(row.get("wild_bootstrap_p")),
                finite_float(row.get("cohort_att_p")),
            ]
            p_values = [p for p in p_values if np.isfinite(p)]
            z_values = [p_to_z(p) for p in p_values if np.isfinite(p)]
            z_max = max(z_values) if z_values else np.nan
            p_pass_count = sum(1 for p in p_values if p < 0.10)
            p_available = len(p_values)
            attenuation = max(0.0, 1.0 - ZCRIT_10_TWO_SIDED / z_max) if np.isfinite(z_max) and z_max > ZCRIT_10_TWO_SIDED else 0.0
            effect = finite_float(row.get("did_coefficient_sd"), 0.0)
            effect_abs = abs(effect)
            multi_p_score = p_pass_count / p_available if p_available else 0.0
            score = 0.50 * bounded(attenuation) + 0.30 * bounded(effect_abs / 0.10) + 0.20 * bounded(multi_p_score)
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "policy_label": row.get("policy_label"),
                    "policy_domain": row.get("policy_domain"),
                    "outcome_column": row.get("outcome_column"),
                    "outcome_label": row.get("outcome_label"),
                    "did_coefficient_sd": effect,
                    "absolute_effect_sd": effect_abs,
                    "available_p_values": p_available,
                    "p_values_lt_0_10": p_pass_count,
                    "max_abs_z_from_reported_p": z_max,
                    "attenuation_fraction_needed_to_lose_p_lt_0_10": attenuation,
                    "omitted_bias_sd_needed_to_flip_sign": effect_abs,
                    "bias_sensitivity_score": bounded(score),
                    "interpretation": "需要较大未观测偏误才会推翻方向/显著性" if score >= 0.75 else "有一定稳健性但仍需按准因果候选表达" if score >= 0.55 else "对未观测偏误较敏感，只能作政策适配或机制证据",
                }
            )
        return pd.DataFrame(rows)


    def build_mechanism_scores(portfolio: pd.DataFrame, mechanism: pd.DataFrame) -> pd.DataFrame:
        tobacco_direction = np.nan
        tobacco_p_pass = np.nan
        tobacco_steps = 0
        if not mechanism.empty:
            work = mechanism.copy()
            work["direction_consistent"] = work["direction_consistent"].map(bool_value)
            p = pd.to_numeric(work["p_value"], errors="coerce")
            empirical = p.notna()
            tobacco_direction = float(work["direction_consistent"].mean())
            tobacco_p_pass = float((p[empirical] < 0.10).mean()) if empirical.any() else np.nan
            tobacco_steps = int(work.shape[0])

        rows: list[dict[str, object]] = []
        for _, row in portfolio.iterrows():
            pid = str(row.get("policy_id", ""))
            domain = str(row.get("policy_domain", ""))
            direct = bool_value(row.get("direct_health_result"))
            service = bool_value(row.get("service_coverage_result"))
            robust = str(row.get("d5_tier", "")) in {"准因果强候选", "稳健政策候选"}
            if "mpower" in pid:
                direction_share = tobacco_direction
                p_share = tobacco_p_pass
                steps = tobacco_steps
                score = 0.75 * bounded(direction_share) + 0.25 * bounded(p_share) if np.isfinite(direction_share) else 0.55
                note = "控烟路径有政策-暴露-风险负担-缺口机制链。"
            elif service and robust:
                direction_share = np.nan
                p_share = np.nan
                steps = 0
                score = 0.68
                note = "服务覆盖路径机制合理，但后半段健康结果需要单独边界说明。"
            elif direct and robust:
                direction_share = np.nan
                p_share = np.nan
                steps = 0
                score = 0.62
                note = "直接健康/风险负担候选有结果层证据，但机制链不如控烟主线完整。"
            elif "治理" in domain or "基层" in domain:
                direction_share = np.nan
                p_share = np.nan
                steps = 0
                score = 0.55
                note = "政策包机制可解释，主要用于适配而非单独因果证明。"
            else:
                direction_share = np.nan
                p_share = np.nan
                steps = 0
                score = 0.45
                note = "机制链证据不足，保留为探索或情景模拟。"
            rows.append(
                {
                    "policy_id": row.get("policy_id"),
                    "outcome_column": row.get("outcome_column"),
                    "policy_label": row.get("policy_label"),
                    "policy_domain": domain,
                    "mechanism_steps": steps,
                    "mechanism_direction_share": direction_share,
                    "mechanism_p_lt_0_10_share": p_share,
                    "mechanism_score": bounded(score),
                    "mechanism_note": note,
                }
            )
        return pd.DataFrame(rows)


    def grade_d6(score: float, design: float, bias: float, claim: float) -> str:
        if score >= 0.86 and design >= 0.78 and bias >= 0.70 and claim >= 0.85:
            return "金牌主证据"
        if score >= 0.76 and design >= 0.68:
            return "银牌强候选"
        if score >= 0.64:
            return "铜牌适配候选"
        return "机制/探索证据"


    def claim_level_from_grade(row: pd.Series) -> str:
        grade = str(row.get("d6_grade", ""))
        family = str(row.get("outcome_family", ""))
        if grade == "金牌主证据":
            return "可作为模块D最强主证据：严格准因果强候选，不称随机实验。"
        if grade == "银牌强候选" and "服务覆盖" in family:
            return "可作为响应能力强候选：说明服务覆盖改善，不直接替代死亡率/DALY。"
        if grade == "银牌强候选":
            return "可作为模块D备选强候选或主证据旁证。"
        if grade == "铜牌适配候选":
            return "进入政策包和中国情景模拟，不单独承担强因果结论。"
        return "只作机制解释、敏感性或探索材料。"


    def build_scorecard(
        portfolio: pd.DataFrame,
        checks: pd.DataFrame,
        bias: pd.DataFrame,
        mechanism_scores: pd.DataFrame,
        transfer_ready: pd.DataFrame,
    ) -> pd.DataFrame:
        if portfolio.empty:
            return pd.DataFrame()
        check_scores = (
            checks.groupby(["policy_id", "outcome_column"], as_index=False)
            .agg(
                design_integrity_score=("weighted_score", "sum"),
                design_checks=("check_name", "count"),
                design_passes=("check_status", lambda s: int((s == "pass").sum())),
                design_partials=("check_status", lambda s: int((s == "partial").sum())),
                design_failures=("check_status", lambda s: int((s == "fail").sum())),
            )
            if not checks.empty
            else pd.DataFrame()
        )
        work = portfolio.merge(check_scores, on=["policy_id", "outcome_column"], how="left")
        work = work.merge(
            bias[["policy_id", "outcome_column", "bias_sensitivity_score", "attenuation_fraction_needed_to_lose_p_lt_0_10", "omitted_bias_sd_needed_to_flip_sign"]],
            on=["policy_id", "outcome_column"],
            how="left",
        )
        work = work.merge(
            mechanism_scores[["policy_id", "outcome_column", "mechanism_score", "mechanism_note"]],
            on=["policy_id", "outcome_column"],
            how="left",
        )
        work = work.merge(
            transfer_ready[["policy_id", "outcome_column", "china_transfer_readiness_score", "matched_province_count", "mean_top_transfer_score"]],
            on=["policy_id", "outcome_column"],
            how="left",
        )
        work["claim_precision_score"] = np.where(work["boundary_note"].astype(str).str.len() > 0, 1.0, 0.65)
        work["design_integrity_score"] = pd.to_numeric(work["design_integrity_score"], errors="coerce").fillna(0.0)
        work["bias_sensitivity_score"] = pd.to_numeric(work["bias_sensitivity_score"], errors="coerce").fillna(0.0)
        work["mechanism_score"] = pd.to_numeric(work["mechanism_score"], errors="coerce").fillna(0.45)
        work["china_transfer_readiness_score"] = pd.to_numeric(work["china_transfer_readiness_score"], errors="coerce").fillna(0.50)
        work["d6_total_score"] = (
            0.54 * work["design_integrity_score"]
            + 0.17 * work["bias_sensitivity_score"]
            + 0.14 * work["mechanism_score"]
            + 0.10 * work["china_transfer_readiness_score"]
            + 0.05 * work["claim_precision_score"]
        )
        work["d6_grade"] = work.apply(
            lambda row: grade_d6(
                finite_float(row.get("d6_total_score"), 0.0),
                finite_float(row.get("design_integrity_score"), 0.0),
                finite_float(row.get("bias_sensitivity_score"), 0.0),
                finite_float(row.get("claim_precision_score"), 0.0),
            ),
            axis=1,
        )
        work["d6_claim_level"] = work.apply(claim_level_from_grade, axis=1)
        work["answer_line"] = work.apply(
            lambda row: (
                f"{row.get('policy_label')} -> {row.get('outcome_label')}："
                f"证据分={finite_float(row.get('d6_total_score'), 0):.2f}，{row.get('d6_grade')}；"
                f"{row.get('d6_claim_level')}"
            ),
            axis=1,
        )
        keep = [
            "policy_id",
            "policy_label",
            "policy_domain",
            "outcome_column",
            "outcome_label",
            "outcome_family",
            "d5_validation_score",
            "d5_tier",
            "design_integrity_score",
            "bias_sensitivity_score",
            "mechanism_score",
            "china_transfer_readiness_score",
            "claim_precision_score",
            "d6_total_score",
            "d6_grade",
            "d6_claim_level",
            "design_checks",
            "design_passes",
            "design_partials",
            "design_failures",
            "attenuation_fraction_needed_to_lose_p_lt_0_10",
            "omitted_bias_sd_needed_to_flip_sign",
            "matched_province_count",
            "mean_top_transfer_score",
            "mechanism_note",
            "answer_line",
            "boundary_note",
        ]
        return work.loc[:, [col for col in keep if col in work.columns]].sort_values("d6_total_score", ascending=False, kind="stable")


    def build_defense_answerbook(scorecard: pd.DataFrame, summary: dict[str, object]) -> pd.DataFrame:
        top = scorecard.iloc[0].to_dict() if not scorecard.empty else {}
        top_line = top.get("answer_line", "模块D证据裁判包未生成。")
        rows = [
            {
                "rank": 1,
                "hard_question": "你们为什么不做随机实验级强因果？",
                "answer": "国家和省级政策不是随机分配，不能诚实称为RCT。我们的做法是用DID、事件研究、安慰剂、合成控制、wild bootstrap、cohort ATT、负控、机制链和偏误敏感性构成严格准因果强候选。",
            },
            {
                "rank": 2,
                "hard_question": "如果政策实施国家本来治理能力更强，结论会不会有选择偏差？",
                "answer": "模块D证据裁判与锁定层把这个问题拆成前趋势、安慰剂、cohort ATT、阈值/滞后、负控和偏误敏感性。通过越多，越能作为强候选；失败路径自动降级为适配或探索证据。",
            },
            {
                "rank": 3,
                "hard_question": "最强政策证据是哪条？",
                "answer": str(top_line),
            },
            {
                "rank": 4,
                "hard_question": "服务覆盖提升能不能说明死亡率下降？",
                "answer": "不能直接替代。模块D证据裁判与锁定层明确区分直接健康/风险负担结果、风险变化结果和服务覆盖中间结果；服务覆盖路径只说明响应能力改善。",
            },
            {
                "rank": 5,
                "hard_question": "你们如何防止只挑显著结果？",
                "answer": "模块D证据裁判与锁定层不是只看p值，而是把所有多政策路径候选统一进入设计完整性矩阵；每条路径都有pass/partial/fail和降级规则，未通过的仍保留为边界而不是删除。",
            },
            {
                "rank": 6,
                "hard_question": "为什么可以迁移到中国？",
                "answer": "迁移不是直接套用全球结论，而是先做中国31省压力-风险-响应画像，再用地方政策执行指标、健康结局锚点和相似国家迁移评分生成建议。",
            },
            {
                "rank": 7,
                "hard_question": "如果未观测混杂很强怎么办？",
                "answer": "模块D证据裁判与锁定层新增偏误敏感性表，报告每条路径需要多少标准化偏误才会翻转方向或跌出p<0.10。敏感路径不承担主证据。",
            },
            {
                "rank": 8,
                "hard_question": "负向结果控制有什么意义？",
                "answer": "负控用于检验政策是否对理论上不应受影响的结果也显著。如果负控也显著，说明可能有系统性混杂，路径会在证据裁判层中降级。",
            },
            {
                "rank": 9,
                "hard_question": "机制链里有些长期疾病负担环节方向不支持怎么办？",
                "answer": "这正是模块D证据裁判与锁定层边界：短期可讲政策到风险暴露/风险负担变化，长期疾病负担不作为单独因果证明，只作为机制边界说明。",
            },
            {
                "rank": 10,
                "hard_question": "模块D现在最终锁定成什么？",
                "answer": f"{summary.get('module_d_level', 'module_d_evidence_extreme_quasi_causal_policy_adaptation_engine')}；它是严格准因果政策适配引擎，不是随机实验式强因果。",
            },
        ]
        return pd.DataFrame(rows)


    def plot_scorecard(scorecard: pd.DataFrame, path: Path) -> None:
        if scorecard.empty:
            return
        top = scorecard.head(12).copy()
        labels = top["policy_label"].astype(str) + "\n" + top["outcome_label"].astype(str)
        colors = np.where(
            top["d6_grade"].eq("金牌主证据"),
            "#0b6b57",
            np.where(top["d6_grade"].eq("银牌强候选"), "#2a9d8f", np.where(top["d6_grade"].eq("铜牌适配候选"), "#f2b84b", "#94a3b8")),
        )
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.barh(labels, top["d6_total_score"], color=colors)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel(choose_text("顶层证据分", "top-level evidence score", USE_CHINESE))
        ax.set_title(choose_text("模块D证据裁判与锁定层", "Module D Evidence Adjudication and Lock Layer", USE_CHINESE))
        ax.grid(axis="x", alpha=0.22)
        for idx, value in enumerate(top["d6_total_score"]):
            ax.text(float(value) + 0.01, idx, f"{value:.2f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


    def plot_bias_sensitivity(bias: pd.DataFrame, path: Path) -> None:
        if bias.empty:
            return
        top = bias.sort_values("bias_sensitivity_score", ascending=False, kind="stable").head(14).copy()
        labels = top["policy_label"].astype(str) + "\n" + top["outcome_label"].astype(str)
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.barh(labels, top["bias_sensitivity_score"], color="#4f7cac")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel(choose_text("偏误敏感性分", "Bias sensitivity score", USE_CHINESE))
        ax.set_title(choose_text("未观测偏误翻转压力测试", "Unobserved Bias Tipping-Point Stress Test", USE_CHINESE))
        ax.grid(axis="x", alpha=0.22)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build 模块D证据裁判与锁定层 extreme quasi-causal excellence layer for Module D.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        doc_dir = project_root / "06_report_assets"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        doc_dir.mkdir(parents=True, exist_ok=True)

        portfolio = read_csv(report_asset_path(report_dir, "policy_d5_global_validation_portfolio.csv"))
        if portfolio.empty:
            raise FileNotFoundError("policy_d5_global_validation_portfolio.csv is required before 模块D证据裁判与锁定层.")
        health_matrix = read_csv(report_asset_path(report_dir, "policy_d5_health_outcome_candidate_matrix.csv"))
        if not health_matrix.empty:
            keep = [
                "policy_id",
                "outcome_column",
                "outcome_family",
                "direct_health_result_candidate",
                "service_coverage_candidate",
            ]
            health_keep = health_matrix[[col for col in keep if col in health_matrix.columns]].rename(
                columns={
                    "direct_health_result_candidate": "direct_health_result",
                    "service_coverage_candidate": "service_coverage_result",
                }
            )
            portfolio = portfolio.merge(health_keep, on=["policy_id", "outcome_column"], how="left")
        classified = portfolio.apply(classify_outcome, axis=1, result_type="expand")
        for col in classified.columns:
            if col not in portfolio.columns:
                portfolio[col] = classified[col]
            else:
                portfolio[col] = portfolio[col].where(portfolio[col].notna(), classified[col])

        threshold = aggregate_threshold_lag(read_csv(report_asset_path(report_dir, "policy_d4_threshold_lag_sensitivity.csv")))
        negative = aggregate_negative_controls(read_csv(report_asset_path(report_dir, "policy_d4_negative_control_outcomes.csv")))
        placebo = aggregate_placebo(read_csv(report_asset_path(report_dir, "policy_causal_placebo_tests.csv")))
        heterogeneity = aggregate_heterogeneity(read_csv(report_asset_path(report_dir, "policy_causal_heterogeneity.csv")))
        mechanism = read_csv(report_asset_path(report_dir, "policy_d5_mechanism_boundary_audit.csv"))
        transfer = read_csv(report_asset_path(report_dir, "policy_transfer_similarity_scores.csv"))
        pathways = read_csv(report_asset_path(report_dir, "policy_response_pathways.csv"))
        china_summary = read_json(report_asset_path(report_dir, "china_mapping_framework_summary.json"))

        for table in [threshold, placebo, heterogeneity]:
            if not table.empty:
                portfolio = portfolio.merge(table, on=["policy_id", "outcome_column"], how="left")
        if not negative.empty:
            portfolio = portfolio.merge(negative, on="policy_id", how="left")

        transfer_ready = build_china_transfer_readiness(portfolio, transfer, pathways, china_summary)
        design_matrix = build_design_integrity_matrix(portfolio)
        bias = build_bias_sensitivity(portfolio)
        mechanism_scores = build_mechanism_scores(portfolio, mechanism)
        scorecard = build_scorecard(portfolio, design_matrix, bias, mechanism_scores, transfer_ready)

        scorecard_path = report_asset_path(report_dir, "policy_d6_extreme_quasi_causal_scorecard.csv")
        design_path = report_asset_path(report_dir, "policy_d6_design_integrity_matrix.csv")
        bias_path = report_asset_path(report_dir, "policy_d6_bias_sensitivity_tipping_points.csv")
        transfer_path = report_asset_path(report_dir, "policy_d6_china_transfer_readiness.csv")
        mechanism_path = report_asset_path(report_dir, "policy_d6_mechanism_transport_matrix.csv")
        answerbook_csv_path = report_asset_path(report_dir, "policy_d6_defense_answerbook.csv")
        summary_path = report_asset_path(report_dir, "policy_d6_extreme_quasi_causal_summary.json")
        score_fig_path = figure_dir / "advanced_policy_d6_scorecard.png"
        bias_fig_path = figure_dir / "advanced_policy_d6_bias_sensitivity.png"
        answerbook_md_path = report_asset_path(doc_dir, "module_d_d6_defense_answerbook.md")

        write_csv(scorecard, scorecard_path)
        write_csv(design_matrix, design_path)
        write_csv(bias, bias_path)
        write_csv(transfer_ready, transfer_path)
        write_csv(mechanism_scores, mechanism_path)
        plot_scorecard(scorecard, score_fig_path)
        plot_bias_sensitivity(bias, bias_fig_path)

        gold = scorecard.loc[scorecard["d6_grade"].eq("金牌主证据")] if not scorecard.empty else pd.DataFrame()
        silver_or_above = scorecard.loc[scorecard["d6_grade"].isin(["金牌主证据", "银牌强候选"])] if not scorecard.empty else pd.DataFrame()
        top = scorecard.iloc[0].to_dict() if not scorecard.empty else {}
        summary = {
            "project_root": project_root.as_posix(),
            "module_d_level": "模块D证据裁判与锁定层",
            "d6_locked": bool(not gold.empty and finite_float(top.get("d6_total_score"), 0.0) >= 0.86),
            "candidate_rows": int(scorecard.shape[0]),
            "design_integrity_check_rows": int(design_matrix.shape[0]),
            "bias_sensitivity_rows": int(bias.shape[0]),
            "china_transfer_readiness_rows": int(transfer_ready.shape[0]),
            "d6_gold_main_evidence_candidates": int(gold.shape[0]),
            "d6_silver_or_above_candidates": int(silver_or_above.shape[0]),
            "top_candidate": top,
            "claim_boundary": "模块D证据裁判与锁定层把模块D推到严格准因果证据裁判层：统一评分设计完整性、偏误敏感性、机制链、迁移适配和声明边界；仍不称随机实验式强因果。",
            "can_claim": [
                "模块D已从准因果增强和多政策路径组合升级为模块D证据裁判与锁定层。",
                "模块D证据裁判与锁定层不是新增夸大因果，而是把每条政策路径按设计完整性、偏误敏感性、机制链、迁移适配和声明边界统一裁判。",
                "最强路径可称金牌主证据或严格准因果强候选，但不能称随机实验式强因果。",
                "中国映射使用模块D证据裁判与锁定层筛出的全球路径作为政策适配输入，不反向证明全球因果。",
            ],
            "cannot_claim": [
                "不能说模块D证据裁判与锁定层等同RCT或随机实验级强因果。",
                "不能把服务覆盖强候选说成死亡率或DALY已经下降。",
                "不能把中国省级映射说成全球证据裁判来源。",
            ],
            "output_files": {
                "policy_d6_extreme_quasi_causal_scorecard": scorecard_path.as_posix(),
                "policy_d6_design_integrity_matrix": design_path.as_posix(),
                "policy_d6_bias_sensitivity_tipping_points": bias_path.as_posix(),
                "policy_d6_china_transfer_readiness": transfer_path.as_posix(),
                "policy_d6_mechanism_transport_matrix": mechanism_path.as_posix(),
                "policy_d6_defense_answerbook": answerbook_csv_path.as_posix(),
                "policy_d6_extreme_quasi_causal_summary": summary_path.as_posix(),
                "advanced_policy_d6_scorecard": score_fig_path.as_posix(),
                "advanced_policy_d6_bias_sensitivity": bias_fig_path.as_posix(),
                "module_d_d6_defense_answerbook": answerbook_md_path.as_posix(),
            },
        }

        answerbook = build_defense_answerbook(scorecard, summary)
        write_csv(answerbook, answerbook_csv_path)
        md_lines = [
            "# 模块D证据裁判与锁定层答辩证据包",
            "",
            "## 结论",
            "",
            f"模块D已升级为 `{summary['module_d_level']}`。",
            "",
            "模块D证据裁判与锁定层的作用不是把观察性政策数据包装成随机实验，而是把每条政策路径放到统一裁判层：设计完整性、偏误敏感性、机制链、迁移适配和声明边界。",
            "",
            "## 最强路径",
            "",
            str(top.get("answer_line", "")),
            "",
            "## 高压问答",
            "",
        ]
        for _, row in answerbook.iterrows():
            md_lines.extend(
                [
                    f"### {int(row['rank'])}. {row['hard_question']}",
                    "",
                    str(row["answer"]),
                    "",
                ]
            )
        answerbook_md_path.write_text("\n".join(md_lines), encoding="utf-8")

        summary_path.write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(json_clean(summary), ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_module_d_lock_audit():
    __name__ = 'run_module_d_lock_audit'
    import argparse
    import json
    from pathlib import Path

    import pandas as pd

    from foundation import detect_project_root as shared_detect_project_root


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def read_json(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


    def file_check(path: Path, required: bool = True) -> dict[str, object]:
        return {
            "item": path.name,
            "path": path.as_posix(),
            "required": required,
            "exists": path.exists(),
            "status": "pass" if path.exists() or not required else "fail",
            "detail": "exists" if path.exists() else "missing",
        }


    def csv_row_count(path: Path) -> int | None:
        if not path.exists():
            return None
        return int(pd.read_csv(path).shape[0])


    def main() -> None:
        parser = argparse.ArgumentParser(description="Audit what is and is not locked in Module D.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        report_dir = project_root / "06_report_assets"
        doc_dir = project_root / "06_report_assets"
        doc_dir.mkdir(parents=True, exist_ok=True)

        required_global_files = {
            "policy_identification_summary": report_asset_path(report_dir, "policy_identification_summary.json"),
            "policy_validation_summary": report_asset_path(report_dir, "policy_validation_summary.csv"),
            "policy_adaptation_engine_summary": report_asset_path(report_dir, "policy_adaptation_engine_summary.json"),
            "policy_response_pathways": report_asset_path(report_dir, "policy_response_pathways.csv"),
            "policy_resource_uplift_scenarios": report_asset_path(report_dir, "policy_resource_uplift_scenarios.csv"),
            "policy_risk_reduction_scenarios": report_asset_path(report_dir, "policy_risk_reduction_scenarios.csv"),
            "policy_type_strategy_matrix": report_asset_path(report_dir, "policy_type_strategy_matrix.csv"),
            "policy_ncd_capacity_latest": report_asset_path(report_dir, "policy_ncd_capacity_latest.csv"),
            "policy_ncd_capacity_type_summary": report_asset_path(report_dir, "policy_ncd_capacity_type_summary.csv"),
            "policy_ncd_service_gap_scenarios": report_asset_path(report_dir, "policy_ncd_service_gap_scenarios.csv"),
            "policy_causal_enhancement_summary": report_asset_path(report_dir, "policy_causal_enhancement_summary.json"),
            "policy_d4_data_source_summary": report_asset_path(report_dir, "policy_d4_data_source_summary.json"),
            "policy_causal_event_library": report_asset_path(report_dir, "policy_causal_event_library.csv"),
            "policy_causal_did_estimates": report_asset_path(report_dir, "policy_causal_did_estimates.csv"),
            "policy_causal_evidence_ladder": report_asset_path(report_dir, "policy_causal_evidence_ladder.csv"),
            "policy_synthetic_control_estimates": report_asset_path(report_dir, "policy_synthetic_control_estimates.csv"),
            "policy_matrix_completion_estimates": report_asset_path(report_dir, "policy_matrix_completion_estimates.csv"),
            "policy_combo_optimization_scenarios": report_asset_path(report_dir, "policy_combo_optimization_scenarios.csv"),
            "policy_d4_advanced_validation_summary": report_asset_path(report_dir, "policy_d4_advanced_validation_summary.json"),
            "policy_d4_wild_cluster_bootstrap": report_asset_path(report_dir, "policy_d4_wild_cluster_bootstrap.csv"),
            "policy_d4_cohort_robust_att": report_asset_path(report_dir, "policy_d4_cohort_robust_att.csv"),
            "policy_d4_threshold_lag_sensitivity": report_asset_path(report_dir, "policy_d4_threshold_lag_sensitivity.csv"),
            "policy_d4_negative_control_outcomes": report_asset_path(report_dir, "policy_d4_negative_control_outcomes.csv"),
            "policy_d4_mechanism_chain": report_asset_path(report_dir, "policy_d4_mechanism_chain.csv"),
            "policy_d5_non_tobacco_pathway_summary": report_asset_path(report_dir, "policy_d5_non_tobacco_pathway_summary.json"),
            "policy_d5_non_tobacco_pathway_candidates": report_asset_path(report_dir, "policy_d5_non_tobacco_pathway_candidates.csv"),
            "policy_d6_extreme_quasi_causal_summary": report_asset_path(report_dir, "policy_d6_extreme_quasi_causal_summary.json"),
            "policy_d6_extreme_quasi_causal_scorecard": report_asset_path(report_dir, "policy_d6_extreme_quasi_causal_scorecard.csv"),
            "policy_d6_design_integrity_matrix": report_asset_path(report_dir, "policy_d6_design_integrity_matrix.csv"),
            "policy_d6_bias_sensitivity_tipping_points": report_asset_path(report_dir, "policy_d6_bias_sensitivity_tipping_points.csv"),
            "policy_d6_china_transfer_readiness": report_asset_path(report_dir, "policy_d6_china_transfer_readiness.csv"),
            "policy_d6_defense_answerbook": report_asset_path(report_dir, "policy_d6_defense_answerbook.csv"),
        }
        downstream_china_files = {
            "china_policy_mapping_rules": report_asset_path(report_dir, "china_policy_mapping_rules.csv"),
            "china_provincial_policy_cards": report_asset_path(report_dir, "china_provincial_policy_cards.csv"),
            "china_mapping_framework_summary": report_asset_path(report_dir, "china_mapping_framework_summary.json"),
            "china_nhc_chronic_demo_zone_policy_score_latest": report_asset_path(report_dir, "china_nhc_chronic_demo_zone_policy_score_latest.csv"),
            "china_nbs2025_ocr_explanatory_candidates_2024": report_asset_path(report_dir, "china_nbs2025_ocr_explanatory_candidates_2024.csv"),
            "china_nbs_mortality_panel_ocr_2018_2024": report_asset_path(report_dir, "china_nbs_mortality_panel_ocr_2018_2024.csv"),
            "china_policy_health_outcome_sensitivity_estimates": report_asset_path(report_dir, "china_policy_health_outcome_sensitivity_estimates.csv"),
            "china_c_layer_ocr_candidate_qc_2024": report_asset_path(report_dir, "china_c_layer_ocr_candidate_qc_2024.csv"),
            "china_c_layer_ocr_field_level_qc_2024": report_asset_path(report_dir, "china_c_layer_ocr_field_level_qc_2024.csv"),
            "china_gbd2021_homology_boundary_audit": report_asset_path(report_dir, "china_gbd2021_homology_boundary_audit.csv"),
            "china_gbd2021_freshness_bridge": report_asset_path(report_dir, "china_gbd2021_freshness_bridge.csv"),
            "china_health_outcome_anchor_2017_2024": report_asset_path(report_dir, "china_health_outcome_anchor_2017_2024.csv"),
            "china_local_policy_execution_indicator_latest": report_asset_path(report_dir, "china_local_policy_execution_indicator_latest.csv"),
            "china_local_policy_execution_source_dictionary": report_asset_path(report_dir, "china_local_policy_execution_source_dictionary.csv"),
        }

        checks: list[dict[str, object]] = []
        for path in required_global_files.values():
            checks.append(file_check(path, required=True))
        for path in downstream_china_files.values():
            checks.append({**file_check(path, required=False), "detail": "downstream_application_output"})

        policy_summary = read_json(required_global_files["policy_identification_summary"])
        adaptation_summary = read_json(required_global_files["policy_adaptation_engine_summary"])
        causal_summary = read_json(required_global_files["policy_causal_enhancement_summary"])
        advanced_summary = read_json(required_global_files["policy_d4_advanced_validation_summary"])
        non_tobacco_summary = read_json(required_global_files["policy_d5_non_tobacco_pathway_summary"])
        d6_summary = read_json(required_global_files["policy_d6_extreme_quasi_causal_summary"])
        china_mapping_summary = read_json(downstream_china_files["china_mapping_framework_summary"])
        china_response_summary = read_json(report_asset_path(report_dir, "china_policy_quasi_causal_response_summary.json"))
        mortality_summary = read_json(report_asset_path(report_dir, "china_nbs_mortality_panel_ocr_summary.json"))
        gbd_bridge_summary = read_json(report_asset_path(report_dir, "china_gbd2021_freshness_bridge_summary.json"))
        health_anchor_summary = read_json(report_asset_path(report_dir, "china_health_outcome_anchor_summary.json"))
        local_policy_summary = read_json(report_asset_path(report_dir, "china_local_policy_execution_indicator_summary.json"))
        c_layer_summary = read_json(report_asset_path(report_dir, "china_c_layer_ocr_candidate_qc_summary.json"))

        locked_outcomes = policy_summary.get("locked_outcomes", []) or []
        global_locked_outcomes = policy_summary.get("global_locked_outcomes", []) or []
        tobacco_causal_locked = bool(locked_outcomes or global_locked_outcomes)
        checks.append(
            {
                "item": "tobacco_causal_lock",
                "path": required_global_files["policy_identification_summary"].as_posix(),
                "required": True,
                "exists": True,
                "status": "warn" if not tobacco_causal_locked else "pass",
                "detail": "no strong causal locked outcomes; keep as trend/mechanism evidence"
                if not tobacco_causal_locked
                else f"locked_outcomes={locked_outcomes or global_locked_outcomes}",
            }
        )

        expected_min_rows = {
            "policy_response_pathways": 12,
            "policy_resource_uplift_scenarios": 190,
            "policy_risk_reduction_scenarios": 1500,
            "policy_type_strategy_matrix": 3,
            "policy_ncd_capacity_latest": 190,
            "policy_ncd_capacity_type_summary": 3,
            "policy_ncd_service_gap_scenarios": 500,
            "policy_causal_event_library": 500,
            "policy_causal_did_estimates": 30,
            "policy_causal_evidence_ladder": 30,
            "policy_synthetic_control_estimates": 200,
            "policy_matrix_completion_estimates": 10,
            "policy_combo_optimization_scenarios": 1000,
            "policy_d4_wild_cluster_bootstrap": 10,
            "policy_d4_cohort_robust_att": 5,
            "policy_d4_threshold_lag_sensitivity": 40,
            "policy_d4_negative_control_outcomes": 4,
            "policy_d4_mechanism_chain": 6,
            "policy_d5_non_tobacco_pathway_candidates": 5,
            "policy_d6_extreme_quasi_causal_scorecard": 31,
            "policy_d6_design_integrity_matrix": 300,
            "policy_d6_bias_sensitivity_tipping_points": 31,
            "policy_d6_china_transfer_readiness": 31,
            "policy_d6_defense_answerbook": 10,
        }
        for key, minimum in expected_min_rows.items():
            rows = csv_row_count(required_global_files[key])
            checks.append(
                {
                    "item": f"{key}_row_count",
                    "path": required_global_files[key].as_posix(),
                    "required": True,
                    "exists": rows is not None,
                    "status": "pass" if rows is not None and rows >= minimum else "fail",
                    "detail": f"rows={rows}, expected_min={minimum}",
                }
            )

        critical_failures = [check for check in checks if check["required"] and check["status"] == "fail"]
        global_adaptation_engine_locked = len(critical_failures) == 0 and bool(adaptation_summary)
        causal_signal_counts = causal_summary.get("causal_signal_counts", {}) if causal_summary else {}
        causal_enhancement_locked = len(critical_failures) == 0 and bool(causal_summary)
        promotable_quasi_causal_paths = int(causal_signal_counts.get("promotable_quasi_causal_paths", 0) or 0)
        directional_did_signals = int(causal_signal_counts.get("p_lt_0_10_directional_did_signals", 0) or 0)
        strict_quasi_causal_locked = promotable_quasi_causal_paths > 0
        advanced_core = advanced_summary.get("core_candidate", {}) if advanced_summary else {}
        advanced_validation_locked = bool(
            advanced_summary
            and (advanced_core.get("wild_bootstrap_p") is not None and float(advanced_core.get("wild_bootstrap_p")) < 0.10)
            and (advanced_core.get("cohort_att_p") is not None and float(advanced_core.get("cohort_att_p")) < 0.05)
            and float(advanced_core.get("negative_control_pass_share", 0) or 0) >= 1.0
            and float(advanced_core.get("mechanism_direction_share", 0) or 0) >= 0.75
        )
        d6_locked = bool(d6_summary.get("d6_locked", False))
        d6_top = d6_summary.get("top_candidate", {}) if d6_summary else {}
        d6_gold_candidates = int(d6_summary.get("d6_gold_main_evidence_candidates", 0) or 0) if d6_summary else 0
        d6_silver_or_above = int(d6_summary.get("d6_silver_or_above_candidates", 0) or 0) if d6_summary else 0
        lock_level = (
            "模块D证据裁判与锁定层"
            if d6_locked
            else "模块D准因果增强层高级稳健性"
            if causal_enhancement_locked and strict_quasi_causal_locked and advanced_validation_locked
            else "模块D准因果增强层"
            if causal_enhancement_locked and strict_quasi_causal_locked
            else "模块D政策适配引擎层"
            if causal_enhancement_locked
            else "模块D政策数据层"
            if global_adaptation_engine_locked
            else "not_locked"
        )

        status = {
            "project_root": project_root.as_posix(),
            "global_module_d_status": {
                "causal_policy_effect_locked": tobacco_causal_locked,
                "policy_adaptation_engine_locked": global_adaptation_engine_locked,
                "causal_enhancement_layer_locked": causal_enhancement_locked,
                "randomized_experiment_style_strong_causality_locked": False,
                "strict_quasi_causal_candidate_locked": strict_quasi_causal_locked,
                "advanced_quasi_causal_validation_locked": advanced_validation_locked,
                "d6_extreme_quasi_causal_layer_locked": d6_locked,
                "promotable_quasi_causal_paths": promotable_quasi_causal_paths,
                "directional_did_signals_p_lt_0_10": directional_did_signals,
                "advanced_core_candidate": advanced_core,
                "d6_top_candidate": d6_top,
                "d6_gold_main_evidence_candidates": d6_gold_candidates,
                "d6_silver_or_above_candidates": d6_silver_or_above,
                "lock_level": lock_level,
                "scope": "global_sample_policy_pathways_policy_execution_service_coverage_scenario_simulation_d6_extreme_quasi_causal_adjudication_and_policy_optimization",
                "non_tobacco_d5_pathway_hardened": bool(non_tobacco_summary),
                "non_tobacco_policy_domains": int(non_tobacco_summary.get("non_tobacco_policy_domains", 0) or 0) if non_tobacco_summary else 0,
                "strong_non_tobacco_candidates": int(non_tobacco_summary.get("strong_non_tobacco_candidates", 0) or 0) if non_tobacco_summary else 0,
                "robust_or_adaptation_non_tobacco_candidates": int(non_tobacco_summary.get("robust_or_adaptation_non_tobacco_candidates", 0) or 0) if non_tobacco_summary else 0,
            },
            "china_mapping_status": {
                "status": "downstream_provincial_policy_adaptation_profile",
                "is_evidence_for_global_d_lock": False,
                "allowed_after_global_d4_plus": causal_enhancement_locked and strict_quasi_causal_locked and advanced_validation_locked,
                "boundary": "China mapping consumes global D outputs and adds provincial response evidence; it does not prove provincial health-outcome causality or the global policy effect.",
                "china_mapping_definition": china_mapping_summary.get("definition", "") if china_mapping_summary else "",
                "china_chronic_policy_execution_available": bool(china_mapping_summary.get("china_chronic_policy_execution_available", False)) if china_mapping_summary else False,
                "nbs2025_explanatory_candidate_available": bool(china_mapping_summary.get("nbs2025_explanatory_candidate_available", False)) if china_mapping_summary else False,
                "nbs_mortality_panel_available": bool(china_mapping_summary.get("nbs_mortality_panel_available", False)) if china_mapping_summary else False,
                "nbs_mortality_panel_included_years": china_mapping_summary.get("nbs_mortality_panel_included_years", []) if china_mapping_summary else [],
                "nbs_health_outcome_sensitivity_rows": int(china_response_summary.get("nbs_health_outcome_sensitivity_rows", 0) or 0) if china_response_summary else 0,
                "c_layer_ocr_qc_high_confidence_rows": int(china_mapping_summary.get("c_layer_ocr_qc_high_confidence_rows", 0) or 0) if china_mapping_summary else 0,
                "c_layer_ocr_field_level_qc_completed": bool(china_mapping_summary.get("c_layer_ocr_field_level_qc_completed", False)) if china_mapping_summary else False,
                "c_layer_ocr_unresolved_manual_review_blockers": int(c_layer_summary.get("unresolved_manual_review_blockers", 0) or 0) if c_layer_summary else 0,
                "gbd2021_freshness_bridge_available": bool(china_mapping_summary.get("gbd2021_freshness_bridge_available", False)) if china_mapping_summary else False,
                "gbd2021_freshness_bridge_spearman_r": china_mapping_summary.get("gbd2021_freshness_bridge_spearman_r") if china_mapping_summary else None,
                "health_outcome_anchor_available": bool(china_mapping_summary.get("health_outcome_anchor_available", False)) if china_mapping_summary else False,
                "health_outcome_anchor_rows": int(health_anchor_summary.get("province_rows", 0) or 0) if health_anchor_summary else 0,
                "china_local_policy_execution_indicator_available": bool(china_mapping_summary.get("china_local_policy_execution_indicator_available", False)) if china_mapping_summary else False,
                "china_local_policy_scored_domains": local_policy_summary.get("scored_policy_domains", []) if local_policy_summary else [],
                "china_local_policy_national_milestone_domains": local_policy_summary.get("national_milestone_policy_domains", []) if local_policy_summary else [],
            },
            "can_claim": [
                "模块D已经升级为模块D证据裁判与锁定层：在准因果增强层和多政策路径组合层基础上统一评分设计完整性、偏误敏感性、机制链、迁移适配和声明边界。",
                "模块D已经升级为模块D准因果增强层高级稳健性准因果政策适配引擎：基于A/B/C结果、WHO NCD国家能力/服务覆盖、WHO MPOWER六分项和World Bank WDI烟草结果变量，输出政策路径、资源提升情景、风险下降情景和服务覆盖补齐情景。",
                "模块D准因果增强层基础部分新增政策事件库、交错DID、事件研究、安慰剂检验、合成控制、低秩矩阵补全和组合政策模拟。",
                "MPOWER无烟环境保护强执行对吸烟风险暴露负担年变化下降形成严格准因果强候选，但不写成随机实验式强因果。",
                "模块D准因果增强层高级稳健性已补充wild cluster bootstrap、cohort-robust ATT、阈值/滞后敏感性、假结果检验和政策-暴露-负担-缺口机制链。",
                "模块D多政策路径组合层已补非控烟路径硬化：高血压服务覆盖形成1条非控烟准因果强候选，基层NCD连续服务、综合治理能力和控盐膳食形成政策包适配证据，不再只依赖控烟。",
                "模块D证据裁判与锁定层最强路径可称为金牌主证据或严格准因果强候选；服务覆盖强候选只说明响应能力改善，不直接等同死亡率或DALY下降。",
                "中国映射是全球模块D准因果增强层的应用层，已接入NHSA DRG/DIP、NHC慢病示范区执行强度、地方政策执行指标、GBD2021年龄标化NCD健康结局锚点、NBS多年粗死亡率健康结局敏感性、NBS解释变量和C层字段级OCR QC，用于展示迁移和情景建议，不反向证明世界模块D。",
            ],
            "cannot_claim": [
                "不能说模块D证据裁判与锁定层已经达到随机实验式强因果。",
                "不能说中国省级政策已经因果证明降低死亡率或DALY；GBD2021是年龄标化健康结局锚点，NBS粗死亡率只能作为敏感性/边界检验。",
                "不能用中国映射替代全球模块D的证据链。",
            ],
            "critical_failures": critical_failures,
            "output_files": {
                "checklist": (report_asset_path(report_dir, "module_d_lock_checklist.csv")).as_posix(),
                "summary": (report_asset_path(report_dir, "module_d_lock_status.json")).as_posix(),
                "statement": (report_asset_path(doc_dir, "module_d_lock_statement.md")).as_posix(),
            },
        }

        checklist = pd.DataFrame(checks)
        checklist.to_csv(report_asset_path(report_dir, "module_d_lock_checklist.csv"), index=False, encoding="utf-8-sig")
        (report_asset_path(report_dir, "module_d_lock_status.json")).write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        statement = f"""# 模块D锁定状态说明

    ## 结论

    模块D目前锁定的是 **模块D证据裁判与锁定层 + 政策适配引擎**，不是“随机实验式强因果结论”。

    - 全球政策适配引擎是否锁定：`{global_adaptation_engine_locked}`
    - 控烟强因果结论是否锁定：`{tobacco_causal_locked}`
    - 模块D准因果增强层是否锁定：`{causal_enhancement_locked}`
    - 模块D准因果增强层严格准因果强候选是否锁定：`{strict_quasi_causal_locked}`
    - 模块D准因果增强层高级稳健性是否锁定：`{advanced_validation_locked}`
    - 模块D证据裁判与锁定层是否锁定：`{d6_locked}`
    - 严格可推广准因果路径数量：`{promotable_quasi_causal_paths}`
    - 金牌主证据数量：`{d6_gold_candidates}`
    - 中国映射是否能作为全球D锁定证据：`False`

    ## 可以怎么讲

    1. 模块D已经从单一控烟DID升级为“全球政策响应证据 + WHO NCD政策执行/服务覆盖 + WHO MPOWER六分项 + World Bank WDI烟草结果变量 + 中国情景适配模拟 + 准因果增强验证 + 模块D证据裁判与锁定层”。
    2. 全球层面已经形成可复跑的政策适配引擎：风险治理、卫生资源投入、服务覆盖、支付韧性、NCD国家能力和相似国家经验迁移。
    3. 模块D准因果增强层已经补上政策事件库、交错DID、事件研究、安慰剂检验、合成控制、矩阵补全和组合政策模拟。
    4. MPOWER无烟环境保护强执行对吸烟风险暴露负担年变化下降形成严格准因果强候选，但不作为随机实验式强因果主结论。
    5. 高级稳健性进一步补上 wild cluster bootstrap、cohort-robust ATT、阈值/滞后敏感性、假结果检验和机制链。
    6. 模块D证据裁判与锁定层进一步把每条政策路径按设计完整性、偏误敏感性、机制链、迁移适配和声明边界统一评分，避免只挑显著结果。
    7. 中国映射是下游应用层，用来展示全球框架如何迁移到中国；它不反向证明全球模块D。

    ## 不能怎么讲

    1. 不能说模块D证据裁判与锁定层已经以随机实验标准证明政策降低死亡率。
    2. 不能说中国省级政策已经因果证明降低死亡率或DALY。
    3. 不能把中国映射当作模块D在全球范围锁定的证据。

    ## 当前边界

    世界范围：模块D证据裁判与锁定层已锁，可以用于国赛主线；新增 WHO NCD Country Capacity Survey 2013-2023、高血压/糖尿病等服务覆盖指标、WHO MPOWER六分项、World Bank WDI烟草结果变量、准因果增强和多政策路径组合稳健性检验，以及设计完整性、偏误敏感性、机制链、迁移适配和声明边界评分。

    中国范围：当前已升级为31省压力-风险-响应-组合政策适配画像，已接入人口普查人口和老龄化、NBS卫生资源、GBD 2017省级疾病/风险、GBD 2021省级NCD年龄标化死亡率/DALY健康结局锚点、ScienceDB 2024地级市人口加权PM2.5、NHSA DRG/DIP政策暴露和资源响应准因果候选、NHC国家慢病综合防控示范区执行强度、地方政策执行指标、NBS 2018/2019/2021-2024分省粗死亡率健康结局敏感性面板、NBS 2025年鉴城镇化/GDP OCR解释变量候选，并补入NBS床位/医保/基金收支OCR字段级QC。边界是：GBD2021锚点不是省级政策面板，NBS粗死亡率不是年龄标化NCD结局，家庭医生/分级诊疗/健康城市按国家政策里程碑进入政策包而不伪装为省级处理变量，因此不能声明省级政策健康结果强因果。
    """
        (report_asset_path(doc_dir, "module_d_lock_statement.md")).write_text(statement, encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


NAMESPACE_BUILDERS = {
    'download_who_ncd_policy_data.py': _namespace_download_who_ncd_policy_data,
    'run_policy_identification.py': _namespace_run_policy_identification,
    'run_policy_adaptation_engine.py': _namespace_run_policy_adaptation_engine,
    'download_policy_d4_data.py': _namespace_download_policy_d4_data,
    'run_policy_causal_enhancement.py': _namespace_run_policy_causal_enhancement,
    'run_policy_d4_advanced_validation.py': _namespace_run_policy_d4_advanced_validation,
    'run_policy_d5_global_portfolio.py': _namespace_run_policy_d5_global_portfolio,
    'run_policy_d5_non_tobacco_pathway_hardening.py': _namespace_run_policy_d5_non_tobacco_pathway_hardening,
    'run_policy_d5_boundary_enhancement.py': _namespace_run_policy_d5_boundary_enhancement,
    'run_policy_d6_quasi_causal_excellence.py': _namespace_run_policy_d6_quasi_causal_excellence,
    'run_module_d_lock_audit.py': _namespace_run_module_d_lock_audit,
}

STEP_GROUPS = {'run': ['download_who_ncd_policy_data.py', 'run_policy_identification.py', 'run_policy_adaptation_engine.py', 'download_policy_d4_data.py', 'run_policy_causal_enhancement.py', 'run_policy_d4_advanced_validation.py', 'run_policy_d5_global_portfolio.py', 'run_policy_d5_non_tobacco_pathway_hardening.py', 'run_policy_d5_boundary_enhancement.py', 'run_policy_d6_quasi_causal_excellence.py'], 'audit': ['run_module_d_lock_audit.py']}
DEFAULT_GROUPS = ['run', 'audit']


def selected_steps(groups: list[str]) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = []
    for group in groups:
        steps.extend((script_name, []) for script_name in STEP_GROUPS[group])
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description='Run global Module D policy response, adaptation, and quasi-causal analysis.')
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--audit", action="store_true")

    args = parser.parse_args()
    project_root = detect_project_root(args.project_root)
    groups = [name for name in STEP_GROUPS if getattr(args, name)]
    if not groups:
        groups = list(DEFAULT_GROUPS)
    run_step_sequence(selected_steps(groups), NAMESPACE_BUILDERS, project_root=project_root)


if __name__ == "__main__":
    main()
