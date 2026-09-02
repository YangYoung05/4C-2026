from __future__ import annotations

import argparse
from pathlib import Path

from a_foundation import detect_project_root, report_asset_glob, report_asset_path, run_step_sequence

def _namespace_clean_health_data():
    __name__ = 'clean_health_data'
    import argparse
    import csv
    import json
    import re
    from dataclasses import dataclass, asdict
    from pathlib import Path
    from typing import Iterable

    import pandas as pd
    from foundation import detect_input_root as shared_detect_input_root
    from foundation import detect_project_root as shared_detect_project_root

    TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin1")
    CSV_EXTENSIONS = {".csv", ".txt"}
    EXCEL_EXTENSIONS = {".xlsx", ".xls"}
    SUPPORTED_EXTENSIONS = CSV_EXTENSIONS | EXCEL_EXTENSIONS | {".gz"}
    YEAR_RE = re.compile(r"^(19|20)\d{2}$")


    @dataclass
    class FileRecord:
        source_key: str
        input_file: str
        rows_in: int | None
        rows_out: int | None
        status: str
        output_file: str | None = None
        note: str | None = None


    SOURCE_CONFIG = {
        "dbd": {
            "prefix": "cleaned_dbd",
            "candidates": ["DBD_CRA47594759"],
            "fallbacks": [],
        },
        "external": {
            "prefix": "external",
            "candidates": ["External_Data28782878"],
            "fallbacks": ["External Data"],
        },
        "gbd": {
            "prefix": "cleaned_gbd",
            "candidates": ["GBDData33943394"],
            "fallbacks": ["external_data/01_GBD2021"],
        },
        "nbs": {
            "prefix": "cleaned_nbs",
            "candidates": ["NBS_HD16661666"],
            "fallbacks": [],
        },
        "wb_hnp": {
            "prefix": "cleaned_wb_hnp",
            "candidates": ["WB_HNP22122212"],
            "fallbacks": ["WB_HNP.csv"],
        },
        "wdi": {
            "prefix": "cleaned_wdi",
            "candidates": ["WDI_SE35873587"],
            "fallbacks": [],
        },
    }

    WDI_INDICATOR_RULES: dict[str, dict[str, list[str]]] = {
        "pm25": {
            "include_name_patterns": ["pm2.5"],
        },
        "diabetes_prevalence": {
            "include_name_patterns": ["diabetes prevalence"],
        },
        "ncd_mortality_30_70": {
            "include_name_patterns": ["mortality from cvd cancer diabetes or crd between exact ages 30 and 70"],
        },
        "gini": {
            "include_name_patterns": ["gini index"],
        },
        "gdp_per_capita": {
            "include_name_patterns": ["gdp per capita"],
        },
        "health_expenditure_pct_gdp": {
            "include_codes": ["SH.XPD.CHEX.GD.ZS"],
            "include_name_patterns": ["current health expenditure (% of gdp)", "current health expenditure gdp"],
            "exclude_name_patterns": ["per capita", "current us$", "ppp"],
        },
        "out_of_pocket_pct": {
            "include_codes": ["SH.XPD.OOPC.CH.ZS"],
            "include_name_patterns": ["out-of-pocket expenditure (% of current health expenditure)"],
            "exclude_codes": ["SH.XPD.OOPC.PC.CD", "SH.XPD.OOPC.PP.CD"],
            "exclude_name_patterns": ["per capita", "current us$", "ppp"],
        },
        "physicians": {
            "include_name_patterns": ["physicians per 1 000 people", "physicians"],
        },
        "nurses_midwives": {
            "include_name_patterns": ["nurses and midwives"],
        },
        "hospital_beds": {
            "include_name_patterns": ["hospital beds"],
        },
        "life_expectancy": {
            "include_name_patterns": ["life expectancy at birth"],
        },
        "under5_mortality": {
            "include_name_patterns": ["mortality rate, under-5", "under-5 mortality rate"],
        },
        "infant_mortality": {
            "include_name_patterns": ["mortality rate, infant"],
        },
        "fertility": {
            "include_name_patterns": ["fertility rate, total"],
        },
        "adolescent_fertility": {
            "include_name_patterns": ["adolescent fertility rate"],
        },
        "population_total": {
            "include_name_patterns": ["population, total"],
        },
        "population_65_plus_pct": {
            "include_name_patterns": ["population ages 65 and above (% of total population)"],
        },
        "urban_population_pct": {
            "include_name_patterns": ["urban population (% of total population)"],
        },
        "health_expenditure_per_capita": {
            "include_codes": ["SH.XPD.CHEX.PC.CD", "SH.XPD.CHEX.PP.CD"],
            "include_name_patterns": ["current health expenditure per capita"],
        },
        "government_health_expenditure_pct": {
            "include_codes": ["SH.XPD.GHED.CH.ZS"],
            "include_name_patterns": ["domestic general government health expenditure (% of current health expenditure)"],
            "exclude_codes": ["SH.XPD.GHED.PC.CD", "SH.XPD.GHED.PP.CD"],
            "exclude_name_patterns": ["per capita", "current us$", "ppp"],
        },
        "external_health_expenditure_pct": {
            "include_codes": ["SH.XPD.EHEX.CH.ZS"],
            "include_name_patterns": ["external health expenditure (% of current health expenditure)"],
            "exclude_codes": ["SH.XPD.EHEX.PC.CD", "SH.XPD.EHEX.PP.CD"],
            "exclude_name_patterns": ["per capita", "current us$", "ppp"],
        },
    }

    WB_HNP_INDICATOR_PATTERNS = {
        "hci": ["human capital index"],
        "life_expectancy": ["life expectancy"],
        "under5_mortality": ["under-5 mortality", "mortality rate, under-5"],
        "fertility": ["fertility rate"],
        "adolescent_fertility": ["adolescent fertility rate"],
        "birth_rate": ["birth rate, crude"],
        "death_rate": ["death rate, crude"],
    }

    WB_HNP_INDICATOR_PATTERNS.update({
        "infant_mortality": ["mortality rate, infant"],
        "neonatal_mortality": ["mortality rate, neonatal"],
        "maternal_mortality": ["maternal mortality ratio"],
        "stunting": ["stunting"],
        "wasting": ["wasting"],
        "low_birthweight": ["low birthweight", "low-birthweight"],
        "anemia_women": ["anemia prevalence among women"],
        "immunization_dpt": ["immunization, dpt"],
        "immunization_measles": ["immunization, measles"],
        "immunization_hepb3": ["immunization, hepb3"],
        "skilled_birth_attendance": ["births attended by skilled health staff"],
        "contraceptive_prevalence": ["contraceptive prevalence"],
    })

    PERCENT_GROUPS = {
        "health_expenditure_pct_gdp",
        "out_of_pocket_pct",
        "diabetes_prevalence",
        "population_65_plus_pct",
        "urban_population_pct",
        "government_health_expenditure_pct",
        "external_health_expenditure_pct",
        "stunting",
        "wasting",
        "low_birthweight",
        "anemia_women",
        "immunization_dpt",
        "immunization_measles",
        "immunization_hepb3",
        "skilled_birth_attendance",
        "contraceptive_prevalence",
    }


    def detect_project_root() -> Path:
        return shared_detect_project_root()


    def detect_input_root(project_root: Path) -> Path:
        del project_root
        return shared_detect_input_root()


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        dirs = {
            "clean": project_root / "09_data_clean",
            "reports": project_root / "06_report_assets",
            "figures": project_root / "05_figures",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def normalize_token(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text


    def normalize_columns(columns: Iterable[str]) -> list[str]:
        seen: dict[str, int] = {}
        normalized: list[str] = []
        for col in columns:
            base = normalize_token(str(col)) or "col"
            count = seen.get(base, 0)
            seen[base] = count + 1
            normalized.append(base if count == 0 else f"{base}_{count}")
        return normalized


    def is_junk_path(path: Path) -> bool:
        parts = path.parts
        if any(part == "__MACOSX" for part in parts):
            return True
        return any(part.startswith("._") or part == ".DS_Store" for part in parts)


    def coerce_year_column(df: pd.DataFrame, fallback_year: int | None = None) -> pd.DataFrame:
        year_candidates = [c for c in df.columns if c in {"year", "time_period", "timedim", "time", "year_id"}]
        if year_candidates:
            year_col = year_candidates[0]
            df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
            if year_col != "year":
                df = df.rename(columns={year_col: "year"})
        elif fallback_year is not None:
            df["year"] = fallback_year
        return df


    def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin({"", "nan", "None", "NULL"}), col] = pd.NA
        return df


    def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        return df


    def convert_numeric_candidates(df: pd.DataFrame) -> pd.DataFrame:
        numeric_like = []
        for col in df.columns:
            if any(token in col for token in ("value", "val", "rate", "pct", "percent", "population", "density", "age", "rank", "number", "count", "upper", "lower", "obs_value")):
                numeric_like.append(col)
        for col in numeric_like:
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                original_non_null = df[col].notna().sum()
                converted_non_null = converted.notna().sum()
                if original_non_null and converted_non_null / original_non_null >= 0.8:
                    df[col] = converted
        return df


    def clean_frame(df: pd.DataFrame, fallback_year: int | None = None) -> pd.DataFrame:
        df = df.copy()
        df.columns = normalize_columns(df.columns)
        df = drop_empty(df)
        df = trim_strings(df)
        df = coerce_year_column(df, fallback_year=fallback_year)
        df = convert_numeric_candidates(df)
        return df


    def read_tabular(path: Path, **kwargs) -> pd.DataFrame:
        ext = path.suffix.lower()
        if ext in EXCEL_EXTENSIONS:
            return pd.read_excel(path, **kwargs)
        if ext == ".gz":
            return pd.read_csv(path, compression="gzip", low_memory=False, **kwargs)
        last_error: Exception | None = None
        for encoding in TEXT_ENCODINGS:
            try:
                return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is None:
            raise RuntimeError(f"Unsupported file type: {path}")
        raise last_error


    def write_manifest(records: list[FileRecord], out_path: Path) -> None:
        pd.DataFrame([asdict(r) for r in records]).to_csv(out_path, index=False, encoding="utf-8-sig")


    def write_text(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")


    def append_csv(df: pd.DataFrame, out_path: Path) -> None:
        compression = "gzip" if out_path.suffix.lower() == ".gz" else None
        df.to_csv(
            out_path,
            mode="a",
            header=not out_path.exists(),
            index=False,
            encoding="utf-8-sig",
            compression=compression,
        )


    def iter_files(root: Path) -> list[Path]:
        files: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if is_junk_path(path):
                continue
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue
            files.append(path)
        return sorted(files)


    def resolve_source_path(source_key: str, input_root: Path) -> Path | None:
        config = SOURCE_CONFIG[source_key]
        for candidate in config["candidates"]:
            path = input_root / candidate
            if path.exists():
                return path
        for fallback in config["fallbacks"]:
            path = input_root / fallback
            if path.exists():
                return path
        return None


    def extract_year_from_name(path: Path) -> int | None:
        stem = path.stem
        if YEAR_RE.match(stem):
            return int(stem)
        match = re.search(r"(19|20)\d{2}", path.name)
        return int(match.group()) if match else None


    def sanitize_output_name(prefix: str, path: Path) -> str:
        stem = normalize_token(path.stem)
        return f"{prefix}_{stem}.csv"


    def compile_pattern_mask(series: pd.Series, patterns: dict[str, list[str]]) -> tuple[pd.Series, pd.Series]:
        key_series = series.fillna("").astype(str).str.lower()
        matched = pd.Series(False, index=series.index)
        labels = pd.Series(pd.NA, index=series.index, dtype="object")
        for label, items in patterns.items():
            mask = pd.Series(False, index=series.index)
            for item in items:
                mask = mask | key_series.str.contains(item, regex=False)
            labels.loc[mask & labels.isna()] = label
            matched = matched | mask
        return matched, labels


    def compile_indicator_rule_mask(
        indicator_names: pd.Series,
        indicator_codes: pd.Series,
        rules: dict[str, dict[str, list[str]]],
    ) -> tuple[pd.Series, pd.Series]:
        name_series = indicator_names.fillna("").astype(str).str.lower()
        code_series = indicator_codes.fillna("").astype(str).str.upper()
        matched = pd.Series(False, index=indicator_names.index)
        labels = pd.Series(pd.NA, index=indicator_names.index, dtype="object")

        for label, rule in rules.items():
            include_mask = pd.Series(False, index=indicator_names.index)
            for pattern in rule.get("include_name_patterns", []):
                include_mask = include_mask | name_series.str.contains(pattern, regex=False)
            for code in rule.get("include_codes", []):
                include_mask = include_mask | code_series.eq(code.upper())

            exclude_mask = pd.Series(False, index=indicator_names.index)
            for pattern in rule.get("exclude_name_patterns", []):
                exclude_mask = exclude_mask | name_series.str.contains(pattern, regex=False)
            for code in rule.get("exclude_codes", []):
                exclude_mask = exclude_mask | code_series.eq(code.upper())

            final_mask = include_mask & ~exclude_mask
            labels.loc[final_mask & labels.isna()] = label
            matched = matched | final_mask
        return matched, labels


    def build_wdi_group_assignment_audit(cleaned_wdi_path: Path, audit_path: Path) -> None:
        if not cleaned_wdi_path.exists():
            return
        df = pd.read_csv(cleaned_wdi_path, encoding="utf-8-sig", low_memory=False)
        required = {"selected_indicator_group", "indicator_code", "indicator_name", "value"}
        if not required.issubset(df.columns):
            return
        df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
        audit_df = (
            df.groupby(["selected_indicator_group", "indicator_code", "indicator_name"], dropna=False)
            .agg(
                rows=("value", "size"),
                non_null_values=("value_num", lambda values: int(values.notna().sum())),
                value_min=("value_num", "min"),
                value_p95=("value_num", lambda values: values.quantile(0.95) if values.notna().any() else pd.NA),
                value_max=("value_num", "max"),
                gt_100_rate=("value_num", lambda values: float((values > 100).mean()) if values.notna().any() else pd.NA),
            )
            .reset_index()
            .sort_values(["selected_indicator_group", "indicator_code", "indicator_name"], kind="stable")
        )
        audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")


    def build_cleaning_raw_qa(clean_dir: Path, audit_path: Path) -> None:
        file_specs = [
            ("gbd", clean_dir / "cleaned_gbd_panel.csv"),
            ("dbd", clean_dir / "cleaned_dbd_panel.csv"),
            ("wdi", clean_dir / "cleaned_wdi.csv"),
            ("wb_hnp", clean_dir / "cleaned_wb_hnp.csv"),
            ("nbs", clean_dir / "cleaned_nbs_health.csv"),
        ]
        rows: list[dict[str, object]] = []
        for dataset_name, path in file_specs:
            if not path.exists():
                continue
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "check_type": "duplicate_full_row",
                    "column_name": "",
                    "detail": f"duplicate_rows={int(df.duplicated().sum())}",
                    "path": path.as_posix(),
                }
            )
            if {"iso3", "year"}.issubset(df.columns):
                key_dup = int(df.duplicated(["iso3", "year"]).sum())
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "check_type": "duplicate_iso3_year",
                        "column_name": "iso3,year",
                        "detail": f"duplicate_pairs={key_dup}",
                        "path": path.as_posix(),
                    }
                )
            if {"country_code", "year", "indicator_code"}.issubset(df.columns):
                key_dup = int(df.duplicated(["country_code", "year", "indicator_code"]).sum())
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "check_type": "duplicate_country_year_indicator",
                        "column_name": "country_code,year,indicator_code",
                        "detail": f"duplicate_pairs={key_dup}",
                        "path": path.as_posix(),
                    }
                )
            if {"value", "selected_indicator_group"}.issubset(df.columns):
                numeric = pd.to_numeric(df["value"], errors="coerce")
                if numeric.notna().any():
                    negative_rate = float((numeric < 0).mean())
                    rows.append(
                        {
                            "dataset_name": dataset_name,
                            "check_type": "negative_value",
                            "column_name": "value",
                            "detail": f"negative_rate={negative_rate:.4%}, non_null={int(numeric.notna().sum())}",
                            "path": path.as_posix(),
                        }
                    )
                for group_name, subset in df.groupby("selected_indicator_group", dropna=False):
                    if str(group_name) not in PERCENT_GROUPS:
                        continue
                    group_numeric = pd.to_numeric(subset["value"], errors="coerce")
                    if not group_numeric.notna().any():
                        continue
                    out_of_range_rate = float(((group_numeric < 0) | (group_numeric > 100)).mean())
                    rows.append(
                        {
                            "dataset_name": dataset_name,
                            "check_type": "percent_range",
                            "column_name": str(group_name),
                            "detail": f"out_of_range_rate={out_of_range_rate:.4%}, non_null={int(group_numeric.notna().sum())}",
                            "path": path.as_posix(),
                        }
                    )
            for column in df.columns:
                if column in {"year", "iso3"}:
                    continue
                if "pct" not in column and "percent" not in column:
                    continue
                numeric = pd.to_numeric(df[column], errors="coerce")
                if not numeric.notna().any():
                    continue
                out_of_range_rate = float(((numeric < 0) | (numeric > 100)).mean())
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "check_type": "percent_range",
                        "column_name": column,
                        "detail": f"out_of_range_rate={out_of_range_rate:.4%}, non_null={int(numeric.notna().sum())}",
                        "path": path.as_posix(),
                    }
                )
        pd.DataFrame(rows).to_csv(audit_path, index=False, encoding="utf-8-sig")


    def clean_external(source_root: Path, clean_dir: Path, report_dir: Path, inventory_only: bool) -> list[FileRecord]:
        records: list[FileRecord] = []
        files = iter_files(source_root)
        expected_csv = [p for p in files if p.suffix.lower() in CSV_EXTENSIONS]
        for path in expected_csv:
            output_path = clean_dir / sanitize_output_name("external", path)
            try:
                df = read_tabular(path)
                cleaned = clean_frame(df)
                if not inventory_only:
                    cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
                records.append(
                    FileRecord(
                        "external",
                        path.as_posix(),
                        len(df),
                        len(cleaned),
                        "success",
                        None if inventory_only else output_path.as_posix(),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                records.append(FileRecord("external", path.as_posix(), None, None, "failed", None, str(exc)))
        summary = [
            f"source_root={source_root}",
            f"files_scanned={len(files)}",
            f"csv_processed={len(expected_csv)}",
            f"success={sum(r.status == 'success' for r in records)}",
            f"failed={sum(r.status == 'failed' for r in records)}",
            f"inventory_only={inventory_only}",
        ]
        write_text(report_asset_path(report_dir, "external_source_summary.txt"), "\n".join(summary) + "\n")
        write_manifest(records, report_asset_path(report_dir, "external_manifest.csv"))
        return records


    def clean_annual_directory(source_key: str, source_root: Path, clean_dir: Path, report_dir: Path, inventory_only: bool) -> list[FileRecord]:
        prefix = SOURCE_CONFIG[source_key]["prefix"]
        records: list[FileRecord] = []
        frames: list[pd.DataFrame] = []
        year_files = [p for p in iter_files(source_root) if p.suffix.lower() in CSV_EXTENSIONS and extract_year_from_name(p) is not None]
        for path in year_files:
            year = extract_year_from_name(path)
            try:
                df = read_tabular(path)
                cleaned = clean_frame(df, fallback_year=year)
                cleaned["source_file"] = path.name
                cleaned["source_group"] = source_key
                frames.append(cleaned)
                records.append(FileRecord(source_key, path.as_posix(), len(df), len(cleaned), "success"))
            except Exception as exc:  # noqa: BLE001
                records.append(FileRecord(source_key, path.as_posix(), None, None, "failed", None, str(exc)))
        merged_path = clean_dir / f"{prefix}_panel.csv"
        if frames and not inventory_only:
            merged = pd.concat(frames, ignore_index=True, sort=False)
            merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
            for record in records:
                if record.status == "success":
                    record.output_file = merged_path.as_posix()
        write_manifest(records, report_asset_path(report_dir, f"{source_key}_manifest.csv"))
        write_text(
            report_asset_path(report_dir, f"{source_key}_source_summary.txt"),
            "\n".join(
                [
                    f"source_root={source_root}",
                    f"year_files={len(year_files)}",
                    f"success={sum(r.status == 'success' for r in records)}",
                    f"failed={sum(r.status == 'failed' for r in records)}",
                    f"merged_output={merged_path.as_posix() if frames and not inventory_only else ''}",
                ]
            )
            + "\n",
        )
        return records


    def clean_nbs(source_root: Path, clean_dir: Path, report_dir: Path, inventory_only: bool) -> list[FileRecord]:
        records: list[FileRecord] = []
        files = [p for p in iter_files(source_root) if p.suffix.lower() in CSV_EXTENSIONS]
        frames: list[pd.DataFrame] = []
        for path in files:
            try:
                df = read_tabular(path)
                cleaned = clean_frame(df)
                if "指标" not in cleaned.columns:
                    cleaned["指标"] = path.stem
                else:
                    cleaned["指标"] = cleaned["指标"].fillna(path.stem)
                cleaned["source_dataset"] = path.stem
                cleaned["source_group"] = "nbs"
                frames.append(cleaned)
                records.append(FileRecord("nbs", path.as_posix(), len(df), len(cleaned), "success"))
            except Exception as exc:  # noqa: BLE001
                records.append(FileRecord("nbs", path.as_posix(), None, None, "failed", None, str(exc)))
        output_path = clean_dir / "cleaned_nbs_health.csv"
        if frames and not inventory_only:
            pd.concat(frames, ignore_index=True, sort=False).to_csv(output_path, index=False, encoding="utf-8-sig")
            for record in records:
                if record.status == "success":
                    record.output_file = output_path.as_posix()
        write_manifest(records, report_asset_path(report_dir, "nbs_manifest.csv"))
        return records


    def clean_wb_hnp(source_root: Path, clean_dir: Path, report_dir: Path, inventory_only: bool, start_year: int, end_year: int) -> list[FileRecord]:
        records: list[FileRecord] = []
        if source_root.is_file():
            wb_file = source_root
            glossary = None
        else:
            candidates = {p.name: p for p in iter_files(source_root)}
            wb_file = candidates.get("WB_HNP.csv")
            glossary = next((p for p in candidates.values() if "glossary" in p.name.lower()), None)
        if wb_file is None:
            return [FileRecord("wb_hnp", source_root.as_posix(), None, None, "failed", None, "WB_HNP.csv not found")]

        selected = [
            "REF_AREA",
            "REF_AREA_LABEL",
            "INDICATOR",
            "INDICATOR_LABEL",
            "SEX",
            "SEX_LABEL",
            "AGE",
            "AGE_LABEL",
            "URBANISATION",
            "URBANISATION_LABEL",
            "UNIT_MEASURE",
            "UNIT_MEASURE_LABEL",
            "FREQ",
            "FREQ_LABEL",
            "TIME_PERIOD",
            "OBS_VALUE",
            "OBS_STATUS",
            "OBS_STATUS_LABEL",
        ]
        output_path = clean_dir / "cleaned_wb_hnp.csv"
        chunk_records = 0
        rows_out = 0
        missing_value_rows_dropped = 0
        non_observed_rows_retained = 0
        try:
            if not inventory_only and output_path.exists():
                output_path.unlink()
            for chunk in pd.read_csv(
                wb_file,
                usecols=lambda c: c in selected or c.lower() in {s.lower() for s in selected},
                chunksize=250000,
                encoding="utf-8-sig",
                low_memory=False,
            ):
                chunk_records += len(chunk)
                cleaned = clean_frame(chunk)
                if "time_period" in cleaned.columns:
                    cleaned["time_period"] = pd.to_numeric(cleaned["time_period"], errors="coerce").astype("Int64")
                    cleaned = cleaned[cleaned["time_period"].between(start_year, end_year, inclusive="both")]
                if "freq" in cleaned.columns:
                    cleaned = cleaned[cleaned["freq"].fillna("A") == "A"]
                if "obs_status_label" in cleaned.columns:
                    status_text = cleaned["obs_status_label"].fillna("").astype(str).str.lower()
                    missing_mask = status_text.str.contains("missing value", regex=False)
                    missing_value_rows_dropped += int(missing_mask.sum())
                    non_observed_rows_retained += int(
                        status_text.str.contains("estimate|interpol|forecast|model|imput", regex=True).sum()
                    )
                    cleaned = cleaned.loc[~missing_mask].copy()
                if cleaned.empty:
                    continue
                indicator_source = cleaned.get("indicator_label", pd.Series(pd.NA, index=cleaned.index)).fillna("").astype(str)
                if "indicator" in cleaned.columns:
                    indicator_source = indicator_source + " " + cleaned["indicator"].fillna("").astype(str)
                matched, labels = compile_pattern_mask(indicator_source, WB_HNP_INDICATOR_PATTERNS)
                cleaned = cleaned.loc[matched].copy()
                if cleaned.empty:
                    continue
                cleaned["selected_indicator_group"] = labels.loc[cleaned.index].values
                rows_out += len(cleaned)
                if not inventory_only:
                    append_csv(cleaned, output_path)
            records.append(
                FileRecord(
                    "wb_hnp",
                    wb_file.as_posix(),
                    chunk_records,
                    rows_out,
                    "success",
                    None if inventory_only else output_path.as_posix(),
                    note=f"missing_value_rows_dropped={missing_value_rows_dropped}; non_observed_rows_retained={non_observed_rows_retained}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            records.append(FileRecord("wb_hnp", wb_file.as_posix(), None, None, "failed", None, str(exc)))

        if glossary is not None:
            glossary_path = clean_dir / "cleaned_wb_hnp_glossary.csv"
            try:
                df = read_tabular(glossary)
                cleaned = clean_frame(df)
                if not inventory_only:
                    cleaned.to_csv(glossary_path, index=False, encoding="utf-8-sig")
                records.append(FileRecord("wb_hnp", glossary.as_posix(), len(df), len(cleaned), "success", None if inventory_only else glossary_path.as_posix()))
            except Exception as exc:  # noqa: BLE001
                records.append(FileRecord("wb_hnp", glossary.as_posix(), None, None, "failed", None, str(exc)))
        write_manifest(records, report_asset_path(report_dir, "wb_hnp_manifest.csv"))
        return records


    def clean_wdi(source_root: Path, clean_dir: Path, report_dir: Path, inventory_only: bool, start_year: int, end_year: int) -> list[FileRecord]:
        records: list[FileRecord] = []
        candidates = {p.name: p for p in iter_files(source_root)}
        data_file = candidates.get("WDICSV.csv")
        country_file = candidates.get("WDICountry.csv")
        series_file = candidates.get("WDISeries.csv")

        if data_file is None:
            return [FileRecord("wdi", source_root.as_posix(), None, None, "failed", None, "WDICSV.csv not found")]

        year_cols = [str(year) for year in range(start_year, end_year + 1)]
        long_output = clean_dir / "cleaned_wdi.csv"
        total_in = 0
        total_out = 0
        try:
            if not inventory_only and long_output.exists():
                long_output.unlink()
            for chunk in pd.read_csv(data_file, chunksize=20000, encoding="utf-8-sig", low_memory=False):
                total_in += len(chunk)
                base_cols = [c for c in ["Country Name", "Country Code", "Indicator Name", "Indicator Code"] if c in chunk.columns]
                usable_years = [c for c in year_cols if c in chunk.columns]
                if not base_cols or not usable_years:
                    continue
                indicator_names = chunk.get("Indicator Name", pd.Series(pd.NA, index=chunk.index))
                indicator_codes = chunk.get("Indicator Code", pd.Series(pd.NA, index=chunk.index))
                matched, labels = compile_indicator_rule_mask(indicator_names, indicator_codes, WDI_INDICATOR_RULES)

                def prepare_long(selected_chunk: pd.DataFrame, selected_labels: pd.Series) -> pd.DataFrame:
                    if selected_chunk.empty:
                        return pd.DataFrame()
                    cleaned = selected_chunk[base_cols + usable_years].copy()
                    cleaned = cleaned.melt(id_vars=base_cols, value_vars=usable_years, var_name="year", value_name="value")
                    cleaned.columns = normalize_columns(cleaned.columns)
                    cleaned = clean_frame(cleaned)
                    cleaned = cleaned.dropna(subset=["value"], how="all")
                    if cleaned.empty:
                        return cleaned
                    label_map = pd.DataFrame(
                        {
                            "country_code": selected_chunk["Country Code"].astype(str).values.repeat(len(usable_years)),
                            "indicator_code": selected_chunk["Indicator Code"].astype(str).values.repeat(len(usable_years)),
                            "selected_indicator_group": selected_labels.astype(str).values.repeat(len(usable_years)),
                            "year": list(usable_years) * len(selected_chunk),
                        }
                    )
                    label_map.columns = normalize_columns(label_map.columns)
                    label_map["year"] = pd.to_numeric(label_map["year"], errors="coerce").astype("Int64")
                    return cleaned.merge(label_map, on=["country_code", "indicator_code", "year"], how="left")

                cleaned_chunk = prepare_long(chunk.loc[matched].copy(), labels.loc[matched])
                if not cleaned_chunk.empty:
                    total_out += len(cleaned_chunk)
                    if not inventory_only:
                        append_csv(cleaned_chunk, long_output)
            if not inventory_only:
                build_wdi_group_assignment_audit(long_output, report_asset_path(report_dir, "wdi_group_assignment_audit.csv"))
            records.append(FileRecord("wdi", data_file.as_posix(), total_in, total_out, "success", None if inventory_only else long_output.as_posix()))
        except Exception as exc:  # noqa: BLE001
            records.append(FileRecord("wdi", data_file.as_posix(), None, None, "failed", None, str(exc)))

        for meta_file, output_name in ((country_file, "cleaned_wdi_country.csv"), (series_file, "cleaned_wdi_series.csv")):
            if meta_file is None:
                continue
            try:
                df = read_tabular(meta_file)
                cleaned = clean_frame(df)
                output_path = clean_dir / output_name
                if not inventory_only:
                    cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
                records.append(FileRecord("wdi", meta_file.as_posix(), len(df), len(cleaned), "success", None if inventory_only else output_path.as_posix()))
            except Exception as exc:  # noqa: BLE001
                records.append(FileRecord("wdi", meta_file.as_posix(), None, None, "failed", None, str(exc)))
        write_manifest(records, report_asset_path(report_dir, "wdi_manifest.csv"))
        return records


    def run_source(source_key: str, source_root: Path, clean_dir: Path, report_dir: Path, inventory_only: bool, start_year: int, end_year: int) -> list[FileRecord]:
        if source_key == "external":
            return clean_external(source_root, clean_dir, report_dir, inventory_only)
        if source_key in {"dbd", "gbd"}:
            return clean_annual_directory(source_key, source_root, clean_dir, report_dir, inventory_only)
        if source_key == "nbs":
            return clean_nbs(source_root, clean_dir, report_dir, inventory_only)
        if source_key == "wb_hnp":
            return clean_wb_hnp(source_root, clean_dir, report_dir, inventory_only, start_year, end_year)
        if source_key == "wdi":
            return clean_wdi(source_root, clean_dir, report_dir, inventory_only, start_year, end_year)
        raise ValueError(f"Unsupported source_key: {source_key}")


    def cleanup_previous_outputs(source_key: str, clean_dir: Path, report_dir: Path) -> None:
        prefix = SOURCE_CONFIG[source_key]["prefix"]
        for directory in (clean_dir, report_dir):
            for path in directory.glob(f"{prefix}*"):
                if path.is_file():
                    path.unlink()
        for path in report_asset_glob(report_dir, f"{source_key}_*"):
            if path.is_file():
                path.unlink()


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Clean six health-data source groups into deterministic outputs.")
        parser.add_argument("--source", choices=["all", "dbd", "external", "gbd", "nbs", "wb_hnp", "wdi"], default="all")
        parser.add_argument("--input-root", type=Path, default=None)
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--inventory-only", action="store_true")
        parser.add_argument("--fresh", action="store_true")
        parser.add_argument("--start-year", type=int, default=2000)
        parser.add_argument("--end-year", type=int, default=2024)
        return parser.parse_args()


    def main() -> None:
        args = parse_args()
        project_root = args.project_root or detect_project_root()
        input_root = args.input_root or detect_input_root(project_root)
        dirs = ensure_dirs(project_root)
        source_keys = list(SOURCE_CONFIG) if args.source == "all" else [args.source]

        run_summary: dict[str, object] = {
            "project_root": project_root.as_posix(),
            "input_root": input_root.as_posix(),
            "source_keys": source_keys,
            "inventory_only": args.inventory_only,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "sources": {},
        }

        for source_key in source_keys:
            source_root = resolve_source_path(source_key, input_root)
            if source_root is None:
                run_summary["sources"][source_key] = {
                    "status": "missing",
                    "message": "source directory not found",
                }
                continue
            if args.fresh:
                cleanup_previous_outputs(source_key, dirs["clean"], dirs["reports"])
            records = run_source(source_key, source_root, dirs["clean"], dirs["reports"], args.inventory_only, args.start_year, args.end_year)
            audit_files: list[str] = []
            if source_key == "wdi":
                wdi_audit = report_asset_path(dirs["reports"], "wdi_group_assignment_audit.csv")
                if wdi_audit.exists():
                    audit_files.append(wdi_audit.as_posix())
            run_summary["sources"][source_key] = {
                "source_root": source_root.as_posix(),
                "success": sum(r.status == "success" for r in records),
                "failed": sum(r.status == "failed" for r in records),
                "outputs": sorted({r.output_file for r in records if r.output_file}),
                "audit_files": audit_files,
            }

        raw_qa_path = report_asset_path(dirs["reports"], "cleaning_raw_qa_report.csv")
        build_cleaning_raw_qa(dirs["clean"], raw_qa_path)
        for source_info in run_summary["sources"].values():
            if isinstance(source_info, dict) and source_info.get("status") != "missing":
                source_info.setdefault("audit_files", [])
                if raw_qa_path.as_posix() not in source_info["audit_files"]:
                    source_info["audit_files"].append(raw_qa_path.as_posix())

        summary_path = report_asset_path(dirs["reports"], "cleaning_run_summary.json")
        summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(run_summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_clean_numeric_like_columns():
    __name__ = 'clean_numeric_like_columns'
    import argparse
    import json
    import re
    from dataclasses import asdict, dataclass
    from pathlib import Path
    from typing import Any

    import pandas as pd
    from foundation import detect_project_root as shared_detect_project_root

    TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin1")
    CSV_EXTENSIONS = {".csv", ".txt"}
    EXCEL_EXTENSIONS = {".xlsx", ".xls"}
    SUPPORTED_EXTENSIONS = CSV_EXTENSIONS | EXCEL_EXTENSIONS | {".gz"}
    DEFAULT_TOKENS = [
        "val",
        "rate",
        "pct",
        "percent",
        "population",
        "density",
        "rank",
        "number",
        "count",
        "upper",
        "lower",
        "obs_value",
        "value",
        "amount",
        "share",
        "ratio",
        "index",
        "score",
        "mean",
        "median",
        "total",
        "expenditure",
        "mortality",
        "prevalence",
        "incidence",
        "coverage",
    ]
    DEFAULT_EXCLUDED_COLUMNS = {
        "country",
        "country_name",
        "country_code",
        "iso2",
        "iso3",
        "location",
        "location_name",
        "ref_area",
        "region",
        "sex",
        "sex_name",
        "source_file",
        "source_group",
    }
    AGE_GUARD_COLUMNS = {"age_group", "age_label", "age_name"}
    IDENTIFIER_PREFIX_GUARDS = ("country_", "location_", "region_")
    IDENTIFIER_SUFFIX_GUARDS = ("_code",)
    MISSING_MARKERS = {"", "nan", "none", "null", "na", "n/a", "n.a.", "-", "--"}
    WHITESPACE_RE = re.compile(r"[\s\u00A0\u200B\u200C\u200D\uFEFF\u3000]+")
    FOOTNOTE_RE = re.compile(r"[*#†‡^~]")
    NOTE_PAREN_RE = re.compile(r"[（(][^0-9eE+\-\.]*[)）]")
    INVALID_NUMERIC_RE = re.compile(r"[^0-9eE+\-\.]")
    NEGATIVE_PAREN_RE = re.compile(r"^\(([-+0-9eE\.,%\s'，]+)\)$")


    @dataclass
    class ColumnRecord:
        file_path: str
        sheet_name: str | None
        output_path: str
        column_original: str
        column_normalized: str
        matched_tokens: str
        matched_by: str | None
        original_dtype: str
        original_non_null: int
        converted_non_null: int
        success_rate: float | None
        modified_cells: int
        risk_level: str
        action_taken: str
        reason: str
        age_token_risk: bool
        rows_in: int
        rows_out: int
        cols_in: int
        cols_out: int
        in_place: bool
        dry_run: bool


    def detect_project_root() -> Path:
        return shared_detect_project_root()


    def normalize_token(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text


    def ensure_report_dir(project_root: Path, report_dir: Path | None) -> Path:
        resolved = report_dir or (project_root / "06_report_assets")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


    def is_junk_path(path: Path) -> bool:
        if any(part == "__MACOSX" for part in path.parts):
            return True
        return any(part == ".DS_Store" or part.startswith("._") for part in path.parts)


    def is_supported_file(path: Path) -> bool:
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return True
        return path.name.lower().endswith(".csv.gz")


    def iter_target_files(target: Path) -> list[Path]:
        if target.is_file():
            if is_junk_path(target) or not is_supported_file(target):
                return []
            return [target]

        files: list[Path] = []
        for path in target.rglob("*"):
            if not path.is_file():
                continue
            if is_junk_path(path) or not is_supported_file(path):
                continue
            files.append(path)
        return sorted(files)


    def read_csv_like(path: Path, **kwargs: Any) -> pd.DataFrame:
        if path.suffix.lower() == ".gz" or path.name.lower().endswith(".csv.gz"):
            return pd.read_csv(path, compression="gzip", low_memory=False, **kwargs)
        last_error: Exception | None = None
        for encoding in TEXT_ENCODINGS:
            try:
                return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is None:
            raise RuntimeError(f"Failed to read file: {path}")
        raise last_error


    def read_file(path: Path) -> dict[str | None, pd.DataFrame]:
        ext = path.suffix.lower()
        if ext in EXCEL_EXTENSIONS:
            return pd.read_excel(path, sheet_name=None)
        return {None: read_csv_like(path)}


    def write_file(path: Path, data: dict[str | None, pd.DataFrame]) -> None:
        ext = path.suffix.lower()
        if ext in EXCEL_EXTENSIONS:
            with pd.ExcelWriter(path) as writer:
                for sheet_name, df in data.items():
                    safe_name = sheet_name or "Sheet1"
                    df.to_excel(writer, sheet_name=safe_name[:31], index=False)
            return

        df = data[None]
        if path.suffix.lower() == ".gz" or path.name.lower().endswith(".csv.gz"):
            df.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
        else:
            df.to_csv(path, index=False, encoding="utf-8-sig")


    def build_output_path(path: Path, in_place: bool, suffix: str) -> Path:
        if in_place:
            return path
        if path.name.lower().endswith(".csv.gz"):
            base = path.name[:-7]
            return path.with_name(f"{base}{suffix}.csv.gz")
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")


    def clean_numeric_text(value: Any) -> tuple[str | None, bool]:
        if pd.isna(value):
            return None, False

        original = str(value)
        stripped = original.strip()
        compact_lower = WHITESPACE_RE.sub("", stripped).lower()
        if compact_lower in MISSING_MARKERS:
            return None, compact_lower != original.lower()

        text = stripped
        negative_match = NEGATIVE_PAREN_RE.match(text)
        if negative_match:
            text = f"-{negative_match.group(1)}"

        text = WHITESPACE_RE.sub("", text)
        text = text.replace("'", "").replace(",", "").replace("，", "").replace("%", "")
        text = NOTE_PAREN_RE.sub("", text)
        text = FOOTNOTE_RE.sub("", text)
        text = INVALID_NUMERIC_RE.sub("", text)

        if text.lower() in MISSING_MARKERS or text in {"", ".", "+", "-", "--"}:
            return None, text != original

        return text, text != original


    def clean_numeric_series(series: pd.Series) -> tuple[pd.Series, int, int, int, float | None]:
        cleaned_texts: list[str | None] = []
        modified_cells = 0
        original_non_null = int(series.notna().sum())

        for value in series.tolist():
            cleaned, changed = clean_numeric_text(value)
            cleaned_texts.append(cleaned)
            if changed and not pd.isna(value):
                modified_cells += 1

        cleaned_series = pd.Series(cleaned_texts, index=series.index, dtype="object")
        numeric = pd.to_numeric(cleaned_series, errors="coerce").astype("float64")
        converted_non_null = int(numeric.notna().sum())
        success_rate = (converted_non_null / original_non_null) if original_non_null else None
        return numeric, modified_cells, original_non_null, converted_non_null, success_rate


    def match_tokens(column_name: str, tokens: list[str]) -> tuple[list[str], str | None]:
        normalized = normalize_token(column_name)
        exact_matches = [token for token in tokens if normalized == token]
        if exact_matches:
            return exact_matches, "exact_name"
        name_tokens = {token for token in normalized.split("_") if token}
        token_matches = [token for token in tokens if "_" not in token and token in name_tokens]
        if token_matches:
            return token_matches, "token"
        return [], None


    def build_record(
        *,
        file_path: Path,
        output_path: Path,
        sheet_name: str | None,
        column: str,
        normalized: str,
        matched_tokens: list[str],
        matched_by: str | None,
        original_dtype: str,
        original_non_null: int,
        converted_non_null: int,
        success_rate: float | None,
        modified_cells: int,
        risk_level: str,
        action_taken: str,
        reason: str,
        rows_in: int,
        rows_out: int,
        cols_in: int,
        cols_out: int,
        in_place: bool,
        dry_run: bool,
    ) -> ColumnRecord:
        return ColumnRecord(
            file_path=file_path.as_posix(),
            sheet_name=sheet_name,
            output_path=output_path.as_posix(),
            column_original=str(column),
            column_normalized=normalized,
            matched_tokens="|".join(matched_tokens),
            matched_by=matched_by,
            original_dtype=original_dtype,
            original_non_null=original_non_null,
            converted_non_null=converted_non_null,
            success_rate=round(success_rate, 6) if success_rate is not None else None,
            modified_cells=modified_cells,
            risk_level=risk_level,
            action_taken=action_taken,
            reason=reason,
            age_token_risk=("age" in matched_tokens),
            rows_in=rows_in,
            rows_out=rows_out,
            cols_in=cols_in,
            cols_out=cols_out,
            in_place=in_place,
            dry_run=dry_run,
        )


    def is_identifier_guard_column(normalized: str, allow_age: bool) -> tuple[bool, str]:
        if normalized in AGE_GUARD_COLUMNS:
            return True, "age label columns are never auto-converted"
        if normalized == "age" and not allow_age:
            return True, "age is disabled by default; use --include-token age to enable it"
        if normalized in DEFAULT_EXCLUDED_COLUMNS:
            return True, "default identifier/category column guard"
        if normalized.startswith(IDENTIFIER_PREFIX_GUARDS):
            return True, "identifier prefix guard"
        if normalized.endswith(IDENTIFIER_SUFFIX_GUARDS):
            return True, "identifier suffix guard"
        return False, ""


    def process_frame(
        df: pd.DataFrame,
        file_path: Path,
        output_path: Path,
        sheet_name: str | None,
        tokens: list[str],
        exclude_columns: set[str],
        warn_threshold: float,
        high_risk_threshold: float,
        unsafe_force_high_risk: bool,
        allow_age: bool,
        in_place: bool,
        dry_run: bool,
    ) -> tuple[pd.DataFrame, list[ColumnRecord]]:
        processed = df.copy()
        records: list[ColumnRecord] = []
        rows_in, cols_in = processed.shape

        for column in processed.columns:
            normalized = normalize_token(column)
            matched_tokens, matched_by = match_tokens(column, tokens)
            if not matched_tokens:
                continue

            original_series = processed[column]
            original_non_null = int(original_series.notna().sum())
            if normalized in exclude_columns:
                records.append(
                    build_record(
                        file_path=file_path,
                        output_path=output_path,
                        sheet_name=sheet_name,
                        column=str(column),
                        normalized=normalized,
                        matched_tokens=matched_tokens,
                        matched_by=matched_by,
                        original_dtype=str(original_series.dtype),
                        original_non_null=original_non_null,
                        converted_non_null=original_non_null,
                        success_rate=None,
                        modified_cells=0,
                        risk_level="ok",
                        action_taken="skipped_excluded_column",
                        reason="explicitly excluded by --exclude-column",
                        rows_in=rows_in,
                        rows_out=processed.shape[0],
                        cols_in=cols_in,
                        cols_out=processed.shape[1],
                        in_place=in_place,
                        dry_run=dry_run,
                    )
                )
                continue

            guarded, guard_reason = is_identifier_guard_column(normalized, allow_age=allow_age)
            if guarded:
                action_taken = "skipped_identifier_guard"
                if normalized in AGE_GUARD_COLUMNS or normalized == "age":
                    action_taken = "skipped_age_column"
                records.append(
                    build_record(
                        file_path=file_path,
                        output_path=output_path,
                        sheet_name=sheet_name,
                        column=str(column),
                        normalized=normalized,
                        matched_tokens=matched_tokens,
                        matched_by=matched_by,
                        original_dtype=str(original_series.dtype),
                        original_non_null=original_non_null,
                        converted_non_null=original_non_null,
                        success_rate=None,
                        modified_cells=0,
                        risk_level="ok",
                        action_taken=action_taken,
                        reason=guard_reason,
                        rows_in=rows_in,
                        rows_out=processed.shape[0],
                        cols_in=cols_in,
                        cols_out=processed.shape[1],
                        in_place=in_place,
                        dry_run=dry_run,
                    )
                )
                continue

            numeric, modified_cells, original_non_null, converted_non_null, success_rate = clean_numeric_series(original_series)
            risk_level = "ok"
            if success_rate is not None and success_rate < high_risk_threshold:
                risk_level = "high_risk"
            elif success_rate is not None and success_rate < warn_threshold:
                risk_level = "warn"

            action_taken = "converted"
            reason = "conversion succeeded within configured thresholds"
            if success_rate is not None and success_rate < high_risk_threshold and not unsafe_force_high_risk:
                action_taken = "blocked_high_risk"
                reason = "conversion success rate below high-risk threshold; original column preserved"
            elif success_rate is not None and success_rate < high_risk_threshold and unsafe_force_high_risk:
                reason = "high-risk conversion forced by --unsafe-force-high-risk"

            if action_taken == "converted":
                processed[column] = numeric

            records.append(
                build_record(
                    file_path=file_path,
                    output_path=output_path,
                    sheet_name=sheet_name,
                    column=str(column),
                    normalized=normalized,
                    matched_tokens=matched_tokens,
                    matched_by=matched_by,
                    original_dtype=str(original_series.dtype),
                    original_non_null=original_non_null,
                    converted_non_null=converted_non_null,
                    success_rate=success_rate,
                    modified_cells=modified_cells,
                    risk_level=risk_level,
                    action_taken=action_taken,
                    reason=reason,
                    rows_in=rows_in,
                    rows_out=processed.shape[0],
                    cols_in=cols_in,
                    cols_out=processed.shape[1],
                    in_place=in_place,
                    dry_run=dry_run,
                )
            )

        return processed, records


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Force-clean numeric-like columns and convert them to float64.")
        parser.add_argument("--path", type=Path, required=True, help="Target file or directory")
        parser.add_argument("--report-dir", type=Path, default=None, help="Directory for manifest and summary outputs")
        parser.add_argument("--suffix", type=str, default="_numeric_cleaned", help="Suffix for non in-place output")
        parser.add_argument("--include-token", action="append", default=[], help="Extra column-name token to treat as numeric-like")
        parser.add_argument("--exclude-column", action="append", default=[], help="Normalized column name to skip")
        parser.add_argument("--dry-run", action="store_true", help="Analyze only; do not write cleaned files")
        parser.add_argument("--warn-threshold", type=float, default=0.80, help="Warn if conversion success rate is below this threshold")
        parser.add_argument("--high-risk-threshold", type=float, default=0.20, help="High risk if conversion success rate is below this threshold")
        parser.add_argument(
            "--unsafe-force-high-risk",
            action="store_true",
            help="Force-write high-risk conversions instead of preserving original columns",
        )
        parser.set_defaults(in_place=True)
        parser.add_argument("--in-place", dest="in_place", action="store_true", help="Overwrite original files (default)")
        parser.add_argument("--no-in-place", dest="in_place", action="store_false", help="Write cleaned copies using --suffix")
        return parser.parse_args()


    def main() -> None:
        args = parse_args()
        project_root = detect_project_root()
        report_dir = ensure_report_dir(project_root, args.report_dir)
        tokens = sorted(set(DEFAULT_TOKENS + [normalize_token(token) for token in args.include_token if token]))
        exclude_columns = {normalize_token(column) for column in args.exclude_column if column}
        allow_age = "age" in {normalize_token(token) for token in args.include_token if token}
        target = args.path.expanduser().resolve()
        files = iter_target_files(target)

        manifest_records: list[ColumnRecord] = []
        failed_files: list[dict[str, str]] = []
        processed_files = 0

        for path in files:
            output_path = build_output_path(path, args.in_place, args.suffix)
            try:
                workbook = read_file(path)
                cleaned_workbook: dict[str | None, pd.DataFrame] = {}
                file_records: list[ColumnRecord] = []
                for sheet_name, df in workbook.items():
                    cleaned_df, records = process_frame(
                        df=df,
                        file_path=path,
                        output_path=output_path,
                        sheet_name=sheet_name,
                        tokens=tokens,
                        exclude_columns=exclude_columns,
                        warn_threshold=args.warn_threshold,
                        high_risk_threshold=args.high_risk_threshold,
                        unsafe_force_high_risk=args.unsafe_force_high_risk,
                        allow_age=allow_age,
                        in_place=args.in_place,
                        dry_run=args.dry_run,
                    )
                    cleaned_workbook[sheet_name] = cleaned_df
                    file_records.extend(records)
                if not args.dry_run:
                    write_file(output_path, cleaned_workbook)
                manifest_records.extend(file_records)
                processed_files += 1
            except Exception as exc:  # noqa: BLE001
                failed_files.append({"file_path": path.as_posix(), "error": str(exc)})

        manifest_path = report_asset_path(report_dir, "numeric_cleaning_manifest.csv")
        manifest_df = pd.DataFrame([asdict(record) for record in manifest_records])
        manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

        high_risk = manifest_df.loc[manifest_df["risk_level"] == "high_risk"].to_dict(orient="records") if not manifest_df.empty else []
        warn_risk = manifest_df.loc[manifest_df["risk_level"] == "warn"].to_dict(orient="records") if not manifest_df.empty else []
        age_risk = manifest_df.loc[manifest_df["age_token_risk"]].to_dict(orient="records") if not manifest_df.empty else []
        converted = manifest_df.loc[manifest_df["action_taken"] == "converted"].to_dict(orient="records") if not manifest_df.empty else []
        blocked_high_risk = manifest_df.loc[manifest_df["action_taken"] == "blocked_high_risk"].to_dict(orient="records") if not manifest_df.empty else []
        skipped_identifier = manifest_df.loc[manifest_df["action_taken"] == "skipped_identifier_guard"].to_dict(orient="records") if not manifest_df.empty else []
        skipped_age = manifest_df.loc[manifest_df["action_taken"] == "skipped_age_column"].to_dict(orient="records") if not manifest_df.empty else []
        skipped_excluded = manifest_df.loc[manifest_df["action_taken"] == "skipped_excluded_column"].to_dict(orient="records") if not manifest_df.empty else []

        summary = {
            "target_path": target.as_posix(),
            "project_root": project_root.as_posix(),
            "report_dir": report_dir.as_posix(),
            "processed_files": processed_files,
            "failed_files_count": len(failed_files),
            "candidate_columns_processed": int(len(manifest_records)),
            "warn_columns_count": int(len(warn_risk)),
            "high_risk_columns_count": int(len(high_risk)),
            "age_token_columns_count": int(len(age_risk)),
            "converted_columns_count": int(len(converted)),
            "blocked_high_risk_columns_count": int(len(blocked_high_risk)),
            "skipped_identifier_columns_count": int(len(skipped_identifier)),
            "skipped_age_columns_count": int(len(skipped_age)),
            "skipped_excluded_columns_count": int(len(skipped_excluded)),
            "in_place": args.in_place,
            "dry_run": args.dry_run,
            "warn_threshold": args.warn_threshold,
            "high_risk_threshold": args.high_risk_threshold,
            "unsafe_force_high_risk": args.unsafe_force_high_risk,
            "tokens": tokens,
            "exclude_columns": sorted(exclude_columns),
            "failed_files": failed_files,
            "high_risk_columns": high_risk,
            "age_token_columns": age_risk,
            "notes": [
                "百分号会被删除，但不会自动缩放到 0-1。",
                "age 默认不自动转换；只有显式传 --include-token age 才会放开 age 列。",
                "age_group、age_label、age_name 默认永远不会自动转数值。",
                "脚本会原地覆盖或输出副本，但不会改变行数和列数。",
                "高风险列默认不落盘覆盖；只有显式传 --unsafe-force-high-risk 才会强制写回。",
            ],
        }
        summary_path = report_asset_path(report_dir, "numeric_cleaning_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(json.dumps(
            {
                "target_path": target.as_posix(),
                "processed_files": processed_files,
                "failed_files_count": len(failed_files),
                "candidate_columns_processed": int(len(manifest_records)),
                "warn_columns_count": int(len(warn_risk)),
                "high_risk_columns_count": int(len(high_risk)),
                "converted_columns_count": int(len(converted)),
                "blocked_high_risk_columns_count": int(len(blocked_high_risk)),
                "manifest_file": manifest_path.as_posix(),
                "summary_file": summary_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_build_global_panel():
    __name__ = 'build_global_panel'
    import argparse
    import json
    import re
    from pathlib import Path
    from typing import Any

    import pandas as pd
    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root


    GBD_CAUSE_SPECS = {
        "gbd_rate_cardiovascular_diseases": {
            "patterns": [
                "cardiovascular disease", "cardiovascular diseases",
                "心血管疾病", "心血管病",
            ],
            "concept_cn": "心血管疾病负担率",
            "unit": "rate_or_number",
            "sex_filter": "both",
            "age_filter": "age-standardized/all ages",
            "measure_filter": "deaths/dalys",
            "metric_filter": "rate/number",
        },
        "gbd_rate_chronic_respiratory_diseases": {
            "patterns": [
                "chronic respiratory", "respiratory diseases",
                "慢性呼吸系统疾病", "慢性呼吸疾病", "呼吸系统疾病",
            ],
            "concept_cn": "慢性呼吸系统疾病负担率",
            "unit": "rate_or_number",
            "sex_filter": "both",
            "age_filter": "age-standardized/all ages",
            "measure_filter": "deaths/dalys",
            "metric_filter": "rate/number",
        },
        "gbd_rate_neoplasms": {
            "patterns": [
                "neoplasm", "neoplasms", "cancer",
                "肿瘤", "癌症",
            ],
            "concept_cn": "肿瘤负担率",
            "unit": "rate_or_number",
            "sex_filter": "both",
            "age_filter": "age-standardized/all ages",
            "measure_filter": "deaths/dalys",
            "metric_filter": "rate/number",
        },
        "gbd_rate_diabetes_kidney": {
            "patterns": [
                "diabetes and kidney diseases", "diabetes", "kidney disease",
                "糖尿病和肾脏疾病", "糖尿病与肾病", "糖尿病与肾脏疾病", "糖尿病和肾病",
                "糖尿病肾病",
            ],
            "concept_cn": "糖尿病与肾病负担率",
            "unit": "rate_or_number",
            "sex_filter": "both",
            "age_filter": "age-standardized/all ages",
            "measure_filter": "deaths/dalys",
            "metric_filter": "rate/number",
        },
    }


    DBD_RISK_SPECS = {
        "dbd_smoking": {
            "patterns": ["smoking", "tobacco", "吸烟", "烟草", "二手烟"],
            "concept_cn": "吸烟相关风险暴露",
            "unit": "rate_or_number",
            "sex_filter": "both",
            "age_filter": "age-standardized/all ages",
            "measure_filter": "deaths/dalys",
            "metric_filter": "rate/number",
        },
        "dbd_pm25": {
            "patterns": [
                "particulate", "pm2.5", "ambient particulate", "pm25",
                "颗粒物", "细颗粒物", "空气污染", "环境颗粒物污染", "环境空气颗粒物污染",
            ],
            "concept_cn": "PM2.5相关风险暴露",
            "unit": "rate_or_number",
            "sex_filter": "both",
            "age_filter": "age-standardized/all ages",
            "measure_filter": "deaths/dalys",
            "metric_filter": "rate/number",
        },
        "dbd_high_bmi": {
            "patterns": [
                "body-mass", "body mass", "bmi",
                "高bmi", "高 bmi", "高体重指数", "高身体质量指数",
            ],
            "concept_cn": "高BMI风险暴露",
            "unit": "rate_or_number",
            "sex_filter": "both",
            "age_filter": "age-standardized/all ages",
            "measure_filter": "deaths/dalys",
            "metric_filter": "rate/number",
        },
        "dbd_high_glucose": {
            "patterns": [
                "fasting plasma glucose", "plasma glucose",
                "高空腹血糖", "高血糖", "高血浆葡萄糖",
            ],
            "concept_cn": "高空腹血糖风险暴露",
            "unit": "rate_or_number",
            "sex_filter": "both",
            "age_filter": "age-standardized/all ages",
            "measure_filter": "deaths/dalys",
            "metric_filter": "rate/number",
        },
        "dbd_high_sbp": {
            "patterns": [
                "systolic blood pressure", "blood pressure",
                "高收缩压", "收缩压", "高血压", "收缩压升高", "高收缩期血压",
            ],
            "concept_cn": "高收缩压风险暴露",
            "unit": "rate_or_number",
            "sex_filter": "both",
            "age_filter": "age-standardized/all ages",
            "measure_filter": "deaths/dalys",
            "metric_filter": "rate/number",
        },
        "dbd_dietary_risks": {
            "patterns": ["dietary risks", "diet", "膳食风险", "饮食风险", "不良饮食", "膳食"],
            "concept_cn": "膳食风险暴露",
            "unit": "rate_or_number",
            "sex_filter": "both",
            "age_filter": "age-standardized/all ages",
            "measure_filter": "deaths/dalys",
            "metric_filter": "rate/number",
        },
    }


    WDI_GROUP_SPECS = {
        "pm25": {"panel_var_name": "wdi_pm25", "concept_cn": "WDI PM2.5年均暴露", "unit": "ug_per_m3"},
        "diabetes_prevalence": {"panel_var_name": "wdi_diabetes_prevalence", "concept_cn": "WDI 糖尿病患病率", "unit": "percent"},
        "ncd_mortality_30_70": {"panel_var_name": "wdi_ncd_mortality_30_70", "concept_cn": "WDI 30-70岁慢病过早死亡率", "unit": "percent"},
        "gini": {"panel_var_name": "wdi_gini", "concept_cn": "WDI 基尼系数", "unit": "index"},
        "gdp_per_capita": {"panel_var_name": "wdi_gdp_per_capita", "concept_cn": "WDI 人均GDP", "unit": "current_usd"},
        "health_expenditure_pct_gdp": {"panel_var_name": "wdi_health_expenditure_pct_gdp", "concept_cn": "WDI 卫生支出占GDP比重", "unit": "percent_gdp"},
        "out_of_pocket_pct": {"panel_var_name": "wdi_out_of_pocket_pct", "concept_cn": "WDI 自付支出占卫生支出比重", "unit": "percent"},
        "physicians": {"panel_var_name": "wdi_physicians", "concept_cn": "WDI 医生密度", "unit": "per_1000"},
        "nurses_midwives": {"panel_var_name": "wdi_nurses_midwives", "concept_cn": "WDI 护士和助产士密度", "unit": "per_1000"},
        "hospital_beds": {"panel_var_name": "wdi_hospital_beds", "concept_cn": "WDI 病床密度", "unit": "per_1000"},
        "life_expectancy": {"panel_var_name": "wdi_life_expectancy", "concept_cn": "WDI 预期寿命", "unit": "years"},
        "under5_mortality": {"panel_var_name": "wdi_under5_mortality", "concept_cn": "WDI 五岁以下死亡率", "unit": "per_1000"},
        "infant_mortality": {"panel_var_name": "wdi_infant_mortality", "concept_cn": "WDI 婴儿死亡率", "unit": "per_1000"},
        "fertility": {"panel_var_name": "wdi_fertility", "concept_cn": "WDI 总和生育率", "unit": "births_per_woman"},
        "adolescent_fertility": {"panel_var_name": "wdi_adolescent_fertility", "concept_cn": "WDI 青少年生育率", "unit": "per_1000"},
        "population_total": {"panel_var_name": "wdi_population_total", "concept_cn": "WDI 总人口", "unit": "persons"},
        "population_65_plus_pct": {"panel_var_name": "wdi_population_65_plus_pct", "concept_cn": "WDI 65岁及以上人口占比", "unit": "percent"},
        "urban_population_pct": {"panel_var_name": "wdi_urban_population_pct", "concept_cn": "WDI 城镇人口占比", "unit": "percent"},
        "health_expenditure_per_capita": {"panel_var_name": "wdi_health_expenditure_per_capita", "concept_cn": "WDI 人均卫生支出", "unit": "current_usd"},
        "government_health_expenditure_pct": {"panel_var_name": "wdi_government_health_expenditure_pct", "concept_cn": "WDI 政府卫生支出占比", "unit": "percent"},
        "external_health_expenditure_pct": {"panel_var_name": "wdi_external_health_expenditure_pct", "concept_cn": "WDI 外部卫生支出占比", "unit": "percent"},
    }


    WB_HNP_GROUP_SPECS = {
        "hci": {"panel_var_name": "wb_hnp_hci", "concept_cn": "WB HNP 人力资本指数", "unit": "index"},
        "life_expectancy": {"panel_var_name": "wb_hnp_life_expectancy", "concept_cn": "WB HNP 预期寿命", "unit": "years"},
        "under5_mortality": {"panel_var_name": "wb_hnp_under5_mortality", "concept_cn": "WB HNP 五岁以下死亡率", "unit": "per_1000"},
        "fertility": {"panel_var_name": "wb_hnp_fertility", "concept_cn": "WB HNP 总和生育率", "unit": "births_per_woman"},
        "adolescent_fertility": {"panel_var_name": "wb_hnp_adolescent_fertility", "concept_cn": "WB HNP 青少年生育率", "unit": "per_1000"},
        "birth_rate": {"panel_var_name": "wb_hnp_birth_rate", "concept_cn": "WB HNP 粗出生率", "unit": "per_1000"},
        "death_rate": {"panel_var_name": "wb_hnp_death_rate", "concept_cn": "WB HNP 粗死亡率", "unit": "per_1000"},
        "infant_mortality": {"panel_var_name": "wb_hnp_infant_mortality", "concept_cn": "WB HNP 婴儿死亡率", "unit": "per_1000"},
        "neonatal_mortality": {"panel_var_name": "wb_hnp_neonatal_mortality", "concept_cn": "WB HNP 新生儿死亡率", "unit": "per_1000"},
        "maternal_mortality": {"panel_var_name": "wb_hnp_maternal_mortality", "concept_cn": "WB HNP 孕产妇死亡率", "unit": "per_100000"},
        "stunting": {"panel_var_name": "wb_hnp_stunting", "concept_cn": "WB HNP 发育迟缓率", "unit": "percent"},
        "wasting": {"panel_var_name": "wb_hnp_wasting", "concept_cn": "WB HNP 消瘦率", "unit": "percent"},
        "low_birthweight": {"panel_var_name": "wb_hnp_low_birthweight", "concept_cn": "WB HNP 低出生体重占比", "unit": "percent"},
        "anemia_women": {"panel_var_name": "wb_hnp_anemia_women", "concept_cn": "WB HNP 女性贫血率", "unit": "percent"},
        "immunization_dpt": {"panel_var_name": "wb_hnp_immunization_dpt", "concept_cn": "WB HNP DPT免疫覆盖率", "unit": "percent"},
        "immunization_measles": {"panel_var_name": "wb_hnp_immunization_measles", "concept_cn": "WB HNP 麻疹免疫覆盖率", "unit": "percent"},
        "immunization_hepb3": {"panel_var_name": "wb_hnp_immunization_hepb3", "concept_cn": "WB HNP HepB3免疫覆盖率", "unit": "percent"},
        "skilled_birth_attendance": {"panel_var_name": "wb_hnp_skilled_birth_attendance", "concept_cn": "WB HNP 专业接生覆盖率", "unit": "percent"},
        "contraceptive_prevalence": {"panel_var_name": "wb_hnp_contraceptive_prevalence", "concept_cn": "WB HNP 避孕使用率", "unit": "percent"},
    }


    WORLD_OR_INCOME_GROUP_CODES = {
        "WLD", "HIC", "LIC", "LMC", "UMC", "LMY", "MIC", "IBD", "IBT", "IDA", "IDB",
        "IDX", "INX", "LDC", "OED", "PST", "PRE", "SST", "FCS", "GLOBAL", "HPC",
        "LTE", "TLA", "TSS",
    }
    AGGREGATE_REGION_CODES = {
        "AFE", "AFR", "AFW", "AMR", "ARB", "CEB", "CSS", "EAP", "EAR", "EAS", "ECA",
        "ECS", "EMR", "EMU", "EUR", "EUU", "LAC", "LCN", "MEA", "MNA", "NAC", "OSS",
        "PSS", "SAS", "SEAR", "SSA", "SSF", "TEA", "TEC", "TMN", "TSA", "WPR",
    }
    NON_SOVEREIGN_CODES = {
        "ABW", "AIA", "ASM", "BES", "BLM", "BMU", "CHI", "CUW", "CYM", "ESH", "FLK",
        "FRO", "GGY", "GIB", "GLP", "GRL", "GUF", "GUM", "HKG", "IMN", "JEY", "MAC",
        "MAF", "MNP", "MSR", "MTQ", "MYT", "NCL", "NIU", "PRI", "PYF", "REU",
        "SHN", "SPM", "SXM", "TCA", "TKL", "VGB", "VIR", "WLF", "XKX",
    }
    FORMAL_ANALYSIS_SCOPE = "UN_193_PLUS_2_OBSERVERS"
    FORMAL_ANALYSIS_EXPECTED_COUNTRIES = 195
    UN_OBSERVER_STATE_CODES = {"PSE", "VAT"}
    OUTSIDE_UN_193_PLUS_2_CODES = {"COK", "TWN"}
    FORMAL_SCOPE_CATEGORIES = {"sovereign_country", "un_observer_state"}
    CHINA_MAINLAND_CODE = "CHN"
    CHINA_TAIWAN_CODE = "TWN"
    CHINA_TAIWAN_MERGE_ENABLED = True
    CHINA_TAIWAN_POPULATION_SUM_COLUMNS = {"population_thousands", "wdi_population_total"}
    WORLD_OR_INCOME_GROUP_MARKERS = (
        "world",
        "income",
        "ida",
        "ibrd",
        "dividend",
        "small states",
        "fragile and conflict",
    )
    AGGREGATE_REGION_MARKERS = (
        "aggregate",
        "regional",
        "arab world",
        "euro area",
        "small states",
    )
    NON_SOVEREIGN_NAME_MARKERS = (
        "channel islands",
        "puerto rico",
        "west bank and gaza",
        "virgin islands",
        "isle of man",
        "bermuda",
        "aruba",
        "gibraltar",
        "greenland",
        "guam",
        "french polynesia",
        "new caledonia",
        "northern mariana",
        "american samoa",
        "cayman islands",
        "turks and caicos",
        "british virgin islands",
        "hong kong",
        "macao",
        "faroe islands",
        "channel islands",
        "sint maarten",
        "st. martin",
    )
    NON_SOVEREIGN_NOTE_MARKERS = (
        "special administrative",
    )


    def detect_project_root() -> Path:
        return shared_detect_project_root()


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        dirs = {
            "clean": project_root / "09_data_clean",
            "report": project_root / "06_report_assets",
            "simulation": project_root / "04_simulation",
            "inventory": external_data_root / "16_Project_Metadata_Registry",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def normalize_token(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text


    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        mapping: dict[str, str] = {}
        seen: dict[str, int] = {}
        for col in df.columns:
            base = normalize_token(col) or "col"
            count = seen.get(base, 0)
            seen[base] = count + 1
            mapping[col] = base if count == 0 else f"{base}_{count}"
        return df.rename(columns=mapping)


    def read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


    def write_text(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")


    def normalize_year(df: pd.DataFrame) -> pd.DataFrame:
        for candidate in ("year", "time_period", "timedim", "time"):
            if candidate in df.columns:
                if candidate != "year":
                    df = df.rename(columns={candidate: "year"})
                break
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        return df


    def collapse_on_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        if df.empty:
            return df
        numeric_cols = [c for c in df.columns if c not in keys and pd.api.types.is_numeric_dtype(df[c])]
        other_cols = [c for c in df.columns if c not in keys and c not in numeric_cols]
        agg = {col: "mean" for col in numeric_cols}
        agg.update({col: "first" for col in other_cols})
        return df.groupby(keys, dropna=False, as_index=False).agg(agg)


    def build_country_lookup(clean_dir: Path) -> tuple[dict[str, str], pd.DataFrame]:
        lookup_rows: list[dict[str, str]] = []
        external_data_root = shared_detect_external_data_root(project_root=clean_dir.parent)
        inventory_dir = external_data_root / "16_Project_Metadata_Registry"
        wdi_country = clean_dir / "cleaned_wdi_country.csv"
        if wdi_country.exists():
            df = normalize_columns(read_csv(wdi_country))
            name_col = next((c for c in ("country_name", "short_name", "table_name", "long_name") if c in df.columns), None)
            code_col = next((c for c in ("country_code", "code", "2_alpha_code", "currency_unit") if c in df.columns), None)
            if name_col and code_col:
                tmp = df[[name_col, code_col]].dropna().rename(columns={name_col: "country_name", code_col: "iso3"})
                lookup_rows.extend(tmp.to_dict("records"))

        ext_pop = clean_dir / "external_un_population.csv"
        if ext_pop.exists():
            df = normalize_columns(read_csv(ext_pop))
            if {"country", "iso3"}.issubset(df.columns):
                lookup_rows.extend(
                    df[["country", "iso3"]]
                    .dropna()
                    .rename(columns={"country": "country_name"})
                    .drop_duplicates()
                    .to_dict("records")
                )

        zh_alias = inventory_dir / "country_alias_zh.csv"
        if zh_alias.exists():
            df = normalize_columns(read_csv(zh_alias))
            name_col = next((c for c in ("country_name_zh", "country_name", "name_zh", "name") if c in df.columns), None)
            code_col = next((c for c in ("iso3", "country_code", "code") if c in df.columns), None)
            if name_col and code_col:
                tmp = (
                    df[[name_col, code_col]]
                    .dropna()
                    .rename(columns={name_col: "country_name", code_col: "iso3"})
                )
                lookup_rows.extend(tmp.to_dict("records"))

        lookup_df = pd.DataFrame(lookup_rows)
        if lookup_df.empty:
            return {}, pd.DataFrame(columns=["country_name", "iso3", "country_key"])
        lookup_df = lookup_df.dropna().drop_duplicates()
        lookup_df["country_key"] = lookup_df["country_name"].map(normalize_token)
        lookup_df["iso3"] = lookup_df["iso3"].astype(str).str.upper()
        mapping = dict(lookup_df[["country_key", "iso3"]].drop_duplicates().values.tolist())
        return mapping, lookup_df


    def load_country_metadata(clean_dir: Path) -> pd.DataFrame:
        wdi_country = clean_dir / "cleaned_wdi_country.csv"
        if not wdi_country.exists():
            return pd.DataFrame(columns=["iso3", "entity_name", "short_name", "table_name", "long_name", "region", "income_group", "special_notes"])
        df = normalize_columns(read_csv(wdi_country))
        if "country_code" not in df.columns:
            return pd.DataFrame(columns=["iso3", "entity_name", "short_name", "table_name", "long_name", "region", "income_group", "special_notes"])
        df["iso3"] = df["country_code"].astype(str).str.upper().str.strip()
        name_col = next((column for column in ("short_name", "table_name", "long_name") if column in df.columns), None)
        df["entity_name"] = df[name_col].astype(str).str.strip() if name_col else df["iso3"]
        for column in ("short_name", "table_name", "long_name", "region", "income_group", "special_notes"):
            if column not in df.columns:
                df[column] = pd.NA
        keep = ["iso3", "entity_name", "short_name", "table_name", "long_name", "region", "income_group", "special_notes"]
        return df.loc[:, keep].drop_duplicates(subset=["iso3"]).reset_index(drop=True)


    def classify_scope_category(iso3: str, meta_row: dict[str, Any] | None) -> tuple[str, str]:
        code = str(iso3).upper().strip()
        if not code:
            return "unknown", "empty_code"
        if code in UN_OBSERVER_STATE_CODES:
            return "un_observer_state", "un_observer_state_included_in_193_plus_2"
        if code in OUTSIDE_UN_193_PLUS_2_CODES:
            return "non_sovereign", "outside_un_193_plus_2_formal_scope"
        if code in NON_SOVEREIGN_CODES:
            return "non_sovereign", "explicit_non_sovereign_code"
        if code.startswith("WB_") or code.startswith("ZZ") or "." in code or "_" in code:
            if "WORLD" in code or "WLD" in code:
                return "world_or_income_group", "pattern_world_code"
            return "aggregate_region", "pattern_aggregate_code"
        if code in WORLD_OR_INCOME_GROUP_CODES:
            return "world_or_income_group", "explicit_world_or_income_group_code"
        if code in AGGREGATE_REGION_CODES:
            return "aggregate_region", "explicit_aggregate_region_code"

        meta_row = meta_row or {}
        name_parts = [
            str(meta_row.get("entity_name", "")),
            str(meta_row.get("short_name", "")),
            str(meta_row.get("table_name", "")),
            str(meta_row.get("long_name", "")),
        ]
        note = str(meta_row.get("special_notes", "")).lower()
        region = str(meta_row.get("region", "")).strip()
        income_group = str(meta_row.get("income_group", "")).strip()
        joined_name = " ".join(name_parts).lower()

        if any(marker in joined_name for marker in NON_SOVEREIGN_NAME_MARKERS) or any(marker in note for marker in NON_SOVEREIGN_NOTE_MARKERS):
            return "non_sovereign", "metadata_non_sovereign_marker"
        if (not region or region.lower() == "nan") and (not income_group or income_group.lower() == "nan"):
            if any(marker in joined_name for marker in WORLD_OR_INCOME_GROUP_MARKERS) or any(marker in note for marker in WORLD_OR_INCOME_GROUP_MARKERS):
                return "world_or_income_group", "metadata_world_or_income_group_marker"
            if "aggregate" in joined_name or "aggregate" in note or any(marker in joined_name for marker in AGGREGATE_REGION_MARKERS):
                return "aggregate_region", "metadata_regionless_aggregate"
            return "sovereign_country", "regionless_default_country"
        return "sovereign_country", "metadata_country_default"


    def build_scope_registry(observed_codes: list[str], country_metadata: pd.DataFrame) -> pd.DataFrame:
        meta_map = (
            country_metadata.drop_duplicates(subset=["iso3"]).set_index("iso3").to_dict(orient="index")
            if not country_metadata.empty
            else {}
        )
        rows: list[dict[str, Any]] = []
        for iso3 in sorted({str(code).upper().strip() for code in observed_codes if str(code).strip()}):
            meta_row = meta_map.get(iso3, {})
            scope_category, reason = classify_scope_category(iso3, meta_row)
            rows.append(
                {
                    "iso3": iso3,
                    "entity_name": meta_row.get("entity_name", iso3),
                    "region": meta_row.get("region", pd.NA),
                    "income_group": meta_row.get("income_group", pd.NA),
                    "scope_category": scope_category,
                    "classification_reason": reason,
                    "formal_analysis_scope": FORMAL_ANALYSIS_SCOPE,
                    "include_in_formal_analysis": scope_category in FORMAL_SCOPE_CATEGORIES,
                    "include_in_un_193_plus_2_analysis": scope_category in FORMAL_SCOPE_CATEGORIES,
                    "include_in_sovereign_analysis": scope_category == "sovereign_country",
                }
            )
        return pd.DataFrame(rows)


    def filter_to_scope(frame: pd.DataFrame, formal_scope_codes: set[str]) -> pd.DataFrame:
        if frame.empty or "iso3" not in frame.columns:
            return frame
        filtered = frame.copy()
        filtered["iso3"] = filtered["iso3"].astype(str).str.upper()
        filtered = filtered.loc[filtered["iso3"].isin(formal_scope_codes)].copy()
        return filtered.reset_index(drop=True)


    def collect_observed_iso3_codes(frames: list[tuple[str, pd.DataFrame]]) -> list[str]:
        codes: set[str] = set()
        for _, frame in frames:
            if frame.empty or "iso3" not in frame.columns:
                continue
            codes.update(frame["iso3"].dropna().astype(str).str.upper().str.strip().tolist())
        return sorted(code for code in codes if code)


    def population_weight_for_china_merge(group: pd.DataFrame) -> pd.Series:
        weight = pd.Series(float("nan"), index=group.index, dtype="float64")
        if "wdi_population_total" in group.columns:
            weight = pd.to_numeric(group["wdi_population_total"], errors="coerce")
        if "population_thousands" in group.columns:
            fallback = pd.to_numeric(group["population_thousands"], errors="coerce") * 1000.0
            weight = weight.where(weight.notna(), fallback)
        return weight.where(weight > 0)


    def weighted_mean(values: pd.Series, weights: pd.Series) -> float | pd.NA:
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.notna()
        if not valid.any():
            return pd.NA
        valid_weights = pd.to_numeric(weights, errors="coerce").where(weights > 0)
        weighted_valid = valid & valid_weights.notna()
        if weighted_valid.any() and float(valid_weights.loc[weighted_valid].sum()) > 0:
            return float((numeric.loc[weighted_valid] * valid_weights.loc[weighted_valid]).sum() / valid_weights.loc[weighted_valid].sum())
        return float(numeric.loc[valid].mean())


    def merge_taiwan_into_china_frame(dataset_name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        if frame.empty or "iso3" not in frame.columns or "year" not in frame.columns:
            return frame, []
        output = frame.copy()
        output["iso3"] = output["iso3"].astype(str).str.upper().str.strip()
        if CHINA_TAIWAN_CODE not in set(output["iso3"].dropna()):
            return output, []

        pair = output.loc[output["iso3"].isin({CHINA_MAINLAND_CODE, CHINA_TAIWAN_CODE})].copy()
        others = output.loc[~output["iso3"].isin({CHINA_MAINLAND_CODE, CHINA_TAIWAN_CODE})].copy()
        numeric_columns = [
            column
            for column in output.columns
            if column not in {"iso3", "year"} and pd.api.types.is_numeric_dtype(output[column])
        ]
        other_columns = [column for column in output.columns if column not in {"iso3", "year"} and column not in numeric_columns]

        merged_rows: list[pd.Series] = []
        audit_rows: list[dict[str, Any]] = []
        for year, group in pair.groupby("year", dropna=False, sort=True):
            chn_rows = group.loc[group["iso3"] == CHINA_MAINLAND_CODE]
            twn_rows = group.loc[group["iso3"] == CHINA_TAIWAN_CODE]
            if twn_rows.empty:
                merged_rows.append(chn_rows.iloc[0].copy())
                continue

            base = chn_rows.iloc[0].copy() if not chn_rows.empty else twn_rows.iloc[0].copy()
            base["iso3"] = CHINA_MAINLAND_CODE
            base["year"] = year
            weights = population_weight_for_china_merge(group)
            chn_weight = float(weights.loc[chn_rows.index].sum()) if not chn_rows.empty else 0.0
            twn_weight = float(weights.loc[twn_rows.index].sum()) if not twn_rows.empty else 0.0

            changed_columns: list[str] = []
            for column in numeric_columns:
                values = group[column]
                if column in CHINA_TAIWAN_POPULATION_SUM_COLUMNS:
                    numeric = pd.to_numeric(values, errors="coerce")
                    value = numeric.sum(min_count=1)
                    base[column] = pd.NA if pd.isna(value) else float(value)
                else:
                    base[column] = weighted_mean(values, weights)
                if not chn_rows.empty and column in chn_rows.columns:
                    chn_value = pd.to_numeric(chn_rows.iloc[0][column], errors="coerce")
                    new_value = pd.to_numeric(base[column], errors="coerce")
                    if pd.notna(chn_value) and pd.notna(new_value) and abs(float(new_value) - float(chn_value)) > 1e-9:
                        changed_columns.append(column)

            for column in other_columns:
                chn_value = base.get(column, pd.NA)
                if pd.isna(chn_value):
                    twn_non_null = twn_rows[column].dropna()
                    if not twn_non_null.empty:
                        base[column] = twn_non_null.iloc[0]

            merged_rows.append(base)
            audit_rows.append(
                {
                    "dataset": dataset_name,
                    "year": year,
                    "mainland_rows": int(len(chn_rows)),
                    "taiwan_rows": int(len(twn_rows)),
                    "mainland_population_weight": chn_weight if chn_weight > 0 else pd.NA,
                    "taiwan_population_weight": twn_weight if twn_weight > 0 else pd.NA,
                    "taiwan_weight_share": (
                        twn_weight / (chn_weight + twn_weight)
                        if (chn_weight + twn_weight) > 0
                        else pd.NA
                    ),
                    "merged_numeric_columns": int(len([column for column in numeric_columns if group[column].notna().any()])),
                    "changed_from_mainland_columns": ";".join(changed_columns),
                    "merge_rule": "TWN merged into CHN; population columns summed; other numeric columns population-weighted when weights are available",
                }
            )

        merged_pair = pd.DataFrame(merged_rows, columns=output.columns) if merged_rows else pd.DataFrame(columns=output.columns)
        merged = pd.concat([others, merged_pair], ignore_index=True, sort=False)
        merged = merged.sort_values(["iso3", "year"], kind="stable").reset_index(drop=True)
        return merged, audit_rows


    def merge_taiwan_into_china_frames(frames: list[tuple[str, pd.DataFrame]]) -> tuple[list[tuple[str, pd.DataFrame]], pd.DataFrame]:
        merged_frames: list[tuple[str, pd.DataFrame]] = []
        audit_rows: list[dict[str, Any]] = []
        if not CHINA_TAIWAN_MERGE_ENABLED:
            return frames, pd.DataFrame()
        for dataset_name, frame in frames:
            merged_frame, rows = merge_taiwan_into_china_frame(dataset_name, frame)
            merged_frames.append((dataset_name, merged_frame))
            audit_rows.extend(rows)
        audit_df = pd.DataFrame(audit_rows)
        return merged_frames, audit_df


    def build_year_coverage_row(
        dataset_name: str,
        df: pd.DataFrame,
        start_year: int,
        end_year: int,
    ) -> dict[str, Any]:
        if df.empty or "year" not in df.columns:
            return {
                "dataset_name": dataset_name,
                "rows": int(df.shape[0]) if df is not None else 0,
                "entities": 0,
                "min_year": pd.NA,
                "max_year": pd.NA,
                "years_present": "",
                "missing_years": ",".join(str(year) for year in range(start_year, end_year + 1)),
                "entity_min_years": pd.NA,
                "entity_median_years": pd.NA,
            }
        years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
        if years.empty:
            return {
                "dataset_name": dataset_name,
                "rows": int(df.shape[0]),
                "entities": int(df["iso3"].nunique()) if "iso3" in df.columns else 0,
                "min_year": pd.NA,
                "max_year": pd.NA,
                "years_present": "",
                "missing_years": ",".join(str(year) for year in range(start_year, end_year + 1)),
                "entity_min_years": pd.NA,
                "entity_median_years": pd.NA,
            }
        observed_years = sorted(years.unique().tolist())
        missing_years = [str(year) for year in range(start_year, end_year + 1) if year not in observed_years]
        entity_year_counts = (
            df.assign(year_num=pd.to_numeric(df["year"], errors="coerce"))
            .dropna(subset=["year_num"])
            .groupby("iso3", dropna=False)["year_num"]
            .nunique()
            if "iso3" in df.columns
            else pd.Series(dtype="float64")
        )
        return {
            "dataset_name": dataset_name,
            "rows": int(df.shape[0]),
            "entities": int(df["iso3"].nunique()) if "iso3" in df.columns else 0,
            "min_year": int(min(observed_years)),
            "max_year": int(max(observed_years)),
            "years_present": ",".join(str(year) for year in observed_years),
            "missing_years": ",".join(missing_years),
            "entity_min_years": int(entity_year_counts.min()) if not entity_year_counts.empty else pd.NA,
            "entity_median_years": float(entity_year_counts.median()) if not entity_year_counts.empty else pd.NA,
        }


    def standardize_panel_keys(df: pd.DataFrame, country_lookup: dict[str, str]) -> pd.DataFrame:
        df = normalize_columns(df)
        df = normalize_year(df)

        zh_rename_map = {
            "地理位置": "location",
            "国家": "country",
            "年龄": "age_name",
            "性别": "sex_name",
            "死亡或受伤原因": "cause",
            "风险因素": "risk",
            "测量": "measure_name",
            "数值": "value",
            "上限": "upper",
            "下限": "lower",
            "人口": "population_label",
        }
        existing_zh = {k: v for k, v in zh_rename_map.items() if k in df.columns and v not in df.columns}
        if existing_zh:
            df = df.rename(columns=existing_zh)

        if "iso3" not in df.columns:
            for candidate in ("country_code", "ref_area", "spatialdim", "iso3_code"):
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "iso3"})
                    break

        if "iso3" not in df.columns:
            for col in df.columns:
                if col in {"year", "country_key", "source_group", "source_file", "population_label"}:
                    continue
                series = df[col]
                if not pd.api.types.is_object_dtype(series):
                    continue
                normalized = series.dropna().astype(str).str.strip().str.upper()
                if normalized.empty:
                    continue
                if normalized.nunique() < 20:
                    continue
                match_rate = normalized.str.fullmatch(r"[A-Z]{3}").mean()
                if match_rate >= 0.8:
                    df = df.rename(columns={col: "iso3"})
                    break

        if "iso3" not in df.columns:
            for candidate in ("country", "country_name", "location", "location_name", "ref_area_label"):
                if candidate in df.columns:
                    df["country_key"] = df[candidate].map(normalize_token)
                    df["iso3"] = df["country_key"].map(country_lookup)
                    break

        if "iso3" in df.columns:
            df["iso3"] = df["iso3"].astype(str).str.upper()
            df.loc[df["iso3"].isin({"<NA>", "NAN", "NONE"}), "iso3"] = pd.NA

        return df.dropna(subset=[c for c in ["iso3", "year"] if c in df.columns], how="any")


    def registry_row(
        panel_var_name: str,
        concept_cn: str,
        source_dataset: str,
        source_file: str,
        source_field: str,
        unit: str = "",
        sex_filter: str = "",
        age_filter: str = "",
        measure_filter: str = "",
        metric_filter: str = "",
        transform_rule: str = "as_is",
        dedupe_rule: str = "groupby_iso3_year_mean_if_duplicates",
        preferred_source_rank: int = 1,
        notes: str = "",
    ) -> dict[str, Any]:
        return {
            "panel_var_name": panel_var_name,
            "concept_cn": concept_cn,
            "source_dataset": source_dataset,
            "source_file": source_file,
            "source_field": source_field,
            "unit": unit,
            "sex_filter": sex_filter,
            "age_filter": age_filter,
            "measure_filter": measure_filter,
            "metric_filter": metric_filter,
            "transform_rule": transform_rule,
            "dedupe_rule": dedupe_rule,
            "preferred_source_rank": preferred_source_rank,
            "notes": notes,
        }


    def aggregate_wide_panel(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        if df.empty:
            return df, []
        key_counts = df.groupby(["iso3", "year"], dropna=False).size()
        dup_pairs = int((key_counts > 1).sum())
        qc_rows = [
            {
                "dataset": dataset_name,
                "panel_var_name": "__dataset_level__",
                "issue_type": "duplicate_iso3_year",
                "duplicate_pairs": dup_pairs,
                "conflicting_pairs": dup_pairs,
                "resolution": "numeric_mean_text_first",
            }
        ]
        return collapse_on_keys(df, ["iso3", "year"]), qc_rows


    def aggregate_long_panel(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        if df.empty:
            return pd.DataFrame(columns=["iso3", "year"]), []
        qc_rows: list[dict[str, Any]] = []
        frames: list[pd.DataFrame] = []
        for panel_var_name, subset in df.groupby("panel_var_name", dropna=False):
            subset = subset.dropna(subset=["iso3", "year", "value"]).copy()
            if subset.empty:
                continue
            subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
            subset = subset.dropna(subset=["value"])
            if subset.empty:
                continue
            grouped = subset.groupby(["iso3", "year"], dropna=False)["value"]
            dup_size = grouped.size()
            dup_pairs = int((dup_size > 1).sum())
            conflict_pairs = int((grouped.nunique() > 1).sum())
            qc_rows.append(
                {
                    "dataset": dataset_name,
                    "panel_var_name": panel_var_name,
                    "issue_type": "duplicate_iso3_year_within_variable",
                    "duplicate_pairs": dup_pairs,
                    "conflicting_pairs": conflict_pairs,
                    "resolution": "mean_after_filtering",
                }
            )
            collapsed = grouped.mean().reset_index().rename(columns={"value": panel_var_name})
            frames.append(collapsed)

        if not frames:
            return pd.DataFrame(columns=["iso3", "year"]), qc_rows
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["iso3", "year"], how="outer")
        return merged, qc_rows


    def load_external_panel(clean_dir: Path, country_lookup: dict[str, str]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
        path = clean_dir / "external_who_panel_wide.csv"
        if not path.exists():
            return pd.DataFrame(columns=["iso3", "year"]), [], []
        df = standardize_panel_keys(read_csv(path), country_lookup)
        keep = ["iso3", "year"] + [c for c in df.columns if c not in {"region", "country_key"} and c not in {"iso3", "year"}]
        df = df[keep]
        collapsed, qc_rows = aggregate_wide_panel(df, "external_panel")
        registry_rows = [
            registry_row(
                panel_var_name=col,
                concept_cn=col,
                source_dataset="external_panel",
                source_file=path.as_posix(),
                source_field=col,
                notes="来自 external_who_panel_wide.csv 的直接字段",
            )
            for col in collapsed.columns
            if col not in {"iso3", "year"}
        ]
        return collapsed, registry_rows, qc_rows


    def load_external_population(clean_dir: Path, country_lookup: dict[str, str]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
        frames: list[pd.DataFrame] = []
        registry_rows: list[dict[str, Any]] = []
        qc_rows: list[dict[str, Any]] = []

        un_path = clean_dir / "external_un_population.csv"
        if un_path.exists():
            df = standardize_panel_keys(read_csv(un_path), country_lookup)
            keep = [
                c
                for c in (
                    "iso3",
                    "year",
                    "population_thousands",
                    "pop_density",
                    "median_age",
                    "pop_growth_rate",
                    "tfr",
                    "life_expectancy",
                    "life_expectancy_male",
                    "life_expectancy_female",
                    "crude_birth_rate",
                    "crude_death_rate",
                    "infant_mortality_rate",
                    "under5_mortality",
                )
                if c in df.columns
            ]
            if keep:
                sub = df[keep]
                collapsed, qc = aggregate_wide_panel(sub, "external_population")
                frames.append(collapsed)
                qc_rows.extend(qc)
                registry_rows.extend(
                    [
                        registry_row(
                            panel_var_name=col,
                            concept_cn=col,
                            source_dataset="external_population",
                            source_file=un_path.as_posix(),
                            source_field=col,
                            notes="来自 UN population",
                        )
                        for col in collapsed.columns
                        if col not in {"iso3", "year"}
                    ]
                )

        hdi_path = clean_dir / "external_hdi_long_format.csv"
        if hdi_path.exists():
            df = normalize_columns(read_csv(hdi_path))
            if {"iso3", "year", "hdi"}.issubset(df.columns):
                melted = df[["iso3", "year", "hdi"]].copy()
            elif {"iso3", "year", "indicator", "value"}.issubset(df.columns):
                tmp = df.copy()
                tmp["indicator"] = tmp["indicator"].astype(str).str.strip().str.lower()
                tmp = tmp[tmp["indicator"].eq("hdi")].copy()
                melted = tmp[["iso3", "year", "value"]].rename(columns={"value": "hdi"})
            else:
                year_cols = [c for c in df.columns if c.startswith("hdi_") and c[4:].isdigit()]
                base_cols = [c for c in ("iso3", "country") if c in df.columns]
                melted = df[base_cols + year_cols].melt(id_vars=base_cols, value_vars=year_cols, var_name="year", value_name="hdi")
                melted["year"] = melted["year"].str.replace("hdi_", "", regex=False)
            melted = standardize_panel_keys(melted, country_lookup)
            collapsed, qc = aggregate_wide_panel(melted[["iso3", "year", "hdi"]], "external_hdi")
            frames.append(collapsed)
            qc_rows.extend(qc)
            registry_rows.append(
                registry_row(
                    panel_var_name="hdi",
                    concept_cn="人类发展指数",
                    source_dataset="external_hdi",
                    source_file=hdi_path.as_posix(),
                    source_field="hdi",
                    unit="index",
                )
            )

        if not frames:
            return pd.DataFrame(columns=["iso3", "year"]), registry_rows, qc_rows

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["iso3", "year"], how="outer")
        return merged, registry_rows, qc_rows


    def pivot_gbd_like(
        clean_dir: Path,
        file_name: str,
        country_lookup: dict[str, str],
        spec_map: dict[str, dict[str, Any]],
        dataset_name: str,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
        path = clean_dir / file_name
        if not path.exists():
            return pd.DataFrame(columns=["iso3", "year"]), [], []

        df = standardize_panel_keys(read_csv(path), country_lookup)
        if dataset_name == "dbd":
            ordered_name_candidates = ("risk", "rei", "rei_name", "cause", "cause_name")
        else:
            ordered_name_candidates = ("cause", "cause_name", "risk", "rei", "rei_name")
        name_candidates = [c for c in ordered_name_candidates if c in df.columns]
        if not name_candidates:
            name_candidates = [
                c
                for c in df.columns
                if any(token in c for token in ("cause", "risk", "rei", "indicator", "疾病", "死因", "风险"))
            ]
        if not name_candidates:
            return pd.DataFrame(columns=["iso3", "year"]), [], []
        name_col = name_candidates[0]

        external_data_root = shared_detect_external_data_root(project_root=clean_dir.parent)
        inventory_dir = external_data_root / "16_Project_Metadata_Registry"
        inventory_dir.mkdir(parents=True, exist_ok=True)
        catalog = (
            df[name_col]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("name_value")
            .reset_index(name="count")
        )
        catalog.insert(0, "dataset", dataset_name)
        catalog.insert(1, "name_column", name_col)
        catalog.to_csv(inventory_dir / f"{dataset_name}_name_catalog.csv", index=False, encoding="utf-8-sig")

        original_df = df.copy()

        if "sex" in df.columns:
            filtered = df[df["sex"].astype(str).str.lower().fillna("").isin(["both", "3", "both sexes"])]
            if not filtered.empty:
                df = filtered
        elif "sex_name" in df.columns:
            filtered = df[df["sex_name"].astype(str).str.lower().fillna("").isin(["both", "合计", "男女合计", "全部"])]
            if not filtered.empty:
                df = filtered

        if "age" in df.columns:
            filtered = df[df["age"].astype(str).str.lower().fillna("").isin(["age_standardized", "age standardized", "age-standardized", "27", "all ages"])]
            if not filtered.empty:
                df = filtered
        elif "age_name" in df.columns:
            filtered = df[df["age_name"].astype(str).str.lower().fillna("").isin(["age-standardized", "all ages", "全部", "全龄", "全人口"])]
            if not filtered.empty:
                df = filtered

        if "metric" in df.columns:
            filtered = df[df["metric"].astype(str).str.lower().fillna("").isin(["rate", "number"])]
            if not filtered.empty:
                df = filtered
        elif "metric_name" in df.columns:
            filtered = df[df["metric_name"].astype(str).str.lower().fillna("").isin(["rate", "number"])]
            if not filtered.empty:
                df = filtered

        if "measure" in df.columns:
            filtered = df[df["measure"].astype(str).str.lower().fillna("").isin(["deaths", "dalys"])]
            if not filtered.empty:
                df = filtered
        elif "measure_name" in df.columns:
            filtered = df[df["measure_name"].astype(str).str.lower().fillna("").isin(["deaths", "dalys", "死亡", "伤残调整寿命年", "daly", "daly率", "死亡率"])]
            if not filtered.empty:
                df = filtered

        value_col = next((c for c in ("value", "val", "obs_value") if c in df.columns), None)
        if value_col is None:
            numeric_candidates = [
                c for c in df.columns
                if c not in {"iso3", "year", name_col, "sex", "sex_name", "age", "age_name", "metric", "metric_name", "measure", "measure_name"}
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            value_col = numeric_candidates[0] if numeric_candidates else None
        if value_col is None:
            return pd.DataFrame(columns=["iso3", "year"]), [], []

        long_rows: list[pd.DataFrame] = []
        name_series = df[name_col].astype(str).str.lower()
        registry_rows = [
            registry_row(
                panel_var_name=panel_var_name,
                concept_cn=spec["concept_cn"],
                source_dataset=dataset_name,
                source_file=path.as_posix(),
                source_field=value_col,
                unit=spec["unit"],
                sex_filter=spec["sex_filter"],
                age_filter=spec["age_filter"],
                measure_filter=spec["measure_filter"],
                metric_filter=spec["metric_filter"],
            )
            for panel_var_name, spec in spec_map.items()
        ]

        for panel_var_name, spec in spec_map.items():
            mask = pd.Series(False, index=df.index)
            for pattern in spec["patterns"]:
                mask = mask | name_series.str.contains(pattern, regex=False)
            if not mask.any() and df is not original_df:
                fallback_name_series = original_df[name_col].astype(str).str.lower()
                fallback_mask = pd.Series(False, index=original_df.index)
                for pattern in spec["patterns"]:
                    fallback_mask = fallback_mask | fallback_name_series.str.contains(pattern, regex=False)
                if fallback_mask.any():
                    fallback_value_col = next((c for c in ("value", "val", "obs_value") if c in original_df.columns), None)
                    if fallback_value_col is None:
                        numeric_candidates = [
                            c for c in original_df.columns
                            if c not in {"iso3", "year", name_col}
                            and pd.api.types.is_numeric_dtype(original_df[c])
                        ]
                        fallback_value_col = numeric_candidates[0] if numeric_candidates else None
                    if fallback_value_col is not None:
                        subset = original_df.loc[fallback_mask, ["iso3", "year", fallback_value_col]].copy()
                        subset = subset.rename(columns={fallback_value_col: "value"})
                        subset["panel_var_name"] = panel_var_name
                        long_rows.append(subset)
                    continue
            subset = df.loc[mask, ["iso3", "year", value_col]].copy()
            if subset.empty:
                continue
            subset = subset.rename(columns={value_col: "value"})
            subset["panel_var_name"] = panel_var_name
            long_rows.append(subset)

        if not long_rows:
            metadata_like = {
                "iso3", "year", "country", "country_name", "location", "location_name",
                "source_file", "source_group", "sex", "sex_name", "age", "age_name",
                "measure", "measure_name", "metric", "metric_name", "cause", "cause_name",
                "risk", "rei", "rei_name",
            }
            wide_rows: list[pd.DataFrame] = []
            numeric_columns = [
                c for c in df.columns
                if c not in metadata_like and pd.api.types.is_numeric_dtype(df[c])
            ]
            normalized_col_map = {c: normalize_token(c) for c in numeric_columns}
            for panel_var_name, spec in spec_map.items():
                matched_cols = []
                normalized_patterns = [normalize_token(p) for p in spec["patterns"]]
                for col, norm_col in normalized_col_map.items():
                    if any(pattern and pattern in norm_col for pattern in normalized_patterns):
                        matched_cols.append(col)
                if not matched_cols:
                    continue
                subset = df[["iso3", "year"] + matched_cols].copy()
                subset[panel_var_name] = subset[matched_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                subset = subset[["iso3", "year", panel_var_name]].dropna(subset=[panel_var_name])
                if subset.empty:
                    continue
                wide_rows.append(subset)
            if wide_rows:
                merged = wide_rows[0]
                qc_rows: list[dict[str, Any]] = []
                for frame in wide_rows[1:]:
                    merged = merged.merge(frame, on=["iso3", "year"], how="outer")
                merged, qc_rows = aggregate_wide_panel(merged, dataset_name)
                return merged, registry_rows, qc_rows
            return pd.DataFrame(columns=["iso3", "year"]), registry_rows, []
        long_df = pd.concat(long_rows, ignore_index=True, sort=False)
        wide, qc_rows = aggregate_long_panel(long_df, dataset_name)
        return wide, registry_rows, qc_rows


    def load_group_labeled_source(
        path: Path,
        dataset_name: str,
        group_specs: dict[str, dict[str, Any]],
        rename_map: dict[str, str],
        status_col: str | None = None,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
        if not path.exists():
            return pd.DataFrame(columns=["iso3", "year"]), [], []
        df = normalize_columns(read_csv(path))
        df = df.rename(columns=rename_map)
        df = normalize_year(df)
        if "iso3" in df.columns:
            df["iso3"] = df["iso3"].astype(str).str.upper()
        if status_col and status_col in df.columns:
            df = df[~df[status_col].fillna("").str.lower().str.contains("missing value", regex=False)]
        required = {"iso3", "year", "value"}
        if not required.issubset(df.columns):
            return pd.DataFrame(columns=["iso3", "year"]), [], []
        if "selected_indicator_group" not in df.columns:
            return pd.DataFrame(columns=["iso3", "year"]), [], []
        df["panel_var_name"] = df["selected_indicator_group"].map(
            {group: spec["panel_var_name"] for group, spec in group_specs.items()}
        )
        df = df.dropna(subset=["panel_var_name"])
        registry_rows = [
            registry_row(
                panel_var_name=spec["panel_var_name"],
                concept_cn=spec["concept_cn"],
                source_dataset=dataset_name,
                source_file=path.as_posix(),
                source_field="value",
                unit=spec["unit"],
                notes="由 selected_indicator_group 标准化映射",
            )
            for spec in group_specs.values()
        ]
        wide, qc_rows = aggregate_long_panel(df[["iso3", "year", "panel_var_name", "value"]], dataset_name)
        return wide, registry_rows, qc_rows


    def load_wdi_subset(clean_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
        path = clean_dir / "cleaned_wdi.csv"
        return load_group_labeled_source(
            path=path,
            dataset_name="wdi",
            group_specs=WDI_GROUP_SPECS,
            rename_map={"country_code": "iso3"},
        )


    def load_wb_hnp_subset(clean_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
        path = clean_dir / "cleaned_wb_hnp.csv"
        return load_group_labeled_source(
            path=path,
            dataset_name="wb_hnp",
            group_specs=WB_HNP_GROUP_SPECS,
            rename_map={"ref_area": "iso3", "time_period": "year", "obs_value": "value"},
            status_col="obs_status_label",
        )


    def build_china_nbs_support(clean_dir: Path, simulation_dir: Path, report_dir: Path) -> Path | None:
        path = clean_dir / "cleaned_nbs_health.csv"
        if not path.exists():
            return None
        df = normalize_columns(read_csv(path))
        if "year" not in df.columns:
            year_col = next((c for c in df.columns if "year" in c or "年份" in c), None)
            if year_col:
                df = df.rename(columns={year_col: "year"})
        if "year" not in df.columns:
            write_text(report_asset_path(report_dir, "china_nbs_support_note.txt"), "NBS 数据缺少可识别年份字段，未生成中国补充面板。\n")
            return None
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["year"])
        numeric_cols = [c for c in df.columns if c != "year" and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            write_text(report_asset_path(report_dir, "china_nbs_support_note.txt"), "NBS 数据缺少可聚合数值字段，未生成中国补充面板。\n")
            return None
        grouped = df.groupby("year", as_index=False)[numeric_cols].mean()
        grouped.insert(0, "iso3", "CHN")
        output = simulation_dir / "china_nbs_support_panel.csv"
        grouped.to_csv(output, index=False, encoding="utf-8-sig")
        write_text(report_asset_path(report_dir, "china_nbs_support_note.txt"), "已按年份对 NBS 数值字段取均值，生成中国补充面板。\n")
        return output


    def merge_frames(frames: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        non_empty = [(name, df.copy()) for name, df in frames if not df.empty]
        if not non_empty:
            return pd.DataFrame(), pd.DataFrame(columns=["dataset", "column", "non_null_count"])

        merged = non_empty[0][1]
        for _, frame in non_empty[1:]:
            merged = merged.merge(frame, on=["iso3", "year"], how="outer", validate="one_to_one")

        merged = merged.sort_values(["iso3", "year"]).reset_index(drop=True)
        variable_rows: list[dict[str, Any]] = []
        for dataset_name, frame in non_empty:
            for column in frame.columns:
                if column in {"iso3", "year"}:
                    continue
                variable_rows.append(
                    {
                        "dataset": dataset_name,
                        "column": column,
                        "non_null_count": int(frame[column].notna().sum()),
                    }
                )
        variable_manifest = pd.DataFrame(variable_rows)
        return merged, variable_manifest


    def build_missingness_report(panel: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        total_rows = len(panel)
        year_count = panel["year"].nunique() if "year" in panel.columns else 0
        country_count = panel["iso3"].nunique() if "iso3" in panel.columns else 0
        for column in panel.columns:
            if column in {"iso3", "year"}:
                continue
            non_null = int(panel[column].notna().sum())
            rows.append(
                {
                    "column": column,
                    "non_null_count": non_null,
                    "missing_count": int(total_rows - non_null),
                    "missing_rate": round((total_rows - non_null) / total_rows, 6) if total_rows else None,
                    "country_coverage": int(panel.loc[panel[column].notna(), "iso3"].nunique()) if "iso3" in panel.columns else None,
                    "year_coverage": int(panel.loc[panel[column].notna(), "year"].nunique()) if "year" in panel.columns else None,
                    "panel_country_count": int(country_count),
                    "panel_year_count": int(year_count),
                }
            )
        return pd.DataFrame(rows).sort_values(["missing_rate", "column"], ascending=[True, True]).reset_index(drop=True)


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Build a registry-driven standardized country-year global health panel.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--start-year", type=int, default=2000)
        parser.add_argument("--end-year", type=int, default=2024)
        parser.add_argument("--with-china-nbs", action="store_true")
        return parser.parse_args()


    def main() -> None:
        args = parse_args()
        project_root = args.project_root or detect_project_root()
        dirs = ensure_dirs(project_root)
        clean_dir = dirs["clean"]
        report_dir = dirs["report"]
        simulation_dir = dirs["simulation"]
        inventory_dir = dirs["inventory"]

        country_lookup, lookup_df = build_country_lookup(clean_dir)
        country_metadata = load_country_metadata(clean_dir)
        if not lookup_df.empty:
            lookup_df.to_csv(inventory_dir / "country_lookup_reference.csv", index=False, encoding="utf-8-sig")

        external_panel, external_panel_registry, external_panel_qc = load_external_panel(clean_dir, country_lookup)
        external_population, external_population_registry, external_population_qc = load_external_population(clean_dir, country_lookup)
        gbd_frame, gbd_registry, gbd_qc = pivot_gbd_like(clean_dir, "cleaned_gbd_panel.csv", country_lookup, GBD_CAUSE_SPECS, "gbd")
        dbd_frame, dbd_registry, dbd_qc = pivot_gbd_like(clean_dir, "cleaned_dbd_panel.csv", country_lookup, DBD_RISK_SPECS, "dbd")
        wdi_frame, wdi_registry, wdi_qc = load_wdi_subset(clean_dir)
        wb_hnp_frame, wb_hnp_registry, wb_hnp_qc = load_wb_hnp_subset(clean_dir)

        frames = [
            ("external_panel", external_panel),
            ("external_population", external_population),
            ("gbd", gbd_frame),
            ("dbd", dbd_frame),
            ("wdi", wdi_frame),
            ("wb_hnp", wb_hnp_frame),
        ]
        observed_codes_before_country_merge = collect_observed_iso3_codes(frames)
        frames, china_taiwan_merge_audit = merge_taiwan_into_china_frames(frames)
        china_taiwan_merge_audit_path = report_asset_path(report_dir, "china_taiwan_merge_audit.csv")
        if china_taiwan_merge_audit.empty:
            china_taiwan_merge_audit = pd.DataFrame(
                columns=[
                    "dataset",
                    "year",
                    "mainland_rows",
                    "taiwan_rows",
                    "mainland_population_weight",
                    "taiwan_population_weight",
                    "taiwan_weight_share",
                    "merged_numeric_columns",
                    "changed_from_mainland_columns",
                    "merge_rule",
                ]
            )
        china_taiwan_merge_audit.to_csv(china_taiwan_merge_audit_path, index=False, encoding="utf-8-sig")

        china_support_output = None
        if args.with_china_nbs:
            china_support_output = build_china_nbs_support(clean_dir, simulation_dir, report_dir)

        merged, variable_manifest = merge_frames(frames)
        if merged.empty:
            raise RuntimeError("No mergeable cleaned source files were found in 09_data_clean.")

        merged = merged[(merged["year"] >= args.start_year) & (merged["year"] <= args.end_year)]
        merged = merged.sort_values(["iso3", "year"]).reset_index(drop=True)

        scope_registry = build_scope_registry(observed_codes_before_country_merge, country_metadata)
        scope_registry_path = inventory_dir / "country_scope_registry.csv"
        scope_registry.to_csv(scope_registry_path, index=False, encoding="utf-8-sig")

        formal_scope_codes = set(
            scope_registry.loc[scope_registry["scope_category"].isin(FORMAL_SCOPE_CATEGORIES), "iso3"]
            .dropna()
            .astype(str)
            .str.upper()
            .tolist()
        )
        dropped_aggregate_entities = sorted(
            scope_registry.loc[scope_registry["scope_category"] == "aggregate_region", "iso3"].dropna().astype(str).tolist()
        )
        dropped_non_sovereign_entities = sorted(
            scope_registry.loc[scope_registry["scope_category"] == "non_sovereign", "iso3"].dropna().astype(str).tolist()
        )
        dropped_world_or_income_groups = sorted(
            scope_registry.loc[scope_registry["scope_category"] == "world_or_income_group", "iso3"].dropna().astype(str).tolist()
        )

        filtered_frames = []
        for name, frame in frames:
            time_filtered = frame.copy()
            if "year" in time_filtered.columns:
                years = pd.to_numeric(time_filtered["year"], errors="coerce")
                time_filtered = time_filtered.loc[years.between(args.start_year, args.end_year, inclusive="both")].copy()
            filtered_frames.append((name, filter_to_scope(time_filtered, formal_scope_codes)))
        merged = filter_to_scope(merged, formal_scope_codes)
        variable_manifest = pd.DataFrame(
            [
                {
                    "dataset": name,
                    "column": column,
                    "non_null_count": int(frame[column].notna().sum()),
                }
                for name, frame in filtered_frames
                for column in frame.columns
                if column not in {"iso3", "year"}
            ]
        )

        output_path = simulation_dir / "global_health_panel_v1.csv"
        merged.to_csv(output_path, index=False, encoding="utf-8-sig")

        registry_rows = (
            external_panel_registry
            + external_population_registry
            + gbd_registry
            + dbd_registry
            + wdi_registry
            + wb_hnp_registry
        )
        registry_df = pd.DataFrame(registry_rows).drop_duplicates().sort_values(["source_dataset", "panel_var_name"]).reset_index(drop=True)
        registry_path = inventory_dir / "global_panel_variable_registry.csv"
        registry_df.to_csv(registry_path, index=False, encoding="utf-8-sig")

        variable_manifest_path = inventory_dir / "global_panel_variable_manifest.csv"
        variable_manifest.to_csv(variable_manifest_path, index=False, encoding="utf-8-sig")

        qc_rows = external_panel_qc + external_population_qc + gbd_qc + dbd_qc + wdi_qc + wb_hnp_qc
        qc_report = pd.DataFrame(qc_rows)
        qc_path = report_asset_path(report_dir, "global_panel_qc_report.csv")
        qc_report.to_csv(qc_path, index=False, encoding="utf-8-sig")

        missingness = build_missingness_report(merged)
        missingness_path = report_asset_path(report_dir, "global_panel_missingness.csv")
        missingness.to_csv(missingness_path, index=False, encoding="utf-8-sig")

        source_year_coverage = pd.DataFrame(
            [build_year_coverage_row(name, df, args.start_year, args.end_year) for name, df in filtered_frames]
        )
        source_year_coverage_path = report_asset_path(report_dir, "global_source_year_coverage.csv")
        source_year_coverage.to_csv(source_year_coverage_path, index=False, encoding="utf-8-sig")

        build_summary = {
            "project_root": project_root.as_posix(),
            "clean_dir": clean_dir.as_posix(),
            "simulation_dir": simulation_dir.as_posix(),
            "inventory_dir": inventory_dir.as_posix(),
            "output_file": output_path.as_posix(),
            "rows": int(len(merged)),
            "columns": int(len(merged.columns)),
            "year_min": int(merged["year"].min()) if not merged.empty else None,
            "year_max": int(merged["year"].max()) if not merged.empty else None,
            "formal_analysis_scope": FORMAL_ANALYSIS_SCOPE,
            "expected_country_count": FORMAL_ANALYSIS_EXPECTED_COUNTRIES,
            "actual_country_count": int(merged["iso3"].nunique()) if "iso3" in merged.columns else 0,
            "included_un_observer_entities": sorted(code for code in UN_OBSERVER_STATE_CODES if code in formal_scope_codes),
            "excluded_outside_un_193_plus_2_entities": sorted(code for code in OUTSIDE_UN_193_PLUS_2_CODES if code not in formal_scope_codes),
            "china_taiwan_merge": {
                "enabled": CHINA_TAIWAN_MERGE_ENABLED,
                "source_iso3": CHINA_TAIWAN_CODE,
                "target_iso3": CHINA_MAINLAND_CODE,
                "audit_file": china_taiwan_merge_audit_path.as_posix(),
                "datasets_with_taiwan_rows": sorted(china_taiwan_merge_audit["dataset"].dropna().astype(str).unique().tolist()),
                "years_merged": int(china_taiwan_merge_audit["year"].nunique()) if "year" in china_taiwan_merge_audit.columns else 0,
                "rule": "population columns are summed; other numeric columns use population-weighted averages when CHN/TWN population weights are available",
            },
            "scope_filter_applied": True,
            "scope_registry_file": scope_registry_path.as_posix(),
            "dropped_aggregate_entities": dropped_aggregate_entities,
            "dropped_non_sovereign_entities": dropped_non_sovereign_entities,
            "dropped_world_or_income_group_entities": dropped_world_or_income_groups,
            "datasets_used": {
                name: {
                    "rows": int(len(df)),
                    "columns": list(df.columns),
                }
                for name, df in filtered_frames
                if not df.empty
            },
            "latest_year_by_source": {
                row["dataset_name"]: {
                    "min_year": None if pd.isna(row["min_year"]) else int(row["min_year"]),
                    "max_year": None if pd.isna(row["max_year"]) else int(row["max_year"]),
                    "missing_years": row["missing_years"],
                    "entity_median_years": None if pd.isna(row["entity_median_years"]) else float(row["entity_median_years"]),
                }
                for row in [
                    build_year_coverage_row(name, df, args.start_year, args.end_year)
                    for name, df in filtered_frames
                ]
            },
            "registry_file": registry_path.as_posix(),
            "variable_manifest_file": variable_manifest_path.as_posix(),
            "qc_report_file": qc_path.as_posix(),
            "missingness_file": missingness_path.as_posix(),
            "source_year_coverage_file": source_year_coverage_path.as_posix(),
            "china_nbs_support_output": None if china_support_output is None else china_support_output.as_posix(),
        }
        summary_path = report_asset_path(report_dir, "global_panel_build_summary.json")
        summary_path.write_text(json.dumps(build_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        note_lines = [
            "标准化规则:",
            "- 统一主键为 iso3 + year",
            "- 各来源先按显式筛选规则抽取变量，再聚合到一国一年一条",
            "- 同一变量若同一国家同一年存在多条记录，先记录到 QC，再按均值聚合",
            "- 最终采用 outer join 合并所有来源，保留完整国家-年份覆盖",
            "- 中国口径已将 TWN 逐年并入 CHN：人口总量类字段求和，其余数值字段在可得人口权重下做人口加权平均",
            "- 正式分析样本限定为 UN 193 个会员国 + 2 个观察员国；区域聚合体、世界/收入组和口径外实体仅保留在 scope registry 中追溯",
            "",
            f"输出主表: {output_path.as_posix()}",
            f"行数: {len(merged)}",
            f"列数: {len(merged.columns)}",
            f"年份范围: {merged['year'].min()} - {merged['year'].max()}",
            f"正式分析口径: {FORMAL_ANALYSIS_SCOPE}",
            f"正式分析实体数: {merged['iso3'].nunique()} / {FORMAL_ANALYSIS_EXPECTED_COUNTRIES}",
            f"国家范围注册表: {scope_registry_path.as_posix()}",
            f"变量注册表: {registry_path.as_posix()}",
            f"变量覆盖清单: {variable_manifest_path.as_posix()}",
            f"QC 报告: {qc_path.as_posix()}",
            f"缺失值报告: {missingness_path.as_posix()}",
            f"年份覆盖报告: {source_year_coverage_path.as_posix()}",
            f"中国台湾合并审计: {china_taiwan_merge_audit_path.as_posix()}",
        ]
        if china_support_output is not None:
            note_lines.append(f"中国补充面板: {china_support_output.as_posix()}")
        write_text(report_asset_path(report_dir, "global_panel_build_note.txt"), "\n".join(note_lines) + "\n")

        print(json.dumps(build_summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_build_analysis_panels():
    __name__ = 'build_analysis_panels'
    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root
    FORMAL_ANALYSIS_SCOPE = "UN_193_PLUS_2_OBSERVERS"
    FORMAL_ANALYSIS_EXPECTED_COUNTRIES = 195

    STAGE_COLUMNS = [
        "iso3",
        "year",
        "analysis_population_total",
        "gbd_rate_cardiovascular_diseases",
        "gbd_rate_chronic_respiratory_diseases",
        "gbd_rate_neoplasms",
        "gbd_rate_diabetes_kidney",
        "gbd_rate_cardiovascular_diseases_per100k",
        "gbd_rate_chronic_respiratory_diseases_per100k",
        "gbd_rate_neoplasms_per100k",
        "gbd_rate_diabetes_kidney_per100k",
        "life_expectancy",
        "life_expectancy_male",
        "life_expectancy_female",
        "hdi",
        "population_thousands",
        "pop_density",
        "median_age",
        "pop_growth_rate",
        "tfr",
        "wdi_population_65_plus_pct",
        "wdi_population_total",
        "wdi_urban_population_pct",
        "wdi_gdp_per_capita",
        "wdi_gini",
    ]

    RISK_COLUMNS = STAGE_COLUMNS + [
        "dbd_smoking",
        "dbd_pm25",
        "dbd_high_bmi",
        "dbd_high_glucose",
        "dbd_high_sbp",
        "dbd_dietary_risks",
        "dbd_smoking_per100k",
        "dbd_pm25_per100k",
        "dbd_high_bmi_per100k",
        "dbd_high_glucose_per100k",
        "dbd_high_sbp_per100k",
        "dbd_dietary_risks_per100k",
        "wdi_diabetes_prevalence",
        "wdi_pm25",
        "mpower_group",
    ]

    RESPONSE_COLUMNS = RISK_COLUMNS + [
        "beds_10k",
        "doctors_10k",
        "nurses_10k",
        "uhc_index",
        "che_pct_gdp",
        "che_pc_usd",
        "govt_he_pct",
        "ext_he_pct",
        "wdi_health_expenditure_per_capita",
        "wdi_government_health_expenditure_pct",
        "wdi_external_health_expenditure_pct",
        "wdi_hospital_beds",
        "wdi_physicians",
        "wdi_nurses_midwives",
        "wdi_out_of_pocket_pct",
        "wb_hnp_hci",
        "wb_hnp_skilled_birth_attendance",
        "wb_hnp_immunization_dpt",
        "wb_hnp_immunization_measles",
        "wb_hnp_immunization_hepb3",
    ]

    BURDEN_COLUMNS = [
        "gbd_rate_cardiovascular_diseases",
        "gbd_rate_chronic_respiratory_diseases",
        "gbd_rate_neoplasms",
        "gbd_rate_diabetes_kidney",
    ]

    RISK_EXPOSURE_COLUMNS = [
        "dbd_smoking",
        "dbd_pm25",
        "dbd_high_bmi",
        "dbd_high_glucose",
        "dbd_high_sbp",
        "dbd_dietary_risks",
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        dirs = {
            "simulation": project_root / "04_simulation",
            "report": project_root / "06_report_assets",
            "clean": project_root / "09_data_clean",
            "inventory": external_data_root / "16_Project_Metadata_Registry",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def load_master_panel(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Master panel not found: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        required = {"iso3", "year"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f"Master panel is missing required columns: {missing}")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        return df


    def derive_analysis_population(df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        population_total = pd.Series(np.nan, index=output.index, dtype="float64")

        if "wdi_population_total" in output.columns:
            population_total = pd.to_numeric(output["wdi_population_total"], errors="coerce")

        if "population_thousands" in output.columns:
            fallback_population = pd.to_numeric(output["population_thousands"], errors="coerce") * 1000.0
            if "iso3" in output.columns:
                china_mask = output["iso3"].astype(str).str.upper().eq("CHN")
                # CHN population_thousands is rebuilt after merging TWN into CHN; WDI CHN population excludes TWN.
                population_total = population_total.where(~(china_mask & fallback_population.notna()), fallback_population)
            population_total = population_total.where(population_total.notna(), fallback_population)

        population_total = population_total.where(population_total > 0)
        output["analysis_population_total"] = population_total.astype("float64")
        return output


    def add_per_100k_columns(df: pd.DataFrame, source_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
        output = df.copy()
        created_columns: list[str] = []
        population = pd.to_numeric(output.get("analysis_population_total"), errors="coerce")
        if population is None:
            return output, created_columns

        valid_population = population.where(population > 0)
        for column in source_columns:
            if column not in output.columns:
                continue
            numeric = pd.to_numeric(output[column], errors="coerce")
            derived_column = f"{column}_per100k"
            output[derived_column] = (numeric / valid_population) * 100000.0
            created_columns.append(derived_column)
        return output, created_columns


    def drop_all_null_nonkeys(df: pd.DataFrame) -> pd.DataFrame:
        non_key_cols = [c for c in df.columns if c not in {"iso3", "year"}]
        if not non_key_cols:
            return df
        return df.loc[df[non_key_cols].notna().any(axis=1)].copy()


    def build_panel(df: pd.DataFrame, requested_columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
        existing_columns = [column for column in requested_columns if column in df.columns]
        missing_columns = [column for column in requested_columns if column not in df.columns]
        panel = df.loc[:, existing_columns].copy()
        panel = drop_all_null_nonkeys(panel)
        panel = panel.sort_values(["iso3", "year"], kind="stable").reset_index(drop=True)
        return panel, existing_columns, missing_columns


    def read_scope_registry(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


    def build_file_year_audit_row(path: Path, label: str, start_year: int, end_year: int) -> dict[str, object]:
        if not path.exists():
            return {
                "dataset_name": label,
                "path": path.as_posix(),
                "exists": False,
                "rows": 0,
                "entities": 0,
                "min_year": None,
                "max_year": None,
                "years_present": "",
                "missing_years": ",".join(str(year) for year in range(start_year, end_year + 1)),
                "entity_min_years": None,
                "entity_median_years": None,
            }
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if "year" not in df.columns:
            return {
                "dataset_name": label,
                "path": path.as_posix(),
                "exists": True,
                "rows": int(df.shape[0]),
                "entities": int(df["iso3"].nunique()) if "iso3" in df.columns else 0,
                "min_year": None,
                "max_year": None,
                "years_present": "",
                "missing_years": ",".join(str(year) for year in range(start_year, end_year + 1)),
                "entity_min_years": None,
                "entity_median_years": None,
            }
        years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
        observed_years = sorted(years.unique().tolist())
        missing_years = [str(year) for year in range(start_year, end_year + 1) if year not in observed_years]
        entity_year_counts = (
            df.assign(year_num=pd.to_numeric(df["year"], errors="coerce"))
            .dropna(subset=["year_num"])
            .groupby("iso3", dropna=False)["year_num"]
            .nunique()
            if "iso3" in df.columns
            else pd.Series(dtype="float64")
        )
        return {
            "dataset_name": label,
            "path": path.as_posix(),
            "exists": True,
            "rows": int(df.shape[0]),
            "entities": int(df["iso3"].nunique()) if "iso3" in df.columns else 0,
            "min_year": int(min(observed_years)) if observed_years else None,
            "max_year": int(max(observed_years)) if observed_years else None,
            "years_present": ",".join(str(year) for year in observed_years),
            "missing_years": ",".join(missing_years),
            "entity_min_years": int(entity_year_counts.min()) if not entity_year_counts.empty else None,
            "entity_median_years": float(entity_year_counts.median()) if not entity_year_counts.empty else None,
        }


    def build_variable_year_audit(df: pd.DataFrame, columns: list[str], panel_label: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["panel_name", "column_name", "first_year", "last_year", "entity_min_years", "entity_median_years", "non_null_rows"])
        rows: list[dict[str, object]] = []
        year_num = pd.to_numeric(df["year"], errors="coerce")
        for column in columns:
            if column not in df.columns:
                continue
            mask = df[column].notna() & year_num.notna()
            if not mask.any():
                rows.append(
                    {
                        "panel_name": panel_label,
                        "column_name": column,
                        "first_year": None,
                        "last_year": None,
                        "entity_min_years": None,
                        "entity_median_years": None,
                        "non_null_rows": 0,
                    }
                )
                continue
            entity_year_counts = (
                df.loc[mask, ["iso3", "year"]]
                .assign(year_num=pd.to_numeric(df.loc[mask, "year"], errors="coerce"))
                .dropna(subset=["year_num"])
                .groupby("iso3", dropna=False)["year_num"]
                .nunique()
            )
            rows.append(
                {
                    "panel_name": panel_label,
                    "column_name": column,
                    "first_year": int(year_num.loc[mask].min()),
                    "last_year": int(year_num.loc[mask].max()),
                    "entity_min_years": int(entity_year_counts.min()) if not entity_year_counts.empty else None,
                    "entity_median_years": float(entity_year_counts.median()) if not entity_year_counts.empty else None,
                    "non_null_rows": int(mask.sum()),
                }
            )
        return pd.DataFrame(rows)


    def latest_year_available(df: pd.DataFrame, required_columns: list[str]) -> int | None:
        usable = [column for column in required_columns if column in df.columns]
        if not usable or df.empty:
            return None
        mask = df[usable].notna().all(axis=1)
        if not mask.any():
            mask = df[usable].notna().any(axis=1)
        if not mask.any():
            return None
        years = pd.to_numeric(df.loc[mask, "year"], errors="coerce").dropna()
        return int(years.max()) if not years.empty else None


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build stage/risk/response analysis panels from the global master panel.")
        parser.add_argument("--project-root", type=Path, default=None, help="Project root; defaults to /home/mw/project/雷霆医疗队")
        parser.add_argument(
            "--input-file",
            type=Path,
            default=None,
            help="Optional explicit master panel path; defaults to <project_root>/04_simulation/global_health_panel_v1.csv",
        )
        parser.add_argument("--start-year", type=int, default=2000)
        parser.add_argument("--end-year", type=int, default=2024)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        input_file = args.input_file.expanduser().resolve() if args.input_file else dirs["simulation"] / "global_health_panel_v1.csv"
        master = load_master_panel(input_file)
        master = derive_analysis_population(master)
        master, derived_burden_columns = add_per_100k_columns(master, BURDEN_COLUMNS)
        master, derived_risk_columns = add_per_100k_columns(master, RISK_EXPOSURE_COLUMNS)

        stage_panel, stage_existing, stage_missing = build_panel(master, STAGE_COLUMNS)
        risk_panel, risk_existing, risk_missing = build_panel(master, RISK_COLUMNS)
        response_panel, response_existing, response_missing = build_panel(master, RESPONSE_COLUMNS)

        stage_path = dirs["simulation"] / "stage_panel.csv"
        risk_path = dirs["simulation"] / "risk_profile_panel.csv"
        response_path = dirs["simulation"] / "response_panel.csv"

        stage_panel.to_csv(stage_path, index=False, encoding="utf-8-sig")
        risk_panel.to_csv(risk_path, index=False, encoding="utf-8-sig")
        response_panel.to_csv(response_path, index=False, encoding="utf-8-sig")

        scope_registry_path = dirs["inventory"] / "country_scope_registry.csv"
        scope_registry = read_scope_registry(scope_registry_path)
        file_year_audit = pd.DataFrame(
            [
                build_file_year_audit_row(dirs["clean"] / "cleaned_gbd_panel.csv", "cleaned_gbd_panel", args.start_year, args.end_year),
                build_file_year_audit_row(dirs["clean"] / "cleaned_dbd_panel.csv", "cleaned_dbd_panel", args.start_year, args.end_year),
                build_file_year_audit_row(dirs["clean"] / "cleaned_wdi.csv", "cleaned_wdi", args.start_year, args.end_year),
                build_file_year_audit_row(dirs["clean"] / "cleaned_wb_hnp.csv", "cleaned_wb_hnp", args.start_year, args.end_year),
                build_file_year_audit_row(dirs["clean"] / "external_who_mpower_policy_score.csv", "external_who_mpower_policy_score", args.start_year, args.end_year),
                build_file_year_audit_row(dirs["clean"] / "external_who_smoking_trend.csv", "external_who_smoking_trend", args.start_year, args.end_year),
                build_file_year_audit_row(dirs["clean"] / "external_who_smoking_prevalence.csv", "external_who_smoking_prevalence", args.start_year, args.end_year),
                build_file_year_audit_row(dirs["clean"] / "external_who_cigarette_prevalence.csv", "external_who_cigarette_prevalence", args.start_year, args.end_year),
                build_file_year_audit_row(input_file, "global_health_panel_v1", args.start_year, args.end_year),
                build_file_year_audit_row(stage_path, "stage_panel", args.start_year, args.end_year),
                build_file_year_audit_row(risk_path, "risk_profile_panel", args.start_year, args.end_year),
                build_file_year_audit_row(response_path, "response_panel", args.start_year, args.end_year),
            ]
        )
        file_year_audit_path = report_asset_path(dirs["report"], "year_continuity_file_audit.csv")
        file_year_audit.to_csv(file_year_audit_path, index=False, encoding="utf-8-sig")

        variable_year_audit = pd.concat(
            [
                build_variable_year_audit(stage_panel, [column for column in STAGE_COLUMNS if column not in {"iso3", "year"}], "stage_panel"),
                build_variable_year_audit(risk_panel, [column for column in RISK_COLUMNS if column not in {"iso3", "year"}], "risk_profile_panel"),
                build_variable_year_audit(response_panel, [column for column in RESPONSE_COLUMNS if column not in {"iso3", "year"}], "response_panel"),
            ],
            ignore_index=True,
        )
        variable_year_audit_path = report_asset_path(dirs["report"], "year_continuity_variable_audit.csv")
        variable_year_audit.to_csv(variable_year_audit_path, index=False, encoding="utf-8-sig")

        summary = {
            "project_root": project_root.as_posix(),
            "input_file": input_file.as_posix(),
            "master_rows": int(master.shape[0]),
            "master_columns": int(master.shape[1]),
            "formal_analysis_scope": FORMAL_ANALYSIS_SCOPE,
            "expected_country_count": FORMAL_ANALYSIS_EXPECTED_COUNTRIES,
            "actual_country_count": int(master["iso3"].nunique()) if "iso3" in master.columns else 0,
            "scope_filter_applied": scope_registry_path.exists(),
            "scope_registry_file": scope_registry_path.as_posix() if scope_registry_path.exists() else "",
            "dropped_aggregate_entities": sorted(
                scope_registry.loc[scope_registry["scope_category"] == "aggregate_region", "iso3"].dropna().astype(str).tolist()
            ) if not scope_registry.empty else [],
            "dropped_non_sovereign_entities": sorted(
                scope_registry.loc[scope_registry["scope_category"] == "non_sovereign", "iso3"].dropna().astype(str).tolist()
            ) if not scope_registry.empty else [],
            "derived_columns": {
                "analysis_population_total": "analysis_population_total" in master.columns,
                "burden_per100k_columns": derived_burden_columns,
                "risk_per100k_columns": derived_risk_columns,
            },
            "latest_year_by_source": {
                row["dataset_name"]: row["max_year"]
                for row in file_year_audit.to_dict(orient="records")
                if row.get("max_year") is not None
            },
            "latest_year_by_module_candidate": {
                "module_A_typology": latest_year_available(response_panel, BURDEN_COLUMNS + RISK_EXPOSURE_COLUMNS),
                "module_B_risk_attribution": latest_year_available(risk_panel, BURDEN_COLUMNS + RISK_EXPOSURE_COLUMNS),
                "module_C_response_priority": latest_year_available(response_panel, BURDEN_COLUMNS + RISK_EXPOSURE_COLUMNS),
            },
            "outputs": {
                "stage_panel": {
                    "path": stage_path.as_posix(),
                    "rows": int(stage_panel.shape[0]),
                    "columns": stage_existing,
                    "missing_requested_columns": stage_missing,
                },
                "risk_profile_panel": {
                    "path": risk_path.as_posix(),
                    "rows": int(risk_panel.shape[0]),
                    "columns": risk_existing,
                    "missing_requested_columns": risk_missing,
                },
                "response_panel": {
                    "path": response_path.as_posix(),
                    "rows": int(response_panel.shape[0]),
                    "columns": response_existing,
                    "missing_requested_columns": response_missing,
                },
            },
            "year_audit_files": {
                "file_level": file_year_audit_path.as_posix(),
                "variable_level": variable_year_audit_path.as_posix(),
            },
        }

        summary_path = report_asset_path(dirs["report"], "analysis_panels_build_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_phase123_data_audit():
    __name__ = 'run_phase123_data_audit'
    import argparse
    import json
    from pathlib import Path

    import pandas as pd

    from clean_health_data import (
        WB_HNP_INDICATOR_PATTERNS,
        WDI_INDICATOR_RULES,
        clean_frame,
        compile_indicator_rule_mask,
        compile_pattern_mask,
        detect_input_root,
        detect_project_root,
        ensure_dirs,
        extract_year_from_name,
        iter_files,
        read_tabular,
        resolve_source_path,
    )


    def choose_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None


    def detect_dirs(project_root: Path) -> dict[str, Path]:
        dirs = ensure_dirs(project_root)
        dirs["audit"] = dirs["reports"]
        return dirs


    def build_year_schema_audit(source_root: Path, source_key: str) -> tuple[pd.DataFrame, dict[str, object]]:
        rows: list[dict[str, object]] = []
        column_sets: dict[int, tuple[str, ...]] = {}
        year_files = [path for path in iter_files(source_root) if path.suffix.lower() in {".csv", ".txt"} and extract_year_from_name(path) is not None]
        for path in year_files:
            year = extract_year_from_name(path)
            try:
                df = clean_frame(read_tabular(path), fallback_year=year)
                columns = tuple(sorted(df.columns.tolist()))
                column_sets[int(year)] = columns
                rows.append(
                    {
                        "source_key": source_key,
                        "year": int(year),
                        "rows": int(df.shape[0]),
                        "columns_count": len(columns),
                        "column_signature": " | ".join(columns),
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "source_key": source_key,
                        "year": int(year),
                        "rows": pd.NA,
                        "columns_count": pd.NA,
                        "column_signature": "",
                        "status": f"failed: {exc}",
                    }
                )
        audit_df = pd.DataFrame(rows).sort_values(["year"], kind="stable")
        valid_signatures = audit_df.loc[audit_df["status"] == "ok", "column_signature"].dropna()
        summary = {
            "source_key": source_key,
            "years_scanned": int(audit_df["year"].nunique()) if not audit_df.empty else 0,
            "years_ok": int((audit_df["status"] == "ok").sum()) if not audit_df.empty else 0,
            "schema_consistent": int(valid_signatures.nunique()) <= 1 if not valid_signatures.empty else False,
            "unique_schema_versions": int(valid_signatures.nunique()) if not valid_signatures.empty else 0,
        }
        return audit_df, summary


    def build_value_domain_audit(source_root: Path, source_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        measure_rows: list[dict[str, object]] = []
        domain_rows: list[dict[str, object]] = []
        year_files = [path for path in iter_files(source_root) if path.suffix.lower() in {".csv", ".txt"} and extract_year_from_name(path) is not None]
        for path in year_files:
            year = extract_year_from_name(path)
            try:
                df = clean_frame(read_tabular(path), fallback_year=year)
            except Exception:
                continue
            measure_col = choose_existing(df, ["measure", "测量"])
            if measure_col is not None:
                counts = df[measure_col].astype(str).value_counts(dropna=False)
                for value, count in counts.items():
                    measure_rows.append(
                        {
                            "source_key": source_key,
                            "year": int(year),
                            "measure_value": value,
                            "records": int(count),
                        }
                    )
            for label, candidates in {
                "population": ["population", "population_name"],
                "age": ["年龄", "age", "age_name", "age_group", "nian_ling"],
                "sex": ["性别", "sex", "sex_name", "xing_bie"],
            }.items():
                column = choose_existing(df, candidates)
                if column is None:
                    continue
                counts = df[column].astype(str).value_counts(dropna=False).head(200)
                for value, count in counts.items():
                    domain_rows.append(
                        {
                            "source_key": source_key,
                            "year": int(year),
                            "domain": label,
                            "column_name": column,
                            "value": value,
                            "records": int(count),
                        }
                    )
        measure_df = pd.DataFrame(measure_rows)
        if not measure_df.empty:
            measure_df = (
                measure_df.groupby(["source_key", "measure_value"], dropna=False, as_index=False)
                .agg(records=("records", "sum"), years_covered=("year", "nunique"))
                .sort_values(["records", "measure_value"], ascending=[False, True], kind="stable")
            )
        domain_df = pd.DataFrame(domain_rows)
        if not domain_df.empty:
            domain_df = (
                domain_df.groupby(["source_key", "domain", "column_name", "value"], dropna=False, as_index=False)
                .agg(records=("records", "sum"), years_covered=("year", "nunique"))
                .sort_values(["domain", "records"], ascending=[True, False], kind="stable")
            )
        return measure_df, domain_df


    def build_candidate_pool(glossary_file: Path, patterns: dict[str, list[str]], source_label: str) -> pd.DataFrame:
        df = clean_frame(read_tabular(glossary_file))
        text_col = choose_existing(
            df,
            [
                "indicator_name",
                "indicator_label",
                "series_name",
                "series_description",
                "long_definition",
                "short_definition",
            ],
        )
        code_col = choose_existing(df, ["code", "indicator", "series_code", "indicator_code"])
        if text_col is None:
            raise RuntimeError(f"{source_label} glossary missing indicator text column")
        matched, labels = compile_pattern_mask(df[text_col], patterns)
        candidate_df = df.loc[matched].copy()
        candidate_df["candidate_group"] = labels.loc[matched].values
        keep_cols = [column for column in [code_col, text_col, "candidate_group"] if column is not None]
        rename_map = {}
        if code_col is not None:
            rename_map[code_col] = "indicator_code"
        rename_map[text_col] = "indicator_name"
        candidate_df = candidate_df.loc[:, keep_cols].rename(columns=rename_map).drop_duplicates()
        candidate_df.insert(0, "source_label", source_label)
        return candidate_df.sort_values(["candidate_group", "indicator_name"], kind="stable")


    def build_wdi_candidate_pool(glossary_file: Path, rules: dict[str, dict[str, list[str]]], source_label: str) -> pd.DataFrame:
        df = clean_frame(read_tabular(glossary_file))
        text_col = choose_existing(
            df,
            [
                "indicator_name",
                "indicator_label",
                "series_name",
                "series_description",
                "long_definition",
                "short_definition",
            ],
        )
        code_col = choose_existing(df, ["code", "indicator", "series_code", "indicator_code"])
        if text_col is None:
            raise RuntimeError(f"{source_label} glossary missing indicator text column")
        if code_col is None:
            df["_empty_indicator_code"] = ""
            code_col = "_empty_indicator_code"
        matched, labels = compile_indicator_rule_mask(df[text_col], df[code_col], rules)
        candidate_df = df.loc[matched].copy()
        candidate_df["candidate_group"] = labels.loc[matched].values
        keep_cols = [code_col, text_col, "candidate_group"]
        candidate_df = (
            candidate_df.loc[:, keep_cols]
            .rename(columns={code_col: "indicator_code", text_col: "indicator_name"})
            .drop_duplicates()
        )
        candidate_df.insert(0, "source_label", source_label)
        return candidate_df.sort_values(["candidate_group", "indicator_name"], kind="stable")


    def build_missingness_audit(panel_file: Path) -> pd.DataFrame:
        df = pd.read_csv(panel_file, encoding="utf-8-sig", low_memory=False)
        rows: list[dict[str, object]] = []
        total_rows = int(df.shape[0])
        for column in df.columns:
            non_null = int(df[column].notna().sum())
            rows.append(
                {
                    "column": column,
                    "non_null_rows": non_null,
                    "missing_rows": total_rows - non_null,
                    "missing_rate": (total_rows - non_null) / total_rows if total_rows else pd.NA,
                }
            )
        return pd.DataFrame(rows).sort_values(["missing_rate", "column"], ascending=[False, True], kind="stable")


    def build_country_coverage(panel_file: Path) -> pd.DataFrame:
        df = pd.read_csv(panel_file, encoding="utf-8-sig", low_memory=False)
        required = {"iso3", "year"}
        if not required.issubset(df.columns):
            raise RuntimeError(f"Panel file missing required country coverage columns: {sorted(required - set(df.columns))}")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        coverage = (
            df.dropna(subset=["iso3", "year"])
            .groupby("iso3", dropna=False)
            .agg(year_min=("year", "min"), year_max=("year", "max"), records=("year", "count"))
            .reset_index()
            .sort_values(["records", "iso3"], ascending=[False, True], kind="stable")
        )
        return coverage


    def main() -> None:
        parser = argparse.ArgumentParser(description="Generate Phase 1-3 governance audit artifacts.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--input-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root() if args.project_root is None else args.project_root.expanduser().resolve()
        input_root = detect_input_root(project_root) if args.input_root is None else args.input_root.expanduser().resolve()
        dirs = detect_dirs(project_root)
        report_dir = dirs["audit"]
        clean_dir = project_root / "09_data_clean"

        dbd_root = resolve_source_path("dbd", input_root)
        gbd_root = resolve_source_path("gbd", input_root)
        wb_hnp_glossary = clean_dir / "cleaned_wb_hnp_glossary.csv"
        wdi_series = clean_dir / "cleaned_wdi_series.csv"
        panel_candidates = [project_root / "04_simulation" / "response_panel.csv", clean_dir / "global_health_panel_v1.csv"]
        panel_file = next((path for path in panel_candidates if path.exists()), None)

        outputs: dict[str, str] = {}
        summary: dict[str, object] = {
            "project_root": project_root.as_posix(),
            "input_root": input_root.as_posix(),
            "phase1_checks": {},
            "phase2_phase3_supporting_checks": {},
        }

        for source_key, root in [("dbd", dbd_root), ("gbd", gbd_root)]:
            if root is None or not root.exists():
                continue
            schema_df, schema_summary = build_year_schema_audit(root, source_key)
            schema_out = report_asset_path(report_dir, f"{source_key}_year_schema_audit.csv")
            schema_df.to_csv(schema_out, index=False, encoding="utf-8-sig")
            outputs[f"{source_key}_year_schema_audit"] = schema_out.as_posix()

            measure_df, domain_df = build_value_domain_audit(root, source_key)
            measure_out = report_asset_path(report_dir, f"{source_key}_measure_values.csv")
            domain_out = report_asset_path(report_dir, f"{source_key}_population_age_sex_domains.csv")
            measure_df.to_csv(measure_out, index=False, encoding="utf-8-sig")
            domain_df.to_csv(domain_out, index=False, encoding="utf-8-sig")
            outputs[f"{source_key}_measure_values"] = measure_out.as_posix()
            outputs[f"{source_key}_population_age_sex_domains"] = domain_out.as_posix()
            summary["phase1_checks"][source_key] = schema_summary

        if wb_hnp_glossary.exists():
            wb_candidates = build_candidate_pool(wb_hnp_glossary, WB_HNP_INDICATOR_PATTERNS, "wb_hnp")
            wb_out = report_asset_path(report_dir, "wb_hnp_candidate_variables.csv")
            wb_candidates.to_csv(wb_out, index=False, encoding="utf-8-sig")
            outputs["wb_hnp_candidate_variables"] = wb_out.as_posix()
            summary["phase1_checks"]["wb_hnp_candidates"] = int(wb_candidates.shape[0])

        if wdi_series.exists():
            wdi_candidates = build_wdi_candidate_pool(wdi_series, WDI_INDICATOR_RULES, "wdi")
            wdi_out = report_asset_path(report_dir, "wdi_candidate_variables.csv")
            wdi_candidates.to_csv(wdi_out, index=False, encoding="utf-8-sig")
            outputs["wdi_candidate_variables"] = wdi_out.as_posix()
            summary["phase1_checks"]["wdi_candidates"] = int(wdi_candidates.shape[0])

        if panel_file is not None:
            missingness_df = build_missingness_audit(panel_file)
            missingness_out = report_asset_path(report_dir, "panel_missingness_audit.csv")
            missingness_df.to_csv(missingness_out, index=False, encoding="utf-8-sig")
            outputs["panel_missingness_audit"] = missingness_out.as_posix()
            summary["phase1_checks"]["panel_missingness_input"] = panel_file.as_posix()
            summary["phase1_checks"]["panel_missingness_variables"] = int(missingness_df.shape[0])

            coverage_df = build_country_coverage(panel_file)
            coverage_out = report_asset_path(report_dir, "country_iso3_panel_coverage.csv")
            coverage_df.to_csv(coverage_out, index=False, encoding="utf-8-sig")
            outputs["country_iso3_panel_coverage"] = coverage_out.as_posix()
            summary["phase1_checks"]["countries_in_panel"] = int(coverage_df["iso3"].nunique())

        summary["output_files"] = outputs
        summary_path = report_asset_path(report_dir, "phase123_data_audit_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_phase123_readiness():
    __name__ = 'run_phase123_readiness'
    import argparse
    import json
    from pathlib import Path

    import pandas as pd
    from foundation import detect_project_root as shared_detect_project_root


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        dirs = {
            "report": project_root / "06_report_assets",
            "figures": project_root / "05_figures",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def load_json(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


    def existing(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0


    def row(phase: str, module: str, deliverable: str, path: Path | None, status: str, notes: str) -> dict[str, object]:
        return {
            "phase": phase,
            "module": module,
            "deliverable": deliverable,
            "path": path.as_posix() if path is not None else "",
            "exists": existing(path) if path is not None else False,
            "status": status,
            "notes": notes,
        }


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build a phase 1-3 readiness checklist aligned to the project PDFs.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        report_dir = dirs["report"]
        figures_dir = dirs["figures"]

        data_audit_summary = load_json(report_asset_path(report_dir, "phase123_data_audit_summary.json"))
        typology_summary = load_json(report_asset_path(report_dir, "vulnerability_typology_summary.json"))
        risk_summary = load_json(report_asset_path(report_dir, "risk_attribution_summary.json"))
        response_summary = load_json(report_asset_path(report_dir, "response_diagnosis_summary.json"))
        policy_summary = load_json(report_asset_path(report_dir, "policy_identification_summary.json"))

        rows: list[dict[str, object]] = []

        rows.extend(
            [
                row(
                    "phase1",
                    "数据治理",
                    "年度字段一致性审计",
                    report_asset_path(report_dir, "dbd_year_schema_audit.csv"),
                    "completed" if data_audit_summary else "pending",
                    "覆盖 DBD/GBD 年度 schema、measure、population/age/sex 域值。",
                ),
                row(
                    "phase1",
                    "数据治理",
                    "候选变量池清单",
                    report_asset_path(report_dir, "wb_hnp_candidate_variables.csv"),
                    "completed" if existing(report_asset_path(report_dir, "wb_hnp_candidate_variables.csv")) and existing(report_asset_path(report_dir, "wdi_candidate_variables.csv")) else "pending",
                    "对应 PDF 中 WB_HNP 与 WDI 指标池筛选。",
                ),
                row(
                    "phase1",
                    "数据治理",
                    "缺失值与国家覆盖审计",
                    report_asset_path(report_dir, "panel_missingness_audit.csv"),
                    "completed" if existing(report_asset_path(report_dir, "panel_missingness_audit.csv")) and existing(report_asset_path(report_dir, "country_iso3_panel_coverage.csv")) else "pending",
                    "包含最终面板缺失率和 ISO3 覆盖情况。",
                ),
            ]
        )

        typology_status = "completed" if typology_summary else "pending"
        risk_status = "completed" if risk_summary else "pending"
        rows.extend(
            [
                row(
                    "phase2",
                    "模块A",
                    "脆弱性分型主结果",
                    report_asset_path(report_dir, "vulnerability_typology_summary.json"),
                    typology_status,
                    f"已输出三类主方案，chosen_n_clusters={typology_summary.get('chosen_n_clusters', 'NA')}。",
                ),
                row(
                    "phase2",
                    "模块A",
                    "分型图表与转移分析",
                    figures_dir / "vulnerability_transition_matrix_heatmap.png",
                    "completed" if existing(figures_dir / "vulnerability_transition_matrix_heatmap.png") else "pending",
                    "包含 PCA、维度画像、类型占比趋势、转移概率热图。",
                ),
                row(
                    "phase2",
                    "模块A",
                    "代表国家与类型画像卡",
                    report_asset_path(report_dir, "vulnerability_country_profile_cards.csv"),
                    "completed" if existing(report_asset_path(report_dir, "vulnerability_country_profile_cards.csv")) and existing(report_asset_path(report_dir, "vulnerability_type_storylines.csv")) else "pending",
                    "可直接进入正文和 PPT。",
                ),
                row(
                    "phase2",
                    "模块B",
                    "风险归因矩阵与疾病卡片",
                    report_asset_path(report_dir, "risk_attribution_type_disease_cards.csv"),
                    risk_status if existing(report_asset_path(report_dir, "risk_attribution_type_disease_cards.csv")) else "pending",
                    "对应类型内主导风险和疾病-风险匹配结果。",
                ),
            ]
        )

        policy_locked = policy_summary.get("locked_outcomes", []) if isinstance(policy_summary, dict) else []
        policy_status = "completed" if policy_summary else "pending"
        policy_notes = "框架完成。"
        if policy_summary:
            policy_notes = f"stacked DID 框架已完成，locked_outcomes={len(policy_locked)}。"
            if len(policy_locked) == 0:
                policy_notes += " 当前真实结果未锁主结论。"

        rows.extend(
            [
                row(
                    "phase3",
                    "模块C",
                    "资源响应失配诊断",
                    report_asset_path(report_dir, "response_diagnosis_summary.json"),
                    "completed" if response_summary else "pending",
                    "包含压力-响应散点、类型矩阵、弱项热图与国家优先卡。",
                ),
                row(
                    "phase3",
                    "模块C",
                    "效率象限与增量配置计划",
                    report_asset_path(report_dir, "response_incremental_allocation_plan.csv"),
                    "completed" if existing(report_asset_path(report_dir, "response_incremental_allocation_plan.csv")) and existing(report_asset_path(report_dir, "response_efficiency_quadrant_summary.csv")) else "pending",
                    "对应 PDF 中效率象限和增量资源配置。",
                ),
                row(
                    "phase3",
                    "模块D",
                    "政策识别与稳健性验证",
                    report_asset_path(report_dir, "policy_validation_summary.csv"),
                    policy_status,
                    policy_notes,
                ),
            ]
        )

        checklist = pd.DataFrame(rows)
        checklist_output = report_asset_path(report_dir, "phase123_completion_checklist.csv")
        checklist.to_csv(checklist_output, index=False, encoding="utf-8-sig")

        completed_rows = checklist["status"].isin(["completed"]).sum()
        readiness_summary = {
            "project_root": project_root.as_posix(),
            "completed_deliverables": int(completed_rows),
            "total_deliverables": int(checklist.shape[0]),
            "phase_completion_rate": (
                checklist.groupby("phase", dropna=False)["status"].apply(lambda values: float((values == "completed").mean())).to_dict()
                if not checklist.empty
                else {}
            ),
            "policy_locked_outcomes": policy_locked,
            "key_outputs": {
                "checklist": checklist_output.as_posix(),
                "data_audit_summary": (report_asset_path(report_dir, "phase123_data_audit_summary.json")).as_posix(),
                "typology_summary": (report_asset_path(report_dir, "vulnerability_typology_summary.json")).as_posix(),
                "risk_summary": (report_asset_path(report_dir, "risk_attribution_summary.json")).as_posix(),
                "response_summary": (report_asset_path(report_dir, "response_diagnosis_summary.json")).as_posix(),
                "policy_summary": (report_asset_path(report_dir, "policy_identification_summary.json")).as_posix(),
            },
        }
        summary_output = report_asset_path(report_dir, "phase123_readiness_summary.json")
        summary_output.write_text(json.dumps(readiness_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        note_lines = [
            "# Phase123 验收清单",
            "",
            f"- 已完成交付项：{completed_rows}/{checklist.shape[0]}",
            f"- Phase1 完成率：{readiness_summary['phase_completion_rate'].get('phase1', 0):.0%}",
            f"- Phase2 完成率：{readiness_summary['phase_completion_rate'].get('phase2', 0):.0%}",
            f"- Phase3 完成率：{readiness_summary['phase_completion_rate'].get('phase3', 0):.0%}",
            "",
            "## 备注",
            f"- 模块D locked_outcomes：{len(policy_locked)}",
            "- 若模块D未锁主结论，正文应降级为验证模块或趋势性证据。",
        ]
        notes_output = report_asset_path(report_dir, "phase123_completion_notes.md")
        notes_output.write_text("\n".join(note_lines), encoding="utf-8")

        print(json.dumps(readiness_summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


NAMESPACE_BUILDERS = {
    'clean_health_data.py': _namespace_clean_health_data,
    'clean_numeric_like_columns.py': _namespace_clean_numeric_like_columns,
    'build_global_panel.py': _namespace_build_global_panel,
    'build_analysis_panels.py': _namespace_build_analysis_panels,
    'run_phase123_data_audit.py': _namespace_run_phase123_data_audit,
    'run_phase123_readiness.py': _namespace_run_phase123_readiness,
}

STEP_GROUPS = {'build': ['clean_health_data.py', 'clean_numeric_like_columns.py', 'build_global_panel.py', 'build_analysis_panels.py'], 'audit': ['run_phase123_data_audit.py', 'run_phase123_readiness.py']}
DEFAULT_GROUPS = ['build', 'audit']
STEP_ARGS = {
    'clean_numeric_like_columns.py': ['--path', '09_data_clean'],
}


def selected_steps(groups: list[str]) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = []
    for group in groups:
        steps.extend((script_name, STEP_ARGS.get(script_name, [])) for script_name in STEP_GROUPS[group])
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description='Build cleaned health data, global panel, and ABCD analysis panels.')
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--skip-clean", action="store_true", help="Skip cleaning and start from panel build steps.")

    args = parser.parse_args()
    project_root = detect_project_root(args.project_root)
    groups = [name for name in STEP_GROUPS if getattr(args, name)]
    if not groups:
        groups = list(DEFAULT_GROUPS)
    steps = selected_steps(groups)
    if args.skip_clean:
        steps = [(name, argv) for name, argv in steps if name not in {"clean_health_data.py", "clean_numeric_like_columns.py"}]
    run_step_sequence(steps, NAMESPACE_BUILDERS, project_root=project_root)


if __name__ == "__main__":
    main()
