from __future__ import annotations

import argparse
from pathlib import Path

from a_foundation import detect_project_root, report_asset_path, run_step_sequence

def _namespace_run_abc_predictive_validation():
    __name__ = 'run_abc_predictive_validation'
    import argparse
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder

    from foundation import choose_text, configure_matplotlib_fonts
    from foundation import detect_project_root as shared_detect_project_root


    USE_CHINESE = configure_matplotlib_fonts()
    RANDOM_STATE = 20260529

    TARGET_SPECS = [
        {
            "module": "A健康转型/脆弱性",
            "target_column": "vulnerability_type_label",
            "task_type": "classification",
            "target_label": "下一年脆弱性类型",
        },
        {
            "module": "B风险归因",
            "target_column": "risk_pressure_score",
            "task_type": "regression",
            "target_label": "下一年风险压力得分",
        },
        {
            "module": "C响应失配",
            "target_column": "adaptation_gap_score",
            "task_type": "regression",
            "target_label": "下一年适配缺口得分",
        },
        {
            "module": "C响应失配",
            "target_column": "response_diagnosis_type",
            "task_type": "classification",
            "target_label": "下一年响应诊断类型",
        },
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def load_panel(project_root: Path) -> pd.DataFrame:
        path = project_root / "04_simulation" / "response_diagnosis_panel.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        panel = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        panel["iso3"] = panel["iso3"].astype(str).str.upper().str.strip()
        panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
        return panel.sort_values(["iso3", "year"], kind="stable").reset_index(drop=True)


    def build_supervised_panel(panel: pd.DataFrame) -> pd.DataFrame:
        work = panel.copy()
        for spec in TARGET_SPECS:
            column = spec["target_column"]
            if column in work.columns:
                work[f"target_next_{column}"] = work.groupby("iso3", sort=False)[column].shift(-1)
        work["target_year"] = work.groupby("iso3", sort=False)["year"].shift(-1)
        work = work.loc[work["target_year"].notna()].copy()
        work["target_year"] = work["target_year"].astype(int)
        return work


    def feature_columns(df: pd.DataFrame) -> list[str]:
        blocked_prefixes = ("target_next_", "component__", "z__")
        blocked_exact = {
            "year",
            "target_year",
            "iso3",
            "vulnerability_type_label",
            "response_diagnosis_type",
            "weakest_resource_components",
            "dominant_risk_triplet",
        }
        numeric_cols = []
        for column in df.columns:
            if column in blocked_exact or column.startswith(blocked_prefixes):
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                numeric_cols.append(column)
        usable = []
        for column in numeric_cols:
            numeric = pd.to_numeric(df[column], errors="coerce")
            if numeric.notna().sum() >= 200 and numeric.nunique(dropna=True) > 3:
                usable.append(column)
        return usable


    def prepare_xy(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_train = train.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x_test = test.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        medians = x_train.median(numeric_only=True)
        x_train = x_train.fillna(medians).fillna(0.0)
        x_test = x_test.fillna(medians).fillna(0.0)
        return x_train, x_test


    def train_classification(
        train: pd.DataFrame,
        test: pd.DataFrame,
        latest: pd.DataFrame,
        target: str,
        features: list[str],
    ) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
        target_col = f"target_next_{target}"
        train = train.dropna(subset=[target_col]).copy()
        test = test.dropna(subset=[target_col]).copy()
        if train.empty or test.empty:
            return {"ok": False, "reason": "empty_train_or_test"}, pd.DataFrame(), pd.DataFrame()
        classes = sorted(set(train[target_col].astype(str)))
        test = test.loc[test[target_col].astype(str).isin(classes)].copy()
        if test.empty or len(classes) < 2:
            return {"ok": False, "reason": "insufficient_classes"}, pd.DataFrame(), pd.DataFrame()
        x_train, x_test = prepare_xy(train, test, features)
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(train[target_col].astype(str))
        y_test = encoder.transform(test[target_col].astype(str))
        model = RandomForestClassifier(
            n_estimators=280,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        baseline_label = train[target_col].astype(str).mode().iloc[0]
        baseline_pred = np.full(test.shape[0], encoder.transform([baseline_label])[0], dtype=int)
        metrics = {
            "ok": True,
            "task_type": "classification",
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "baseline_accuracy": float(accuracy_score(y_test, baseline_pred)),
            "baseline_macro_f1": float(f1_score(y_test, baseline_pred, average="macro")),
            "classes": int(len(classes)),
            "train_rows": int(train.shape[0]),
            "test_rows": int(test.shape[0]),
            "test_year_min": int(test["target_year"].min()),
            "test_year_max": int(test["target_year"].max()),
        }
        x_latest, _ = prepare_xy(latest, latest, features)
        latest_pred = model.predict(x_latest)
        forecast = latest.loc[:, ["iso3", "year"]].copy()
        forecast["forecast_target_year"] = forecast["year"].astype(int) + 1
        forecast[f"pred_next_{target}"] = encoder.inverse_transform(latest_pred)
        if hasattr(model, "predict_proba"):
            forecast[f"pred_next_{target}_confidence"] = model.predict_proba(x_latest).max(axis=1)
        importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        return metrics, forecast, importance


    def train_regression(
        train: pd.DataFrame,
        test: pd.DataFrame,
        latest: pd.DataFrame,
        target: str,
        features: list[str],
    ) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
        target_col = f"target_next_{target}"
        train = train.dropna(subset=[target_col]).copy()
        test = test.dropna(subset=[target_col]).copy()
        if train.empty or test.empty:
            return {"ok": False, "reason": "empty_train_or_test"}, pd.DataFrame(), pd.DataFrame()
        y_train = pd.to_numeric(train[target_col], errors="coerce")
        y_test = pd.to_numeric(test[target_col], errors="coerce")
        valid_train = y_train.notna()
        valid_test = y_test.notna()
        train = train.loc[valid_train].copy()
        test = test.loc[valid_test].copy()
        y_train = y_train.loc[valid_train]
        y_test = y_test.loc[valid_test]
        x_train, x_test = prepare_xy(train, test, features)
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        persistence = pd.to_numeric(test.get(target), errors="coerce").fillna(y_train.median()).to_numpy()
        mse = float(mean_squared_error(y_test, pred))
        persistence_mse = float(mean_squared_error(y_test, persistence))
        metrics = {
            "ok": True,
            "task_type": "regression",
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
            "baseline_persistence_rmse": float(np.sqrt(persistence_mse)),
            "baseline_persistence_mae": float(mean_absolute_error(y_test, persistence)),
            "baseline_persistence_r2": float(r2_score(y_test, persistence)),
            "train_rows": int(train.shape[0]),
            "test_rows": int(test.shape[0]),
            "test_year_min": int(test["target_year"].min()),
            "test_year_max": int(test["target_year"].max()),
        }
        x_latest, _ = prepare_xy(latest, latest, features)
        forecast = latest.loc[:, ["iso3", "year"]].copy()
        forecast["forecast_target_year"] = forecast["year"].astype(int) + 1
        forecast[f"pred_next_{target}"] = model.predict(x_latest)
        importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        return metrics, forecast, importance


    def run_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
        supervised = build_supervised_panel(panel)
        features = feature_columns(supervised)
        train = supervised.loc[supervised["target_year"] <= 2018].copy()
        test = supervised.loc[supervised["target_year"] >= 2019].copy()
        latest_year = int(panel["year"].max())
        latest = panel.loc[panel["year"] == latest_year].copy()
        metrics_rows: list[dict[str, object]] = []
        forecasts: list[pd.DataFrame] = []
        importances: list[pd.DataFrame] = []

        for spec in TARGET_SPECS:
            target = spec["target_column"]
            if target not in panel.columns or f"target_next_{target}" not in supervised.columns:
                metrics_rows.append({**spec, "ok": False, "reason": "target_missing"})
                continue
            if spec["task_type"] == "classification":
                metrics, forecast, importance = train_classification(train, test, latest, target, features)
            else:
                metrics, forecast, importance = train_regression(train, test, latest, target, features)
            metrics_rows.append({**spec, **metrics})
            if not forecast.empty:
                forecasts.append(forecast)
            if not importance.empty:
                importance["module"] = spec["module"]
                importance["target_column"] = target
                importance["target_label"] = spec["target_label"]
                importances.append(importance)

        forecast_out = latest.loc[:, ["iso3", "year"]].copy()
        forecast_out["forecast_target_year"] = latest_year + 1
        for forecast in forecasts:
            value_cols = [c for c in forecast.columns if c not in {"iso3", "year", "forecast_target_year"}]
            forecast_out = forecast_out.merge(forecast.loc[:, ["iso3", *value_cols]], on="iso3", how="left")
        metrics_df = pd.DataFrame(metrics_rows)
        importance_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
        if not importance_df.empty:
            importance_df = importance_df.sort_values(["target_column", "importance"], ascending=[True, False], kind="stable")
        summary = {
            "latest_year": latest_year,
            "feature_count": int(len(features)),
            "train_rows": int(train.shape[0]),
            "test_rows": int(test.shape[0]),
            "test_target_year_min": int(test["target_year"].min()) if not test.empty else None,
            "test_target_year_max": int(test["target_year"].max()) if not test.empty else None,
        }
        return metrics_df, forecast_out, importance_df, summary


    def plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
        if metrics.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        cls = metrics.loc[metrics["task_type"].eq("classification") & metrics["ok"].astype(bool)].copy()
        reg = metrics.loc[metrics["task_type"].eq("regression") & metrics["ok"].astype(bool)].copy()
        if not cls.empty:
            x = np.arange(cls.shape[0])
            axes[0].bar(x - 0.18, cls["accuracy"], width=0.36, label="model", color="#0b6b57")
            axes[0].bar(x + 0.18, cls["baseline_accuracy"], width=0.36, label="baseline", color="#b8c2cc")
            axes[0].set_xticks(x, labels=cls["target_label"], rotation=20, ha="right")
            axes[0].set_ylim(0, 1)
            axes[0].set_title(choose_text("分类预测准确率", "Classification Accuracy", USE_CHINESE))
            axes[0].legend(frameon=False)
        if not reg.empty:
            x = np.arange(reg.shape[0])
            axes[1].bar(x - 0.18, reg["mae"], width=0.36, label="model MAE", color="#4f7cac")
            axes[1].bar(x + 0.18, reg["baseline_persistence_mae"], width=0.36, label="persistence MAE", color="#b8c2cc")
            axes[1].set_xticks(x, labels=reg["target_label"], rotation=20, ha="right")
            axes[1].set_title(choose_text("回归预测误差", "Regression Error", USE_CHINESE))
            axes[1].legend(frameon=False)
        for ax in axes:
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
        parser = argparse.ArgumentParser(description="Run predictive validation for modules A/B/C.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        panel = load_panel(project_root)
        metrics, forecast, importance, summary_extra = run_models(panel)

        metrics_path = report_asset_path(report_dir, "abc_predictive_validation_metrics.csv")
        forecast_path = report_asset_path(report_dir, "abc_next_year_country_forecast_latest.csv")
        importance_path = report_asset_path(report_dir, "abc_predictive_feature_importance.csv")
        summary_path = report_asset_path(report_dir, "abc_predictive_validation_summary.json")
        figure_path = figure_dir / "advanced_abc_predictive_validation.png"

        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        forecast.to_csv(forecast_path, index=False, encoding="utf-8-sig")
        importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
        plot_metrics(metrics, figure_path)

        summary = {
            "project_root": project_root.as_posix(),
            "validation_layer": "ABC_next_year_predictive_validation",
            "definition": "用t年全球面板指标预测t+1年的A脆弱性类型、B风险压力、C适配缺口和响应诊断类型；用于证明A/B/C不仅能描述，还能做短期预警。",
            **summary_extra,
            "metrics": metrics.to_dict(orient="records"),
            "claim_boundary": "预测模型是时序外推验证，不是因果识别；它补的是A/B/C的预警能力和模型泛化证据。",
            "output_files": {
                "abc_predictive_validation_metrics": metrics_path.as_posix(),
                "abc_next_year_country_forecast_latest": forecast_path.as_posix(),
                "abc_predictive_feature_importance": importance_path.as_posix(),
                "abc_predictive_validation_summary": summary_path.as_posix(),
                "advanced_abc_predictive_validation": figure_path.as_posix(),
            },
        }
        clean_summary = json_clean(summary)
        summary_path.write_text(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_abc_predictive_robustness():
    __name__ = 'run_abc_predictive_robustness'
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
    from run_abc_predictive_validation import (
        TARGET_SPECS,
        build_supervised_panel,
        feature_columns,
        load_panel,
        train_classification,
        train_regression,
    )


    USE_CHINESE = configure_matplotlib_fonts()

    SYNTHETIC_SCORE_COLUMNS = {
        "vulnerability_type_code",
        "overall_vulnerability_score",
        "burden_pressure_score",
        "risk_pressure_score",
        "combined_pressure_score",
        "response_score",
        "adaptation_gap_score",
        "resource_shortage_score",
        "resource_response_score",
        "gap_percentile",
    }

    SYNTHETIC_SCORE_PATTERNS = (
        "_score",
        "_type_code",
        "_percentile",
    )

    RAW_DOMAIN_PREFIXES = (
        "dbd_",
        "gbd_rate_",
        "beds_",
        "doctors_",
        "nurses_",
        "who_",
        "wdi_",
        "ncd_",
        "population",
        "age",
        "uhc_",
        "current_health",
        "government_health",
        "out_of_pocket",
        "che_",
        "oop_",
    )


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def hardened_features(features: list[str], mode: str) -> list[str]:
        if mode == "full_current_features":
            return features
        if mode == "no_current_synthetic_scores":
            return [
                column
                for column in features
                if column not in SYNTHETIC_SCORE_COLUMNS and not any(column.endswith(pattern) for pattern in SYNTHETIC_SCORE_PATTERNS)
            ]
        if mode == "raw_domain_only":
            return [
                column
                for column in features
                if column.startswith(RAW_DOMAIN_PREFIXES)
                and column not in SYNTHETIC_SCORE_COLUMNS
                and not any(column.endswith(pattern) for pattern in SYNTHETIC_SCORE_PATTERNS)
            ]
        raise ValueError(mode)


    def run_variant(panel: pd.DataFrame, variant: str, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        supervised = build_supervised_panel(panel)
        train = supervised.loc[supervised["target_year"] <= 2018].copy()
        test = supervised.loc[supervised["target_year"] >= 2019].copy()
        latest = panel.loc[panel["year"].eq(panel["year"].max())].copy()
        metrics_rows: list[dict[str, object]] = []
        importance_rows: list[pd.DataFrame] = []

        for spec in TARGET_SPECS:
            target = spec["target_column"]
            if target not in panel.columns or f"target_next_{target}" not in supervised.columns:
                metrics_rows.append({**spec, "variant": variant, "ok": False, "reason": "target_missing", "feature_count": len(features)})
                continue
            if len(features) < 3:
                metrics_rows.append({**spec, "variant": variant, "ok": False, "reason": "too_few_features", "feature_count": len(features)})
                continue
            if spec["task_type"] == "classification":
                metrics, _, importance = train_classification(train, test, latest, target, features)
            else:
                metrics, _, importance = train_regression(train, test, latest, target, features)
            metrics_rows.append({**spec, "variant": variant, "feature_count": len(features), **metrics})
            if not importance.empty:
                importance["variant"] = variant
                importance["module"] = spec["module"]
                importance["target_column"] = target
                importance["target_label"] = spec["target_label"]
                importance_rows.append(importance)

        metrics = pd.DataFrame(metrics_rows)
        importances = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()
        return metrics, importances


    def add_degradation(metrics: pd.DataFrame) -> pd.DataFrame:
        if metrics.empty:
            return metrics
        out = metrics.copy()
        full = out.loc[out["variant"].eq("full_current_features")].copy()
        baselines = {}
        for _, row in full.iterrows():
            key = row["target_column"]
            if row["task_type"] == "classification":
                baselines[key] = pd.to_numeric(pd.Series([row.get("macro_f1")]), errors="coerce").iloc[0]
            else:
                baselines[key] = pd.to_numeric(pd.Series([row.get("mae")]), errors="coerce").iloc[0]
        degradation = []
        for _, row in out.iterrows():
            base = baselines.get(row["target_column"], np.nan)
            if row["task_type"] == "classification":
                value = pd.to_numeric(pd.Series([row.get("macro_f1")]), errors="coerce").iloc[0]
                degradation.append(float(base - value) if pd.notna(base) and pd.notna(value) else np.nan)
            else:
                value = pd.to_numeric(pd.Series([row.get("mae")]), errors="coerce").iloc[0]
                degradation.append(float(value - base) if pd.notna(base) and pd.notna(value) else np.nan)
        out["degradation_vs_full"] = degradation
        out["robustness_interpretation"] = np.where(
            out["variant"].eq("full_current_features"),
            "原始下一年预测验证，允许使用当前综合得分，存在自相关优势。",
            np.where(
                out["variant"].eq("no_current_synthetic_scores"),
                "去掉当前合成得分/分位数后的鲁棒性测试，主要依赖原始指标。",
                "仅使用原始领域变量的压力测试，适合答辩说明模型不是纯靠标签滞后。",
            ),
        )
        return out


    def plot_robustness(metrics: pd.DataFrame, path: Path) -> None:
        if metrics.empty:
            return
        ok = metrics.loc[metrics["ok"].astype(bool)].copy()
        if ok.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.4))
        cls = ok.loc[ok["task_type"].eq("classification")].copy()
        reg = ok.loc[ok["task_type"].eq("regression")].copy()
        label_map = {
            "full_current_features": "完整特征",
            "no_current_synthetic_scores": "去当前合成得分",
            "raw_domain_only": "仅原始领域变量",
        }
        if not cls.empty:
            pivot = cls.pivot_table(index="target_label", columns="variant", values="macro_f1", aggfunc="first")
            pivot = pivot.rename(columns=label_map)
            pivot.plot(kind="bar", ax=axes[0], color=["#0b6b57", "#4f7cac", "#f2b84b"])
            axes[0].set_title(choose_text("去自相关后分类Macro-F1", "Classification Macro-F1 After Hardening", USE_CHINESE))
            axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis="x", rotation=18, labelsize=10)
        if not reg.empty:
            pivot = reg.pivot_table(index="target_label", columns="variant", values="mae", aggfunc="first")
            pivot = pivot.rename(columns=label_map)
            pivot.plot(kind="bar", ax=axes[1], color=["#0b6b57", "#4f7cac", "#f2b84b"])
            axes[1].set_title(choose_text("去自相关后回归MAE", "Regression MAE After Hardening", USE_CHINESE))
            axes[1].tick_params(axis="x", rotation=18, labelsize=10)
        legend_handles, legend_labels = [], []
        for ax in axes:
            ax.grid(axis="y", alpha=0.22)
            ax.set_xlabel("")
            legend = ax.get_legend()
            if legend is not None:
                if not legend_handles:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()
                legend.remove()
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=min(3, len(legend_labels)),
                frameon=False,
                fontsize=10,
                bbox_to_anchor=(0.5, 0.01),
            )
        fig.tight_layout(rect=(0, 0.13, 1, 1))
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
        parser = argparse.ArgumentParser(description="Run leakage-hardened predictive robustness tests for ABC modules.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        report_dir = project_root / "06_report_assets"
        figure_dir = project_root / "05_figures"
        report_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        panel = load_panel(project_root)
        supervised = build_supervised_panel(panel)
        base_features = feature_columns(supervised)
        variants = {
            "full_current_features": hardened_features(base_features, "full_current_features"),
            "no_current_synthetic_scores": hardened_features(base_features, "no_current_synthetic_scores"),
            "raw_domain_only": hardened_features(base_features, "raw_domain_only"),
        }

        metric_frames = []
        importance_frames = []
        for variant, features in variants.items():
            metrics, importance = run_variant(panel, variant, features)
            metric_frames.append(metrics)
            if not importance.empty:
                importance_frames.append(importance)
        metrics_all = add_degradation(pd.concat(metric_frames, ignore_index=True))
        importances = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
        if not importances.empty:
            importances = importances.sort_values(["variant", "target_column", "importance"], ascending=[True, True, False], kind="stable")

        metrics_path = report_asset_path(report_dir, "abc_predictive_robustness_metrics.csv")
        importance_path = report_asset_path(report_dir, "abc_predictive_robustness_feature_importance.csv")
        summary_path = report_asset_path(report_dir, "abc_predictive_robustness_summary.json")
        figure_path = figure_dir / "advanced_abc_predictive_robustness.png"

        metrics_all.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        importances.to_csv(importance_path, index=False, encoding="utf-8-sig")
        plot_robustness(metrics_all, figure_path)

        raw_ok = metrics_all.loc[metrics_all["variant"].eq("raw_domain_only") & metrics_all["ok"].astype(bool)].copy()
        summary = {
            "project_root": project_root.as_posix(),
            "robustness_layer": "ABC leakage-hardened next-year predictive validation",
            "variant_feature_counts": {variant: len(features) for variant, features in variants.items()},
            "metric_rows": int(metrics_all.shape[0]),
            "raw_domain_only_successful_targets": int(raw_ok.shape[0]),
            "raw_domain_only_metrics": raw_ok.to_dict(orient="records"),
            "claim_boundary": "原始ABC预测验证有当前得分自相关优势；本脚本补充去合成得分和仅原始领域变量两档压力测试，用于证明预警能力不是完全靠标签滞后。",
            "output_files": {
                "abc_predictive_robustness_metrics": metrics_path.as_posix(),
                "abc_predictive_robustness_feature_importance": importance_path.as_posix(),
                "abc_predictive_robustness_summary": summary_path.as_posix(),
                "advanced_abc_predictive_robustness": figure_path.as_posix(),
            },
        }
        clean_summary = json_clean(summary)
        summary_path.write_text(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(clean_summary, ensure_ascii=False, indent=2, allow_nan=False))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_vulnerability_typology():
    __name__ = 'run_vulnerability_typology'
    import argparse
    import json
    from dataclasses import dataclass
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin, silhouette_score
    from sklearn.preprocessing import StandardScaler

    from foundation import (
        choose_text,
        configure_matplotlib_fonts,
        country_display_name,
        load_country_zh_labels,
        set_centered_suptitle,
    )
    from foundation import detect_project_root as shared_detect_project_root
    USE_CHINESE = configure_matplotlib_fonts()


    @dataclass(frozen=True)
    class FeatureSpec:
        column: str
        dimension: str
        direction: str
        concept_cn: str
        fallback_columns: tuple[str, ...] = ()
        transform: str = "identity"


    @dataclass(frozen=True)
    class ResolvedFeature:
        analysis_column: str
        source_column: str
        dimension: str
        direction: str
        concept_cn: str
        transform: str


    FEATURE_SPECS = [
        FeatureSpec(
            "gbd_rate_cardiovascular_diseases_per100k",
            "health_burden",
            "high",
            "心血管疾病负担（每10万人）",
            fallback_columns=("gbd_rate_cardiovascular_diseases",),
            transform="log1p",
        ),
        FeatureSpec(
            "gbd_rate_chronic_respiratory_diseases_per100k",
            "health_burden",
            "high",
            "慢性呼吸系统疾病负担（每10万人）",
            fallback_columns=("gbd_rate_chronic_respiratory_diseases",),
            transform="log1p",
        ),
        FeatureSpec(
            "gbd_rate_neoplasms_per100k",
            "health_burden",
            "high",
            "肿瘤负担（每10万人）",
            fallback_columns=("gbd_rate_neoplasms",),
            transform="log1p",
        ),
        FeatureSpec(
            "gbd_rate_diabetes_kidney_per100k",
            "health_burden",
            "high",
            "糖尿病与肾病负担（每10万人）",
            fallback_columns=("gbd_rate_diabetes_kidney",),
            transform="log1p",
        ),
        FeatureSpec(
            "dbd_smoking_per100k",
            "risk_exposure",
            "high",
            "吸烟风险暴露（每10万人）",
            fallback_columns=("dbd_smoking",),
            transform="log1p",
        ),
        FeatureSpec(
            "dbd_pm25_per100k",
            "risk_exposure",
            "high",
            "PM2.5 风险暴露（每10万人）",
            fallback_columns=("dbd_pm25",),
            transform="log1p",
        ),
        FeatureSpec(
            "dbd_high_bmi_per100k",
            "risk_exposure",
            "high",
            "高 BMI 风险暴露（每10万人）",
            fallback_columns=("dbd_high_bmi",),
            transform="log1p",
        ),
        FeatureSpec(
            "dbd_high_glucose_per100k",
            "risk_exposure",
            "high",
            "高血糖风险暴露（每10万人）",
            fallback_columns=("dbd_high_glucose",),
            transform="log1p",
        ),
        FeatureSpec(
            "dbd_high_sbp_per100k",
            "risk_exposure",
            "high",
            "高收缩压风险暴露（每10万人）",
            fallback_columns=("dbd_high_sbp",),
            transform="log1p",
        ),
        FeatureSpec(
            "dbd_dietary_risks_per100k",
            "risk_exposure",
            "high",
            "膳食风险暴露（每10万人）",
            fallback_columns=("dbd_dietary_risks",),
            transform="log1p",
        ),
        FeatureSpec("beds_10k", "system_fragility", "low", "病床密度"),
        FeatureSpec("doctors_10k", "system_fragility", "low", "医生密度"),
        FeatureSpec("nurses_10k", "system_fragility", "low", "护士密度"),
        FeatureSpec("uhc_index", "system_fragility", "low", "全民健康覆盖指数"),
        FeatureSpec("che_pct_gdp", "system_fragility", "low", "卫生支出占 GDP 比重"),
        FeatureSpec("che_pc_usd", "system_fragility", "low", "人均卫生支出", transform="log1p"),
        FeatureSpec("govt_he_pct", "system_fragility", "low", "政府卫生支出占比"),
        FeatureSpec("ext_he_pct", "system_fragility", "high", "外部卫生支出依赖"),
        FeatureSpec("wdi_health_expenditure_per_capita", "system_fragility", "low", "WDI 人均卫生支出", transform="log1p"),
        FeatureSpec("wdi_government_health_expenditure_pct", "system_fragility", "low", "WDI 政府卫生支出占比"),
        FeatureSpec("wdi_external_health_expenditure_pct", "system_fragility", "high", "WDI 外部卫生支出占比"),
        FeatureSpec("wdi_hospital_beds", "system_fragility", "low", "WDI 病床密度"),
        FeatureSpec("wdi_physicians", "system_fragility", "low", "WDI 医生密度"),
        FeatureSpec("wdi_nurses_midwives", "system_fragility", "low", "WDI 护士和助产士密度"),
        FeatureSpec("wdi_out_of_pocket_pct", "system_fragility", "high", "WDI 自付卫生支出占比"),
        FeatureSpec("wb_hnp_hci", "system_fragility", "low", "人力资本指数"),
        FeatureSpec("wb_hnp_skilled_birth_attendance", "system_fragility", "low", "专业接生覆盖率"),
        FeatureSpec("wb_hnp_immunization_dpt", "system_fragility", "low", "DPT 免疫覆盖率"),
        FeatureSpec("wb_hnp_immunization_measles", "system_fragility", "low", "麻疹免疫覆盖率"),
        FeatureSpec("wb_hnp_immunization_hepb3", "system_fragility", "low", "乙肝免疫覆盖率"),
        FeatureSpec("hdi", "socioeconomic_fragility", "low", "人类发展指数"),
        FeatureSpec("wdi_gdp_per_capita", "socioeconomic_fragility", "low", "人均 GDP", transform="log1p"),
        FeatureSpec("wdi_gini", "socioeconomic_fragility", "high", "基尼系数"),
        FeatureSpec("wdi_population_65_plus_pct", "socioeconomic_fragility", "high", "65 岁及以上人口占比"),
        FeatureSpec("wdi_urban_population_pct", "socioeconomic_fragility", "low", "城镇化率"),
    ]

    DIMENSION_ORDER = [
        "health_burden",
        "risk_exposure",
        "system_fragility",
        "socioeconomic_fragility",
    ]
    DIMENSION_CN = {
        "health_burden": "健康负担",
        "risk_exposure": "风险暴露",
        "system_fragility": "系统脆弱性",
        "socioeconomic_fragility": "社会经济脆弱性",
    }
    DIMENSION_EN = {
        "health_burden": "Health burden",
        "risk_exposure": "Risk exposure",
        "system_fragility": "System fragility",
        "socioeconomic_fragility": "Socioeconomic fragility",
    }
    DIMENSION_SCORE_COLUMNS = [f"{dimension}_score" for dimension in DIMENSION_ORDER]


    BASE_LABELS_EN = {
        "相对稳健型": "Relatively resilient",
        "高负担高风险低承载型": "High burden / high risk / low capacity",
        "低缓冲低承载型": "Low buffer / low capacity",
        "风险暴露前移型": "Risk exposure leading",
        "高负担转型压力型": "High-burden transition pressure",
        "过渡脆弱型": "Transitional vulnerability",
    }

    FINAL_LABEL_SCHEME_3 = {
        1: "类型1-高负担转型承压型",
        2: "类型2-低缓冲低承载型",
        3: "类型3-相对稳健型",
    }

    FINAL_LABEL_SCHEME_3_EN = {
        "类型1-高负担转型承压型": "Type 1 - High-burden transition pressure",
        "类型2-低缓冲低承载型": "Type 2 - Low buffer / low capacity",
        "类型3-相对稳健型": "Type 3 - Relatively resilient",
    }


    def dimension_display_name(dimension: str) -> str:
        return DIMENSION_CN[dimension] if USE_CHINESE else DIMENSION_EN[dimension]


    def vulnerability_display_label(label: str) -> str:
        if USE_CHINESE or "-" not in label:
            return label
        if label in FINAL_LABEL_SCHEME_3_EN:
            return FINAL_LABEL_SCHEME_3_EN[label]
        prefix, base = label.split("-", 1)
        suffix = ""
        if "_" in base:
            base, suffix = base.split("_", 1)
            suffix = f"_{suffix}"
        return f"{prefix}-{BASE_LABELS_EN.get(base, base)}{suffix}"


    def apply_final_label_scheme(label_map: dict[int, str], chosen_k: int) -> dict[int, str]:
        if chosen_k == 3:
            return {code: FINAL_LABEL_SCHEME_3.get(code, label) for code, label in label_map.items()}
        return label_map


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


    def parse_cluster_values(raw: str) -> list[int]:
        values = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            value = int(part)
            if value < 2:
                raise ValueError("Cluster count must be >= 2")
            values.append(value)
        unique = sorted(set(values))
        if not unique:
            raise ValueError("At least one cluster count must be provided")
        return unique


    def resolve_feature_specs(df: pd.DataFrame) -> tuple[list[ResolvedFeature], list[str]]:
        resolved: list[ResolvedFeature] = []
        missing: list[str] = []
        for spec in FEATURE_SPECS:
            candidates = (spec.column,) + spec.fallback_columns
            selected = next((column for column in candidates if column in df.columns), None)
            if selected is None:
                missing.append(spec.column)
                continue
            resolved.append(
                ResolvedFeature(
                    analysis_column=spec.column,
                    source_column=selected,
                    dimension=spec.dimension,
                    direction=spec.direction,
                    concept_cn=spec.concept_cn,
                    transform=spec.transform,
                )
            )
        return resolved, missing


    def filter_sparse_feature_specs(
        df: pd.DataFrame,
        specs: list[ResolvedFeature],
        threshold: float,
    ) -> tuple[list[ResolvedFeature], list[dict[str, object]], dict[str, list[str]]]:
        retained: list[ResolvedFeature] = []
        dropped: list[dict[str, object]] = []
        retained_by_dimension: dict[str, list[str]] = {dimension: [] for dimension in DIMENSION_ORDER}
        for spec in specs:
            numeric = pd.to_numeric(df[spec.source_column], errors="coerce")
            missing_rate = float(numeric.isna().mean())
            if missing_rate > threshold:
                dropped.append(
                    {
                        "analysis_column": spec.analysis_column,
                        "source_column": spec.source_column,
                        "dimension": spec.dimension,
                        "dimension_cn": DIMENSION_CN[spec.dimension],
                        "missing_rate": missing_rate,
                    }
                )
                continue
            retained.append(spec)
            retained_by_dimension[spec.dimension].append(spec.analysis_column)
        return retained, dropped, retained_by_dimension


    def winsorize_series(series: pd.Series, lower_quantile: float, upper_quantile: float) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        mask = numeric.notna()
        if not mask.any():
            return numeric.astype("float64")
        lower = numeric.loc[mask].quantile(lower_quantile)
        upper = numeric.loc[mask].quantile(upper_quantile)
        return numeric.clip(lower=lower, upper=upper).astype("float64")


    def apply_feature_transform(series: pd.Series, transform: str, enable_log_transform: bool) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        if transform == "log1p" and enable_log_transform:
            numeric = numeric.where(numeric >= 0)
            return np.log1p(numeric)
        return numeric


    def safe_zscore(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        output = pd.Series(np.nan, index=series.index, dtype="float64")
        mask = numeric.notna()
        if not mask.any():
            return output
        std = numeric[mask].std(ddof=0)
        if pd.isna(std) or std == 0:
            output.loc[mask] = 0.0
            return output
        mean = numeric[mask].mean()
        output.loc[mask] = (numeric.loc[mask] - mean) / std
        return output


    def bootstrap_stability(
        matrix: np.ndarray,
        base_labels: np.ndarray,
        n_clusters: int,
        iterations: int,
        random_state: int,
    ) -> tuple[float, float]:
        rng = np.random.default_rng(random_state)
        scores: list[float] = []
        n_rows = matrix.shape[0]
        for _ in range(iterations):
            sample_idx = rng.integers(0, n_rows, size=n_rows)
            sampled = matrix[sample_idx]
            model = KMeans(n_clusters=n_clusters, random_state=int(rng.integers(0, 1_000_000)), n_init=20)
            model.fit(sampled)
            predicted = pairwise_distances_argmin(matrix, model.cluster_centers_)
            scores.append(adjusted_rand_score(base_labels, predicted))
        return float(np.mean(scores)), float(np.std(scores, ddof=0))


    def make_cluster_base_label(center_row: pd.Series) -> str:
        burden = center_row["health_burden_score"]
        risk = center_row["risk_exposure_score"]
        system = center_row["system_fragility_score"]
        socio = center_row["socioeconomic_fragility_score"]
        overall = center_row[DIMENSION_SCORE_COLUMNS].mean()

        if overall <= -0.2 and system <= 0 and socio <= 0:
            return "相对稳健型"
        if burden >= 0.35 and risk >= 0.25 and system >= 0.2:
            return "高负担高风险低承载型"
        if system >= 0.35 and socio >= 0.2:
            return "低缓冲低承载型"
        if risk >= 0.35 and burden < 0.25:
            return "风险暴露前移型"
        if burden >= 0.35:
            return "高负担转型压力型"
        return "过渡脆弱型"


    def build_unique_cluster_labels(centers: pd.DataFrame) -> dict[int, str]:
        seen: dict[str, int] = {}
        label_map: dict[int, str] = {}
        for cluster_code, row in centers.iterrows():
            base_label = make_cluster_base_label(row)
            seen[base_label] = seen.get(base_label, 0) + 1
            suffix = f"_{seen[base_label]}" if seen[base_label] > 1 else ""
            label_map[int(cluster_code)] = f"类型{int(cluster_code)}-{base_label}{suffix}"
        return label_map


    def plot_cluster_selection(evaluation_df: pd.DataFrame, chosen_k: int, output_path: Path) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        axes[0].plot(evaluation_df["n_clusters"], evaluation_df["silhouette_score"], marker="o", color="#457b9d")
        axes[0].axvline(chosen_k, color="#e76f51", linestyle="--", linewidth=1.5)
        axes[0].set_title("Silhouette")
        axes[0].set_xlabel(choose_text("聚类数", "Number of clusters", USE_CHINESE))

        axes[1].plot(evaluation_df["n_clusters"], evaluation_df["bootstrap_stability_mean"], marker="o", color="#2a9d8f")
        axes[1].axvline(chosen_k, color="#e76f51", linestyle="--", linewidth=1.5)
        axes[1].set_title(choose_text("Bootstrap 稳定性", "Bootstrap stability", USE_CHINESE))
        axes[1].set_xlabel(choose_text("聚类数", "Number of clusters", USE_CHINESE))

        axes[2].plot(evaluation_df["n_clusters"], evaluation_df["kmeans_hierarchical_ari"], marker="o", color="#8d99ae")
        axes[2].axvline(chosen_k, color="#e76f51", linestyle="--", linewidth=1.5)
        axes[2].set_title("KMeans vs Hierarchical ARI")
        axes[2].set_xlabel(choose_text("聚类数", "Number of clusters", USE_CHINESE))

        for ax in axes:
            ax.grid(alpha=0.2)
        set_centered_suptitle(fig, choose_text("健康脆弱性分型聚类数选择", "Cluster count selection for vulnerability typology", USE_CHINESE), y=0.99)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_latest_type_counts(latest_df: pd.DataFrame, output_path: Path) -> None:
        counts = latest_df["vulnerability_type_label"].value_counts()
        latest_year = int(pd.to_numeric(latest_df["year"], errors="coerce").dropna().max())
        fig, ax = plt.subplots(figsize=(10, 5.5))
        display_labels = [vulnerability_display_label(label) for label in counts.index]
        bars = ax.bar(display_labels, counts.values, color="#457b9d")
        ax.set_title(choose_text(f"{latest_year}年健康脆弱性类型分布", f"Distribution of vulnerability types, {latest_year}", USE_CHINESE))
        ax.set_ylabel(choose_text("国家数量", "Number of countries", USE_CHINESE))
        ax.bar_label(bars, padding=3)
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_pca_scatter(latest_df: pd.DataFrame, output_path: Path) -> None:
        latest_year = int(pd.to_numeric(latest_df["year"], errors="coerce").dropna().max())
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, subset in latest_df.groupby("vulnerability_type_label", dropna=True):
            ax.scatter(subset["pc1"], subset["pc2"], label=vulnerability_display_label(label), alpha=0.7, s=28)
        ax.set_title(choose_text(f"健康脆弱性分型 PCA 散点图（{latest_year}年）", f"PCA scatter of vulnerability types ({latest_year})", USE_CHINESE))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_dimension_profiles(profile_df: pd.DataFrame, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(profile_df.shape[0])
        width = 0.18
        for idx, dimension in enumerate(DIMENSION_SCORE_COLUMNS):
            ax.bar(
                x + idx * width,
                profile_df[dimension].values,
                width=width,
                label=dimension_display_name(dimension.replace("_score", "")),
            )
        ax.set_xticks(x + width * (len(DIMENSION_SCORE_COLUMNS) - 1) / 2)
        ax.set_xticklabels([vulnerability_display_label(label) for label in profile_df["vulnerability_type_label"]], rotation=20, ha="right")
        ax.set_ylabel(choose_text("维度均值（标准化后）", "Mean dimension score (standardized)", USE_CHINESE))
        ax.set_title(choose_text("不同脆弱性类型的四维画像", "Four-dimension profile by vulnerability type", USE_CHINESE))
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_transition_trends(transition_df: pd.DataFrame, output_path: Path) -> None:
        pivot = transition_df.pivot(index="year", columns="vulnerability_type_label", values="records").fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        for column in pivot.columns:
            ax.plot(pivot.index.astype(int), pivot[column], linewidth=2, label=vulnerability_display_label(column))
        ax.set_title(choose_text("健康脆弱性类型年度变化", "Annual transition in vulnerability types", USE_CHINESE))
        ax.set_xlabel(choose_text("年份", "Year", USE_CHINESE))
        ax.set_ylabel(choose_text("国家-年份记录数", "Country-year records", USE_CHINESE))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_transition_share_trends(share_df: pd.DataFrame, output_path: Path) -> None:
        pivot = share_df.pivot(index="year", columns="vulnerability_type_label", values="share").fillna(0.0)
        if pivot.empty:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        for column in pivot.columns:
            ax.plot(pivot.index.astype(int), pivot[column], linewidth=2, label=vulnerability_display_label(column))
        ax.set_title(choose_text("健康脆弱性类型年度占比变化", "Annual share of vulnerability types", USE_CHINESE))
        ax.set_xlabel(choose_text("年份", "Year", USE_CHINESE))
        ax.set_ylabel(choose_text("类型占比", "Type share", USE_CHINESE))
        ax.set_ylim(0, min(1.0, pivot.to_numpy().max() * 1.1 if pivot.to_numpy().size else 1.0))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_transition_matrix_heatmap(transition_probability_df: pd.DataFrame, output_path: Path) -> None:
        heatmap = transition_probability_df.pivot(
            index="vulnerability_type_label",
            columns="next_type",
            values="transition_probability",
        ).fillna(0.0)
        if heatmap.empty:
            return
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        im = ax.imshow(heatmap.values, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(np.arange(heatmap.shape[1]))
        ax.set_xticklabels([vulnerability_display_label(label) for label in heatmap.columns], rotation=25, ha="right")
        ax.set_yticks(np.arange(heatmap.shape[0]))
        ax.set_yticklabels([vulnerability_display_label(label) for label in heatmap.index])
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                ax.text(j, i, f"{heatmap.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
        ax.set_title(choose_text("健康脆弱性类型转移概率矩阵", "Transition probability matrix of vulnerability types", USE_CHINESE))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(choose_text("转移概率（颜色越深概率越高）", "Transition probability (darker means higher)", USE_CHINESE))
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_representative_countries(representative_df: pd.DataFrame, output_path: Path) -> None:
        subset = representative_df.head(12).copy()
        if subset.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 7))
        country_labels = load_country_zh_labels(output_path.parents[1])
        labels = subset["iso3"].map(lambda code: country_display_name(code, country_labels)) + " | " + subset["vulnerability_type_label"].map(vulnerability_display_label)
        ax.barh(labels, subset["distance_to_cluster_center"], color="#2a9d8f")
        ax.invert_yaxis()
        ax.set_title(choose_text("代表国家画像卡候选（越靠上越接近类型中心）", "Representative country candidates (closest to cluster center)", USE_CHINESE))
        ax.set_xlabel(choose_text("到类型中心的距离", "Distance to cluster center", USE_CHINESE))
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def build_type_storylines(cluster_profiles: pd.DataFrame, representative_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for _, profile in cluster_profiles.sort_values("vulnerability_type_code", kind="stable").iterrows():
            type_code = int(profile["vulnerability_type_code"])
            type_label = str(profile["vulnerability_type_label"])
            dimension_scores = {
                "health_burden_score": float(profile["health_burden_score"]),
                "risk_exposure_score": float(profile["risk_exposure_score"]),
                "system_fragility_score": float(profile["system_fragility_score"]),
                "socioeconomic_fragility_score": float(profile["socioeconomic_fragility_score"]),
            }
            strongest_dimensions = sorted(dimension_scores.items(), key=lambda item: item[1], reverse=True)
            lead_dims = [dimension_display_name(name.replace("_score", "")) for name, _ in strongest_dimensions[:2]]
            support_dims = [dimension_display_name(name.replace("_score", "")) for name, _ in strongest_dimensions[2:]]
            representatives = (
                representative_df.loc[representative_df["vulnerability_type_code"] == type_code, "iso3"]
                .head(3)
                .astype(str)
                .tolist()
            )
            rows.append(
                {
                    "vulnerability_type_code": type_code,
                    "vulnerability_type_label": type_label,
                    "primary_pressure_dimensions": " / ".join(lead_dims),
                    "secondary_dimensions": " / ".join(support_dims),
                    "representative_countries": " / ".join(representatives),
                    "overall_vulnerability_score": float(profile["overall_vulnerability_score"]),
                }
            )
        return pd.DataFrame(rows)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Run formal vulnerability typology analysis for Module A.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--input-file", type=Path, default=None, help="Optional explicit response panel path")
        parser.add_argument("--latest-year", type=int, default=None, help="Optional explicit latest year for summary exports")
        parser.add_argument("--cluster-values", type=str, default="3,4,5,6", help="Candidate cluster counts, comma-separated")
        parser.add_argument("--n-clusters", type=int, default=3, help="Fixed cluster count; defaults to the locked 3-cluster main scheme")
        parser.add_argument("--bootstrap-iterations", type=int, default=40, help="Bootstrap iterations for KMeans stability")
        parser.add_argument("--random-state", type=int, default=42)
        parser.add_argument("--min-dimensions-observed", type=int, default=3, help="Minimum observed dimension scores required per row")
        parser.add_argument("--representatives-per-cluster", type=int, default=6, help="Number of representative countries to keep for each cluster")
        parser.add_argument("--winsor-lower", type=float, default=0.01, help="Lower winsorization quantile for continuous features")
        parser.add_argument("--winsor-upper", type=float, default=0.99, help="Upper winsorization quantile for continuous features")
        parser.add_argument("--sparse-feature-threshold", type=float, default=0.65, help="Drop features whose missingness is strictly above this threshold")
        parser.add_argument("--disable-log-transform", action="store_true", help="Disable log1p transform on configured skewed features")
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        input_file = args.input_file.expanduser().resolve() if args.input_file else dirs["simulation"] / "response_panel.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Response panel not found: {input_file}")

        df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)
        required = {"iso3", "year"}
        missing_required = sorted(required - set(df.columns))
        if missing_required:
            raise RuntimeError(f"Response panel is missing required columns: {missing_required}")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        if not (0.0 <= args.winsor_lower < args.winsor_upper <= 1.0):
            raise ValueError("Winsor quantiles must satisfy 0 <= lower < upper <= 1")
        if not (0.0 <= args.sparse_feature_threshold < 1.0):
            raise ValueError("sparse-feature-threshold must satisfy 0 <= threshold < 1")

        resolved_specs, missing_feature_columns = resolve_feature_specs(df)
        available_specs, dropped_sparse_features, retained_features_by_dimension = filter_sparse_feature_specs(
            df,
            resolved_specs,
            args.sparse_feature_threshold,
        )

        dimension_counts = {
            dimension: sum(1 for spec in available_specs if spec.dimension == dimension)
            for dimension in DIMENSION_ORDER
        }
        missing_dimensions = [dimension for dimension, count in dimension_counts.items() if count == 0]
        if missing_dimensions:
            raise RuntimeError(f"Missing full dimensions for typology analysis: {missing_dimensions}")

        source_columns = []
        for spec in available_specs:
            if spec.source_column not in source_columns:
                source_columns.append(spec.source_column)
        work_df = df.loc[:, ["iso3", "year"] + source_columns].copy()

        variable_manifest_rows: list[dict[str, object]] = []
        zscore_columns_by_dimension: dict[str, list[str]] = {dimension: [] for dimension in DIMENSION_ORDER}
        for spec in available_specs:
            numeric = pd.to_numeric(work_df[spec.source_column], errors="coerce")
            transformed = apply_feature_transform(numeric, spec.transform, enable_log_transform=not args.disable_log_transform)
            winsorized = winsorize_series(transformed, args.winsor_lower, args.winsor_upper)
            work_df[spec.analysis_column] = winsorized
            oriented = winsorized if spec.direction == "high" else -winsorized
            z_column = f"z__{spec.analysis_column}"
            work_df[z_column] = safe_zscore(oriented)
            zscore_columns_by_dimension[spec.dimension].append(z_column)
            variable_manifest_rows.append(
                {
                    "analysis_column": spec.analysis_column,
                    "source_column": spec.source_column,
                    "concept_cn": spec.concept_cn,
                    "dimension": spec.dimension,
                    "dimension_cn": DIMENSION_CN[spec.dimension],
                    "direction": spec.direction,
                    "transform": spec.transform if not args.disable_log_transform else "identity" if spec.transform == "log1p" else spec.transform,
                    "winsor_lower": args.winsor_lower,
                    "winsor_upper": args.winsor_upper,
                    "non_null_rows": int(numeric.notna().sum()),
                    "missing_rate": float(numeric.isna().mean()),
                }
            )

        for dimension in DIMENSION_ORDER:
            score_column = f"{dimension}_score"
            z_columns = zscore_columns_by_dimension[dimension]
            work_df[score_column] = work_df[z_columns].mean(axis=1)
            work_df[f"{dimension}_observed_vars"] = work_df[z_columns].notna().sum(axis=1).astype("Int64")

        work_df["dimensions_observed"] = work_df[DIMENSION_SCORE_COLUMNS].notna().sum(axis=1).astype("Int64")
        eligible_mask = work_df["dimensions_observed"] >= args.min_dimensions_observed
        eligible_df = work_df.loc[eligible_mask].copy()
        if eligible_df.empty:
            raise RuntimeError("No rows satisfy the minimum observed dimension threshold for clustering")

        dimension_impute_values = eligible_df[DIMENSION_SCORE_COLUMNS].median(skipna=True).fillna(0.0)
        feature_matrix_raw = eligible_df[DIMENSION_SCORE_COLUMNS].fillna(dimension_impute_values)
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix_raw)

        cluster_values = [args.n_clusters] if args.n_clusters else parse_cluster_values(args.cluster_values)
        evaluation_rows: list[dict[str, float]] = []
        for cluster_count in cluster_values:
            if feature_matrix.shape[0] <= cluster_count:
                continue
            kmeans = KMeans(n_clusters=cluster_count, random_state=args.random_state, n_init=30)
            kmeans_labels = kmeans.fit_predict(feature_matrix)
            silhouette = float(silhouette_score(feature_matrix, kmeans_labels))
            hierarchical = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward")
            hierarchical_labels = hierarchical.fit_predict(feature_matrix)
            between_method_ari = float(adjusted_rand_score(kmeans_labels, hierarchical_labels))
            bootstrap_mean, bootstrap_std = bootstrap_stability(
                feature_matrix,
                kmeans_labels,
                cluster_count,
                args.bootstrap_iterations,
                args.random_state,
            )
            composite_score = silhouette + 0.25 * bootstrap_mean + 0.15 * between_method_ari
            evaluation_rows.append(
                {
                    "n_clusters": cluster_count,
                    "silhouette_score": silhouette,
                    "kmeans_inertia": float(kmeans.inertia_),
                    "kmeans_hierarchical_ari": between_method_ari,
                    "bootstrap_stability_mean": bootstrap_mean,
                    "bootstrap_stability_std": bootstrap_std,
                    "selection_score": composite_score,
                }
            )

        evaluation_df = pd.DataFrame(evaluation_rows).sort_values("n_clusters", kind="stable").reset_index(drop=True)
        if evaluation_df.empty:
            raise RuntimeError("Unable to evaluate any cluster counts; check available data size")
        chosen_row = evaluation_df.sort_values(
            ["selection_score", "silhouette_score", "bootstrap_stability_mean"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
        chosen_k = int(chosen_row["n_clusters"])

        final_kmeans = KMeans(n_clusters=chosen_k, random_state=args.random_state, n_init=50)
        raw_labels = final_kmeans.fit_predict(feature_matrix)
        eligible_df["raw_cluster_code"] = raw_labels

        centers = eligible_df.groupby("raw_cluster_code", dropna=False)[DIMENSION_SCORE_COLUMNS].mean()
        order = centers.mean(axis=1).sort_values(ascending=False).index.tolist()
        code_map = {old_code: idx + 1 for idx, old_code in enumerate(order)}
        eligible_df["vulnerability_type_code"] = eligible_df["raw_cluster_code"].map(code_map).astype("Int64")

        centers = centers.loc[order].copy()
        centers.index = [code_map[idx] for idx in order]
        centers.index.name = "vulnerability_type_code"
        label_map = apply_final_label_scheme(build_unique_cluster_labels(centers), chosen_k)
        eligible_df["vulnerability_type_label"] = eligible_df["vulnerability_type_code"].map(label_map)
        centers["vulnerability_type_label"] = centers.index.map(label_map)
        centers["overall_vulnerability_score"] = centers[DIMENSION_SCORE_COLUMNS].mean(axis=1)

        pca = PCA(n_components=min(4, len(DIMENSION_SCORE_COLUMNS)), random_state=args.random_state)
        pca_scores = pca.fit_transform(feature_matrix)
        eligible_df["pc1"] = pca_scores[:, 0]
        eligible_df["pc2"] = pca_scores[:, 1] if pca_scores.shape[1] > 1 else 0.0
        eligible_df["overall_vulnerability_score"] = eligible_df[DIMENSION_SCORE_COLUMNS].mean(axis=1)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=DIMENSION_SCORE_COLUMNS,
            columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
        ).reset_index(names="dimension_score")
        explained = pd.DataFrame(
            {
                "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }
        )

        final_panel = work_df.copy()
        final_panel["vulnerability_type_code"] = pd.Series(pd.NA, index=final_panel.index, dtype="Int64")
        final_panel["vulnerability_type_label"] = pd.Series(pd.NA, index=final_panel.index, dtype="object")
        final_panel["pc1"] = np.nan
        final_panel["pc2"] = np.nan
        final_panel["overall_vulnerability_score"] = np.nan

        final_panel.loc[eligible_df.index, "vulnerability_type_code"] = eligible_df["vulnerability_type_code"].to_numpy()
        final_panel.loc[eligible_df.index, "vulnerability_type_label"] = eligible_df["vulnerability_type_label"].to_numpy()
        final_panel.loc[eligible_df.index, "pc1"] = eligible_df["pc1"].to_numpy()
        final_panel.loc[eligible_df.index, "pc2"] = eligible_df["pc2"].to_numpy()
        final_panel.loc[eligible_df.index, "overall_vulnerability_score"] = eligible_df["overall_vulnerability_score"].to_numpy()

        latest_country_labels = (
            final_panel.dropna(subset=["vulnerability_type_label"])
            .sort_values(["iso3", "year"], kind="stable")
            .groupby("iso3", as_index=False)
            .tail(1)
        )
        latest_year = args.latest_year if args.latest_year is not None else int(latest_country_labels["year"].dropna().max())
        latest_year_df = latest_country_labels.loc[latest_country_labels["year"] == latest_year].copy()

        cluster_profiles = (
            eligible_df.groupby(["vulnerability_type_code", "vulnerability_type_label"], dropna=False)[DIMENSION_SCORE_COLUMNS + ["overall_vulnerability_score"]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("vulnerability_type_code", kind="stable")
        )

        transition_summary = (
            final_panel.dropna(subset=["vulnerability_type_label"])
            .groupby(["year", "vulnerability_type_label"], dropna=False)
            .size()
            .rename("records")
            .reset_index()
        )
        type_share_by_year = transition_summary.copy()
        type_share_by_year["share"] = type_share_by_year.groupby("year", dropna=False)["records"].transform(
            lambda values: values / values.sum() if float(values.sum()) > 0 else 0.0
        )

        transitions = final_panel.loc[:, ["iso3", "year", "vulnerability_type_label"]].dropna().copy()
        transitions = transitions.sort_values(["iso3", "year"], kind="stable")
        transitions["next_type"] = transitions.groupby("iso3")["vulnerability_type_label"].shift(-1)
        transition_matrix = (
            transitions.dropna(subset=["next_type"])
            .groupby(["vulnerability_type_label", "next_type"], dropna=False)
            .size()
            .rename("transition_count")
            .reset_index()
        )
        transition_probability_matrix = transition_matrix.copy()
        if not transition_probability_matrix.empty:
            transition_probability_matrix["transition_probability"] = transition_probability_matrix.groupby(
                "vulnerability_type_label",
                dropna=False,
            )["transition_count"].transform(lambda values: values / values.sum() if float(values.sum()) > 0 else 0.0)

        type_distribution_latest = (
            latest_year_df["vulnerability_type_label"]
            .value_counts(dropna=False)
            .rename_axis("vulnerability_type_label")
            .reset_index(name="countries")
        )
        if not type_distribution_latest.empty:
            type_distribution_latest["share"] = type_distribution_latest["countries"] / float(type_distribution_latest["countries"].sum())

        representative_pool = latest_year_df.merge(
            eligible_df[["iso3", "year", "vulnerability_type_code"] + DIMENSION_SCORE_COLUMNS],
            on=["iso3", "year", "vulnerability_type_code"] + DIMENSION_SCORE_COLUMNS,
            how="left",
        )
        center_lookup = centers[DIMENSION_SCORE_COLUMNS].to_dict(orient="index")
        representative_pool["distance_to_cluster_center"] = representative_pool.apply(
            lambda row: float(
                np.linalg.norm(
                    np.array([row[column] for column in DIMENSION_SCORE_COLUMNS], dtype=float)
                    - np.array([center_lookup[int(row["vulnerability_type_code"])][column] for column in DIMENSION_SCORE_COLUMNS], dtype=float)
                )
            ),
            axis=1,
        )
        representative_df = (
            representative_pool.sort_values(["vulnerability_type_code", "distance_to_cluster_center"], kind="stable")
            .groupby("vulnerability_type_code", as_index=False)
            .head(args.representatives_per_cluster)
            .reset_index(drop=True)
        )
        representative_df["rank_within_cluster"] = representative_df.groupby("vulnerability_type_code").cumcount() + 1
        profile_card_columns = [
            "iso3",
            "year",
            "rank_within_cluster",
            "vulnerability_type_code",
            "vulnerability_type_label",
            "overall_vulnerability_score",
            "health_burden_score",
            "risk_exposure_score",
            "system_fragility_score",
            "socioeconomic_fragility_score",
            "distance_to_cluster_center",
            "gbd_rate_cardiovascular_diseases_per100k",
            "gbd_rate_chronic_respiratory_diseases_per100k",
            "gbd_rate_neoplasms_per100k",
            "gbd_rate_diabetes_kidney_per100k",
            "dbd_smoking_per100k",
            "dbd_pm25_per100k",
            "dbd_high_bmi_per100k",
            "dbd_high_glucose_per100k",
            "dbd_high_sbp_per100k",
            "dbd_dietary_risks_per100k",
            "beds_10k",
            "doctors_10k",
            "nurses_10k",
            "uhc_index",
            "che_pct_gdp",
            "hdi",
            "wdi_gdp_per_capita",
            "wdi_gini",
            "wdi_population_65_plus_pct",
            "wdi_urban_population_pct",
        ]
        profile_cards = representative_df.loc[:, [column for column in profile_card_columns if column in representative_df.columns]].copy()
        type_storylines = build_type_storylines(cluster_profiles, representative_df)

        output_panel = dirs["simulation"] / "vulnerability_typology_panel.csv"
        latest_output = report_asset_path(dirs["report"], "vulnerability_country_labels_latest.csv")
        evaluation_output = report_asset_path(dirs["report"], "vulnerability_cluster_evaluation.csv")
        pca_output = report_asset_path(dirs["report"], "vulnerability_pca_loadings.csv")
        explained_output = report_asset_path(dirs["report"], "vulnerability_pca_explained_variance.csv")
        profile_output = report_asset_path(dirs["report"], "vulnerability_cluster_profiles.csv")
        transition_output = report_asset_path(dirs["report"], "vulnerability_transition_summary.csv")
        transition_matrix_output = report_asset_path(dirs["report"], "vulnerability_transition_matrix.csv")
        transition_probability_output = report_asset_path(dirs["report"], "vulnerability_transition_probability_matrix.csv")
        latest_distribution_output = report_asset_path(dirs["report"], "vulnerability_type_distribution_latest.csv")
        share_by_year_output = report_asset_path(dirs["report"], "vulnerability_type_share_by_year.csv")
        representatives_output = report_asset_path(dirs["report"], "vulnerability_representative_countries.csv")
        profile_cards_output = report_asset_path(dirs["report"], "vulnerability_country_profile_cards.csv")
        type_storylines_output = report_asset_path(dirs["report"], "vulnerability_type_storylines.csv")
        variable_manifest_output = report_asset_path(dirs["report"], "vulnerability_variable_manifest.csv")

        final_panel.to_csv(output_panel, index=False, encoding="utf-8-sig")
        latest_country_labels.to_csv(latest_output, index=False, encoding="utf-8-sig")
        evaluation_df.to_csv(evaluation_output, index=False, encoding="utf-8-sig")
        loadings.to_csv(pca_output, index=False, encoding="utf-8-sig")
        explained.to_csv(explained_output, index=False, encoding="utf-8-sig")
        cluster_profiles.to_csv(profile_output, index=False, encoding="utf-8-sig")
        transition_summary.to_csv(transition_output, index=False, encoding="utf-8-sig")
        transition_matrix.to_csv(transition_matrix_output, index=False, encoding="utf-8-sig")
        transition_probability_matrix.to_csv(transition_probability_output, index=False, encoding="utf-8-sig")
        type_distribution_latest.to_csv(latest_distribution_output, index=False, encoding="utf-8-sig")
        type_share_by_year.to_csv(share_by_year_output, index=False, encoding="utf-8-sig")
        representative_df.to_csv(representatives_output, index=False, encoding="utf-8-sig")
        profile_cards.to_csv(profile_cards_output, index=False, encoding="utf-8-sig")
        type_storylines.to_csv(type_storylines_output, index=False, encoding="utf-8-sig")
        pd.DataFrame(variable_manifest_rows).to_csv(variable_manifest_output, index=False, encoding="utf-8-sig")

        plot_cluster_selection(evaluation_df, chosen_k, dirs["figures"] / "vulnerability_cluster_selection.png")
        if not latest_year_df.empty:
            plot_latest_type_counts(latest_year_df, dirs["figures"] / "vulnerability_type_counts_latest.png")
            plot_pca_scatter(latest_year_df, dirs["figures"] / "vulnerability_pca_scatter_latest.png")
            plot_representative_countries(representative_df, dirs["figures"] / "vulnerability_representative_countries.png")
        plot_dimension_profiles(cluster_profiles, dirs["figures"] / "vulnerability_dimension_profiles.png")
        plot_transition_trends(transition_summary, dirs["figures"] / "vulnerability_transition_trends.png")
        plot_transition_share_trends(type_share_by_year, dirs["figures"] / "vulnerability_type_share_trends.png")
        plot_transition_matrix_heatmap(transition_probability_matrix, dirs["figures"] / "vulnerability_transition_matrix_heatmap.png")

        summary = {
            "project_root": project_root.as_posix(),
            "input_file": input_file.as_posix(),
            "output_panel": output_panel.as_posix(),
            "rows_total": int(final_panel.shape[0]),
            "rows_clustered": int(eligible_df.shape[0]),
            "latest_year_used": latest_year,
            "available_feature_columns": [spec.analysis_column for spec in available_specs],
            "available_feature_columns_pre_sparse_filter": [spec.analysis_column for spec in resolved_specs],
            "feature_source_columns": {spec.analysis_column: spec.source_column for spec in available_specs},
            "missing_feature_columns": missing_feature_columns,
            "sparse_feature_threshold": args.sparse_feature_threshold,
            "dropped_sparse_features": dropped_sparse_features,
            "retained_features_by_dimension": retained_features_by_dimension,
            "dimension_feature_counts": dimension_counts,
            "min_dimensions_observed": args.min_dimensions_observed,
            "winsorization": {"lower": args.winsor_lower, "upper": args.winsor_upper},
            "log_transform_enabled": not args.disable_log_transform,
            "chosen_n_clusters": chosen_k,
            "cluster_selection_metrics": chosen_row.to_dict(),
            "bootstrap_iterations": args.bootstrap_iterations,
            "countries_with_labels": int(latest_country_labels["iso3"].nunique()) if not latest_country_labels.empty else 0,
            "cluster_sizes_latest_year": latest_year_df["vulnerability_type_label"].value_counts(dropna=False).to_dict(),
            "cluster_sizes_all_years": eligible_df["vulnerability_type_label"].value_counts(dropna=False).to_dict(),
            "representatives_per_cluster": args.representatives_per_cluster,
            "final_label_scheme_applied": chosen_k == 3,
            "output_files": {
                "latest_labels": latest_output.as_posix(),
                "cluster_evaluation": evaluation_output.as_posix(),
                "pca_loadings": pca_output.as_posix(),
                "explained_variance": explained_output.as_posix(),
                "cluster_profiles": profile_output.as_posix(),
                "transition_summary": transition_output.as_posix(),
                "transition_matrix": transition_matrix_output.as_posix(),
                "transition_probability_matrix": transition_probability_output.as_posix(),
                "type_distribution_latest": latest_distribution_output.as_posix(),
                "type_share_by_year": share_by_year_output.as_posix(),
                "representative_countries": representatives_output.as_posix(),
                "country_profile_cards": profile_cards_output.as_posix(),
                "type_storylines": type_storylines_output.as_posix(),
                "variable_manifest": variable_manifest_output.as_posix(),
            },
        }
        summary_path = report_asset_path(dirs["report"], "vulnerability_typology_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_stage_identification():
    __name__ = 'run_stage_identification'
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

    GBD_COLUMNS = [
        "gbd_rate_cardiovascular_diseases",
        "gbd_rate_chronic_respiratory_diseases",
        "gbd_rate_neoplasms",
        "gbd_rate_diabetes_kidney",
    ]
    AGEING_COLUMNS = [
        "hdi",
        "life_expectancy",
        "median_age",
        "wdi_population_65_plus_pct",
    ]
    STAGE_LABELS = {
        1: "转型早期",
        2: "负担上升期",
        3: "非传染病高压期",
        4: "高龄稳定期",
    }
    STAGE_LABELS_EN = {
        "转型早期": "Early transition",
        "负担上升期": "Rising burden",
        "非传染病高压期": "NCD high-pressure",
        "高龄稳定期": "Aging-stable",
    }
    STAGE_COLORS = {
        "转型早期": "#457b9d",
        "负担上升期": "#e9c46a",
        "非传染病高压期": "#e76f51",
        "高龄稳定期": "#2a9d8f",
    }
    GBD_LABELS_CN = {
        "gbd_rate_cardiovascular_diseases": "心血管疾病负担",
        "gbd_rate_chronic_respiratory_diseases": "慢性呼吸系统疾病负担",
        "gbd_rate_neoplasms": "肿瘤负担",
        "gbd_rate_diabetes_kidney": "糖尿病与肾病负担",
    }
    GBD_LABELS_EN = {
        "gbd_rate_cardiovascular_diseases": "Cardiovascular burden",
        "gbd_rate_chronic_respiratory_diseases": "Chronic respiratory burden",
        "gbd_rate_neoplasms": "Neoplasm burden",
        "gbd_rate_diabetes_kidney": "Diabetes-kidney burden",
    }


    def stage_display_label(label: str) -> str:
        return label if USE_CHINESE else STAGE_LABELS_EN.get(label, label)


    def gbd_display_label(column: str) -> str:
        labels = GBD_LABELS_CN if USE_CHINESE else GBD_LABELS_EN
        return labels.get(column, column)


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


    def percentile_score(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.rank(pct=True, method="average")


    def choose_existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
        return [column for column in columns if column in df.columns]


    def classify_stage(ageing_score: float | None, burden_score: float | None) -> tuple[int | None, str | None]:
        if pd.isna(ageing_score) or pd.isna(burden_score):
            return None, None
        if ageing_score < 0.5 and burden_score < 0.5:
            code = 1
        elif ageing_score < 0.5 and burden_score >= 0.5:
            code = 2
        elif ageing_score >= 0.5 and burden_score >= 0.5:
            code = 3
        else:
            code = 4
        return code, STAGE_LABELS[code]


    def plot_stage_counts(stage_df: pd.DataFrame, output_path: Path) -> None:
        counts = (
            stage_df.dropna(subset=["transition_stage_label"])
            .groupby(["year", "transition_stage_label"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=list(STAGE_COLORS.keys()), fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        for label in counts.columns:
            ax.plot(counts.index.astype(int), counts[label], label=stage_display_label(label), color=STAGE_COLORS[label], linewidth=2)
        ax.set_title(choose_text("健康转型阶段年度数量变化", "Annual counts by transition stage", USE_CHINESE))
        ax.set_xlabel(choose_text("年份", "Year", USE_CHINESE))
        ax.set_ylabel(choose_text("国家-年份记录数", "Country-year records", USE_CHINESE))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_stage_distribution(latest_df: pd.DataFrame, output_path: Path) -> None:
        counts = latest_df["transition_stage_label"].value_counts().reindex(STAGE_COLORS.keys(), fill_value=0)
        latest_year = int(pd.to_numeric(latest_df["year"], errors="coerce").dropna().max())
        fig, ax = plt.subplots(figsize=(9, 5))
        display_labels = [stage_display_label(label) for label in counts.index]
        bars = ax.bar(display_labels, counts.values, color=[STAGE_COLORS[k] for k in counts.index])
        ax.set_title(choose_text(f"{latest_year}年健康转型阶段国家分布", f"Country distribution by stage, {latest_year}", USE_CHINESE))
        ax.set_ylabel(choose_text("国家数量", "Number of countries", USE_CHINESE))
        ax.bar_label(bars, padding=3)
        plt.xticks(rotation=20)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_stage_scatter(latest_df: pd.DataFrame, output_path: Path) -> None:
        latest_year = int(pd.to_numeric(latest_df["year"], errors="coerce").dropna().max())
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, color in STAGE_COLORS.items():
            subset = latest_df.loc[latest_df["transition_stage_label"] == label]
            if subset.empty:
                continue
            ax.scatter(subset["ageing_axis_score"], subset["burden_axis_score"], label=stage_display_label(label), color=color, alpha=0.7, s=28)
        ax.set_title(choose_text(f"健康转型阶段散点图（{latest_year}年）", f"Transition-stage scatter ({latest_year})", USE_CHINESE))
        ax.set_xlabel(choose_text("老龄化/发展轴得分", "Aging/development axis score", USE_CHINESE))
        ax.set_ylabel(choose_text("负担轴得分", "Burden axis score", USE_CHINESE))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_stage_profiles(latest_df: pd.DataFrame, gbd_columns: list[str], output_path: Path) -> None:
        profile = latest_df.groupby("transition_stage_label", dropna=True)[gbd_columns].mean().reindex(STAGE_COLORS.keys())
        if profile.empty:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(profile.index))
        width = 0.18 if len(gbd_columns) <= 4 else 0.12
        for idx, column in enumerate(gbd_columns):
            ax.bar(x + idx * width, profile[column].values, width=width, label=gbd_display_label(column))
        ax.set_xticks(x + width * (len(gbd_columns) - 1) / 2)
        ax.set_xticklabels([stage_display_label(label) for label in profile.index], rotation=15)
        ax.set_title(choose_text("不同健康转型阶段的疾病负担画像", "Burden profile by transition stage", USE_CHINESE))
        ax.set_ylabel(choose_text("平均疾病负担指标", "Mean burden indicator", USE_CHINESE))
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Identify health transition stages from the stage panel.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--input-file", type=Path, default=None, help="Optional explicit stage panel path")
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        input_file = args.input_file.expanduser().resolve() if args.input_file else dirs["simulation"] / "stage_panel.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Stage panel not found: {input_file}")

        df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)
        if "year" not in df.columns or "iso3" not in df.columns:
            raise RuntimeError("Stage panel must contain iso3 and year")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        gbd_columns = choose_existing(df, GBD_COLUMNS)
        ageing_columns = choose_existing(df, AGEING_COLUMNS)
        if not gbd_columns:
            raise RuntimeError("Stage panel is missing all GBD burden columns")
        if not ageing_columns:
            raise RuntimeError("Stage panel is missing all ageing/development columns")

        stage_df = df.copy()
        stage_df["gbd_burden_total"] = stage_df[gbd_columns].sum(axis=1, min_count=1)
        stage_df["gbd_burden_mean"] = stage_df[gbd_columns].mean(axis=1)
        stage_df["ageing_axis_score"] = pd.concat([percentile_score(stage_df[col]) for col in ageing_columns], axis=1).mean(axis=1)
        stage_df["burden_axis_score"] = percentile_score(stage_df["gbd_burden_total"])

        code_label = stage_df.apply(
            lambda row: classify_stage(row["ageing_axis_score"], row["burden_axis_score"]),
            axis=1,
            result_type="expand",
        )
        stage_df["transition_stage_code"] = code_label[0].astype("Int64")
        stage_df["transition_stage_label"] = code_label[1]

        output_panel = dirs["simulation"] / "stage_identification_panel.csv"
        stage_df.to_csv(output_panel, index=False, encoding="utf-8-sig")

        labeled_rows = stage_df.dropna(subset=["transition_stage_label"]).copy()
        latest_country_labels = (
            labeled_rows.sort_values(["iso3", "year"], kind="stable")
            .groupby("iso3", as_index=False)
            .tail(1)
            .sort_values(["transition_stage_code", "iso3"], kind="stable")
        )
        latest_labels_path = report_asset_path(dirs["report"], "stage_country_labels_latest.csv")
        latest_country_labels.to_csv(latest_labels_path, index=False, encoding="utf-8-sig")

        transition_summary = (
            labeled_rows.groupby(["year", "transition_stage_label"], dropna=False)
            .size()
            .rename("records")
            .reset_index()
        )
        transition_summary_path = report_asset_path(dirs["report"], "stage_transition_summary.csv")
        transition_summary.to_csv(transition_summary_path, index=False, encoding="utf-8-sig")

        if not latest_country_labels.empty:
            latest_year = int(latest_country_labels["year"].max())
            latest_year_df = latest_country_labels.loc[latest_country_labels["year"] == latest_year].copy()
            plot_stage_distribution(latest_year_df, dirs["figures"] / "stage_distribution_latest_year.png")
            plot_stage_scatter(latest_year_df, dirs["figures"] / "stage_burden_scatter_latest.png")
            plot_stage_profiles(latest_country_labels, gbd_columns, dirs["figures"] / "stage_group_burden_profiles.png")
        else:
            latest_year = None
        plot_stage_counts(labeled_rows, dirs["figures"] / "stage_trend_counts.png")

        summary = {
            "project_root": project_root.as_posix(),
            "input_file": input_file.as_posix(),
            "output_panel": output_panel.as_posix(),
            "rows": int(stage_df.shape[0]),
            "staged_rows": int(labeled_rows.shape[0]),
            "countries_with_labels": int(latest_country_labels["iso3"].nunique()) if not latest_country_labels.empty else 0,
            "latest_year_used": latest_year,
            "gbd_columns_used": gbd_columns,
            "ageing_columns_used": ageing_columns,
            "stage_counts_latest": latest_country_labels["transition_stage_label"].value_counts(dropna=False).to_dict(),
            "latest_labels_file": latest_labels_path.as_posix(),
            "transition_summary_file": transition_summary_path.as_posix(),
        }
        summary_path = report_asset_path(dirs["report"], "stage_identification_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_risk_attribution_matrix():
    __name__ = 'run_risk_attribution_matrix'
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

    DISEASE_COLUMNS = [
        "gbd_rate_cardiovascular_diseases_per100k",
        "gbd_rate_chronic_respiratory_diseases_per100k",
        "gbd_rate_neoplasms_per100k",
        "gbd_rate_diabetes_kidney_per100k",
    ]

    RISK_COLUMNS = [
        "dbd_smoking_per100k",
        "dbd_pm25_per100k",
        "dbd_high_bmi_per100k",
        "dbd_high_glucose_per100k",
        "dbd_high_sbp_per100k",
        "dbd_dietary_risks_per100k",
    ]

    DISEASE_LABELS_CN = {
        "gbd_rate_cardiovascular_diseases_per100k": "心血管疾病负担",
        "gbd_rate_chronic_respiratory_diseases_per100k": "慢性呼吸系统疾病负担",
        "gbd_rate_neoplasms_per100k": "肿瘤负担",
        "gbd_rate_diabetes_kidney_per100k": "糖尿病与肾病负担",
    }

    DISEASE_LABELS_EN = {
        "gbd_rate_cardiovascular_diseases_per100k": "Cardiovascular burden",
        "gbd_rate_chronic_respiratory_diseases_per100k": "Chronic respiratory burden",
        "gbd_rate_neoplasms_per100k": "Neoplasm burden",
        "gbd_rate_diabetes_kidney_per100k": "Diabetes-kidney burden",
    }

    RISK_LABELS_CN = {
        "dbd_smoking_per100k": "吸烟",
        "dbd_pm25_per100k": "PM2.5",
        "dbd_high_bmi_per100k": "高BMI",
        "dbd_high_glucose_per100k": "高血糖",
        "dbd_high_sbp_per100k": "高收缩压",
        "dbd_dietary_risks_per100k": "膳食风险",
    }

    RISK_LABELS_EN = {
        "dbd_smoking_per100k": "Smoking",
        "dbd_pm25_per100k": "PM2.5",
        "dbd_high_bmi_per100k": "High BMI",
        "dbd_high_glucose_per100k": "High glucose",
        "dbd_high_sbp_per100k": "High SBP",
        "dbd_dietary_risks_per100k": "Dietary risks",
    }


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


    def winsorize_series(series: pd.Series, lower_quantile: float, upper_quantile: float) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        mask = numeric.notna()
        if not mask.any():
            return numeric
        lower = numeric.loc[mask].quantile(lower_quantile)
        upper = numeric.loc[mask].quantile(upper_quantile)
        return numeric.clip(lower=lower, upper=upper)


    def prepare_continuous(series: pd.Series, lower_quantile: float, upper_quantile: float, log_transform: bool) -> pd.Series:
        clipped = winsorize_series(series, lower_quantile, upper_quantile)
        if log_transform:
            clipped = clipped.where(clipped >= 0)
            clipped = np.log1p(clipped)
        return clipped.astype("float64")


    def zscore_array(values: np.ndarray) -> np.ndarray:
        values = values.astype("float64")
        std = values.std(ddof=0)
        if np.isnan(std) or std == 0:
            return np.zeros_like(values, dtype="float64")
        return (values - values.mean()) / std


    def disease_display(column: str) -> str:
        return DISEASE_LABELS_CN[column] if USE_CHINESE else DISEASE_LABELS_EN[column]


    def risk_display(column: str) -> str:
        return RISK_LABELS_CN[column] if USE_CHINESE else RISK_LABELS_EN[column]


    def run_type_disease_model(
        subset: pd.DataFrame,
        disease_column: str,
        risk_columns: list[str],
        winsor_lower: float,
        winsor_upper: float,
        log_transform: bool,
    ) -> pd.DataFrame:
        required_columns = ["year", disease_column] + risk_columns
        model_df = subset.loc[:, required_columns].dropna().copy()
        min_required_rows = max(18, len(risk_columns) * 3)
        if model_df.shape[0] < min_required_rows:
            return pd.DataFrame()

        y = prepare_continuous(model_df[disease_column], winsor_lower, winsor_upper, log_transform)
        X_risk = pd.DataFrame(
            {
                column: prepare_continuous(model_df[column], winsor_lower, winsor_upper, log_transform)
                for column in risk_columns
            }
        )
        full_df = pd.concat([model_df[["year"]], y.rename("y"), X_risk], axis=1).dropna()
        if full_df.shape[0] < min_required_rows:
            return pd.DataFrame()

        y_z = zscore_array(full_df["y"].to_numpy())
        x_z = np.column_stack([zscore_array(full_df[column].to_numpy()) for column in risk_columns])

        year_dummies = pd.get_dummies(full_df["year"].astype(int), prefix="year", drop_first=True, dtype=float)
        design_matrix = np.column_stack([np.ones(len(full_df)), x_z, year_dummies.to_numpy(dtype=float)])
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y_z, rcond=None)
        risk_betas = coefficients[1 : 1 + len(risk_columns)]

        correlation_rows = []
        beta_abs_sum = float(np.abs(risk_betas).sum())
        for idx, column in enumerate(risk_columns):
            risk_values = x_z[:, idx]
            corr = float(np.corrcoef(y_z, risk_values)[0, 1]) if len(risk_values) > 1 else np.nan
            contribution = abs(float(risk_betas[idx])) / beta_abs_sum if beta_abs_sum > 0 else 0.0
            correlation_rows.append(
                {
                    "risk_column": column,
                    "pairwise_corr": corr,
                    "standardized_beta": float(risk_betas[idx]),
                    "absolute_beta": abs(float(risk_betas[idx])),
                    "contribution_share": contribution,
                    "n_obs": int(len(full_df)),
                    "year_min": int(full_df["year"].min()),
                    "year_max": int(full_df["year"].max()),
                }
            )
        result = pd.DataFrame(correlation_rows).sort_values(
            ["contribution_share", "absolute_beta", "pairwise_corr"],
            ascending=[False, False, False],
            kind="stable",
        )
        result["rank_in_disease"] = np.arange(1, len(result) + 1)
        return result


    def plot_type_heatmaps(matrix_df: pd.DataFrame, output_path: Path) -> None:
        type_labels = matrix_df["vulnerability_type_label"].dropna().unique().tolist()
        if not type_labels:
            return
        fig, axes = plt.subplots(1, len(type_labels), figsize=(6 * len(type_labels), 5.9), squeeze=False, constrained_layout=False)
        axes = axes[0]
        for idx, type_label in enumerate(type_labels):
            ax = axes[idx]
            subset = matrix_df.loc[matrix_df["vulnerability_type_label"] == type_label]
            heatmap = subset.pivot(index="disease_label", columns="risk_label", values="contribution_share").fillna(0.0)
            im = ax.imshow(heatmap.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(0.35, float(heatmap.values.max())))
            ax.set_xticks(np.arange(heatmap.shape[1]))
            ax.set_xticklabels(heatmap.columns, rotation=25, ha="right")
            ax.set_yticks(np.arange(heatmap.shape[0]))
            ax.set_yticklabels(heatmap.index)
            ax.set_title(type_label, fontsize=15, loc="center", pad=10)
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    ax.text(j, i, f"{heatmap.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
        fig.subplots_adjust(left=0.10, right=0.86, top=0.82, bottom=0.18, wspace=0.55)
        cax = fig.add_axes([0.91, 0.20, 0.018, 0.56])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(choose_text("归因权重占比", "Contribution share", USE_CHINESE))
        set_centered_suptitle(fig, choose_text("脆弱性类型内的疾病-风险归因矩阵", "Disease-risk attribution matrix within vulnerability types", USE_CHINESE), y=0.98)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_top_risk_counts(top_df: pd.DataFrame, output_path: Path) -> None:
        summary = (
            top_df.groupby(["vulnerability_type_label", "risk_label"], dropna=False)
            .size()
            .rename("top_count")
            .reset_index()
        )
        if summary.empty:
            return
        type_labels = summary["vulnerability_type_label"].dropna().unique().tolist()
        fig, axes = plt.subplots(
            len(type_labels),
            1,
            figsize=(10, 4.2 * len(type_labels)),
            squeeze=False,
            constrained_layout=False,
        )
        axes = axes[:, 0]
        for idx, type_label in enumerate(type_labels):
            ax = axes[idx]
            subset = summary.loc[summary["vulnerability_type_label"] == type_label].sort_values("top_count", ascending=False, kind="stable")
            ax.bar(subset["risk_label"], subset["top_count"], color="#457b9d")
            ax.set_title(type_label, fontsize=15, loc="center", pad=10)
            ax.set_ylabel(choose_text("成为首位风险的疾病数", "Diseases where risk ranks #1", USE_CHINESE))
            ax.tick_params(axis="x", rotation=20)
        set_centered_suptitle(fig, choose_text("各脆弱性类型的主导风险出现频次", "Dominant risk frequency by vulnerability type", USE_CHINESE), y=0.985)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92], h_pad=2.8)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def build_type_storylines(top_risks_df: pd.DataFrame, type_risk_summary: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        type_order = (
            type_risk_summary.loc[:, ["vulnerability_type_code", "vulnerability_type_label"]]
            .drop_duplicates()
            .sort_values("vulnerability_type_code", kind="stable")
        )
        for _, type_row in type_order.iterrows():
            type_code = type_row["vulnerability_type_code"]
            type_label = type_row["vulnerability_type_label"]
            top3 = (
                type_risk_summary.loc[type_risk_summary["vulnerability_type_code"] == type_code]
                .sort_values("rank_within_type", kind="stable")
                .head(3)
            )
            disease_top = (
                top_risks_df.loc[top_risks_df["vulnerability_type_code"] == type_code]
                .sort_values("disease_label", kind="stable")
            )
            rows.append(
                {
                    "vulnerability_type_code": type_code,
                    "vulnerability_type_label": type_label,
                    "top3_risks": " / ".join(top3["risk_label"].tolist()),
                    "cardiovascular_top_risk": next(
                        (row["risk_label"] for _, row in disease_top.iterrows() if row["disease_label"] == disease_display(DISEASE_COLUMNS[0])),
                        None,
                    ),
                    "respiratory_top_risk": next(
                        (row["risk_label"] for _, row in disease_top.iterrows() if row["disease_label"] == disease_display(DISEASE_COLUMNS[1])),
                        None,
                    ),
                    "neoplasm_top_risk": next(
                        (row["risk_label"] for _, row in disease_top.iterrows() if row["disease_label"] == disease_display(DISEASE_COLUMNS[2])),
                        None,
                    ),
                    "diabetes_kidney_top_risk": next(
                        (row["risk_label"] for _, row in disease_top.iterrows() if row["disease_label"] == disease_display(DISEASE_COLUMNS[3])),
                        None,
                    ),
                }
            )
        return pd.DataFrame(rows)


    def build_type_disease_cards(top_risks_df: pd.DataFrame) -> pd.DataFrame:
        cards = top_risks_df.loc[
            :,
            [
                "vulnerability_type_code",
                "vulnerability_type_label",
                "disease_column",
                "disease_label",
                "risk_column",
                "risk_label",
                "contribution_share",
                "standardized_beta",
                "pairwise_corr",
                "n_obs",
                "year_min",
                "year_max",
            ],
        ].copy()
        cards = cards.rename(
            columns={
                "risk_column": "top_risk_column",
                "risk_label": "top_risk_label",
                "contribution_share": "top_risk_contribution_share",
                "standardized_beta": "top_risk_standardized_beta",
                "pairwise_corr": "top_risk_pairwise_corr",
            }
        )
        return cards.sort_values(["vulnerability_type_code", "disease_label"], kind="stable").reset_index(drop=True)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Build Module B risk attribution matrix from vulnerability typology results.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--response-panel", type=Path, default=None, help="Optional explicit response panel path")
        parser.add_argument("--typology-panel", type=Path, default=None, help="Optional explicit vulnerability typology panel path")
        parser.add_argument("--latest-year", type=int, default=None, help="Optional explicit latest year for latest-year summaries")
        parser.add_argument("--winsor-lower", type=float, default=0.01)
        parser.add_argument("--winsor-upper", type=float, default=0.99)
        parser.add_argument("--disable-log-transform", action="store_true")
        args = parser.parse_args()

        if not (0.0 <= args.winsor_lower < args.winsor_upper <= 1.0):
            raise ValueError("Winsor quantiles must satisfy 0 <= lower < upper <= 1")

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        response_panel = args.response_panel.expanduser().resolve() if args.response_panel else dirs["simulation"] / "response_panel.csv"
        typology_panel = args.typology_panel.expanduser().resolve() if args.typology_panel else dirs["simulation"] / "vulnerability_typology_panel.csv"

        if not response_panel.exists():
            raise FileNotFoundError(f"Response panel not found: {response_panel}")
        if not typology_panel.exists():
            raise FileNotFoundError(f"Vulnerability typology panel not found: {typology_panel}")

        response_df = pd.read_csv(response_panel, encoding="utf-8-sig", low_memory=False)
        typology_df = pd.read_csv(typology_panel, encoding="utf-8-sig", low_memory=False)

        required_typology = {"iso3", "year", "vulnerability_type_code", "vulnerability_type_label"}
        missing_typology = sorted(required_typology - set(typology_df.columns))
        if missing_typology:
            raise RuntimeError(f"Typology panel missing required columns: {missing_typology}")

        merge_columns = ["iso3", "year", "vulnerability_type_code", "vulnerability_type_label"]
        analysis_df = response_df.merge(
            typology_df[merge_columns].drop_duplicates(["iso3", "year"], keep="last"),
            on=["iso3", "year"],
            how="left",
        )
        analysis_df = analysis_df.dropna(subset=["vulnerability_type_label"]).copy()
        if analysis_df.empty:
            raise RuntimeError("No rows with vulnerability type labels were found after merging response and typology panels")

        latest_year = args.latest_year
        if latest_year is None:
            latest_year = int(pd.to_numeric(analysis_df["year"], errors="coerce").dropna().max())

        missing_disease_columns = [column for column in DISEASE_COLUMNS if column not in analysis_df.columns]
        missing_risk_columns = [column for column in RISK_COLUMNS if column not in analysis_df.columns]
        if missing_disease_columns or missing_risk_columns:
            raise RuntimeError(
                f"Analysis input is missing required columns. diseases={missing_disease_columns}, risks={missing_risk_columns}"
            )

        matrix_rows: list[dict[str, object]] = []
        for type_label, type_subset in analysis_df.groupby("vulnerability_type_label", dropna=False):
            for disease_column in DISEASE_COLUMNS:
                result = run_type_disease_model(
                    type_subset,
                    disease_column,
                    RISK_COLUMNS,
                    args.winsor_lower,
                    args.winsor_upper,
                    log_transform=not args.disable_log_transform,
                )
                if result.empty:
                    continue
                result["vulnerability_type_label"] = type_label
                result["vulnerability_type_code"] = type_subset["vulnerability_type_code"].dropna().mode().iloc[0]
                result["disease_column"] = disease_column
                result["disease_label"] = disease_display(disease_column)
                result["risk_label"] = result["risk_column"].map(risk_display)
                matrix_rows.extend(result.to_dict(orient="records"))

        matrix_df = pd.DataFrame(matrix_rows)
        if matrix_df.empty:
            raise RuntimeError("Risk attribution matrix is empty; no disease/type combination had sufficient data")

        matrix_df = matrix_df.sort_values(
            ["vulnerability_type_code", "disease_column", "rank_in_disease"],
            kind="stable",
        ).reset_index(drop=True)

        top_risks_df = matrix_df.loc[matrix_df["rank_in_disease"] == 1].copy()
        type_risk_summary = (
            matrix_df.groupby(["vulnerability_type_code", "vulnerability_type_label", "risk_column", "risk_label"], dropna=False)[
                ["contribution_share", "absolute_beta"]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["vulnerability_type_code", "contribution_share", "absolute_beta"], ascending=[True, False, False], kind="stable")
        )
        type_risk_summary["rank_within_type"] = type_risk_summary.groupby("vulnerability_type_code").cumcount() + 1
        type_storylines = build_type_storylines(top_risks_df, type_risk_summary)
        type_disease_cards = build_type_disease_cards(top_risks_df)

        latest_snapshot = analysis_df.loc[analysis_df["year"] == latest_year, ["iso3", "year", "vulnerability_type_label"]].copy()
        latest_counts = latest_snapshot["vulnerability_type_label"].value_counts(dropna=False).to_dict()

        matrix_output = report_asset_path(dirs["report"], "risk_attribution_matrix.csv")
        top_risks_output = report_asset_path(dirs["report"], "risk_attribution_top_risks.csv")
        type_risk_output = report_asset_path(dirs["report"], "risk_attribution_type_risk_summary.csv")
        type_storylines_output = report_asset_path(dirs["report"], "risk_attribution_type_storylines.csv")
        type_disease_cards_output = report_asset_path(dirs["report"], "risk_attribution_type_disease_cards.csv")
        latest_snapshot_output = report_asset_path(dirs["report"], "risk_attribution_latest_type_snapshot.csv")

        matrix_df.to_csv(matrix_output, index=False, encoding="utf-8-sig")
        top_risks_df.to_csv(top_risks_output, index=False, encoding="utf-8-sig")
        type_risk_summary.to_csv(type_risk_output, index=False, encoding="utf-8-sig")
        type_storylines.to_csv(type_storylines_output, index=False, encoding="utf-8-sig")
        type_disease_cards.to_csv(type_disease_cards_output, index=False, encoding="utf-8-sig")
        latest_snapshot.to_csv(latest_snapshot_output, index=False, encoding="utf-8-sig")

        plot_type_heatmaps(matrix_df, dirs["figures"] / "risk_attribution_matrix_heatmap.png")
        plot_top_risk_counts(top_risks_df, dirs["figures"] / "risk_attribution_top_risk_counts.png")

        summary = {
            "project_root": project_root.as_posix(),
            "response_panel": response_panel.as_posix(),
            "typology_panel": typology_panel.as_posix(),
            "rows_with_type_labels": int(analysis_df.shape[0]),
            "latest_year_used": latest_year,
            "latest_year_type_counts": latest_counts,
            "disease_columns": DISEASE_COLUMNS,
            "risk_columns": RISK_COLUMNS,
            "vulnerability_types": sorted(analysis_df["vulnerability_type_label"].dropna().unique().tolist()),
            "winsorization": {"lower": args.winsor_lower, "upper": args.winsor_upper},
            "log_transform_enabled": not args.disable_log_transform,
            "method_note": choose_text(
                "本矩阵使用类型内的标准化回归系数与相关系数构建关联型风险归因权重，不等同于严格因果识别。",
                "This matrix uses within-type standardized regression weights and correlations as an association-based risk attribution indicator, not strict causal identification.",
                USE_CHINESE,
            ),
            "output_files": {
                "risk_attribution_matrix": matrix_output.as_posix(),
                "top_risks": top_risks_output.as_posix(),
                "type_risk_summary": type_risk_output.as_posix(),
                "type_storylines": type_storylines_output.as_posix(),
                "type_disease_cards": type_disease_cards_output.as_posix(),
                "latest_type_snapshot": latest_snapshot_output.as_posix(),
                "matrix_heatmap": (dirs["figures"] / "risk_attribution_matrix_heatmap.png").as_posix(),
                "top_risk_counts": (dirs["figures"] / "risk_attribution_top_risk_counts.png").as_posix(),
            },
        }
        summary_path = report_asset_path(dirs["report"], "risk_attribution_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_response_diagnosis():
    __name__ = 'run_response_diagnosis'
    import argparse
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from foundation import choose_text, configure_matplotlib_fonts, country_display_name, load_country_zh_labels
    from foundation import detect_project_root as shared_detect_project_root
    USE_CHINESE = configure_matplotlib_fonts()

    GBD_COLUMNS = [
        "gbd_rate_cardiovascular_diseases_per100k",
        "gbd_rate_chronic_respiratory_diseases_per100k",
        "gbd_rate_neoplasms_per100k",
        "gbd_rate_diabetes_kidney_per100k",
    ]

    RISK_COLUMNS = [
        "dbd_smoking_per100k",
        "dbd_pm25_per100k",
        "dbd_high_bmi_per100k",
        "dbd_high_glucose_per100k",
        "dbd_high_sbp_per100k",
        "dbd_dietary_risks_per100k",
    ]

    RESOURCE_POSITIVE_COLUMNS = [
        "beds_10k",
        "doctors_10k",
        "nurses_10k",
        "uhc_index",
        "che_pct_gdp",
        "che_pc_usd",
        "govt_he_pct",
        "wdi_health_expenditure_per_capita",
        "wdi_government_health_expenditure_pct",
        "wdi_hospital_beds",
        "wdi_physicians",
        "wdi_nurses_midwives",
        "wb_hnp_hci",
        "wb_hnp_skilled_birth_attendance",
        "wb_hnp_immunization_dpt",
        "wb_hnp_immunization_measles",
        "wb_hnp_immunization_hepb3",
    ]

    RESOURCE_NEGATIVE_COLUMNS = [
        "ext_he_pct",
        "wdi_external_health_expenditure_pct",
        "wdi_out_of_pocket_pct",
    ]

    CONTEXT_COLUMNS = [
        "hdi",
        "wdi_gdp_per_capita",
        "wdi_gini",
        "wdi_population_65_plus_pct",
        "wdi_urban_population_pct",
    ]

    RESOURCE_LABELS_CN = {
        "beds_10k": "病床",
        "doctors_10k": "医生",
        "nurses_10k": "护士",
        "uhc_index": "UHC",
        "che_pct_gdp": "卫生支出/GDP",
        "che_pc_usd": "人均卫生支出",
        "govt_he_pct": "政府卫生支出占比",
        "wdi_health_expenditure_per_capita": "WDI人均卫生支出",
        "wdi_government_health_expenditure_pct": "WDI政府卫生支出占比",
        "wdi_hospital_beds": "WDI病床",
        "wdi_physicians": "WDI医生",
        "wdi_nurses_midwives": "WDI护士助产士",
        "wb_hnp_hci": "HCI",
        "wb_hnp_skilled_birth_attendance": "专业接生",
        "wb_hnp_immunization_dpt": "DPT免疫",
        "wb_hnp_immunization_measles": "麻疹免疫",
        "wb_hnp_immunization_hepb3": "乙肝免疫",
        "ext_he_pct": "外部卫生支出依赖",
        "wdi_external_health_expenditure_pct": "WDI外部支出依赖",
        "wdi_out_of_pocket_pct": "自付卫生支出",
    }

    RESOURCE_LABELS_EN = {
        "beds_10k": "Beds",
        "doctors_10k": "Doctors",
        "nurses_10k": "Nurses",
        "uhc_index": "UHC",
        "che_pct_gdp": "Health spend/GDP",
        "che_pc_usd": "Health spend pc",
        "govt_he_pct": "Gov health %",
        "wdi_health_expenditure_per_capita": "WDI health spend pc",
        "wdi_government_health_expenditure_pct": "WDI gov health %",
        "wdi_hospital_beds": "WDI beds",
        "wdi_physicians": "WDI physicians",
        "wdi_nurses_midwives": "WDI nurses-midwives",
        "wb_hnp_hci": "HCI",
        "wb_hnp_skilled_birth_attendance": "Skilled birth attendance",
        "wb_hnp_immunization_dpt": "DPT immunization",
        "wb_hnp_immunization_measles": "Measles immunization",
        "wb_hnp_immunization_hepb3": "HepB3 immunization",
        "ext_he_pct": "External health dependence",
        "wdi_external_health_expenditure_pct": "WDI external dependence",
        "wdi_out_of_pocket_pct": "Out-of-pocket share",
    }

    TYPE_COLORS = {
        "高压低响应型": "#e76f51",
        "高压高响应型": "#2a9d8f",
        "低压低响应型": "#8d99ae",
        "相对均衡型": "#457b9d",
    }

    TYPE_LABELS_EN = {
        "高压低响应型": "High pressure, low response",
        "高压高响应型": "High pressure, high response",
        "低压低响应型": "Low pressure, low response",
        "相对均衡型": "Relatively balanced",
    }

    STAGE_LABELS_EN = {
        "转型早期": "Early transition",
        "负担上升期": "Rising burden",
        "非传染病高压期": "NCD high-pressure",
        "高龄稳定期": "Aging-stable",
    }


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


    def choose_existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
        return [column for column in columns if column in df.columns]


    def filter_sparse_resource_columns(
        df: pd.DataFrame,
        positive_cols: list[str],
        negative_cols: list[str],
        sparse_threshold: float,
        main_missingness_threshold: float,
    ) -> tuple[list[str], list[str], list[dict[str, object]], list[dict[str, object]], list[str]]:
        retained_positive: list[str] = []
        retained_negative: list[str] = []
        dropped_sparse: list[dict[str, object]] = []
        dropped_moderate_missing: list[dict[str, object]] = []
        for column, direction, bucket in (
            *((column, "positive", retained_positive) for column in positive_cols),
            *((column, "negative", retained_negative) for column in negative_cols),
        ):
            missing_rate = float(df[column].isna().mean())
            drop_record = {
                "column": column,
                "direction": direction,
                "label": resource_display_label(column),
                "missing_rate": missing_rate,
            }
            if missing_rate > sparse_threshold:
                dropped_sparse.append(
                    {
                        **drop_record,
                        "drop_reason": "missing_rate_above_sparse_threshold",
                    }
                )
                continue
            if missing_rate > main_missingness_threshold:
                dropped_moderate_missing.append(
                    {
                        **drop_record,
                        "drop_reason": "missing_rate_above_main_model_threshold",
                    }
                )
                continue
            bucket.append(column)
        retained_all = retained_positive + retained_negative
        return retained_positive, retained_negative, dropped_sparse, dropped_moderate_missing, retained_all


    def percentile_score(series: pd.Series, reverse: bool = False) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if reverse:
            numeric = -numeric
        return numeric.rank(pct=True, method="average")


    def type_display_label(label: str) -> str:
        return label if USE_CHINESE else TYPE_LABELS_EN.get(label, label)


    def stage_display_label(label: str) -> str:
        return label if USE_CHINESE else STAGE_LABELS_EN.get(label, label)


    def resource_display_label(column: str) -> str:
        if USE_CHINESE:
            return RESOURCE_LABELS_CN.get(column, column)
        return RESOURCE_LABELS_EN.get(column, column)


    def latest_year_from_frame(df: pd.DataFrame) -> int:
        years = pd.to_numeric(df.get("year"), errors="coerce").dropna()
        return int(years.max()) if not years.empty else 0


    def classify_response_type(pressure: float | None, response: float | None) -> str | None:
        if pd.isna(pressure) or pd.isna(response):
            return None
        if pressure >= 0.67 and response < 0.50:
            return "高压低响应型"
        if pressure >= 0.67 and response >= 0.50:
            return "高压高响应型"
        if pressure < 0.50 and response < 0.50:
            return "低压低响应型"
        return "相对均衡型"


    def component_score_frame(df: pd.DataFrame, positive_cols: list[str], negative_cols: list[str]) -> pd.DataFrame:
        frames = {}
        for column in positive_cols:
            frames[column] = percentile_score(df[column], reverse=False)
        for column in negative_cols:
            frames[column] = percentile_score(df[column], reverse=True)
        return pd.DataFrame(frames, index=df.index)


    def summarize_weak_components(row: pd.Series, component_columns: list[str], top_n: int = 3) -> str:
        available = [(column, row[column]) for column in component_columns if pd.notna(row[column])]
        available.sort(key=lambda item: item[1])
        chosen = [resource_display_label(column) for column, _ in available[:top_n]]
        return " / ".join(chosen)


    def weakest_component_names(row: pd.Series, component_columns: list[str], top_n: int = 3) -> list[str]:
        available = [(column.replace("component__", ""), row[column]) for column in component_columns if pd.notna(row[column])]
        available.sort(key=lambda item: item[1])
        return [column for column, _ in available[:top_n]]


    def build_type_storylines(type_risk_summary: pd.DataFrame) -> dict[str, str]:
        storyline_map: dict[str, str] = {}
        if type_risk_summary.empty:
            return storyline_map
        for type_label, subset in type_risk_summary.groupby("vulnerability_type_label", dropna=False):
            top3 = (
                subset.sort_values("rank_within_type", kind="stable")
                .head(3)["risk_label"]
                .astype(str)
                .tolist()
            )
            storyline_map[type_label] = " / ".join(top3)
        return storyline_map


    def simulate_component_upgrade(
        df: pd.DataFrame,
        component_columns: list[str],
        context_weight: float,
        scenario_step: float,
        top_n: int,
    ) -> pd.DataFrame:
        scenario_df = df.copy()
        selected_components: list[list[str]] = []
        optimized_scores: list[float] = []
        optimized_adjusted: list[float] = []
        optimized_gap: list[float] = []
        gap_reduction: list[float] = []

        for _, row in scenario_df.iterrows():
            weakest = weakest_component_names(row, component_columns, top_n=top_n)
            selected_components.append(weakest)
            updated_values = []
            for column in component_columns:
                value = row[column]
                if pd.isna(value):
                    continue
                raw_column = column.replace("component__", "")
                if raw_column in weakest:
                    updated_values.append(min(1.0, float(value) + scenario_step))
                else:
                    updated_values.append(float(value))
            if updated_values:
                new_resource = float(np.mean(updated_values))
            else:
                new_resource = np.nan
            optimized_scores.append(new_resource)
            if pd.notna(new_resource):
                context_support = row["context_support_score"] if pd.notna(row["context_support_score"]) else new_resource
                new_adjusted = (1 - context_weight) * new_resource + context_weight * float(context_support)
            else:
                new_adjusted = np.nan
            optimized_adjusted.append(new_adjusted)
            old_gap = row["adaptation_gap_score"]
            new_gap = row["combined_pressure_score"] - new_adjusted if pd.notna(new_adjusted) else np.nan
            optimized_gap.append(new_gap)
            gap_reduction.append(old_gap - new_gap if pd.notna(old_gap) and pd.notna(new_gap) else np.nan)

        scenario_df["scenario_target_components"] = [
            " / ".join(resource_display_label(column) for column in weakest) for weakest in selected_components
        ]
        scenario_df["optimized_resource_response_score"] = optimized_scores
        scenario_df["optimized_adjusted_response_score"] = optimized_adjusted
        scenario_df["optimized_adaptation_gap_score"] = optimized_gap
        scenario_df["gap_reduction_if_upgraded"] = gap_reduction
        scenario_df["response_lift_needed_to_balance"] = (0.50 - scenario_df["adjusted_response_score"]).clip(lower=0)
        return scenario_df


    def plot_pressure_response_scatter(latest_df: pd.DataFrame, output_path: Path) -> None:
        latest_year = latest_year_from_frame(latest_df)
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        for label, color in TYPE_COLORS.items():
            subset = latest_df.loc[latest_df["response_diagnosis_type"] == label]
            if subset.empty:
                continue
            ax.scatter(
                subset["combined_pressure_score"],
                subset["adjusted_response_score"],
                label=type_display_label(label),
                color=color,
                alpha=0.8,
                s=30,
            )
        ax.axvline(0.67, color="#999999", linestyle="--", linewidth=1)
        ax.axhline(0.50, color="#999999", linestyle="--", linewidth=1)
        ax.set_title(choose_text(f"风险压力与响应能力诊断散点图（{latest_year}年）", f"Pressure-response diagnosis scatter ({latest_year})", USE_CHINESE))
        ax.set_xlabel(choose_text("综合压力得分", "Combined pressure score", USE_CHINESE))
        ax.set_ylabel(choose_text("调整后响应得分", "Adjusted response score", USE_CHINESE))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_type_counts(latest_df: pd.DataFrame, output_path: Path) -> None:
        latest_year = latest_year_from_frame(latest_df)
        counts = latest_df["response_diagnosis_type"].value_counts().reindex(TYPE_COLORS.keys(), fill_value=0)
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar([type_display_label(label) for label in counts.index], counts.values, color=[TYPE_COLORS[label] for label in counts.index])
        ax.bar_label(bars, padding=3)
        ax.set_title(choose_text(f"{latest_year}年响应失配类型分布", f"Distribution of response mismatch types, {latest_year}", USE_CHINESE))
        ax.set_ylabel(choose_text("国家数量", "Number of countries", USE_CHINESE))
        plt.xticks(rotation=15)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_gap_top(latest_df: pd.DataFrame, output_path: Path) -> None:
        latest_year = latest_year_from_frame(latest_df)
        subset = latest_df.dropna(subset=["adaptation_gap_score"]).nlargest(20, "adaptation_gap_score")
        if subset.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 7))
        country_labels = load_country_zh_labels(output_path.parents[1])
        display_labels = subset["iso3"].map(lambda code: country_display_name(code, country_labels))
        ax.barh(display_labels, subset["adaptation_gap_score"], color="#e76f51")
        ax.invert_yaxis()
        ax.set_title(choose_text(f"{latest_year}年适配缺口最高国家 Top 20", f"Top 20 countries with largest adaptation gaps, {latest_year}", USE_CHINESE))
        ax.set_xlabel(choose_text("适配缺口得分", "Adaptation gap score", USE_CHINESE))
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_vulnerability_response_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
        if matrix.empty:
            return
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        im = ax.imshow(matrix.values, cmap="Blues")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([type_display_label(label) for label in matrix.columns], rotation=20, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(int(matrix.iloc[i, j])), ha="center", va="center", color="black")
        ax.set_title(choose_text("脆弱性类型 × 响应失配类型", "Vulnerability type × response mismatch type", USE_CHINESE))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(choose_text("国家数量（颜色越深数量越多）", "Country count (darker means more)", USE_CHINESE))
        fig.subplots_adjust(left=0.30, right=0.88, bottom=0.20, top=0.86)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_weak_component_heatmap(type_component_df: pd.DataFrame, output_path: Path) -> None:
        if type_component_df.empty:
            return
        heatmap = type_component_df.pivot(index="response_diagnosis_type", columns="resource_component_label", values="mean_component_score").fillna(0.0)
        fig, ax = plt.subplots(figsize=(12, 5.8))
        im = ax.imshow(heatmap.values, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(np.arange(heatmap.shape[1]))
        ax.set_xticklabels(heatmap.columns, rotation=25, ha="right")
        ax.set_yticks(np.arange(heatmap.shape[0]))
        ax.set_yticklabels([type_display_label(label) for label in heatmap.index])
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                ax.text(j, i, f"{heatmap.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
        ax.set_title(choose_text("各响应类型的资源短板结构", "Resource weakness structure by response type", USE_CHINESE))
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        cbar.set_label(choose_text("资源响应得分（颜色越深得分越高，浅色表示短板）", "Resource response score (darker means higher)", USE_CHINESE))
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_stage_response_heatmap(matrix: pd.DataFrame, output_path: Path, latest_year: int) -> None:
        if matrix.empty:
            return
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        im = ax.imshow(matrix.values, cmap="Blues", aspect="auto")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([type_display_label(label) for label in matrix.columns], rotation=20, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([stage_display_label(label) for label in matrix.index])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(int(matrix.iloc[i, j])), ha="center", va="center", color="black")
        ax.set_title(choose_text(f"健康转型阶段 × 响应诊断类型（{latest_year}年）", f"Transition stage × response diagnosis ({latest_year})", USE_CHINESE))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(choose_text("国家数量（颜色越深数量越多）", "Country count (darker means more)", USE_CHINESE))
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_scenario_reduction(type_summary_df: pd.DataFrame, output_path: Path) -> None:
        if type_summary_df.empty:
            return
        ordered = type_summary_df.sort_values("mean_gap_reduction", ascending=False, kind="stable")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bars = ax.bar(
            [type_display_label(label) for label in ordered["response_diagnosis_type"]],
            ordered["mean_gap_reduction"],
            color="#2a9d8f",
        )
        ax.bar_label(bars, padding=3, fmt="%.3f")
        ax.set_title(choose_text("补最短板后的平均缺口缩减", "Average gap reduction after weakest-link upgrade", USE_CHINESE))
        ax.set_ylabel(choose_text("平均缺口缩减", "Mean gap reduction", USE_CHINESE))
        plt.xticks(rotation=15)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def plot_efficiency_quadrant_summary(quadrant_summary: pd.DataFrame, output_path: Path) -> None:
        if quadrant_summary.empty:
            return
        ordered = quadrant_summary.sort_values("countries", ascending=False, kind="stable")
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        labels = [type_display_label(label) for label in ordered["response_diagnosis_type"]]
        bars = ax.bar(labels, ordered["countries"], color=[TYPE_COLORS.get(label, "#457b9d") for label in ordered["response_diagnosis_type"]])
        ax.bar_label(bars, padding=3)
        ax.set_title(choose_text("效率象限国家分布", "Country distribution by efficiency quadrant", USE_CHINESE))
        ax.set_ylabel(choose_text("国家数量", "Number of countries", USE_CHINESE))
        plt.xticks(rotation=15)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


    def main() -> None:
        parser = argparse.ArgumentParser(description="Run lockable Module C response mismatch diagnosis.")
        parser.add_argument("--project-root", type=Path, default=None)
        parser.add_argument("--input-file", type=Path, default=None, help="Optional explicit response panel path")
        parser.add_argument("--typology-file", type=Path, default=None, help="Optional explicit vulnerability typology panel path")
        parser.add_argument("--risk-summary-file", type=Path, default=None, help="Optional explicit Module B type risk summary path")
        parser.add_argument("--latest-year", type=int, default=None)
        parser.add_argument("--context-weight", type=float, default=0.2, help="Weight of context support in adjusted response score")
        parser.add_argument("--scenario-step", type=float, default=0.10, help="Scenario uplift added to weakest component percentile scores")
        parser.add_argument("--scenario-top-components", type=int, default=3, help="Number of weakest components to improve in optimization scenario")
        parser.add_argument("--sparse-feature-threshold", type=float, default=0.65, help="Drop resource columns whose missingness is strictly above this threshold")
        parser.add_argument(
            "--resource-main-missingness-threshold",
            type=float,
            default=0.40,
            help="Drop Module C resource columns from the main response score when missingness is strictly above this threshold",
        )
        args = parser.parse_args()

        if not (0 <= args.context_weight <= 0.5):
            raise ValueError("context-weight must be between 0 and 0.5")
        if not (0 < args.scenario_step <= 0.5):
            raise ValueError("scenario-step must be between 0 and 0.5")
        if args.scenario_top_components < 1:
            raise ValueError("scenario-top-components must be >= 1")
        if not (0.0 <= args.sparse_feature_threshold < 1.0):
            raise ValueError("sparse-feature-threshold must satisfy 0 <= threshold < 1")
        if not (0.0 <= args.resource_main_missingness_threshold <= args.sparse_feature_threshold):
            raise ValueError("resource-main-missingness-threshold must satisfy 0 <= threshold <= sparse-feature-threshold")

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        input_file = args.input_file.expanduser().resolve() if args.input_file else dirs["simulation"] / "response_panel.csv"
        typology_file = args.typology_file.expanduser().resolve() if args.typology_file else dirs["simulation"] / "vulnerability_typology_panel.csv"
        risk_summary_file = (
            args.risk_summary_file.expanduser().resolve()
            if args.risk_summary_file
            else report_asset_path(dirs["report"], "risk_attribution_type_risk_summary.csv")
        )

        if not input_file.exists():
            raise FileNotFoundError(f"Response panel not found: {input_file}")
        if not typology_file.exists():
            raise FileNotFoundError(f"Vulnerability typology panel not found: {typology_file}")

        df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)
        typology_df = pd.read_csv(typology_file, encoding="utf-8-sig", low_memory=False)

        if not {"iso3", "year"}.issubset(df.columns):
            raise RuntimeError("Response panel must contain iso3 and year")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        merge_columns = ["iso3", "year", "vulnerability_type_code", "vulnerability_type_label", "overall_vulnerability_score"]
        missing_typology_cols = sorted(set(merge_columns) - set(typology_df.columns))
        if missing_typology_cols:
            raise RuntimeError(f"Vulnerability typology panel missing columns: {missing_typology_cols}")

        response_df = df.merge(
            typology_df[merge_columns].drop_duplicates(["iso3", "year"], keep="last"),
            on=["iso3", "year"],
            how="left",
        )
        response_df = response_df.dropna(subset=["vulnerability_type_label"]).copy()
        if response_df.empty:
            raise RuntimeError("No rows with vulnerability type labels after merging typology information")

        gbd_columns = choose_existing(response_df, GBD_COLUMNS)
        risk_columns = choose_existing(response_df, RISK_COLUMNS)
        resource_positive_pre = choose_existing(response_df, RESOURCE_POSITIVE_COLUMNS)
        resource_negative_pre = choose_existing(response_df, RESOURCE_NEGATIVE_COLUMNS)
        (
            resource_positive,
            resource_negative,
            dropped_sparse_resource_columns,
            dropped_moderate_missing_resource_columns,
            retained_resource_columns,
        ) = filter_sparse_resource_columns(
            response_df,
            resource_positive_pre,
            resource_negative_pre,
            args.sparse_feature_threshold,
            args.resource_main_missingness_threshold,
        )
        context_columns = choose_existing(response_df, CONTEXT_COLUMNS)

        if not gbd_columns:
            raise RuntimeError("No per100k GBD burden columns available for Module C")
        if not risk_columns:
            raise RuntimeError("No per100k DBD risk columns available for Module C")
        if not (resource_positive or resource_negative):
            raise RuntimeError("No resource response columns available for Module C")

        response_df["burden_pressure_score"] = pd.concat([percentile_score(response_df[column]) for column in gbd_columns], axis=1).mean(axis=1)
        response_df["risk_pressure_score"] = pd.concat([percentile_score(response_df[column]) for column in risk_columns], axis=1).mean(axis=1)
        response_df["combined_pressure_score"] = pd.concat(
            [response_df["burden_pressure_score"], response_df["risk_pressure_score"]],
            axis=1,
        ).mean(axis=1)

        resource_component_scores = component_score_frame(response_df, resource_positive, resource_negative)
        response_df["resource_response_score"] = resource_component_scores.mean(axis=1)

        if context_columns:
            context_scores = pd.concat(
                [percentile_score(response_df[column], reverse=(column == "wdi_gini")) for column in context_columns],
                axis=1,
            )
            response_df["context_support_score"] = context_scores.mean(axis=1)
            response_df["adjusted_response_score"] = (
                (1 - args.context_weight) * response_df["resource_response_score"]
                + args.context_weight * response_df["context_support_score"]
            )
        else:
            response_df["context_support_score"] = np.nan
            response_df["adjusted_response_score"] = response_df["resource_response_score"]

        response_df["adaptation_gap_score"] = response_df["combined_pressure_score"] - response_df["adjusted_response_score"]
        response_df["response_diagnosis_type"] = response_df.apply(
            lambda row: classify_response_type(row["combined_pressure_score"], row["adjusted_response_score"]),
            axis=1,
        )

        component_score_columns = []
        for column in resource_component_scores.columns:
            score_column = f"component__{column}"
            response_df[score_column] = resource_component_scores[column]
            component_score_columns.append(score_column)

        response_df["weakest_resource_components"] = response_df.apply(
            lambda row: summarize_weak_components(row, component_score_columns, top_n=3),
            axis=1,
        )

        dominant_risk_storylines = {}
        if risk_summary_file.exists():
            risk_summary_df = pd.read_csv(risk_summary_file, encoding="utf-8-sig", low_memory=False)
            dominant_risk_storylines = build_type_storylines(risk_summary_df)
            response_df["dominant_risk_triplet"] = response_df["vulnerability_type_label"].map(dominant_risk_storylines)
        else:
            response_df["dominant_risk_triplet"] = np.nan

        output_panel = dirs["simulation"] / "response_diagnosis_panel.csv"
        response_df.to_csv(output_panel, index=False, encoding="utf-8-sig")

        latest_df = (
            response_df.dropna(subset=["response_diagnosis_type"])
            .sort_values(["iso3", "year"], kind="stable")
            .groupby("iso3", as_index=False)
            .tail(1)
        )
        latest_year = args.latest_year if args.latest_year is not None else int(pd.to_numeric(latest_df["year"], errors="coerce").dropna().max())
        latest_year_df = latest_df.loc[latest_df["year"] == latest_year].copy()

        latest_output = report_asset_path(dirs["report"], "country_response_diagnosis_latest.csv")
        latest_year_df.to_csv(latest_output, index=False, encoding="utf-8-sig")

        vulnerability_response_matrix = pd.crosstab(
            latest_year_df["vulnerability_type_label"],
            latest_year_df["response_diagnosis_type"],
        )
        vulnerability_response_matrix_output = report_asset_path(dirs["report"], "response_diagnosis_matrix.csv")
        vulnerability_response_matrix.to_csv(vulnerability_response_matrix_output, encoding="utf-8-sig")

        stage_response_matrix = pd.DataFrame()
        stage_response_matrix_output = report_asset_path(dirs["report"], "response_stage_type_matrix.csv")
        stage_file = dirs["simulation"] / "stage_identification_panel.csv"
        if stage_file.exists() and not latest_year_df.empty:
            stage_df = pd.read_csv(stage_file, encoding="utf-8-sig", low_memory=False)
            if {"iso3", "year", "transition_stage_label"}.issubset(stage_df.columns):
                stage_df["year"] = pd.to_numeric(stage_df["year"], errors="coerce")
                stage_latest = stage_df.loc[
                    stage_df["year"] == latest_year,
                    ["iso3", "transition_stage_label"],
                ].dropna(subset=["transition_stage_label"])
                stage_response_df = latest_year_df.loc[:, ["iso3", "response_diagnosis_type"]].merge(
                    stage_latest,
                    on="iso3",
                    how="inner",
                )
                stage_response_matrix = pd.crosstab(
                    stage_response_df["transition_stage_label"],
                    stage_response_df["response_diagnosis_type"],
                )
        stage_response_matrix.to_csv(stage_response_matrix_output, encoding="utf-8-sig")

        response_type_profiles = (
            latest_year_df.groupby("response_diagnosis_type", dropna=False)[
                [
                    "burden_pressure_score",
                    "risk_pressure_score",
                    "combined_pressure_score",
                    "resource_response_score",
                    "context_support_score",
                    "adjusted_response_score",
                    "adaptation_gap_score",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        response_type_profiles_output = report_asset_path(dirs["report"], "response_type_profiles.csv")
        response_type_profiles.to_csv(response_type_profiles_output, index=False, encoding="utf-8-sig")

        component_long = []
        for column in component_score_columns:
            raw_column = column.replace("component__", "")
            component_long.append(
                latest_year_df.loc[:, ["response_diagnosis_type", column]]
                .rename(columns={column: "mean_component_score"})
                .assign(resource_component=raw_column, resource_component_label=resource_display_label(raw_column))
            )
        type_component_df = (
            pd.concat(component_long, axis=0, ignore_index=True)
            .groupby(["response_diagnosis_type", "resource_component", "resource_component_label"], dropna=False)["mean_component_score"]
            .mean()
            .reset_index()
            .sort_values(["response_diagnosis_type", "mean_component_score"], ascending=[True, True], kind="stable")
        )
        type_component_output = report_asset_path(dirs["report"], "response_priority_components_by_type.csv")
        type_component_df.to_csv(type_component_output, index=False, encoding="utf-8-sig")

        priority_cards = (
            latest_year_df.sort_values(["response_diagnosis_type", "adaptation_gap_score"], ascending=[True, False], kind="stable")
            .groupby("response_diagnosis_type", as_index=False)
            .head(6)
            .copy()
        )
        priority_cards["rank_within_response_type"] = priority_cards.groupby("response_diagnosis_type").cumcount() + 1
        priority_card_columns = [
            "iso3",
            "year",
            "rank_within_response_type",
            "vulnerability_type_label",
            "response_diagnosis_type",
            "dominant_risk_triplet",
            "weakest_resource_components",
            "combined_pressure_score",
            "adjusted_response_score",
            "adaptation_gap_score",
            "burden_pressure_score",
            "risk_pressure_score",
            "resource_response_score",
            "context_support_score",
            "beds_10k",
            "doctors_10k",
            "nurses_10k",
            "uhc_index",
            "che_pct_gdp",
            "hdi",
            "wdi_gdp_per_capita",
            "wdi_gini",
            "wdi_population_65_plus_pct",
            "wdi_urban_population_pct",
        ]
        priority_cards = priority_cards.loc[:, [column for column in priority_card_columns if column in priority_cards.columns]]
        priority_cards_output = report_asset_path(dirs["report"], "response_country_priority_cards.csv")
        priority_cards.to_csv(priority_cards_output, index=False, encoding="utf-8-sig")

        optimization_df = simulate_component_upgrade(
            latest_year_df,
            component_score_columns,
            context_weight=args.context_weight,
            scenario_step=args.scenario_step,
            top_n=args.scenario_top_components,
        )
        optimization_columns = [
            "iso3",
            "year",
            "vulnerability_type_label",
            "response_diagnosis_type",
            "scenario_target_components",
            "combined_pressure_score",
            "adjusted_response_score",
            "optimized_adjusted_response_score",
            "adaptation_gap_score",
            "optimized_adaptation_gap_score",
            "gap_reduction_if_upgraded",
            "response_lift_needed_to_balance",
        ]
        optimization_output = report_asset_path(dirs["report"], "response_optimization_scenarios.csv")
        optimization_df.loc[:, [column for column in optimization_columns if column in optimization_df.columns]].to_csv(
            optimization_output,
            index=False,
            encoding="utf-8-sig",
        )

        optimization_type_summary = (
            optimization_df.groupby("response_diagnosis_type", dropna=False)
            .agg(
                countries=("iso3", "count"),
                mean_gap=("adaptation_gap_score", "mean"),
                mean_gap_reduction=("gap_reduction_if_upgraded", "mean"),
                mean_optimized_gap=("optimized_adaptation_gap_score", "mean"),
                typical_target_components=("scenario_target_components", "first"),
            )
            .reset_index()
            .sort_values("mean_gap_reduction", ascending=False, kind="stable")
        )
        optimization_type_output = report_asset_path(dirs["report"], "response_optimization_type_summary.csv")
        optimization_type_summary.to_csv(optimization_type_output, index=False, encoding="utf-8-sig")

        quadrant_summary = (
            optimization_df.groupby("response_diagnosis_type", dropna=False)
            .agg(
                countries=("iso3", "count"),
                mean_combined_pressure=("combined_pressure_score", "mean"),
                mean_adjusted_response=("adjusted_response_score", "mean"),
                mean_gap=("adaptation_gap_score", "mean"),
                mean_gap_reduction=("gap_reduction_if_upgraded", "mean"),
                mean_response_lift_needed=("response_lift_needed_to_balance", "mean"),
            )
            .reset_index()
            .sort_values("mean_gap", ascending=False, kind="stable")
        )
        if not quadrant_summary.empty:
            quadrant_summary["country_share"] = quadrant_summary["countries"] / float(quadrant_summary["countries"].sum())
        quadrant_summary_output = report_asset_path(dirs["report"], "response_efficiency_quadrant_summary.csv")
        quadrant_summary.to_csv(quadrant_summary_output, index=False, encoding="utf-8-sig")

        allocation_plan = (
            optimization_df.groupby(["response_diagnosis_type", "vulnerability_type_label"], dropna=False)
            .agg(
                countries=("iso3", "count"),
                dominant_risk_triplet=("dominant_risk_triplet", "first"),
                typical_target_components=("scenario_target_components", "first"),
                mean_gap=("adaptation_gap_score", "mean"),
                mean_gap_reduction=("gap_reduction_if_upgraded", "mean"),
                mean_optimized_gap=("optimized_adaptation_gap_score", "mean"),
                mean_response_lift_needed=("response_lift_needed_to_balance", "mean"),
            )
            .reset_index()
            .sort_values(["mean_gap_reduction", "mean_gap"], ascending=[False, False], kind="stable")
        )
        if not allocation_plan.empty:
            allocation_plan["rank_within_response_type"] = allocation_plan.groupby("response_diagnosis_type").cumcount() + 1
        allocation_plan_output = report_asset_path(dirs["report"], "response_incremental_allocation_plan.csv")
        allocation_plan.to_csv(allocation_plan_output, index=False, encoding="utf-8-sig")

        response_storylines = (
            priority_cards.groupby("response_diagnosis_type", dropna=False)
            .agg(
                countries=("iso3", lambda values: " / ".join(values.astype(str).tolist())),
                dominant_risk_triplet=("dominant_risk_triplet", "first"),
                weakest_resource_components=("weakest_resource_components", "first"),
                mean_gap=("adaptation_gap_score", "mean"),
            )
            .reset_index()
            .sort_values("mean_gap", ascending=False, kind="stable")
        )
        response_storylines_output = report_asset_path(dirs["report"], "response_type_storylines.csv")
        response_storylines.to_csv(response_storylines_output, index=False, encoding="utf-8-sig")

        if not latest_year_df.empty:
            plot_pressure_response_scatter(latest_year_df, dirs["figures"] / "response_pressure_vs_capacity_latest.png")
            plot_type_counts(latest_year_df, dirs["figures"] / "response_type_counts.png")
            plot_gap_top(latest_year_df, dirs["figures"] / "response_gap_top20.png")
            if not vulnerability_response_matrix.empty:
                plot_vulnerability_response_heatmap(vulnerability_response_matrix.fillna(0), dirs["figures"] / "response_vulnerability_type_heatmap.png")
            if not stage_response_matrix.empty:
                plot_stage_response_heatmap(stage_response_matrix.fillna(0), dirs["figures"] / "response_stage_type_heatmap.png", latest_year)
            if not type_component_df.empty:
                plot_weak_component_heatmap(type_component_df, dirs["figures"] / "response_weak_components_heatmap.png")
            if not optimization_type_summary.empty:
                plot_scenario_reduction(optimization_type_summary, dirs["figures"] / "response_gap_reduction_scenarios.png")
            if not quadrant_summary.empty:
                plot_efficiency_quadrant_summary(quadrant_summary, dirs["figures"] / "response_efficiency_quadrant_summary.png")

        summary = {
            "project_root": project_root.as_posix(),
            "input_file": input_file.as_posix(),
            "typology_file": typology_file.as_posix(),
            "risk_summary_file": risk_summary_file.as_posix() if risk_summary_file.exists() else None,
            "output_panel": output_panel.as_posix(),
            "rows_with_type_labels": int(response_df.shape[0]),
            "latest_year_used": latest_year,
            "latest_country_rows": int(latest_year_df.shape[0]),
            "gbd_columns_used": gbd_columns,
            "risk_columns_used": risk_columns,
            "resource_positive_columns_used": resource_positive,
            "resource_negative_columns_used": resource_negative,
            "resource_positive_columns_pre_sparse_filter": resource_positive_pre,
            "resource_negative_columns_pre_sparse_filter": resource_negative_pre,
            "sparse_feature_threshold": args.sparse_feature_threshold,
            "resource_main_missingness_threshold": args.resource_main_missingness_threshold,
            "dropped_sparse_resource_columns": dropped_sparse_resource_columns,
            "dropped_moderate_missing_resource_columns": dropped_moderate_missing_resource_columns,
            "dropped_resource_columns": dropped_sparse_resource_columns + dropped_moderate_missing_resource_columns,
            "retained_resource_columns": retained_resource_columns,
            "context_columns_used": context_columns,
            "context_weight": args.context_weight,
            "scenario_step": args.scenario_step,
            "scenario_top_components": args.scenario_top_components,
            "response_type_counts_latest": latest_year_df["response_diagnosis_type"].value_counts(dropna=False).to_dict(),
            "vulnerability_type_counts_latest": latest_year_df["vulnerability_type_label"].value_counts(dropna=False).to_dict(),
            "output_files": {
                "latest_diagnosis": latest_output.as_posix(),
                "vulnerability_response_matrix": vulnerability_response_matrix_output.as_posix(),
                "stage_response_matrix": stage_response_matrix_output.as_posix(),
                "response_type_profiles": response_type_profiles_output.as_posix(),
                "priority_components_by_type": type_component_output.as_posix(),
                "country_priority_cards": priority_cards_output.as_posix(),
                "response_type_storylines": response_storylines_output.as_posix(),
                "efficiency_quadrant_summary": quadrant_summary_output.as_posix(),
                "incremental_allocation_plan": allocation_plan_output.as_posix(),
                "optimization_scenarios": optimization_output.as_posix(),
                "optimization_type_summary": optimization_type_output.as_posix(),
                "pressure_response_scatter": (dirs["figures"] / "response_pressure_vs_capacity_latest.png").as_posix(),
                "response_type_counts": (dirs["figures"] / "response_type_counts.png").as_posix(),
                "gap_top20": (dirs["figures"] / "response_gap_top20.png").as_posix(),
                "vulnerability_response_heatmap": (dirs["figures"] / "response_vulnerability_type_heatmap.png").as_posix(),
                "stage_response_heatmap": (dirs["figures"] / "response_stage_type_heatmap.png").as_posix(),
                "weak_components_heatmap": (dirs["figures"] / "response_weak_components_heatmap.png").as_posix(),
                "gap_reduction_scenarios": (dirs["figures"] / "response_gap_reduction_scenarios.png").as_posix(),
                "efficiency_quadrant_summary_figure": (dirs["figures"] / "response_efficiency_quadrant_summary.png").as_posix(),
            },
        }
        summary_path = report_asset_path(dirs["report"], "response_diagnosis_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


def _namespace_run_result_contamination_audit():
    __name__ = 'run_result_contamination_audit'
    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from foundation import detect_external_data_root as shared_detect_external_data_root
    from foundation import detect_project_root as shared_detect_project_root

    KEY_COLUMNS = ["iso3", "year"]
    FORMAL_ANALYSIS_SCOPE = "UN_193_PLUS_2_OBSERVERS"
    FORMAL_EXPECTED_COUNTRY_COUNT = 195
    FORMAL_REQUIRED_CODES = {"PSE", "VAT"}
    FORMAL_EXCLUDED_CODES = {"COK", "TWN"}
    FORMAL_COMPLETE_PANEL_LABELS = {"master_panel", "response_panel", "policy_identification_panel"}

    NONNEGATIVE_COLUMNS = [
        "analysis_population_total",
        "wdi_population_total",
        "population_thousands",
        "gbd_rate_cardiovascular_diseases",
        "gbd_rate_chronic_respiratory_diseases",
        "gbd_rate_neoplasms",
        "gbd_rate_diabetes_kidney",
        "gbd_rate_cardiovascular_diseases_per100k",
        "gbd_rate_chronic_respiratory_diseases_per100k",
        "gbd_rate_neoplasms_per100k",
        "gbd_rate_diabetes_kidney_per100k",
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
        "beds_10k",
        "doctors_10k",
        "nurses_10k",
        "wdi_hospital_beds",
        "wdi_physicians",
        "wdi_nurses_midwives",
        "wdi_gdp_per_capita",
    ]

    RANGE_RULES = {
        "hdi": (0.0, 1.0),
        "uhc_index": (0.0, 100.0),
        "wdi_gini": (0.0, 100.0),
        "wdi_population_65_plus_pct": (0.0, 100.0),
        "wdi_urban_population_pct": (0.0, 100.0),
        "govt_he_pct": (0.0, 100.0),
        "ext_he_pct": (0.0, 100.0),
        "wdi_out_of_pocket_pct": (0.0, 100.0),
        "wdi_government_health_expenditure_pct": (0.0, 100.0),
        "wdi_external_health_expenditure_pct": (0.0, 100.0),
        "che_pct_gdp": (0.0, 100.0),
        "wb_hnp_skilled_birth_attendance": (0.0, 100.0),
        "wb_hnp_immunization_dpt": (0.0, 100.0),
        "wb_hnp_immunization_measles": (0.0, 100.0),
        "wb_hnp_immunization_hepb3": (0.0, 100.0),
    }

    PER100K_PAIRS = [
        ("gbd_rate_cardiovascular_diseases", "gbd_rate_cardiovascular_diseases_per100k"),
        ("gbd_rate_chronic_respiratory_diseases", "gbd_rate_chronic_respiratory_diseases_per100k"),
        ("gbd_rate_neoplasms", "gbd_rate_neoplasms_per100k"),
        ("gbd_rate_diabetes_kidney", "gbd_rate_diabetes_kidney_per100k"),
        ("dbd_smoking", "dbd_smoking_per100k"),
        ("dbd_pm25", "dbd_pm25_per100k"),
        ("dbd_high_bmi", "dbd_high_bmi_per100k"),
        ("dbd_high_glucose", "dbd_high_glucose_per100k"),
        ("dbd_high_sbp", "dbd_high_sbp_per100k"),
        ("dbd_dietary_risks", "dbd_dietary_risks_per100k"),
    ]

    A_DIMENSION_GROUPS = {
        "health_burden": [
            "gbd_rate_cardiovascular_diseases_per100k",
            "gbd_rate_chronic_respiratory_diseases_per100k",
            "gbd_rate_neoplasms_per100k",
            "gbd_rate_diabetes_kidney_per100k",
            "gbd_rate_cardiovascular_diseases",
            "gbd_rate_chronic_respiratory_diseases",
            "gbd_rate_neoplasms",
            "gbd_rate_diabetes_kidney",
        ],
        "risk_exposure": [
            "dbd_smoking_per100k",
            "dbd_pm25_per100k",
            "dbd_high_bmi_per100k",
            "dbd_high_glucose_per100k",
            "dbd_high_sbp_per100k",
            "dbd_dietary_risks_per100k",
            "dbd_smoking",
            "dbd_pm25",
            "dbd_high_bmi",
            "dbd_high_glucose",
            "dbd_high_sbp",
            "dbd_dietary_risks",
        ],
        "system_fragility": [
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
        ],
        "socioeconomic_fragility": [
            "hdi",
            "wdi_gdp_per_capita",
            "wdi_gini",
            "wdi_population_65_plus_pct",
            "wdi_urban_population_pct",
        ],
    }

    B_DISEASE_COLUMNS = [
        "gbd_rate_cardiovascular_diseases_per100k",
        "gbd_rate_chronic_respiratory_diseases_per100k",
        "gbd_rate_neoplasms_per100k",
        "gbd_rate_diabetes_kidney_per100k",
    ]
    B_RISK_COLUMNS = [
        "dbd_smoking_per100k",
        "dbd_pm25_per100k",
        "dbd_high_bmi_per100k",
        "dbd_high_glucose_per100k",
        "dbd_high_sbp_per100k",
        "dbd_dietary_risks_per100k",
    ]

    C_GBD_COLUMNS = B_DISEASE_COLUMNS
    C_RISK_COLUMNS = B_RISK_COLUMNS
    C_RESOURCE_COLUMNS = [
        "beds_10k",
        "doctors_10k",
        "nurses_10k",
        "uhc_index",
        "che_pct_gdp",
        "che_pc_usd",
        "govt_he_pct",
        "wdi_health_expenditure_per_capita",
        "wdi_government_health_expenditure_pct",
        "wdi_hospital_beds",
        "wdi_physicians",
        "wdi_nurses_midwives",
        "wb_hnp_hci",
        "wb_hnp_skilled_birth_attendance",
        "wb_hnp_immunization_dpt",
        "wb_hnp_immunization_measles",
        "wb_hnp_immunization_hepb3",
        "ext_he_pct",
        "wdi_external_health_expenditure_pct",
        "wdi_out_of_pocket_pct",
    ]


    def detect_project_root(explicit: Path | None) -> Path:
        return shared_detect_project_root(explicit)


    def ensure_dirs(project_root: Path) -> dict[str, Path]:
        dirs = {
            "simulation": project_root / "04_simulation",
            "report": project_root / "06_report_assets",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs


    def read_json(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


    def read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


    def resolve_module_a_dimension_groups(vulnerability_summary: dict[str, object]) -> dict[str, list[str]]:
        retained = vulnerability_summary.get("retained_features_by_dimension")
        source_map = vulnerability_summary.get("feature_source_columns", {})
        if not isinstance(retained, dict):
            return A_DIMENSION_GROUPS
        dimension_groups: dict[str, list[str]] = {}
        for dimension, default_columns in A_DIMENSION_GROUPS.items():
            analysis_columns = retained.get(dimension)
            if not isinstance(analysis_columns, list):
                dimension_groups[dimension] = default_columns
                continue
            resolved_columns: list[str] = []
            for analysis_column in analysis_columns:
                source_column = source_map.get(analysis_column, analysis_column) if isinstance(source_map, dict) else analysis_column
                if isinstance(source_column, str) and source_column not in resolved_columns:
                    resolved_columns.append(source_column)
            dimension_groups[dimension] = resolved_columns
        return dimension_groups


    def resolve_module_c_resource_columns(response_summary: dict[str, object]) -> list[str]:
        retained = response_summary.get("retained_resource_columns")
        if isinstance(retained, list):
            return [str(column) for column in retained if isinstance(column, str)]
        return C_RESOURCE_COLUMNS


    def add_row(
        rows: list[dict[str, object]],
        category: str,
        item: str,
        status: str,
        severity: str,
        detail: str,
        path: Path | str = "",
    ) -> None:
        rows.append(
            {
                "category": category,
                "item": item,
                "status": status,
                "severity": severity,
                "detail": detail,
                "path": str(path) if path else "",
            }
        )


    def status_from_rate(rate: float, warn_at: float, fail_at: float) -> tuple[str, str]:
        if rate >= fail_at:
            return "fail", "high"
        if rate >= warn_at:
            return "warn", "medium"
        return "pass", "info"


    def audit_structure(df: pd.DataFrame, label: str, path: Path, rows: list[dict[str, object]]) -> None:
        if df.empty:
            add_row(rows, "structure", f"{label}存在性", "fail", "high", "文件缺失或为空", path)
            return
        missing = sorted(set(KEY_COLUMNS) - set(df.columns))
        if missing:
            add_row(rows, "structure", f"{label}主键字段", "fail", "high", f"缺少字段 {missing}", path)
            return
        dup_count = int(df.duplicated(KEY_COLUMNS).sum())
        null_iso3 = int(df["iso3"].isna().sum())
        null_year = int(pd.to_numeric(df["year"], errors="coerce").isna().sum())
        status = "pass" if dup_count == 0 and null_iso3 == 0 and null_year == 0 else "fail"
        severity = "info" if status == "pass" else "high"
        add_row(
            rows,
            "structure",
            f"{label}主键完整性",
            status,
            severity,
            f"duplicate_iso3_year={dup_count}, null_iso3={null_iso3}, null_year={null_year}",
            path,
        )


    def audit_nonnegative_and_ranges(
        df: pd.DataFrame,
        label: str,
        path: Path,
        rows: list[dict[str, object]],
        columns: list[str],
    ) -> None:
        for column in columns:
            if column not in df.columns:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            valid = numeric.notna()
            if not valid.any():
                continue
            if column in NONNEGATIVE_COLUMNS:
                rate = float((numeric.loc[valid] < 0).mean())
                status, severity = status_from_rate(rate, warn_at=0.001, fail_at=0.01)
                add_row(
                    rows,
                    "values",
                    f"{label}:{column}:negative_check",
                    status,
                    severity,
                    f"negative_rate={rate:.4%}, non_null={int(valid.sum())}",
                    path,
                )
            if column in RANGE_RULES:
                lower, upper = RANGE_RULES[column]
                out_of_range = ((numeric.loc[valid] < lower) | (numeric.loc[valid] > upper)).mean()
                rate = float(out_of_range)
                status, severity = status_from_rate(rate, warn_at=0.001, fail_at=0.01)
                add_row(
                    rows,
                    "values",
                    f"{label}:{column}:range_check",
                    status,
                    severity,
                    f"out_of_range_rate={rate:.4%}, expected=[{lower}, {upper}], non_null={int(valid.sum())}",
                    path,
                )


    def audit_per100k_consistency(
        df: pd.DataFrame,
        label: str,
        path: Path,
        rows: list[dict[str, object]],
    ) -> None:
        if "analysis_population_total" not in df.columns:
            add_row(rows, "consistency", f"{label}:analysis_population_total", "warn", "medium", "缺少 analysis_population_total，无法校验 per100k 一致性", path)
            return
        population = pd.to_numeric(df["analysis_population_total"], errors="coerce")
        for source, derived in PER100K_PAIRS:
            if source not in df.columns or derived not in df.columns:
                continue
            source_num = pd.to_numeric(df[source], errors="coerce")
            derived_num = pd.to_numeric(df[derived], errors="coerce")
            mask = source_num.notna() & derived_num.notna() & population.notna() & (population > 0)
            if not mask.any():
                continue
            recomputed = (source_num.loc[mask] / population.loc[mask]) * 100000.0
            mismatch_rate = float((~np.isclose(derived_num.loc[mask], recomputed, rtol=1e-6, atol=1e-6)).mean())
            status, severity = status_from_rate(mismatch_rate, warn_at=0.001, fail_at=0.01)
            add_row(
                rows,
                "consistency",
                f"{label}:{derived}:per100k_consistency",
                status,
                severity,
                f"mismatch_rate={mismatch_rate:.4%}, checked_rows={int(mask.sum())}",
                path,
            )


    def compare_master_to_response(
        master: pd.DataFrame,
        response: pd.DataFrame,
        master_path: Path,
        response_path: Path,
        rows: list[dict[str, object]],
    ) -> None:
        if master.empty or response.empty:
            return
        shared_columns = [
            column
            for column in [
                "analysis_population_total",
                "dbd_smoking_per100k",
                "gbd_rate_cardiovascular_diseases_per100k",
                "hdi",
                "beds_10k",
                "uhc_index",
            ]
            if column in master.columns and column in response.columns
        ]
        if not shared_columns:
            return
        merged = response[KEY_COLUMNS + shared_columns].merge(
            master[KEY_COLUMNS + shared_columns],
            on=KEY_COLUMNS,
            how="left",
            suffixes=("_response", "_master"),
        )
        if merged.empty:
            return
        for column in shared_columns:
            left = pd.to_numeric(merged[f"{column}_response"], errors="coerce")
            right = pd.to_numeric(merged[f"{column}_master"], errors="coerce")
            mask = left.notna() | right.notna()
            if not mask.any():
                continue
            mismatch_rate = float((~np.isclose(left.loc[mask], right.loc[mask], rtol=1e-9, atol=1e-9)).mean())
            status, severity = ("pass", "info") if mismatch_rate == 0.0 else status_from_rate(mismatch_rate, warn_at=0.0001, fail_at=0.001)
            add_row(
                rows,
                "consistency",
                f"response_vs_master:{column}",
                status,
                severity,
                f"mismatch_rate={mismatch_rate:.4%}, compared_rows={int(mask.sum())}",
                f"{response_path} <- {master_path}",
            )


    def audit_missingness(
        df: pd.DataFrame,
        label: str,
        path: Path,
        rows: list[dict[str, object]],
        columns: list[str],
        warn_at: float = 0.4,
        fail_at: float = 0.7,
    ) -> None:
        usable = [column for column in columns if column in df.columns]
        if not usable:
            add_row(rows, "coverage", f"{label}:required_columns", "warn", "medium", "无可用列参与此模块检查", path)
            return
        for column in usable:
            rate = float(df[column].isna().mean())
            status, severity = status_from_rate(rate, warn_at=warn_at, fail_at=fail_at)
            add_row(
                rows,
                "coverage",
                f"{label}:{column}:missingness",
                status,
                severity,
                f"missing_rate={rate:.4%}",
                path,
            )


    def audit_scope_contamination(
        df: pd.DataFrame,
        label: str,
        path: Path,
        rows: list[dict[str, object]],
        scope_registry: pd.DataFrame,
    ) -> None:
        if df.empty or scope_registry.empty or "iso3" not in df.columns or "iso3" not in scope_registry.columns:
            return
        merged = (
            df.loc[:, ["iso3"]]
            .dropna()
            .assign(iso3=lambda frame: frame["iso3"].astype(str).str.upper())
            .drop_duplicates()
            .merge(
                scope_registry.loc[:, ["iso3", "scope_category"]].drop_duplicates().assign(iso3=lambda frame: frame["iso3"].astype(str).str.upper()),
                on="iso3",
                how="left",
            )
        )
        offending = merged.loc[merged["scope_category"].isin(["aggregate_region", "non_sovereign", "world_or_income_group"])].copy()
        status = "pass" if offending.empty else "fail"
        severity = "info" if offending.empty else "high"
        add_row(
            rows,
            "scope",
            f"{label}:formal_scope_only",
            status,
            severity,
            f"offending_entities={sorted(offending['iso3'].tolist()) if not offending.empty else []}",
            path,
        )
        present_codes = set(merged["iso3"].dropna().astype(str).str.upper())
        excluded_present = sorted(FORMAL_EXCLUDED_CODES.intersection(present_codes))
        add_row(
            rows,
            "scope",
            f"{label}:excluded_entities_absent",
            "pass" if not excluded_present else "fail",
            "info" if not excluded_present else "high",
            f"formal_scope={FORMAL_ANALYSIS_SCOPE}, excluded_present={excluded_present}",
            path,
        )
        if label in FORMAL_COMPLETE_PANEL_LABELS:
            country_count = len(present_codes)
            missing_required = sorted(FORMAL_REQUIRED_CODES.difference(present_codes))
            add_row(
                rows,
                "scope",
                f"{label}:formal_country_count",
                "pass" if country_count == FORMAL_EXPECTED_COUNTRY_COUNT else "fail",
                "info" if country_count == FORMAL_EXPECTED_COUNTRY_COUNT else "high",
                f"formal_scope={FORMAL_ANALYSIS_SCOPE}, unique_iso3={country_count}, expected={FORMAL_EXPECTED_COUNTRY_COUNT}",
                path,
            )
            add_row(
                rows,
                "scope",
                f"{label}:required_observer_entities_present",
                "pass" if not missing_required else "fail",
                "info" if not missing_required else "high",
                f"required={sorted(FORMAL_REQUIRED_CODES)}, missing={missing_required}",
                path,
            )


    def audit_module_a(
        df: pd.DataFrame,
        path: Path,
        rows: list[dict[str, object]],
        dimension_groups: dict[str, list[str]],
    ) -> None:
        observed = {}
        for dimension, columns in dimension_groups.items():
            usable = [column for column in columns if column in df.columns]
            if not usable:
                observed[dimension] = pd.Series(False, index=df.index)
                add_row(rows, "module_A", f"{dimension}:availability", "fail", "high", "该维度没有可用列", path)
                continue
            observed[dimension] = df[usable].notna().any(axis=1)
            coverage = float(observed[dimension].mean())
            status, severity = status_from_rate(1.0 - coverage, warn_at=0.2, fail_at=0.5)
            add_row(
                rows,
                "module_A",
                f"{dimension}:dimension_observed",
                status,
                severity,
                f"row_coverage={coverage:.4%}, usable_columns={usable}",
                path,
            )
        if observed:
            dim_df = pd.DataFrame(observed)
            eligible_rate = float((dim_df.sum(axis=1) >= 3).mean())
            status, severity = ("pass", "info") if eligible_rate >= 0.6 else ("warn", "medium") if eligible_rate >= 0.3 else ("fail", "high")
            add_row(
                rows,
                "module_A",
                "rows_with_min_3_dimensions",
                status,
                severity,
                f"eligible_rate={eligible_rate:.4%}",
                path,
            )


    def audit_module_d(
        policy_panel: pd.DataFrame,
        validation_df: pd.DataFrame,
        policy_summary: dict[str, object],
        policy_panel_path: Path,
        validation_path: Path,
        rows: list[dict[str, object]],
    ) -> None:
        if validation_df.empty:
            add_row(rows, "module_D", "validation_summary", "fail", "high", "policy_validation_summary.csv 缺失或为空", validation_path)
            return
        locked_outcomes = list(policy_summary.get("global_locked_outcomes", []))
        main_rows = validation_df.loc[
            (validation_df["analysis_scope"] == "global")
            & (validation_df["sample_variant"] == "balanced_main")
            & (validation_df["outcome_column"] == "dbd_smoking_per100k")
        ]
        if main_rows.empty:
            add_row(rows, "module_D", "main_locked_row", "fail", "high", "缺少 global+balanced_main+dbd_smoking_per100k 行", validation_path)
            return
        row = main_rows.iloc[0]
        locked = str(row.get("validation_status", "")) == "锁定"
        summary_locked = "dbd_smoking_per100k" in locked_outcomes
        lock_state_consistent = locked == summary_locked
        add_row(
            rows,
            "module_D",
            "locked_outcome_consistency",
            "pass" if lock_state_consistent else "warn",
            "info" if lock_state_consistent else "medium",
            f"validation_status={row.get('validation_status')}, summary_locked={locked_outcomes}",
            validation_path,
        )
        if policy_panel.empty:
            add_row(rows, "module_D", "policy_panel_presence", "fail", "high", "policy_identification_panel.csv 缺失或为空", policy_panel_path)
            return
        required = ["dbd_smoking_per100k", "policy_strength", "iso3", "year"]
        missing = [column for column in required if column not in policy_panel.columns]
        if missing:
            add_row(rows, "module_D", "policy_panel_columns", "fail", "high", f"缺少列 {missing}", policy_panel_path)
            return
        core = policy_panel[required].copy()
        usable_rate = float(core.dropna().shape[0] / max(len(core), 1))
        status, severity = ("pass", "info") if usable_rate >= 0.3 else ("warn", "medium") if usable_rate >= 0.15 else ("fail", "high")
        add_row(
            rows,
            "module_D",
            "locked_outcome_usable_rows",
            status,
            severity,
            f"usable_rate={usable_rate:.4%}, rows={int(core.dropna().shape[0])}",
            policy_panel_path,
        )


    def build_notes(result_df: pd.DataFrame, summary: dict[str, object]) -> str:
        failed = result_df.loc[result_df["status"] == "fail"]
        warned = result_df.loc[result_df["status"] == "warn"]
        lines = [
            "# 结果污染审计说明",
            "",
            f"- 总检查项: {summary['checks_total']}",
            f"- 通过项: {summary['checks_passed']}",
            f"- 警告项: {summary['checks_warned']}",
            f"- 失败项: {summary['checks_failed']}",
            f"- 总体判断: {summary['overall_assessment']}",
            "",
            "## 判断原则",
            "- outer join 产生的高缺失不自动等于结果错误。",
            "- 真正高风险的是主键错配、重复记录、非法数值范围、per100k 派生不一致。",
            "- 模块D单独检查，因为它决定政策识别证据应锁定为主结论还是降级为趋势性证据。",
            "- 如果 A/C 模块已经在 summary 中显式剔除了稀疏特征，本审计按剔除后的有效输入口径检查。",
            "",
            "## 高风险项",
        ]
        if failed.empty:
            lines.append("- 无")
        else:
            for row in failed.itertuples():
                lines.append(f"- [{row.category}] {row.item}: {row.detail}")
        lines.extend(["", "## 中风险提醒"])
        if warned.empty:
            lines.append("- 无")
        else:
            for row in warned.itertuples():
                lines.append(f"- [{row.category}] {row.item}: {row.detail}")
        lines.extend(
            [
                "",
                "## 解释建议",
                "- 如果高风险项只集中在 coverage/missingness，通常说明样本稀疏，而不是结果被机械污染。",
                "- 如果高风险项出现在 per100k 一致性或主键结构，相关模块应重跑并优先复核源数据。",
            ]
        )
        return "\n".join(lines) + "\n"


    def main() -> None:
        parser = argparse.ArgumentParser(description="Audit whether the final analysis results are structurally contaminated by master-panel issues.")
        parser.add_argument("--project-root", type=Path, default=None)
        args = parser.parse_args()

        project_root = detect_project_root(args.project_root)
        dirs = ensure_dirs(project_root)
        simulation_dir = dirs["simulation"]
        report_dir = dirs["report"]

        master_path = simulation_dir / "global_health_panel_v1.csv"
        response_path = simulation_dir / "response_panel.csv"
        typology_path = simulation_dir / "vulnerability_typology_panel.csv"
        response_diagnosis_path = simulation_dir / "response_diagnosis_panel.csv"
        policy_panel_path = simulation_dir / "policy_identification_panel.csv"
        external_data_root = shared_detect_external_data_root(project_root=project_root)
        scope_registry_path = external_data_root / "16_Project_Metadata_Registry" / "country_scope_registry.csv"

        validation_path = report_asset_path(report_dir, "policy_validation_summary.csv")
        policy_summary_path = report_asset_path(report_dir, "policy_identification_summary.json")
        vulnerability_summary_path = report_asset_path(report_dir, "vulnerability_typology_summary.json")
        response_summary_path = report_asset_path(report_dir, "response_diagnosis_summary.json")

        master_df = read_csv(master_path)
        response_df = read_csv(response_path)
        typology_df = read_csv(typology_path)
        response_diagnosis_df = read_csv(response_diagnosis_path)
        policy_panel_df = read_csv(policy_panel_path)
        scope_registry_df = read_csv(scope_registry_path)
        validation_df = read_csv(validation_path)
        policy_summary = read_json(policy_summary_path)
        vulnerability_summary = read_json(vulnerability_summary_path)
        response_summary = read_json(response_summary_path)
        module_a_dimension_groups = resolve_module_a_dimension_groups(vulnerability_summary)
        module_c_resource_columns = resolve_module_c_resource_columns(response_summary)

        rows: list[dict[str, object]] = []

        audit_structure(master_df, "master_panel", master_path, rows)
        audit_structure(response_df, "response_panel", response_path, rows)
        audit_structure(response_diagnosis_df, "response_diagnosis_panel", response_diagnosis_path, rows)
        audit_structure(policy_panel_df, "policy_identification_panel", policy_panel_path, rows)
        audit_scope_contamination(master_df, "master_panel", master_path, rows, scope_registry_df)
        audit_scope_contamination(response_df, "response_panel", response_path, rows, scope_registry_df)
        audit_scope_contamination(response_diagnosis_df, "response_diagnosis_panel", response_diagnosis_path, rows, scope_registry_df)
        audit_scope_contamination(policy_panel_df, "policy_identification_panel", policy_panel_path, rows, scope_registry_df)

        if not master_df.empty:
            audit_nonnegative_and_ranges(master_df, "master_panel", master_path, rows, list(set(NONNEGATIVE_COLUMNS + list(RANGE_RULES.keys()))))
        if not response_df.empty:
            audit_nonnegative_and_ranges(response_df, "response_panel", response_path, rows, list(set(NONNEGATIVE_COLUMNS + list(RANGE_RULES.keys()))))
            audit_per100k_consistency(response_df, "response_panel", response_path, rows)
            compare_master_to_response(master_df, response_df, master_path, response_path, rows)
            audit_module_a(response_df, response_path, rows, module_a_dimension_groups)
            audit_missingness(response_df, "module_B_diseases", response_path, rows, B_DISEASE_COLUMNS)
            audit_missingness(response_df, "module_B_risks", response_path, rows, B_RISK_COLUMNS)
            module_c_df = response_diagnosis_df if not response_diagnosis_df.empty else response_df
            module_c_path = response_diagnosis_path if not response_diagnosis_df.empty else response_path
            audit_missingness(module_c_df, "module_C_burdens", module_c_path, rows, C_GBD_COLUMNS)
            audit_missingness(module_c_df, "module_C_risks", module_c_path, rows, C_RISK_COLUMNS)
            audit_missingness(module_c_df, "module_C_resources", module_c_path, rows, module_c_resource_columns)

        dropped_sparse_features = vulnerability_summary.get("dropped_sparse_features", [])
        if isinstance(dropped_sparse_features, list) and dropped_sparse_features:
            feature_names = [str(item.get("analysis_column")) for item in dropped_sparse_features if isinstance(item, dict)]
            add_row(
                rows,
                "module_A",
                "dropped_sparse_features",
                "pass",
                "info",
                f"count={len(dropped_sparse_features)}, features={feature_names}",
                vulnerability_summary_path,
            )

        dropped_sparse_resources = response_summary.get("dropped_sparse_resource_columns", [])
        if isinstance(dropped_sparse_resources, list) and dropped_sparse_resources:
            resource_names = [str(item.get('column')) for item in dropped_sparse_resources if isinstance(item, dict)]
            add_row(
                rows,
                "module_C",
                "dropped_sparse_resource_columns",
                "pass",
                "info",
                f"count={len(dropped_sparse_resources)}, columns={resource_names}",
                response_summary_path,
            )

        if not typology_df.empty and {"iso3", "year"}.issubset(typology_df.columns):
            coverage = float(
                response_df[KEY_COLUMNS]
                .merge(
                    typology_df[KEY_COLUMNS].drop_duplicates(),
                    on=KEY_COLUMNS,
                    how="left",
                    indicator=True,
                )["_merge"]
                .eq("both")
                .mean()
            ) if not response_df.empty else 0.0
            status, severity = ("pass", "info") if coverage >= 0.8 else ("warn", "medium") if coverage >= 0.5 else ("fail", "high")
            add_row(rows, "module_A", "typology_panel_merge_coverage", status, severity, f"coverage={coverage:.4%}", typology_path)

        audit_module_d(policy_panel_df, validation_df, policy_summary, policy_panel_path, validation_path, rows)

        result_df = pd.DataFrame(rows)
        if result_df.empty:
            raise RuntimeError("No audit rows were generated.")

        checks_failed = int((result_df["status"] == "fail").sum())
        checks_warned = int((result_df["status"] == "warn").sum())
        checks_passed = int((result_df["status"] == "pass").sum())

        if checks_failed > 0:
            overall = "发现结构性污染风险，至少有一个模块的结果需要复核或重跑。"
            overall_risk = "high"
        elif checks_warned > 0:
            overall = "未发现明确结构性污染证据，但存在稀疏覆盖或边缘风险，需要在报告中谨慎表述。"
            overall_risk = "medium"
        else:
            overall = "未发现会机械污染主分析结论的结构性问题；当前主要风险不是脏表，而是覆盖不均。"
            overall_risk = "low"

        summary = {
            "project_root": str(project_root),
            "checks_total": int(result_df.shape[0]),
            "checks_passed": checks_passed,
            "checks_warned": checks_warned,
            "checks_failed": checks_failed,
            "overall_risk": overall_risk,
            "overall_assessment": overall,
            "locked_outcomes": policy_summary.get("global_locked_outcomes", []),
            "output_files": {
                "summary": str(report_asset_path(report_dir, "result_contamination_audit_summary.json")),
                "checklist": str(report_asset_path(report_dir, "result_contamination_audit_checklist.csv")),
                "notes": str(report_asset_path(report_dir, "result_contamination_audit_notes.md")),
            },
        }

        checklist_path = report_asset_path(report_dir, "result_contamination_audit_checklist.csv")
        summary_path = report_asset_path(report_dir, "result_contamination_audit_summary.json")
        notes_path = report_asset_path(report_dir, "result_contamination_audit_notes.md")

        result_df.to_csv(checklist_path, index=False, encoding="utf-8-sig")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        notes_path.write_text(build_notes(result_df, summary), encoding="utf-8")

        print(json.dumps(summary, ensure_ascii=False, indent=2))


    if __name__ == "__main__":
        main()

    return locals()


NAMESPACE_BUILDERS = {
    'run_abc_predictive_validation.py': _namespace_run_abc_predictive_validation,
    'run_abc_predictive_robustness.py': _namespace_run_abc_predictive_robustness,
    'run_vulnerability_typology.py': _namespace_run_vulnerability_typology,
    'run_stage_identification.py': _namespace_run_stage_identification,
    'run_risk_attribution_matrix.py': _namespace_run_risk_attribution_matrix,
    'run_response_diagnosis.py': _namespace_run_response_diagnosis,
    'run_result_contamination_audit.py': _namespace_run_result_contamination_audit,
}

STEP_GROUPS = {'run': ['run_vulnerability_typology.py', 'run_stage_identification.py', 'run_risk_attribution_matrix.py', 'run_response_diagnosis.py', 'run_abc_predictive_validation.py', 'run_abc_predictive_robustness.py'], 'audit': ['run_result_contamination_audit.py']}
DEFAULT_GROUPS = ['run', 'audit']


def selected_steps(groups: list[str]) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = []
    for group in groups:
        steps.extend((script_name, []) for script_name in STEP_GROUPS[group])
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description='Run modules A/B/C: vulnerability, stage, risk, response, and predictive validation.')
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
