"""构造 10.3 的规则化监管资金暴露代理 Dep_reg。

底层计算仍调用 10.2 已审计的 multi_city_dep_model.calculate_city；本层只把
市场价格、去化周期、施工时序和 LPR 固定为 41 城样本中位数，同时保留每城
原始释放节点、比例和 cohort 窗口。这样 Dep_reg 的城市差异来自规则结构，
而不是把城市市场规模直接重复塞入结果变量。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config


def load_engine():
    if config.DEP_ENGINE_PATH is None or not config.DEP_ENGINE_PATH.exists():
        raise FileNotFoundError(
            "The separate Modelling Dep engine is not bundled with this release. "
            "Set JHBE_DEP_DIR to its 41_Cities_Dep directory for raw rebuilds."
        )
    spec = importlib.util.spec_from_file_location("jhbe_dep_engine_103", config.DEP_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入 Dep 引擎：{config.DEP_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _median(catalog: dict[str, Any], getter, default: float | None = None) -> float:
    values = []
    for city in catalog["city_names"]:
        value = getter(catalog["cities"][city])
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    if not values:
        if default is None:
            raise ValueError("无法从 Dep catalog 得到固定中位数")
        return float(default)
    return float(np.median(values))


def fixed_inputs(catalog: dict[str, Any]) -> dict[str, float]:
    def timing(key: str, item: dict[str, Any]):
        return item.get("timing_parameters", {}).get(key)

    return {
        "lpr_annual": _median(catalog, lambda item: item.get("lpr_annual"), config.BASE_LPR_ANNUAL),
        "floors": _median(catalog, lambda item: item.get("floors_baseline"), 30.0),
        "completion_months": _median(catalog, lambda item: timing("completion_months", item), 0.0),
        "pre_sale_months": _median(catalog, lambda item: timing("pre_sale_months", item), 0.0),
        "top_out_months": _median(catalog, lambda item: timing("top_out_months", item), 0.0),
        "decoration_months": _median(catalog, lambda item: timing("decoration_months", item), 0.0),
        "record_duration_months": _median(catalog, lambda item: timing("record_duration_months", item), 0.0),
        "registration_lag_months": _median(catalog, lambda item: timing("registration_lag_months", item), 0.0),
        "average_price": _median(catalog, lambda item: item.get("average_price")),
        "absorption_months": _median(catalog, lambda item: item.get("absorption_months")),
    }


def build_regulated_outcomes(
    catalog: dict[str, Any],
    *,
    lpr_multiplier: float = config.BASE_LPR_MULTIPLIER,
    engine=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    engine = load_engine() if engine is None else engine
    fixed = fixed_inputs(catalog)
    rows: list[dict[str, Any]] = []
    for city in catalog["city_names"]:
        item = catalog["cities"][city]
        result = engine.calculate_city(
            city,
            catalog,
            lpr_annual=fixed["lpr_annual"],
            lpr_multiplier=lpr_multiplier,
            floors=fixed["floors"],
            completion_months=fixed["completion_months"],
            pre_sale_months=fixed["pre_sale_months"],
            top_out_months=fixed["top_out_months"],
            decoration_months=fixed["decoration_months"],
            record_duration_months=fixed["record_duration_months"],
            registration_lag_months=fixed["registration_lag_months"],
            average_price=fixed["average_price"],
            absorption_months=fixed["absorption_months"],
        )
        row: dict[str, Any] = {
            "city": city,
            "city_key": city_key(city),
            "Dep_reg": float(result["dep"]),
            "Dep_engine_original": float(engine.calculate_city(city, catalog, lpr_multiplier=lpr_multiplier)["dep"]),
            "peak_exposure": result.get("peak_exposure"),
            "peak_funding_cost": result.get("peak_funding_cost"),
            "n_stages": len(item.get("stages", [])),
            "lpr_multiplier": float(lpr_multiplier),
            "monthly_rate": float(result["monthly_rate"]),
        }
        for index, stage in enumerate(item.get("stages", []), 1):
            key = stage["key"]
            row[f"stage_{index}_ratio"] = stage.get("baseline_release_ratio")
            row[f"stage_{index}_label"] = stage.get("name")
            row[f"stage_{index}_first_cohort"] = stage.get("first_cohort_month")
            row[f"stage_{index}_last_cohort"] = stage.get("last_cohort_month")
            stage_cost = 0.0
            holds = []
            for scenario_row in result.get("rows", []):
                entry = scenario_row.get("stages", {}).get(key, {})
                amount = float(entry.get("amount", 0.0) or 0.0)
                hold = float(entry.get("hold", 0.0) or 0.0)
                stage_cost += amount * hold * float(result["monthly_rate"])
                if hold > 0:
                    holds.append(hold)
            row[f"stage_{index}_cost"] = stage_cost
            row[f"stage_{index}_hold_mean"] = float(np.mean(holds)) if holds else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    audit = {
        "definition": "Dep_reg = deterministic rule-based supervised-fund exposure proxy",
        "formula": "Dep_reg_c = sum_i sum_k amount_i,k(fixed market inputs, city release rules) * hold_i,k(fixed schedule, city cohort windows) * (fixed annual LPR * multiplier / 12)",
        "variation_source": ["city-specific release-stage labels", "city-specific release proportions", "city-specific cohort windows and source rows"],
        "fixed_inputs": fixed,
        "lpr_multiplier": float(lpr_multiplier),
        "n_cities": int(len(frame)),
        "engine_path": config.display_path(config.DEP_ENGINE_PATH),
        "calculation_function": "calculate_city",
        "iv3_role": "descriptive only; not predictor/rank/SHAP input",
        "iv4_role": "descriptive overlap audit only; not predictor/rank/SHAP input",
    }
    return frame, audit


def city_key(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return text[:-1] if text.endswith(("市", "县")) else text


def write_audit(audit: dict[str, Any], *, name: str = "dep_reg_fixed_input_audit.json") -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (config.LOG_DIR / name).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
