"""Bridge the 41-city Dep pipeline to an auditable operational H1 layer.

The policy-observed attributes in :mod:`rule_attributes` remain unchanged.
This module derives a second layer from ``Modelling Dep/41_Cities_Dep`` and
keeps the provenance and translation limits explicit.  A Dep stage ratio is
not silently treated as a policy percentage: common-denominator checks and
stage-name mapping determine whether an operational proxy is estimable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

import config
from .data_io import city_key


EPS = 1e-10
MAPPING_VERSION = "dep-operational-h1-v1"


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _stage_semantic(name: object) -> str:
    text = str(name or "").strip()
    if any(token in text for token in ("主体", "封顶")):
        return "topout"
    if "装修" in text:
        return "decoration"
    if any(token in text for token in ("竣备", "备案")):
        return "record"
    if any(token in text for token in ("确权", "登记")):
        return "registration"
    return "unmapped"


def _active_stage_keys(model: dict) -> list[str]:
    return [
        str(stage["key"])
        for stage in model.get("stages", [])
        if stage.get("first_cohort_month") is not None
    ]


def _within_cohort_share(model: dict, stage_key: str) -> float:
    values: list[float] = []
    for row in model.get("rows", []):
        stages = row.get("stages", {})
        numerator = stages.get(stage_key, {}).get("amount", 0.0)
        denominator = sum(float(entry.get("amount", 0.0) or 0.0) for entry in stages.values())
        if _finite(numerator) and float(numerator) > EPS and denominator > EPS:
            values.append(float(numerator) / denominator)
    return float(median(values)) if values else np.nan


def _stage_share(model: dict, stage: dict) -> tuple[float, str]:
    ratio = stage.get("baseline_release_ratio")
    base_type = str(stage.get("calculation_base_type", ""))
    if base_type == "monthly_revenue" and _finite(ratio):
        return float(ratio), "monthly_revenue_stage_ratio"
    if base_type == "source_amount":
        value = _within_cohort_share(model, str(stage["key"]))
        if _finite(value):
            return value, "within_cohort_amount_share"
    return np.nan, "not_estimable"


def _judgement(coverage: int, n_cities: int, *, reason: str = "") -> str:
    if coverage == n_cities:
        return "complete_under_declared_dep_definition"
    if coverage > 0:
        return "partial_dep_operational_coverage" + (f":{reason}" if reason else "")
    return "not_estimable" + (f":{reason}" if reason else "")


def load_dep_catalog() -> dict:
    if config.DEP_CATALOG_PATH is None or not config.DEP_CATALOG_PATH.exists():
        raise FileNotFoundError(f"缺少41城 Dep catalog：{config.DEP_CATALOG_PATH}")
    payload = json.loads(config.DEP_CATALOG_PATH.read_text(encoding="utf-8"))
    names = payload.get("city_names", [])
    cities = payload.get("cities", {})
    if len(names) != config.N_CITIES or len(cities) != config.N_CITIES:
        raise ValueError("Dep catalog 必须包含41个唯一城市")
    keys = [city_key(name) for name in names]
    if len(set(keys)) != config.N_CITIES:
        raise ValueError("Dep catalog 城市键不唯一")
    return payload


def _load_cached_bridge() -> tuple[pd.DataFrame, dict] | None:
    """Load the audited bridge shipped with this release when no Dep engine is present."""
    path = config.DEP_OPERATIONAL_BRIDGE_PATH
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {
        "city_key",
        "dep_n_release_steps",
        "dep_share_drawable_at_topout",
        "dep_final_retained_share",
        "dep_topout_status",
        "dep_final_retained_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dep operational bridge 缺少字段：{missing}")
    if len(frame) != config.N_CITIES or frame.city_key.nunique() != config.N_CITIES:
        raise ValueError("Dep operational bridge 必须包含41个唯一城市键")
    if config.DEP_OPERATIONAL_AUDIT_PATH.exists():
        audit = json.loads(config.DEP_OPERATIONAL_AUDIT_PATH.read_text(encoding="utf-8"))
    else:
        audit = {
            "n_cities": int(len(frame)),
            "mapping_version": MAPPING_VERSION,
            "coverage": {
                column: int(frame[column].notna().sum())
                for column in ("dep_n_release_steps", "dep_share_drawable_at_topout", "dep_final_retained_share")
            },
        }
    audit["source_file"] = config.display_path(path)
    audit["source_workbook"] = "separately published Modelling Dep project; raw engine not bundled"
    audit.setdefault("mapping_version", MAPPING_VERSION)
    return frame, audit


def build_dep_operational_attributes() -> tuple[pd.DataFrame, dict]:
    cached = _load_cached_bridge()
    if cached is not None and (config.DEP_CATALOG_PATH is None or not config.DEP_CATALOG_PATH.exists()):
        return cached
    catalog = load_dep_catalog()
    rows: list[dict] = []
    for name in catalog["city_names"]:
        model = catalog["cities"][name]
        stages = model.get("stages", [])
        active_keys = set(_active_stage_keys(model))
        stage_records = []
        for index, stage in enumerate(stages):
            key = str(stage["key"])
            semantic = _stage_semantic(stage.get("name"))
            share, share_method = _stage_share(model, stage)
            stage_records.append(
                {
                    "index": index,
                    "key": key,
                    "name": str(stage.get("name", "")),
                    "semantic": semantic,
                    "active": key in active_keys,
                    "share": share,
                    "share_method": share_method,
                    "base_type": str(stage.get("calculation_base_type", "")),
                }
            )

        active = [item for item in stage_records if item["active"]]
        topout = [item for item in active if item["semantic"] == "topout"]
        registration = [item for item in active if item["semantic"] == "registration"]
        active_order = {item["key"]: item["index"] for item in active}
        topout_share = np.nan
        topout_method = "not_estimable"
        if len(topout) == 1:
            topout_index = topout[0]["index"]
            prefix = [item for item in active if item["index"] <= topout_index]
            methods = {item["share_method"] for item in prefix}
            values = [item["share"] for item in prefix]
            if len(methods) == 1 and methods != {"not_estimable"} and all(_finite(value) for value in values):
                candidate = float(sum(values))
                if candidate <= 1.0 + 1e-8:
                    topout_share = min(1.0, candidate)
                    topout_method = "cumulative_same_denominator_stage_shares"
                else:
                    topout_method = "not_estimable:stage_shares_exceed_common_denominator"
            else:
                topout_method = "not_estimable:mixed_or_missing_stage_share_methods"
        elif not topout:
            topout_method = "not_estimable:topout_stage_name_unmapped"
        else:
            topout_method = "not_estimable:multiple_topout_stage_names"

        terminal_share = np.nan
        terminal_method = "not_estimable"
        if len(registration) == 1:
            terminal = registration[0]
            if _finite(terminal["share"]):
                terminal_share = float(terminal["share"])
                terminal_method = f"terminal_stage_share:{terminal['share_method']}"
            else:
                terminal_method = "not_estimable:terminal_stage_share_missing"
        elif not registration:
            terminal_method = "not_estimable:registration_stage_name_unmapped"
        else:
            terminal_method = "not_estimable:multiple_registration_stage_names"

        mapping_status = "complete_named_milestones" if topout and registration else "partial_named_milestones"
        rows.append(
            {
                "city_key": city_key(name),
                "city": name,
                "dep_template_id": model.get("template_id", ""),
                "dep_stage_count_total": len(stages),
                "dep_active_stage_count": len(active),
                "dep_stage_keys": "|".join(item["key"] for item in stage_records),
                "dep_stage_names": "|".join(item["name"] for item in stage_records),
                "dep_active_stage_names": "|".join(item["name"] for item in active),
                "dep_stage_mapping_status": mapping_status,
                "dep_topout_stage_key": topout[0]["key"] if len(topout) == 1 else "",
                "dep_registration_stage_key": registration[0]["key"] if len(registration) == 1 else "",
                "dep_n_release_steps": len(active),
                "dep_share_drawable_at_topout": topout_share,
                "dep_final_retained_share": terminal_share,
                "dep_topout_status": topout_method,
                "dep_final_retained_status": terminal_method,
                "dep_attribute_layer": "expert_calibrated_operational_proxy_pending_source_mapping",
                "dep_stage_ratio_definition": "monthly-revenue stage ratio or within-cohort amount share; not automatically a policy percentage",
                "dep_source_file": config.display_path(config.DEP_CATALOG_PATH),
                "dep_source_workbook": catalog.get("workbook_path", ""),
                "dep_mapping_version": MAPPING_VERSION,
            }
        )

    result = pd.DataFrame(rows).sort_values("city_key").reset_index(drop=True)
    audit = {
        "source_file": config.display_path(config.DEP_CATALOG_PATH),
        "source_workbook": catalog.get("workbook_path", ""),
        "n_cities": int(len(result)),
        "mapping_version": MAPPING_VERSION,
        "stage_semantics": {"topout": "主体/封顶", "decoration": "装修", "record": "竣备/备案", "registration": "确权/登记"},
        "coverage": {
            "dep_n_release_steps": int(result["dep_n_release_steps"].notna().sum()),
            "dep_share_drawable_at_topout": int(result["dep_share_drawable_at_topout"].notna().sum()),
            "dep_final_retained_share": int(result["dep_final_retained_share"].notna().sum()),
        },
        "stage_mapping_status_counts": result["dep_stage_mapping_status"].value_counts(dropna=False).to_dict(),
        "topout_status_counts": result["dep_topout_status"].value_counts(dropna=False).to_dict(),
        "final_retained_status_counts": result["dep_final_retained_status"].value_counts(dropna=False).to_dict(),
        "interpretation_boundary": "These are Dep-derived operational proxies. They must not be reported as policy-observed values without city-level source or expert mapping evidence.",
    }
    return result, audit


def dep_attribute_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"attribute": "Dep operational stage count", "category": "active release stages", "n_cities": int(frame["dep_n_release_steps"].notna().sum()), "share": float(frame["dep_n_release_steps"].notna().mean()), "n_absent": int(frame["dep_n_release_steps"].isna().sum()), "note": "counted from active stage columns in the Dep catalog"})
    for column, label in [("dep_share_drawable_at_topout", "cumulative share through mapped topout"), ("dep_final_retained_share", "mapped terminal-stage share")]:
        rows.append({"attribute": "Dep operational quantitative proxy", "category": label, "n_cities": int(frame[column].notna().sum()), "share": float(frame[column].notna().mean()), "n_absent": int(frame[column].isna().sum()), "note": "not policy-observed; see status and mapping columns"})
    return pd.DataFrame(rows)
