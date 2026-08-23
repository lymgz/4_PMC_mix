"""Dep_reg 的 1x/1.5x/2x/3x LPR 水平敏感性分析。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import config
from src.data_io import load_combined, city_key
from src.dep_reg import build_regulated_outcomes, load_engine


def _run_cached():
    """Reuse the audited sensitivity bundle when the separate Dep engine is absent."""
    required = [
        config.RATE_SENSITIVITY_PATH,
        config.RATE_SENSITIVITY_CITY_PATH,
        config.RATE_SENSITIVITY_RATIO_PATH,
    ]
    missing = [config.display_path(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少随包提供的 rate-sensitivity cache：{missing}")

    summary = pd.read_csv(config.RATE_SENSITIVITY_PATH)
    city_level = pd.read_csv(config.RATE_SENSITIVITY_CITY_PATH)
    ratio_check = pd.read_csv(config.RATE_SENSITIVITY_RATIO_PATH)
    summary.to_csv(config.TABLE_DIR / "rate_sensitivity.csv", index=False, encoding="utf-8-sig")
    city_level.to_csv(config.TABLE_DIR / "rate_sensitivity_city_level.csv", index=False, encoding="utf-8-sig")
    ratio_check.to_csv(config.TABLE_DIR / "rate_sensitivity_ratio_check.csv", index=False, encoding="utf-8-sig")

    audit = {}
    if config.DEP_PIPELINE_AUDIT_PATH.exists():
        audit = json.loads(config.DEP_PIPELINE_AUDIT_PATH.read_text(encoding="utf-8"))
    audit.update(
        {
            "definition": "cached audited sensitivity of deterministic Dep_reg; raw engine not bundled",
            "engine_path": config.display_path(config.DEP_ENGINE_PATH),
            "cache_mode": True,
            "cache_source": config.display_path(config.RATE_SENSITIVITY_PATH),
        }
    )
    (config.LOG_DIR / "dep_pipeline_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if config.DEP_FIXED_INPUT_AUDIT_PATH.exists():
        fixed = json.loads(config.DEP_FIXED_INPUT_AUDIT_PATH.read_text(encoding="utf-8"))
        fixed["engine_path"] = config.display_path(config.DEP_ENGINE_PATH)
        (config.LOG_DIR / "dep_reg_fixed_input_audit.json").write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
    if config.RATE_JUSTIFICATION_PATH.exists():
        (config.LOG_DIR / "rate_justification.md").write_text(config.RATE_JUSTIFICATION_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return summary, ratio_check


def run():
    if config.DEP_ENGINE_PATH is None or not config.DEP_ENGINE_PATH.exists():
        return _run_cached()
    config.ensure_directories(); engine = load_engine(); catalog = engine.extract_catalog(config.PMC_DATA_PATH)
    rows, audits = [], []
    for multiplier in config.RATE_MULTIPLIERS:
        frame, audit = build_regulated_outcomes(catalog, lpr_multiplier=multiplier, engine=engine)
        audits.append(audit)
        keep = frame[["city", "city_key", "Dep_reg", "Dep_engine_original", "peak_exposure", "peak_funding_cost", "monthly_rate"]].copy()
        keep["rate_multiple"] = multiplier; keep["annual_lpr"] = audit["fixed_inputs"]["lpr_annual"]; keep["annual_rate"] = keep.annual_lpr * multiplier; rows.append(keep)
    city_level = pd.concat(rows, ignore_index=True)
    summary = city_level.groupby(["rate_multiple", "annual_rate"], as_index=False).agg(Dep_reg_mean=("Dep_reg", "mean"), Dep_reg_sd=("Dep_reg", "std"), Dep_reg_min=("Dep_reg", "min"), Dep_reg_max=("Dep_reg", "max"), Dep_reg_p50=("Dep_reg", "median"), peak_exposure_mean=("peak_exposure", "mean"))
    baseline = city_level[city_level.rate_multiple == 1.0].set_index("city_key").Dep_reg
    ratio = city_level.copy(); ratio["baseline_Dep_reg_1x"] = ratio.city_key.map(baseline); ratio["Dep_reg_ratio_to_1x"] = ratio.apply(lambda r: float(r.Dep_reg / r.baseline_Dep_reg_1x) if abs(float(r.baseline_Dep_reg_1x)) > config.DEP_TOLERANCE else float("nan"), axis=1); ratio["Dep_reg_scale_error"] = ratio.Dep_reg - ratio.rate_multiple * ratio.baseline_Dep_reg_1x
    ratio_check = ratio.groupby("rate_multiple").agg(min=("Dep_reg_ratio_to_1x", "min"), max=("Dep_reg_ratio_to_1x", "max"), mean=("Dep_reg_ratio_to_1x", "mean"), sd=("Dep_reg_ratio_to_1x", "std"), max_abs_scale_error=("Dep_reg_scale_error", lambda s: float(s.abs().max())), zero_baseline_cities=("baseline_Dep_reg_1x", lambda s: int((s.abs() <= config.DEP_TOLERANCE).sum()))).reset_index()
    ratio_check["expected_ratio"] = ratio_check.rate_multiple
    ratio_check["max_abs_error"] = (ratio_check["mean"] - ratio_check["expected_ratio"]).abs()
    pure = bool((ratio_check.max_abs_scale_error < config.DEP_TOLERANCE).all())
    summary["pure_Dep_reg_multiplier"] = pure
    summary.to_csv(config.TABLE_DIR / "rate_sensitivity.csv", index=False, encoding="utf-8-sig")
    city_level.to_csv(config.TABLE_DIR / "rate_sensitivity_city_level.csv", index=False, encoding="utf-8-sig")
    ratio_check.to_csv(config.TABLE_DIR / "rate_sensitivity_ratio_check.csv", index=False, encoding="utf-8-sig")
    combined = load_combined(); base = city_level[city_level.rate_multiple == config.BASE_LPR_MULTIPLIER].set_index("city_key").Dep_reg; current = combined.set_index("city_key")[config.TARGET]; keys = sorted(set(base.index) & set(current.index)); diff = (base.loc[keys] - current.loc[keys]).abs()
    audit = {"definition": "level sensitivity of deterministic Dep_reg, not an independent robustness design", "formula": "Dep_reg = sum(stage amount * holding months) * (fixed annual LPR * multiplier / 12)", "rate_multipliers": list(config.RATE_MULTIPLIERS), "baseline_multiplier": config.BASE_LPR_MULTIPLIER, "fixed_inputs": audits[-1]["fixed_inputs"], "pure_multiplier": pure, "baseline_match_count": len(keys), "max_abs_diff_vs_combined_3x": float(diff.max()) if len(diff) else None, "n_city_level_rows": len(city_level)}
    (config.LOG_DIR / "dep_pipeline_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.LOG_DIR / "dep_reg_fixed_input_audit.json").write_text(json.dumps(audits[-1], ensure_ascii=False, indent=2), encoding="utf-8")
    (config.LOG_DIR / "rate_justification.md").write_text("基准为 3×LPR；1×、1.5×、2×、3× 只改变融资利率水平，市场输入和规则结构固定。该部分是水平敏感性分析，不被表述为独立因果稳健性证据。\n", encoding="utf-8")
    return summary, ratio_check


if __name__ == "__main__": print(run()[0].to_string(index=False))
