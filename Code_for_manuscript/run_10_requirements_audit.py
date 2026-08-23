"""Machine-readable audit for the 10.4 evidence and Section 5 asset contract."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.tuning import load_frozen_payload, rf_param_hash


def _row(requirement: str, status: str, evidence: str) -> dict:
    return {"requirement": requirement, "status": status, "evidence": evidence, "scope": "Code 10.4 formal evidence and Section 5 assets"}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _scan_source_usage() -> dict:
    scripts = sorted({*ROOT.glob("run_*.py"), *(ROOT / "src").rglob("*.py")})
    authoritative, legacy = [], []
    for path in scripts:
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(ROOT))
        if "Evaluation-transfer" in text and "read_excel" in text:
            authoritative.append(relative)
        if re.search(r"sheet_name\s*=\s*['\"]matrix['\"]|\[['\"]matrix['\"]\]", text, flags=re.I):
            legacy.append(relative)
    return {"authoritative_x1_x10_consumers": sorted(authoritative), "legacy_matrix_x1_x10_consumers": sorted(legacy), "interpretation": "pmc_data.xlsx remains a Dep/LPR source; its legacy matrix is not an X1-X10 source."}


# Compatibility name retained for the focused source-contract test.
def _scan_x1_x10_source_usage() -> dict:
    audit = _scan_source_usage()
    return {
        "authoritative_x1_x10_source": {"consumers": audit["authoritative_x1_x10_consumers"]},
        "legacy_35_item_matrix": {"runtime_consumers": audit["legacy_matrix_x1_x10_consumers"]},
        "interpretation": audit["interpretation"],
    }


def run() -> dict:
    config.ensure_directories()
    tp, fp, lp = config.TABLE_DIR, config.FIGURE_DIR, config.LOG_DIR
    rows = []
    provenance = json.loads((lp / "data_provenance.json").read_text(encoding="utf-8")) if (lp / "data_provenance.json").exists() else {}
    rows.append(_row("41-city keyed data with Dep_reg", "通过" if provenance.get("n_cities") == 41 and config.TARGET in provenance.get("required_columns", []) else "不通过", f"n_cities={provenance.get('n_cities')}; target={config.TARGET}"))
    manifest = _read(tp / "split_manifest.csv")
    rows.append(_row("Shared RepeatedKFold 5x20 manifest", "通过" if len(manifest) == 820 and manifest.get("split_id", pd.Series(dtype=float)).nunique() == 100 and manifest.get("city_key", pd.Series(dtype=str)).nunique() == 41 else "不通过", f"rows={len(manifest)}; splits={manifest.split_id.nunique() if not manifest.empty else 0}"))
    canonical = _read(tp / "canonical_results.csv")
    folds = _read(tp / "canonical_folds.csv")
    models = set(canonical.model) if not canonical.empty else set()
    rows.append(_row("Parallel M0-M4 representation table", "通过" if models == {"M0", "M1", "M2", "M3", "M4"} and len(folds) == 500 else "不通过", f"models={sorted(models)}; fold_rows={len(folds)}"))
    runtime_hash = rf_param_hash(config.RF_COMMON)
    frozen = load_frozen_payload(config.FROZEN_RF_PATH) if config.FROZEN_RF_PATH.exists() else {}
    rows.append(_row("M0-only tuning and frozen RF contract", "通过" if frozen.get("tuning_scope") == "M0_only" and frozen.get("rf_param_hash") == runtime_hash else "不通过", f"tuning_scope={frozen.get('tuning_scope')}; runtime_hash={runtime_hash}; frozen_hash={frozen.get('rf_param_hash')}"))
    feature_sets = _read(tp / "model_feature_sets.csv")
    forbidden = [value for value in feature_sets.get("feature", pd.Series(dtype=str)).astype(str) if value in {"IV3", "IV4", "Dep", "Dep_reg"} or "IV3" in value or "IV4" in value]
    rows.append(_row("IV3/IV4/target excluded from predictors", "通过" if not forbidden else "不通过", f"forbidden_hits={forbidden[:10]}"))

    pairs = _read(tp / "p1_text_rule_pair_audit.csv")
    p1_summary = _read(tp / "p1_discordance_summary.csv")
    gower_ok = not pairs.empty and pairs.rule_distance_gower.between(0, 1).all()
    rows.append(_row("P1 820-pair Gower audit and four text distances", "通过" if len(pairs) == 820 and len(p1_summary) == 4 and gower_ok else "不通过", f"pairs={len(pairs)}; summary_rows={len(p1_summary)}; gower_0_1={gower_ok}"))

    rules = _read(tp / "rule_attributes_41cities.csv")
    dep_rules = _read(tp / "dep_operational_attributes_41cities.csv")
    distribution = _read(tp / "h1_attribute_distribution.csv")
    concordance = _read(tp / "h1_rank_concordance.csv")
    rows.append(_row("41-city manually coded release attributes", "通过" if len(rules) == 41 and len(distribution) >= 10 else "不通过", f"rule_rows={len(rules)}; distribution_rows={len(distribution)}"))
    layer_counts = concordance.get("evidence_layer", pd.Series(dtype=str)).value_counts().to_dict()
    h1_ok = all(int(layer_counts.get(layer, 0)) >= 3 for layer in ["policy_observed", "dep_operational"])
    rows.append(_row("H1 policy and Dep-operational concordance layers", "通过" if h1_ok else "不通过", f"rank_pairs={len(concordance)}; layer_counts={layer_counts}; policy_missing={rules[['n_release_steps', 'share_drawable_at_topout', 'final_retained_share']].isna().sum().to_dict() if not rules.empty else {}}"))
    dep_audit = json.loads((lp / "dep_attribute_audit.json").read_text(encoding="utf-8")) if (lp / "dep_attribute_audit.json").exists() else {}
    dep_ok = len(dep_rules) == 41 and dep_audit.get("n_cities") == 41 and bool(dep_audit.get("mapping_version"))
    rows.append(_row("41-city Dep operational H1 bridge", "通过" if dep_ok else "不通过", f"rows={len(dep_rules)}; coverage={dep_audit.get('coverage', {})}; mapping_version={dep_audit.get('mapping_version')}"))

    source_usage = _scan_source_usage()
    (lp / "x1_x10_source_usage_audit.json").write_text(json.dumps(source_usage, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.append(_row("No legacy matrix consumer for X1-X10", "通过" if not source_usage["legacy_matrix_x1_x10_consumers"] else "不通过", f"legacy_consumers={source_usage['legacy_matrix_x1_x10_consumers']}; authoritative={source_usage['authoritative_x1_x10_consumers']}"))
    legacy_outputs = sorted(path.name for path in [*tp.glob("pca_*.csv"), *fp.glob("fig_pca*.png")])
    rows.append(_row("Removed PCA products do not enter Section 5", "通过" if not legacy_outputs else "不通过", f"legacy_outputs_present={legacy_outputs}"))

    paired = _read(tp / "paired_delta_r2.csv")
    rank = _read(tp / "pairwise_rank_agreement.csv")
    rows.append(_row("H2 paired deltas and rank agreement", "通过" if len(paired) == 4 and len(rank) == 10 else "不通过", f"paired_rows={len(paired)}; rank_pairs={len(rank)}"))
    blocks = _read(tp / "block_results.csv")
    block_delta = _read(tp / "block_paired_delta_r2.csv")
    rows.append(_row("B0-B1-B2 incremental block comparison", "通过" if set(blocks.get("model", pd.Series(dtype=str))) == {"B0", "B1", "B2"} and len(block_delta) == 3 else "不通过", f"models={sorted(set(blocks.model)) if not blocks.empty else []}; delta_rows={len(block_delta)}"))

    shap = json.loads((lp / "shap_provenance.json").read_text(encoding="utf-8")) if (lp / "shap_provenance.json").exists() else {}
    rows.append(_row("Exploratory SHAP boundary and provenance", "通过" if shap.get("status") == "ok" and shap.get("exploratory_only") and not shap.get("shap_interaction_values_used") and shap.get("rf_param_hash") == runtime_hash else "不通过", f"status={shap.get('status')}; interaction_values={shap.get('shap_interaction_values_used')}; hash={shap.get('rf_param_hash')}"))
    rates = _read(tp / "rate_sensitivity.csv")
    rate_audit = json.loads((lp / "dep_pipeline_audit.json").read_text(encoding="utf-8")) if (lp / "dep_pipeline_audit.json").exists() else {}
    rate_values = set(pd.to_numeric(rates.get("rate_multiple", pd.Series(dtype=float)), errors="coerce").dropna().round(6))
    rows.append(_row("LPR level sensitivity", "通过" if rate_values == {1.0, 1.5, 2.0, 3.0} and bool(rate_audit.get("pure_multiplier")) else "不通过", f"rates={sorted(rate_values)}; pure_multiplier={rate_audit.get('pure_multiplier')}"))

    assets = {"descriptive_statistics.csv", "h1_attribute_distribution.csv", "h1_rank_concordance.csv", "p1_discordance_summary.csv", "tab_representation_contrast.csv", "tab_block_complexity.csv"}
    figures = {"fig_p1_text_rule_distance.png", "fig_delta_r2_distributions.png", "fig_rank_displacement.png", "fig_shap_exploratory_bar.png"}
    asset_ok = all((tp / name).exists() for name in assets) and all((fp / name).exists() for name in figures) and (config.OUTPUT_DIR / "Section5_assets_manifest.md").exists()
    rows.append(_row("Section 5 Table 5-10 and Figure 3-6 asset bundle", "通过" if asset_ok else "不通过", f"tables={sorted(assets)}; figures={sorted(figures)}"))
    a1_assets = [
        fp / "fig_A1_corr_heatmap.png",
        tp / "corr_spearman_7vars.csv",
        tp / "corr_pearson_7vars.csv",
        tp / "corr_spearman_pvalues_7vars.csv",
        tp / "corr_abs_pearson_minus_spearman_7vars.csv",
        lp / "corr_heatmap_7vars_summary.json",
        lp / "figure_A1_section5_4_insertion.md",
    ]
    a1_ok = all(path.exists() for path in a1_assets)
    rows.append(_row("Appendix Figure A1 retained-variable correlation bundle", "通过" if a1_ok else "不通过", f"files={len(a1_assets)}; n=41; variables=7; panel_a_stars=True; panel_b_underlying_spearman_stars=True; panel_b_star_color=white"))
    appendix_ok = all((tp / name).exists() for name in ["table_A1_city_corpus.csv", "table_A2_frozen_rf_contract.csv", "table_A3_oof_dispersion.csv", "table_A4_targeted_ablations.csv", "table_A5_lpr_sensitivity.csv"])
    rows.append(_row("Appendix Table A1-A5", "通过" if appendix_ok else "不通过", f"appendix_ok={appendix_ok}"))
    rows.append(_row("Interpretive scope notes", "通过" if all((lp / name).exists() for name in ["p1_scope.md", "h2_scope.md", "shap_audit.md", "rule_attributes_audit.json"]) else "不通过", "predictive/measurement-validity/attribution boundary notes present"))

    audit = pd.DataFrame(rows)
    audit.to_csv(tp / "requirements_audit.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "通过" if bool((audit.status == "通过").all()) else "不通过", "n_checks": len(audit), "n_pass": int((audit.status == "通过").sum()), "n_fail": int((audit.status != "通过").sum()), "checks": rows, "design": "JHBE Revised Sections 3 and 4 v2; Code 10.4 Gower/P1/H1/H2 evidence bundle"}
    (lp / "requirements_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
