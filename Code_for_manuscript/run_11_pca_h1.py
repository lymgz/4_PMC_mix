"""H1: clause-level release-attribute distributions and ranking concordance.

The historical filename is retained so existing local entry points do not
break.  This 10.4 implementation no longer performs dimension reduction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.data_io import city_key, load_combined
from src.dep_attributes import build_dep_operational_attributes, dep_attribute_distribution
from src.plot_style import colors, save
from src.rule_attributes import attribute_distribution, build_rule_attributes


X_GROUPS = {
    "x1": ["X1: 1", "X1: 2", "X1: 3"],
    "x2": ["X2: 1", "X2: 2", "X2: 3", "X2: 4"],
    "x3": ["X3: 1", "X3: 2", "X3: 3"],
    "x4": ["X4: 1", "X4: 2"],
    "x5": ["X5: 1", "X5: 2", "X5: 3", "X5: 4", "X5: 5"],
    "x6": ["X6: 1", "X6: 2", "X6: 3"],
    "x7": ["X7: 1", "X7: 2", "X7: 3", "X7: 4", "X7: 5"],
    "x8": ["X8: 1", "X8: 2"],
    "x9": ["X9: 1", "X9: 2", "X9: 3", "X9: 4", "X9: 5"],
    "x10": ["X10"],
}
X_GROUP_NAMES = {
    "x1": "Policy issuing authority",
    "x2": "Regulatory period",
    "x3": "Regulatory requirements",
    "x4": "Disbursement conditions",
    "x5": "Fund-use nodes",
    "x6": "Regulated parties",
    "x7": "Developer obligations",
    "x8": "Relaxation clauses",
    "x9": "Additional security",
    "x10": "Credit-grade classification",
}


def _evaluation_transfer_matrix():
    """Read the authoritative 33-column Evaluation-transfer city matrix."""
    raw = pd.read_excel(config.VALIDITY_PATH, sheet_name="Evaluation-transfer", header=None)
    expected_labels = [label for labels in X_GROUPS.values() for label in labels]
    header_labels = [str(raw.iloc[1, col]).strip() for col in range(1, 1 + len(expected_labels))]
    if header_labels != expected_labels:
        raise ValueError("Evaluation-transfer 二级表头不符合33条X1-X10契约")
    block = raw.iloc[2 : 2 + config.N_CITIES, : 1 + len(expected_labels)].copy()
    pmc_codes = block.iloc[:, 0].astype(str).str.strip().tolist()
    expected_codes = [f"P{i}" for i in range(1, config.N_CITIES + 1)]
    if pmc_codes != expected_codes:
        raise ValueError("Evaluation-transfer 第一数据块必须按 P1-P41 排列")
    source = pd.read_excel(config.VALIDITY_PATH, sheet_name="machine_learning", header=1)
    source.columns = [str(column).strip() for column in source.columns]
    source = source[["PMCcode", "City"]].dropna().copy()
    source["PMCcode"] = source["PMCcode"].astype(str).str.strip()
    source["city_key"] = source["City"].map(city_key)
    if set(source.PMCcode) != set(expected_codes) or source.PMCcode.duplicated().any() or source.city_key.duplicated().any():
        raise ValueError("machine_learning 必须提供唯一的 P1-P41 到城市键映射")
    code_to_city = source.set_index("PMCcode")["city_key"].to_dict()
    records = []
    wide = pd.DataFrame({"city_key": [code_to_city[code] for code in pmc_codes], "pmc_code": pmc_codes})
    indicator_columns = []
    indicator_labels = {}
    for index, (group, label) in enumerate(
        ((group, label) for group, labels in X_GROUPS.items() for label in labels), start=1
    ):
        stable = f"{group}_{X_GROUPS[group].index(label) + 1}"
        indicator_columns.append(stable)
        indicator_labels[stable] = label
        values = pd.to_numeric(block.iloc[:, index], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Evaluation-transfer 指标 {label} 存在缺失值")
        wide[stable] = values.to_numpy(dtype=float)
        for row_index, code in enumerate(pmc_codes):
            records.append({"city_key": code_to_city[code], "pmc_code": code, "indicator": stable, "source_label": label, "dimension": group, "value": float(values.iloc[row_index])})
    metadata = {
        "source_file": config.VALIDITY_PATH.name,
        "source_sheet": "Evaluation-transfer",
        "n_cities": config.N_CITIES,
        "n_indicator_columns": len(indicator_columns),
        "indicator_columns": indicator_columns,
        "indicator_labels": indicator_labels,
        "indicator_counts": {group: len(labels) for group, labels in X_GROUPS.items()},
        "indicator_names": X_GROUP_NAMES,
        "taxonomy_basis": "approved 33-item X1-X10 codebook; retained for descriptive source traceability, not for H1 dimension reduction",
        "legacy_matrix_source": f"{config.PMC_DATA_PATH.name}/matrix; not used for X1-X10 classification",
    }
    return pd.DataFrame(records), wide, metadata


def _kendall_bootstrap(left: pd.Series, right: pd.Series, *, seed: int = config.SEED, n_boot: int = config.H1_BOOTSTRAP_N) -> dict:
    pair = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    n = len(pair)
    if n < 3 or pair.left.nunique() < 2 or pair.right.nunique() < 2:
        return {"kendall_tau": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": int(n_boot), "n_cities_ranked": n}
    point = float(kendalltau(pair.left, pair.right).statistic)
    rng = np.random.default_rng(seed)
    boot = []
    values = pair.to_numpy(float)
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        tau = kendalltau(sample[:, 0], sample[:, 1]).statistic
        if np.isfinite(tau):
            boot.append(float(tau))
    return {"kendall_tau": point, "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)), "n_boot": int(n_boot), "n_cities_ranked": n}


def _rank_concordance(rules: pd.DataFrame, rankings: dict[str, str] | None = None) -> pd.DataFrame:
    rankings = rankings or {
        "release steps count": "n_release_steps",
        "share drawable at main-structure acceptance": "share_drawable_at_topout",
        "final retained share": "final_retained_share",
    }
    rows = []
    items = list(rankings.items())
    for index, (left_label, left_col) in enumerate(items):
        for right_label, right_col in items[index + 1 :]:
            boot = _kendall_bootstrap(rules[left_col], rules[right_col])
            decision = bool(np.isfinite(boot["kendall_tau"]) and boot["kendall_tau"] >= 0.7 and boot["ci_low"] > 0.5)
            rows.append({"ranking_a": left_label, "ranking_b": right_label, "n_cities_ranked": boot["n_cities_ranked"], "kendall_tau": boot["kendall_tau"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "decision_rule_met": decision, "field_a": left_col, "field_b": right_col, "bootstrap_n": boot["n_boot"]})
    return pd.DataFrame(rows)


def run():
    config.ensure_directories()
    df = load_combined()
    rules, policy_audit = build_rule_attributes()
    dep_rules, dep_audit = build_dep_operational_attributes()
    if set(rules.city_key) != set(df.city_key):
        raise ValueError("人工规则编码与41城模型数据的城市键不一致")
    if set(dep_rules.city_key) != set(df.city_key):
        raise ValueError("Dep operational 属性与41城模型数据的城市键不一致")
    long, xwide, taxonomy = _evaluation_transfer_matrix()
    rules = rules.merge(dep_rules, on="city_key", how="left", validate="one_to_one", suffixes=("", "_dep"))
    if "city_dep" in rules:
        rules = rules.drop(columns=["city_dep"])
    rules = rules.sort_values("city_key").reset_index(drop=True)
    xwide = xwide.sort_values("city_key").reset_index(drop=True)
    combined_rules = rules.merge(df[["city_key"]], on="city_key", how="inner", validate="one_to_one")
    policy_distribution = attribute_distribution(combined_rules)
    policy_distribution["evidence_layer"] = "policy_observed"
    operational_distribution = dep_attribute_distribution(dep_rules)
    operational_distribution["evidence_layer"] = "dep_operational"
    distribution = pd.concat([policy_distribution, operational_distribution], ignore_index=True)
    policy_rankings = {
        "release steps count": "n_release_steps",
        "share drawable at main-structure acceptance": "share_drawable_at_topout",
        "final retained share": "final_retained_share",
    }
    operational_rankings = {
        "active Dep release stages": "dep_n_release_steps",
        "cumulative Dep share through mapped topout": "dep_share_drawable_at_topout",
        "mapped Dep terminal-stage share": "dep_final_retained_share",
    }
    policy_concordance = _rank_concordance(combined_rules, policy_rankings)
    policy_concordance["evidence_layer"] = "policy_observed"
    policy_concordance["ranking_definition"] = "direct or explicitly parseable policy attributes"
    operational_concordance = _rank_concordance(dep_rules, operational_rankings)
    operational_concordance["evidence_layer"] = "dep_operational"
    operational_concordance["ranking_definition"] = "Dep-derived proxy; requires stage mapping and common-denominator status"
    concordance = pd.concat([policy_concordance, operational_concordance], ignore_index=True)
    rule_data = combined_rules.merge(df, on="city_key", how="left", suffixes=("", "_model"), validate="one_to_one")

    rules.to_csv(config.TABLE_DIR / "rule_attributes_41cities.csv", index=False, encoding="utf-8-sig")
    dep_rules.to_csv(config.TABLE_DIR / "dep_operational_attributes_41cities.csv", index=False, encoding="utf-8-sig")
    mapping_template = dep_rules[["city_key", "city", "dep_stage_keys", "dep_stage_names", "dep_active_stage_names", "dep_topout_stage_key", "dep_registration_stage_key"]].copy()
    mapping_template["expert_confirmed_topout_stage_key"] = ""
    mapping_template["expert_confirmed_registration_stage_key"] = ""
    mapping_template["expert_confirmed_n_release_steps"] = ""
    mapping_template["expert_confirmed_share_drawable_at_topout"] = ""
    mapping_template["expert_confirmed_final_retained_share"] = ""
    mapping_template["common_denominator_definition"] = ""
    mapping_template["expert_source_id_or_page"] = ""
    mapping_template["expert_evidence_quote"] = ""
    mapping_template["expert_validation_status"] = "pending_review"
    mapping_template.to_csv(config.TABLE_DIR / "h1_dep_expert_mapping_template.csv", index=False, encoding="utf-8-sig")
    rule_data.to_csv(config.TABLE_DIR / "h1_operational_features.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(config.TABLE_DIR / "h1_attribute_distribution.csv", index=False, encoding="utf-8-sig")
    concordance.to_csv(config.TABLE_DIR / "h1_rank_concordance.csv", index=False, encoding="utf-8-sig")
    concordance.to_csv(config.TABLE_DIR / "h1_rank_concordance_audit.csv", index=False, encoding="utf-8-sig")
    policy_concordance.to_csv(config.TABLE_DIR / "h1_rank_concordance_policy.csv", index=False, encoding="utf-8-sig")
    operational_concordance.to_csv(config.TABLE_DIR / "h1_rank_concordance_dep_operational.csv", index=False, encoding="utf-8-sig")
    long.to_csv(config.TABLE_DIR / "x1_x10_indicator_long.csv", index=False, encoding="utf-8-sig")
    xwide.to_csv(config.TABLE_DIR / "x1_x10_city_matrix.csv", index=False, encoding="utf-8-sig")
    (config.LOG_DIR / "h1_indicator_taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "n_cities": config.N_CITIES,
        "policy_observed": policy_audit,
        "dep_operational": dep_audit,
        "interpretation_boundary": "Dep operational proxies do not replace policy-observed attributes; their stage mapping, denominator and expert provenance must be reported separately.",
    }
    (config.LOG_DIR / "rule_attributes_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.LOG_DIR / "dep_attribute_audit.json").write_text(json.dumps(dep_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    def _layer_judgement(layer_frame: pd.DataFrame) -> dict:
        available = layer_frame["decision_rule_met"].notna() & layer_frame["kendall_tau"].notna()
        if available.sum() < 3:
            judgement = "qualified_incomplete_quantitative_coding"
        elif bool(layer_frame.loc[available, "decision_rule_met"].all()):
            judgement = "H1_not_supported_by_concordance_rule"
        else:
            judgement = "H1_supported_or_qualified_by_divergent_rankings"
        return {"h1_judgement": judgement, "n_pairs_with_estimable_tau": int(available.sum()), "n_rank_pairs": int(len(layer_frame))}

    policy_judgement = _layer_judgement(policy_concordance)
    operational_judgement = _layer_judgement(operational_concordance)
    h1_judgement = {
        "criterion": "H1 fails only if all three ranking pairs have Kendall tau >= 0.7 and bootstrap interval excludes 0.5; missing policy values are not imputed and Dep proxies are reported as a separate layer.",
        "h1_judgement": policy_judgement["h1_judgement"],
        "layers": {"policy_observed": policy_judgement, "dep_operational": operational_judgement},
        "n_cities": int(len(rules)),
        "bootstrap_n": config.H1_BOOTSTRAP_N,
        "seed": config.SEED,
        "policy_quantitative_missing_counts": policy_audit["missing_counts"],
        "dep_operational_coverage": dep_audit["coverage"],
        "interpretation_scope": "descriptive rule heterogeneity; Dep layer is an operational proxy and carries no causal policy effect or mechanism claim",
    }
    (config.LOG_DIR / "h1_judgement.json").write_text(json.dumps(h1_judgement, ensure_ascii=False, indent=2), encoding="utf-8")

    plot = distribution[distribution.attribute.isin(["Class of release trigger", "Form of the schedule", "Intermediate steps"])].copy()
    plot["label"] = plot["attribute"] + "\n" + plot["category"]
    plot = plot.sort_values("n_cities")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bar_colors = colors(len(plot))
    bars = ax.barh(plot["label"], plot["n_cities"], color=bar_colors)
    ax.set_xlabel("Number of cities")
    ax.set_xlim(0, max(1, float(plot.n_cities.max()) * 1.16))
    for bar, value in zip(bars, plot.n_cities):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=8)
    save(fig, config.FIGURE_DIR / "fig_h1_attribute_distribution.png")
    return distribution, concordance


if __name__ == "__main__":
    print(run()[1].to_string(index=False))
