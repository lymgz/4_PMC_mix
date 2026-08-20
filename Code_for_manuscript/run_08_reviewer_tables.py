"""Create the reviewer-requested descriptive and institutional tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.data_io import city_key, load_combined
from src.plot_style import colors, save
from run_11_pca_h1 import X_GROUPS, X_GROUP_NAMES


def _indicator_summary() -> pd.DataFrame:
    transfer = pd.read_excel(config.VALIDITY_PATH, sheet_name="Evaluation-transfer", header=None)
    expected_labels = [label for labels in X_GROUPS.values() for label in labels]
    header_labels = [str(transfer.iloc[1, col]).strip() for col in range(1, 1 + len(expected_labels))]
    if header_labels != expected_labels:
        raise ValueError("Evaluation-transfer 二级表头不符合 33 条 X1-X10 契约")
    groups: dict[str, list[int]] = {}
    offset = 1
    for group, labels in X_GROUPS.items():
        groups[group] = list(range(offset, offset + len(labels)))
        offset += len(labels)

    sheet1 = pd.read_excel(config.VALIDITY_PATH, sheet_name="Sheet1", header=0)
    primary = sheet1.iloc[:, 0].astype(str).str.strip()
    source_counts = primary[primary.str.fullmatch(r"x(?:10|[1-9])", case=False)].str.lower().value_counts()

    rows = []
    for index in range(1, 11):
        code = f"x{index}"
        cols = groups.get(code, [])
        values = []
        for col in cols:
            values.extend(pd.to_numeric(transfer.iloc[2 : 2 + config.N_CITIES, col], errors="coerce").dropna().tolist())
        unique = sorted({float(value) for value in values})
        if unique and set(unique).issubset({0.0, 1.0}):
            method = "binary 0/1 coding"
        elif unique:
            method = "numeric source coding; inspect codebook"
        else:
            method = "not available"
        rows.append(
            {
                "indicator_code": code,
                "indicator_name": X_GROUP_NAMES[code],
                "subindicator_count": len(cols) or int(source_counts.get(code, 0)),
                "assignment_method": method,
                "observed_code_values": "|".join(str(int(value)) if value.is_integer() else str(value) for value in unique),
                "data_source": "测试信效度的数据_.xlsx: Evaluation-transfer first P1-P41 block; names from approved 33-item X1-X10 codebook",
                "semantic_name_status": "来自 33-item X1-X10 codebook",
            }
        )
    return pd.DataFrame(rows)


def _representative_release_table(policy: pd.DataFrame) -> pd.DataFrame:
    if config.DEP_CATALOG_PATH is None or not config.DEP_CATALOG_PATH.exists():
        if not config.REPRESENTATIVE_RELEASE_PATH.exists():
            raise FileNotFoundError(
                "缺少 representative release cache；如需从原始 Dep 阶段重建，请设置 JHBE_DEP_DIR。"
            )
        return pd.read_csv(config.REPRESENTATIVE_RELEASE_PATH)

    catalog_path = config.DEP_DIR / "city_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_by_key = {city_key(name): catalog["cities"][name] for name in catalog["city_names"]}

    policy = policy.copy()
    policy["text_chars"] = policy["content"].fillna("").astype(str).str.len()
    keywords = ("监管", "节点", "竣工", "交付", "登记", "封顶", "拨付", "释放", "比例", "尾款")
    policy["milestone_keyword_hits"] = policy["content"].fillna("").astype(str).map(
        lambda text: sum(text.count(keyword) for keyword in keywords)
    )
    selected = (
        policy.sort_values(["Region", "milestone_keyword_hits", "text_chars"], ascending=[True, False, False])
        .groupby("Region", as_index=False, sort=False)
        .head(1)
    )
    rows = []
    for _, item in selected.iterrows():
        key = city_key(item["city"])
        dep_item = catalog_by_key.get(key, {})
        stages = dep_item.get("stages", [])
        max_hold = max(
            (
                float(row.get("stages", {}).get(stage.get("key"), {}).get("hold", 0) or 0)
                for row in dep_item.get("rows", [])
                for stage in stages
            ),
            default=0.0,
        )
        row = {
            "city": item["city"],
            "tier": item["Region"],
            "selection_rule": "tier内政策文本里程碑关键词数优先、文本长度次优",
            "policy_text_chars": int(item["text_chars"]),
            "milestone_keyword_hits": int(item["milestone_keyword_hits"]),
            "max_occupation_months_in_engine": max_hold,
            "final_payment_condition": "stage_5 condition retained as source stage; semantic label requires manuscript cross-check",
            "source": f"{config.POLICIES_PATH.name}; {catalog_path.name}",
            "semantic_mapping_status": "raw engine stages, not re-labelled as policy milestones",
        }
        row.update({f"stage_{index + 1}_release_ratio": stage.get("baseline_release_ratio") for index, stage in enumerate(stages)})
        row.update({f"stage_{index + 1}_first_cohort_month": stage.get("first_cohort_month") for index, stage in enumerate(stages)})
        row.update({f"stage_{index + 1}_last_cohort_month": stage.get("last_cohort_month") for index, stage in enumerate(stages)})
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> dict:
    config.ensure_directories()
    df = load_combined()
    indicator = _indicator_summary()
    indicator.to_csv(config.TABLE_DIR / "x1_x10_summary.csv", index=False, encoding="utf-8-sig")

    long_rows = []
    for variable in config.MODERATORS:
        for value in pd.to_numeric(df[variable], errors="raise"):
            long_rows.append({"variable": config.LABELS.get(variable, variable), "value": float(value)})
    long = pd.DataFrame(long_rows)
    long.to_csv(config.TABLE_DIR / "pmc_score_distribution.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    order = [config.LABELS[item] for item in config.MODERATORS]
    box = ax.boxplot([long.loc[long["variable"] == label, "value"] for label in order], labels=order, patch_artist=True)
    for patch, color in zip(box["boxes"], colors(len(order))):
        patch.set_facecolor(color)
        patch.set_alpha(.75)
    ax.set_ylabel("PMC score")
    ax.set_title("Distribution of four PMC measures across 41 cities")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, config.FIGURE_DIR / "fig_pmc_distribution.png")

    policy = pd.read_excel(config.POLICIES_PATH, sheet_name="Sheet1")
    representative = _representative_release_table(policy)
    representative.to_csv(config.TABLE_DIR / "representative_release_milestones.csv", index=False, encoding="utf-8-sig")
    note = """# Reviewer tables

x1_x10_summary.csv reports the ten primary indicators and source-defined
sub-indicator counts. representative_release_milestones.csv is an auditable
three-tier extraction from the Dep engine and policy text. Its stage columns
retain raw engine stage labels; they are not silently renamed to semantic
milestones such as top-out or registration. The manuscript author must cross-
check those labels against the policy text before inserting the table in prose.
"""
    (config.LOG_DIR / "reviewer_tables.md").write_text(note, encoding="utf-8")
    return {
        "x1_x10_rows": len(indicator),
        "pmc_distribution_rows": len(long),
        "representative_rows": len(representative),
    }


if __name__ == "__main__":
    print(run())

