"""P1: Gower rule distance against four separate text distances."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.data_io import load_combined
from src.plot_style import colors, save
from src.rule_attributes import build_rule_attributes


TEXT_MEASURES = {
    "T": "PMC_T",
    "I": "PMC_I",
    "F": "PMC_Frequency",
    "BERT": "PMC_BERT",
}
GOWER_CATEGORICAL = ["basis_category", "release_trigger_category", "schedule_form_category"]
GOWER_NUMERIC = ["n_release_steps", "final_retained_share"]


def _gower_distance(left: pd.Series, right: pd.Series, frame: pd.DataFrame) -> float:
    contributions = []
    for column in GOWER_CATEGORICAL:
        a, b = left.get(column), right.get(column)
        if pd.isna(a) or pd.isna(b):
            continue
        contributions.append(0.0 if str(a) == str(b) else 1.0)
    for column in GOWER_NUMERIC:
        numeric_pair = pd.to_numeric(pd.Series([left.get(column), right.get(column)]), errors="coerce")
        if numeric_pair.isna().any():
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        span = float(values.max() - values.min())
        contributions.append(0.0 if span == 0 else abs(float(numeric_pair.iloc[0]) - float(numeric_pair.iloc[1])) / span)
    return float(np.mean(contributions)) if contributions else np.nan


def _pair_frame(df: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    text = df.set_index("city_key")
    rule = rules.set_index("city_key")
    dep = pd.to_numeric(text[config.TARGET], errors="coerce")
    keys = list(text.index)
    rows = []
    for index, left_key in enumerate(keys):
        for right_key in keys[index + 1 :]:
            left_rule, right_rule = rule.loc[left_key], rule.loc[right_key]
            row = {
                "city_a": left_key,
                "city_b": right_key,
                "rule_distance_gower": _gower_distance(left_rule, right_rule, rules),
                "Dep_reg_gap": abs(float(dep.loc[left_key]) - float(dep.loc[right_key])),
                "rule_distance_components": ";".join(GOWER_CATEGORICAL + GOWER_NUMERIC),
                "scope": "descriptive P1 measurement-validity evidence; not causal",
            }
            for label, column in TEXT_MEASURES.items():
                row[f"text_dist_{label}"] = abs(float(text.loc[left_key, column]) - float(text.loc[right_key, column]))
            rows.append(row)
    return pd.DataFrame(rows)


def _summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in TEXT_MEASURES:
        text_col = f"text_dist_{label}"
        valid = pairs[["rule_distance_gower", text_col]].dropna()
        q1 = float(valid[text_col].quantile(0.25))
        q3 = float(valid.rule_distance_gower.quantile(0.75))
        discordant = valid[text_col].le(q1) & valid.rule_distance_gower.ge(q3)
        selected = valid.loc[discordant].index
        city_counts = pd.concat([pairs.loc[selected, "city_a"], pairs.loc[selected, "city_b"]]).value_counts()
        top = ", ".join(f"{city} ({count})" for city, count in city_counts.head(5).items())
        rho = spearmanr(valid[text_col], valid.rule_distance_gower).statistic if len(valid) >= 3 else np.nan
        rows.append(
            {
                "measure": f"PMC_{label}",
                "n_pairs": int(len(valid)),
                "q1_text_cut": q1,
                "q3_rule_cut": q3,
                "n_discordant": int(discordant.sum()),
                "chance_benchmark": config.P1_CHANCE_BENCHMARK,
                "ratio_to_chance": float(discordant.sum() / config.P1_CHANCE_BENCHMARK),
                "top_cities": top,
                "spearman_text_rule": rho,
                "scope": "counts are descriptive because each city occurs in 40 pairs; no pair-level significance test",
            }
        )
        pairs.loc[:, f"discordant_{label}"] = False
        pairs.loc[valid.index[discordant], f"discordant_{label}"] = True
    return pd.DataFrame(rows)


def _plot(pairs: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0), sharex=True, sharey=True)
    palette = colors(4, start=0.12, stop=0.78)
    neutral = "#9b9b9b"

    def percentile_rank(values: pd.Series) -> pd.Series:
        return pd.to_numeric(values, errors="coerce").rank(method="average", pct=True) * 100.0

    def cut_percentile(values: pd.Series, cut: float) -> float:
        numeric = pd.to_numeric(values, errors="coerce")
        ranks = percentile_rank(numeric)
        exact = np.isclose(numeric.to_numpy(float), float(cut))
        if exact.any():
            return float(ranks.loc[exact].mean())
        order = np.argsort(numeric.to_numpy(float))
        return float(np.interp(float(cut), numeric.iloc[order], ranks.iloc[order]))

    for index, (label, _) in enumerate(TEXT_MEASURES.items()):
        ax = axes.flat[index]
        text_col = f"text_dist_{label}"
        valid = pairs[["rule_distance_gower", text_col]].dropna()
        row = summary.iloc[index]
        x_percentile = percentile_rank(valid.rule_distance_gower)
        y_percentile = percentile_rank(valid[text_col])
        x_cut = cut_percentile(valid.rule_distance_gower, float(row.q3_rule_cut))
        y_cut = cut_percentile(valid[text_col], float(row.q1_text_cut))
        discordant = valid[text_col].le(row.q1_text_cut) & valid.rule_distance_gower.ge(row.q3_rule_cut)

        # Fixed-seed horizontal jitter reveals pairs that share the same Gower
        # distance without changing the values used for classification.
        rng = np.random.default_rng(config.SEED + index)
        x_plot = np.clip(x_percentile.to_numpy(float) + rng.uniform(-0.75, 0.75, len(valid)), 0.0, 100.0)
        y_plot = y_percentile.to_numpy(float)

        ax.fill_between([x_cut, 102], -2, y_cut, color=palette[index], alpha=0.07, linewidth=0)
        ax.scatter(
            x_plot[~discordant.to_numpy()],
            y_plot[~discordant.to_numpy()],
            s=9,
            alpha=0.25,
            color=neutral,
            edgecolors="none",
            rasterized=True,
        )
        ax.scatter(
            x_plot[discordant.to_numpy()],
            y_plot[discordant.to_numpy()],
            s=22,
            alpha=0.88,
            color=palette[index],
            edgecolors="white",
            linewidths=0.25,
            rasterized=True,
        )
        ax.axvline(x_cut, color=palette[index], linewidth=1.0, linestyle="--")
        ax.axhline(y_cut, color=palette[index], linewidth=1.0, linestyle=":")
        rho = float(row.spearman_text_rule)
        ax.text(
            0.03,
            0.97,
            (
                f"({chr(97 + index)}) PMC-{label}\n"
                f"820 city pairs; discordant = {int(row.n_discordant)}\n"
                f"benchmark = {int(row.chance_benchmark)}; Spearman $\\rho$ = {rho:.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.set_xlabel("Gower rule-distance percentile")
        ax.set_ylabel("Text-distance percentile")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="none", markerfacecolor=neutral, markeredgecolor="none", alpha=0.45, markersize=5, label="Other city pairs"),
        Line2D([0], [0], marker="o", linestyle="none", color="none", markerfacecolor=palette[2], markeredgecolor="white", markersize=6, label="Discordant: rule $\\geq$ Q3 and text $\\leq$ Q1"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save(fig, config.FIGURE_DIR / "fig_p1_text_rule_distance.png")


def run():
    config.ensure_directories()
    df = load_combined()
    rules, audit = build_rule_attributes()
    pairs = _pair_frame(df, rules)
    summary = _summary(pairs)
    required = ["city_a", "city_b", "rule_distance_gower", "text_dist_T", "text_dist_I", "text_dist_F", "text_dist_BERT", "discordant_T", "discordant_I", "discordant_F", "discordant_BERT"]
    pairs = pairs[required + ["Dep_reg_gap", "rule_distance_components", "scope"]]
    pairs.to_csv(config.TABLE_DIR / "p1_text_rule_pair_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(config.TABLE_DIR / "p1_discordance_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(config.TABLE_DIR / "p1_pair_audit_summary.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(config.TABLE_DIR / "overlap_diagnostics.csv", index=False, encoding="utf-8-sig")
    (config.LOG_DIR / "p1_scope.md").write_text(
        "P1 uses a five-component Gower distance: three categorical release-rule fields and two quantitative fields. Text distance is computed separately for PMC-T, PMC-I, PMC-Frequency and PMC-BERT. Discordant-pair counts are descriptive measurement-validity evidence, not causal tests.\n",
        encoding="utf-8",
    )
    (config.LOG_DIR / "p1_rule_distance_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(pairs, summary)
    return pairs, summary


if __name__ == "__main__":
    print(run()[1].to_string(index=False))
