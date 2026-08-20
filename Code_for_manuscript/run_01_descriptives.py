"""Table 5 and descriptive diagnostics for the 10.4 evidence bundle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.data_io import load_combined
from src.plot_style import colors, save


VARIABLES = [
    ("Dep_reg", "Dep_reg", "CNY/m²", "outcome"),
    ("Dep", "Dep", "CNY/m²", "outcome"),
    ("IV1", "IV1", "CNY/m²", "contextual covariate"),
    ("IV2", "IV2", "months", "contextual covariate"),
    ("IV3", "IV3", "months", "descriptive only"),
    ("IV4", "IV4", "CNY/m²", "excluded (mechanical overlap)"),
    ("PMC_T", "PMC_T", "index", "text measure"),
    ("PMC_I", "PMC_I", "index", "text measure"),
    ("PMC_Frequency", "PMC_F", "index", "text measure"),
    ("PMC_BERT", "PMC_BERT", "index", "text measure"),
]


def run() -> dict:
    config.ensure_directories()
    df = load_combined()
    rows = []
    for source, label, unit, role in VARIABLES:
        values = pd.to_numeric(df[source], errors="coerce")
        desc = values.describe(percentiles=[.25, .5, .75])
        rows.append(
            {
                "variable": label,
                "unit": unit,
                "role": role,
                "n": int(values.notna().sum()),
                "n_missing": int(values.isna().sum()),
                "mean": desc.get("mean"),
                "sd": values.std(ddof=1),
                "min": desc.get("min"),
                "p25": desc.get("25%"),
                "median": desc.get("50%"),
                "p75": desc.get("75%"),
                "max": desc.get("max"),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(config.TABLE_DIR / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    categorical_rows = []
    for variable, values in [("CV1", df["Region"]), ("CV2", pd.to_numeric(df["year"], errors="raise"))]:
        counts = values.value_counts(dropna=False).sort_index()
        for category, n in counts.items():
            categorical_rows.append({"variable": variable, "category": category, "n_cities": int(n), "share": float(n / len(df))})
    categorical = pd.DataFrame(categorical_rows)
    categorical.to_csv(config.TABLE_DIR / "descriptive_statistics_categorical.csv", index=False, encoding="utf-8-sig")
    unique_iv3 = int(pd.to_numeric(df["IV3"], errors="coerce").nunique(dropna=True))
    overlap = df[[config.TARGET, "IV4"]].apply(pd.to_numeric, errors="coerce").dropna()
    rho = spearmanr(overlap[config.TARGET], overlap["IV4"]).statistic if len(overlap) >= 3 else float("nan")
    audit = {"iv3_unique_value_count": unique_iv3, "dep_reg_iv4_spearman_rho": rho, "iv4_role": "descriptive overlap audit only; excluded from predictors", "n_cities": len(df)}
    (config.LOG_DIR / "descriptive_overlap_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    plot = df[["city_key", *config.MODERATORS]].melt(id_vars="city_key", var_name="representation", value_name="score")
    groups = [plot.loc[plot.representation == column, "score"].to_numpy(float) for column in config.MODERATORS]
    box = ax.boxplot(groups, patch_artist=True, showfliers=False, labels=[config.LABELS.get(column, column) for column in config.MODERATORS])
    for patch, color in zip(box["boxes"], colors(len(groups))):
        patch.set_facecolor(color)
        patch.set_alpha(.78)
        patch.set_edgecolor(color)
    ax.set_ylabel("PMC score")
    ax.tick_params(axis="x", rotation=20)
    save(fig, config.FIGURE_DIR / "fig_pmc_distribution.png")
    return {"n": len(df), "target": config.TARGET, "iv3_unique_value_count": unique_iv3, "dep_reg_iv4_spearman_rho": rho}


if __name__ == "__main__":
    print(run())
