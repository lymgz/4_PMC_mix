"""Build the Section 5 main tables, figures, appendices and asset manifest."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.data_io import load_combined
from src.evaluate import paired_bootstrap
from src.plot_style import colors, save


def _read(name: str) -> pd.DataFrame:
    path = config.TABLE_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _paired_values(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty or right.empty:
        return pd.DataFrame(columns=["repeat", "delta_R2"])
    from sklearn.metrics import r2_score

    merged = left.merge(right, on=["repeat", "city_key"], suffixes=("_left", "_right"), validate="one_to_one")
    rows = []
    for repeat, group in merged.groupby("repeat", sort=True):
        rows.append({"repeat": int(repeat), "delta_R2": float(r2_score(group.y_true_left, group.y_pred_left) - r2_score(group.y_true_right, group.y_pred_right))})
    return pd.DataFrame(rows)


def _save_paired_value_tables() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    canonical = _read("canonical_oof_predictions.csv")
    block = _read("block_oof_predictions.csv")
    parallel_values, block_values = {}, {}
    if not canonical.empty:
        base = canonical[canonical.model == "M0"]
        for model in ["M1", "M2", "M3", "M4"]:
            values = _paired_values(canonical[canonical.model == model], base)
            values["comparison"] = f"{model} - M0"
            parallel_values[model] = values
        pd.concat(parallel_values.values(), ignore_index=True).to_csv(config.TABLE_DIR / "paired_delta_r2_values.csv", index=False, encoding="utf-8-sig")
    if not block.empty:
        for model, reference in [("B1", "B0"), ("B2", "B1"), ("B2", "B0")]:
            values = _paired_values(block[block.model == model], block[block.model == reference])
            values["comparison"] = f"{model} - {reference}"
            block_values[f"{model}-{reference}"] = values
        pd.concat(block_values.values(), ignore_index=True).to_csv(config.TABLE_DIR / "block_paired_delta_r2_values.csv", index=False, encoding="utf-8-sig")
    return parallel_values, block_values


def _table9(parallel_values: dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = _read("canonical_results.csv")
    delta_summary = _read("paired_delta_r2.csv")
    rank = _read("pairwise_rank_agreement.csv")
    displacement = _read("rank_displacement.csv")
    labels = {"M0": "none", "M1": "PMC_T", "M2": "PMC_I", "M3": "PMC_F", "M4": "PMC_BERT"}
    rows = []
    for model in ["M0", "M1", "M2", "M3", "M4"]:
        item = results.loc[results.model == model].iloc[0] if not results.loc[results.model == model].empty else pd.Series(dtype=object)
        values = parallel_values.get(model, pd.DataFrame())
        dsummary = delta_summary.loc[delta_summary.comparison == f"{model} - M0"] if not delta_summary.empty else pd.DataFrame()
        rrow = rank.loc[((rank.representation_a == "M0") & (rank.representation_b == model)) | ((rank.representation_a == model) & (rank.representation_b == "M0"))] if not rank.empty else pd.DataFrame()
        disp = displacement.loc[displacement.representation == model] if not displacement.empty else pd.DataFrame()
        boot = paired_bootstrap(values.delta_R2.to_numpy(float)) if not values.empty else {}
        rows.append(
            {
                "model": model,
                "added_index": labels[model],
                "n_features": item.get("n_features", np.nan),
                "oof_r2_mean": item.get("R2_mean", np.nan),
                "oof_r2_sd_across_folds": item.get("R2_fold_sd", np.nan),
                "paired_delta_r2_median": dsummary.iloc[0].get("median_delta_R2", boot.get("median", np.nan)) if not dsummary.empty else np.nan,
                "ci_low": dsummary.iloc[0].get("q025", boot.get("q025", np.nan)) if not dsummary.empty else np.nan,
                "ci_high": dsummary.iloc[0].get("q975", boot.get("q975", np.nan)) if not dsummary.empty else np.nan,
                "share_splits_positive": float((values.delta_R2 > 0).mean()) if not values.empty else np.nan,
                "kendall_tau_vs_M0_ranking": rrow.iloc[0].get("kendall", np.nan) if not rrow.empty else np.nan,
                "median_rank_displacement": disp.iloc[0].get("median_abs_rank_displacement_vs_M0", 0.0) if not disp.empty else np.nan,
                "max_rank_displacement": disp.iloc[0].get("max_abs_rank_displacement_vs_M0", 0.0) if not disp.empty else np.nan,
                "note": "100 paired splits; M0-only tuning and frozen RF parameters" if model != "M0" else "baseline",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(config.TABLE_DIR / "tab_representation_contrast.csv", index=False, encoding="utf-8-sig")
    return table


def _table10(block_values: dict[str, pd.DataFrame]) -> pd.DataFrame:
    blocks = _read("block_results.csv")
    feature_counts = {str(row.model): int(row.n_features) for _, row in blocks.iterrows()} if not blocks.empty else {}
    rows = []
    for comparison, added in [("B1-B0", feature_counts.get("B1", np.nan) - feature_counts.get("B0", np.nan)), ("B2-B1", feature_counts.get("B2", np.nan) - feature_counts.get("B1", np.nan)), ("B2-B0", feature_counts.get("B2", np.nan) - feature_counts.get("B0", np.nan))]:
        values = block_values.get(comparison, pd.DataFrame())
        boot = paired_bootstrap(values.delta_R2.to_numpy(float)) if not values.empty else {}
        low, high = boot.get("q025", np.nan), boot.get("q975", np.nan)
        reading = "gain" if np.isfinite(low) and low > 0 else "loss" if np.isfinite(high) and high < 0 else "no gain (interval spans zero)"
        rows.append({"contrast": comparison.replace("-", "−"), "n_features_added": added, "paired_delta_r2_median": boot.get("median", np.nan), "ci_low": low, "ci_high": high, "share_splits_positive": float((values.delta_R2 > 0).mean()) if not values.empty else np.nan, "reading": reading})
    table = pd.DataFrame(rows)
    table.to_csv(config.TABLE_DIR / "tab_block_complexity.csv", index=False, encoding="utf-8-sig")
    return table


def _figure4(parallel_values: dict[str, pd.DataFrame], block_values: dict[str, pd.DataFrame]) -> None:
    specs = [
        ("M1−M0", "+ PMC-T", parallel_values.get("M1", pd.DataFrame())),
        ("M2−M0", "+ PMC-I", parallel_values.get("M2", pd.DataFrame())),
        ("M3−M0", "+ PMC-F", parallel_values.get("M3", pd.DataFrame())),
        ("M4−M0", "+ PMC-BERT", parallel_values.get("M4", pd.DataFrame())),
        ("B1−B0", "+ four PMC indices", block_values.get("B1-B0", pd.DataFrame())),
        ("B2−B1", "+ IV × PMC interactions", block_values.get("B2-B1", pd.DataFrame())),
        ("B2−B0", "+ indices and interactions", block_values.get("B2-B0", pd.DataFrame())),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 5.15), sharey=True)
    palette = colors(len(specs))
    for index, (label, addition, values) in enumerate(specs):
        row, col = divmod(index, 4)
        ax = axes[row, col]
        data = values.delta_R2.to_numpy(float) if not values.empty else np.array([])
        data = data[np.isfinite(data)]
        if data.size:
            violin = ax.violinplot(
                data,
                positions=[0],
                widths=0.34,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                points=120,
            )
            for body in violin["bodies"]:
                body.set_facecolor(palette[index])
                body.set_edgecolor(palette[index])
                body.set_linewidth(0.8)
                body.set_alpha(.68)
            boot = paired_bootstrap(data)
            ax.errorbar(
                [0],
                [boot["median"]],
                yerr=[[boot["median"] - boot["q025"]], [boot["q975"] - boot["median"]]],
                fmt="o",
                color="#111111",
                markersize=3.4,
                capsize=3,
                linewidth=.85,
                zorder=4,
            )
        ax.axhline(0, color="#222222", linewidth=1.0)
        ax.set_xticks([0], [label])
        ax.set_xlim(-0.48, 0.48)
        ax.grid(axis="y", alpha=.16, linewidth=.6)
        ax.text(
            0.04,
            0.96,
            f"({chr(97 + index)}) {addition}\n$n$ = {data.size} repeats",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
        )
        if col == 0:
            ax.set_ylabel("Paired Δ OOF R²")
    # Seven contrasts occupy a 2×4 grid; remove the unused eighth axes instead
    # of leaving an unexplained empty panel. M4−M0 remains visible at top right.
    fig.delaxes(axes[1, 3])
    fig.text(
        0.5,
        0.015,
        "Black point: median; whiskers: bootstrap 95% CI. Each estimate pools five OOF folds within one repeat (20 repeats; 100 splits total).",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=1.1, h_pad=1.25)
    save(fig, config.FIGURE_DIR / "fig_delta_r2_distributions.png")


def _figure5() -> None:
    ranks = _read("city_representation_ranks.csv")
    if ranks.empty:
        return
    wide = ranks.pivot(index="city_key", columns="representation", values="pred_rank")
    order = [name for name in ["M1", "M2", "M3", "M4"] if name in wide.columns]
    if not order:
        return
    span = wide[order].max(axis=1) - wide[order].min(axis=1)
    highlight = set(span.sort_values(ascending=False).head(5).index)
    city_pinyin = {
        "万宁": "Wanning", "上海": "Shanghai", "上饶": "Shangrao", "东方": "Dongfang",
        "东莞": "Dongguan", "东营": "Dongying", "中卫": "Zhongwei", "中山": "Zhongshan",
        "临高": "Lingao", "丽水": "Lishui", "义乌": "Yiwu", "乐东": "Ledong",
        "乐山": "Leshan", "九江": "Jiujiang", "五指山": "Wuzhishan", "佛山": "Foshan",
        "信阳": "Xinyang", "儋州": "Danzhou", "六安": "Lu'an", "兰州": "Lanzhou",
        "北京": "Beijing", "北海": "Beihai", "合肥": "Hefei", "定安": "Ding'an",
        "屯昌县": "Tunchang", "广州": "Guangzhou", "惠州": "Huizhou", "文昌": "Wenchang",
        "昌江": "Changjiang", "汕头": "Shantou", "海口": "Haikou", "深圳": "Shenzhen",
        "清远": "Qingyuan", "澄迈": "Chengmai", "珠海": "Zhuhai", "琼中": "Qiongzhong",
        "琼海": "Qionghai", "白沙": "Baisha", "肇庆": "Zhaoqing", "陵水": "Lingshui",
        "韶关": "Shaoguan",
    }
    missing_pinyin = sorted(str(city) for city in highlight if str(city) not in city_pinyin)
    if missing_pinyin:
        raise ValueError(f"Figure 5 highlighted cities lack pinyin labels: {missing_pinyin}")
    # Offset the two adjacent right-edge ranks so their labels remain legible.
    label_y_offsets = {"六安": -9, "昌江": 10}
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    base_color = (0.55, 0.55, 0.55, .26)
    for city, row in wide.iterrows():
        values = row[order].to_numpy(float)
        if city in highlight:
            line_color = colors(5)[sorted(highlight).index(city)]
            ax.plot(order, values, color=line_color, linewidth=1.7, alpha=.9)
            ax.annotate(
                city_pinyin[str(city)],
                xy=(order[-1], values[-1]),
                xytext=(5, label_y_offsets.get(str(city), 0)),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=line_color,
                clip_on=False,
            )
        else:
            ax.plot(order, values, color=base_color, linewidth=.65)
    ax.invert_yaxis()
    ax.set_ylabel("Predicted city rank (1 = highest Dep_reg)")
    ax.set_xlabel("PMC representation")
    ax.set_ylim(41.5, .5)
    ax.grid(axis="y", alpha=.14, linewidth=.5)
    fig.tight_layout()
    save(fig, config.FIGURE_DIR / "fig_rank_displacement.png")


def _figure6() -> None:
    shap = _read("shap_exploratory_global.csv")
    if shap.empty or set(["B1", "B2"]) - set(shap.model):
        return
    shap = shap.copy()
    shap["feature_class"] = np.select(
        [
            shap.feature.str.contains("_x_", na=False),
            shap.feature.isin(["IV1", "IV2"]),
            shap.feature.isin(["year_1", "Region_1"]),
            shap.feature.str.startswith("PMC_", na=False),
        ],
        ["IV × PMC interactions", "Market covariates", "Controls", "PMC main effects"],
        default="Other",
    )
    totals = shap.groupby("model")["mean_abs_SHAP"].sum().to_dict()
    shap["share_total_mean_abs"] = shap.apply(
        lambda row: float(row.mean_abs_SHAP) / float(totals[row.model]) if totals.get(row.model, 0) else np.nan,
        axis=1,
    )
    focus = shap[shap.model == "B2"].sort_values("mean_abs_SHAP", ascending=False).head(10)["feature"].tolist()
    display = shap[shap.feature.isin(focus) & shap.model.isin(["B1", "B2"])].copy()
    labels = display.drop_duplicates("feature").set_index("feature")["label"].to_dict()

    def clean_label(feature: str) -> str:
        label = str(labels.get(feature, feature.replace("_x_", " × ")))
        return label.replace("¡Á", "×").replace("_x_", " × ")

    order = list(reversed(focus))
    fig = plt.figure(figsize=(11.7, 5.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.42, 1.20, 0.92], wspace=0.16)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    axes[1].sharey(axes[0])
    composition_ax = fig.add_subplot(grid[0, 2])
    palette = colors(2)
    for panel, (ax, metric, xlabel) in enumerate(
        [(axes[0], "mean_abs_SHAP", "Mean |SHAP|"), (axes[1], "signed_mean_SHAP", "Signed mean SHAP")]
    ):
        for offset, model in enumerate(["B1", "B2"]):
            sub = display[display.model == model].set_index("feature").reindex(order)
            y = np.arange(len(order)) + (-.18 if offset == 0 else .18)
            values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(float)
            finite = np.isfinite(values)
            bars = ax.barh(y[finite], values[finite], height=.30, color=palette[offset], label=model, zorder=2)
            for local_index, (bar, value) in enumerate(zip(bars, values[finite])):
                source_index = np.flatnonzero(finite)[local_index]
                if metric == "mean_abs_SHAP":
                    share = pd.to_numeric(sub["share_total_mean_abs"], errors="coerce").to_numpy(float)[source_index]
                    text = f"{value:.1f} | {share * 100:.1f}%"
                    x = value + max(0.8, np.nanmax(values) * 0.018)
                    ha = "left"
                else:
                    text = f"{value:+.2f}"
                    pad = 0.13
                    x = value + (pad if value >= 0 else -pad)
                    ha = "left" if value >= 0 else "right"
                ax.text(x, bar.get_y() + bar.get_height() / 2, text, va="center", ha=ha, fontsize=6.6, color="#222222")
            if model == "B1":
                for missing_y in y[~finite]:
                    ax.text(0, missing_y, "n/a", ha="center", va="center", fontsize=6.3, color="#777777", style="italic")
        ax.axvline(0, color="#222222", linewidth=.8)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.14, linewidth=.5)
        ax.text(0.02, 0.985, f"({chr(97 + panel)})", transform=ax.transAxes, va="top", fontweight="bold", fontsize=8)
    axes[0].set_yticks(np.arange(len(order)), [clean_label(feature) for feature in order])
    axes[0].set_xlim(0, max(1.0, float(display.mean_abs_SHAP.max()) * 1.36))
    signed_limit = max(1.0, float(display.signed_mean_SHAP.abs().max()) * 1.38)
    axes[1].set_xlim(-signed_limit, signed_limit)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].legend(frameon=False, loc="lower right", ncol=2)

    class_order = ["Market covariates", "Controls", "PMC main effects", "IV × PMC interactions"]
    class_colors = colors(len(class_order), start=0.10, stop=0.92)
    composition = (
        shap[shap.model.isin(["B1", "B2"])]
        .groupby(["model", "feature_class"], as_index=False)["mean_abs_SHAP"]
        .sum()
    )
    composition["share"] = composition.apply(
        lambda row: float(row.mean_abs_SHAP) / float(totals[row.model]) if totals.get(row.model, 0) else np.nan,
        axis=1,
    )
    composition.to_csv(config.TABLE_DIR / "shap_figure6_composition.csv", index=False, encoding="utf-8-sig")
    y_models = np.arange(2)
    left = np.zeros(2, dtype=float)
    for class_name, class_color in zip(class_order, class_colors):
        class_values = []
        for model in ["B1", "B2"]:
            match = composition[(composition.model == model) & (composition.feature_class == class_name)]
            class_values.append(float(match.share.iloc[0]) if not match.empty else 0.0)
        class_values = np.asarray(class_values)
        bars = composition_ax.barh(y_models, class_values * 100, left=left * 100, height=.48, color=class_color, label=class_name)
        for bar, value in zip(bars, class_values):
            if value >= 0.055:
                composition_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="white" if class_name in class_order[:2] else "#202020",
                    fontweight="bold",
                )
        left += class_values
    composition_ax.set_yticks(y_models, ["B1", "B2"])
    composition_ax.invert_yaxis()
    composition_ax.set_xlim(0, 100)
    composition_ax.set_xlabel("Share of total mean |SHAP| (%)")
    composition_ax.grid(axis="x", alpha=.14, linewidth=.5)
    composition_ax.text(0.02, 0.985, "(c)", transform=composition_ax.transAxes, va="top", fontweight="bold", fontsize=8)
    for y, model in zip(y_models, ["B1", "B2"]):
        composition_ax.text(101.0, y, f"Σ={totals[model]:.1f}", va="center", fontsize=6.8, clip_on=False)
    class_handles, class_labels = composition_ax.get_legend_handles_labels()
    fig.legend(
        class_handles,
        class_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.66, 0.075),
        ncol=4,
        fontsize=6.8,
        columnspacing=1.2,
        handlelength=1.8,
    )

    fig.text(
        0.5,
        0.018,
        "Bars are full-sample exploratory model attributions for N = 41 cities. Labels in (a): value | share of the model total; n/a: interaction absent from B1. Signed means indicate average attribution direction, not causal effects.",
        ha="center",
        fontsize=7.2,
    )
    fig.subplots_adjust(left=0.19, right=0.95, bottom=0.20, top=0.98, wspace=0.18)
    save(fig, config.FIGURE_DIR / "fig_shap_exploratory_bar.png")


def _appendices(df: pd.DataFrame) -> None:
    city = df[["city_key", "city", "Region", "year"]].copy()
    city["policy_text_source"] = config.display_path(config.POLICIES_PATH)
    city.to_csv(config.TABLE_DIR / "table_A1_city_corpus.csv", index=False, encoding="utf-8-sig")
    frozen_path = config.FROZEN_RF_PATH
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else {}
    rows = [{"parameter": key, "value": value} for key, value in frozen.get("frozen_params", {}).items()]
    rows += [{"parameter": "rf_param_hash", "value": frozen.get("rf_param_hash")}, {"parameter": "tuning_scope", "value": frozen.get("tuning_scope")}, {"parameter": "tuning_metadata", "value": json.dumps(frozen.get("tuning_metadata", {}), ensure_ascii=False)}]
    pd.DataFrame(rows).to_csv(config.TABLE_DIR / "table_A2_frozen_rf_contract.csv", index=False, encoding="utf-8-sig")
    folds = _read("canonical_folds.csv")
    if not folds.empty:
        grouped = folds.groupby("model")["R2"].agg(["mean", "std", "min", "max", lambda s: s.quantile(.75) - s.quantile(.25), "count"]).reset_index()
        grouped.columns = ["model", "mean", "sd", "min", "max", "IQR", "n_folds"]
    else:
        grouped = pd.DataFrame()
    grouped.to_csv(config.TABLE_DIR / "table_A3_oof_dispersion.csv", index=False, encoding="utf-8-sig")
    ablation = _read("targeted_ablation_results.csv")
    registry = _read("targeted_ablation_registry.csv")
    if not ablation.empty and not registry.empty and "model" in registry:
        ablation = ablation.merge(registry.drop_duplicates("model"), on="model", how="left", suffixes=("", "_registry"))
    ablation.to_csv(config.TABLE_DIR / "table_A4_targeted_ablations.csv", index=False, encoding="utf-8-sig")
    rate = _read("rate_sensitivity.csv")
    ratio = _read("rate_sensitivity_ratio_check.csv")
    if not rate.empty and not ratio.empty:
        rate = rate.merge(ratio, on="rate_multiple", how="left", suffixes=("", "_ratio"))
    rate.to_csv(config.TABLE_DIR / "table_A5_lpr_sensitivity.csv", index=False, encoding="utf-8-sig")
    for source, target in [("fig_corr_heatmap.png", "fig_A1_corr_heatmap.png"), ("fig_shapley_r2.png", "fig_A2_shapley_r2.png")]:
        src, dst = config.FIGURE_DIR / source, config.FIGURE_DIR / target
        if src.exists():
            shutil.copy2(src, dst)


def _manifest() -> Path:
    assets = [
        ("Table 5", "descriptive_statistics.csv", "Sample-level descriptive statistics; IV3 and IV4 roles remain auditable."),
        ("Table 6", "h1_attribute_distribution.csv", "Distribution of the five release attributes and absent/flag counts."),
        ("Table 7", "h1_rank_concordance.csv", "Policy-observed and Dep-operational ranking concordance; Kendall tau with 1,000 bootstrap resamples and explicit evidence-layer labels."),
        ("Table 8", "p1_discordance_summary.csv", "Four separate text distances against the same Gower rule distance."),
        ("Table 9", "tab_representation_contrast.csv", "M0-M4 parallel representation contrast using 100 paired splits."),
        ("Table 10", "tab_block_complexity.csv", "B0-B2 incremental complexity comparison."),
        ("Figure 3", "fig_p1_text_rule_distance.png", "Four-panel Viridis scatter plot; dotted/ dashed lines mark pre-specified quartiles."),
        ("Figure 4", "fig_delta_r2_distributions.png", "Paired ΔR² distributions for four parallel PMC-representation contrasts and three block-model contrasts. Each violin contains 20 repeat-level estimates pooling five OOF folds per repeat (100 splits total); black points and whiskers show medians and bootstrap 95% intervals."),
        ("Figure 5", "fig_rank_displacement.png", "41-city predicted rank displacement across four PMC representations."),
        ("Figure 6", "fig_shap_exploratory_bar.png", "Exploratory B1/B2 attribution for N=41 cities: top B2 features by mean absolute SHAP with values and within-model shares, signed mean SHAP with direction labels, and total attribution composition across market covariates, controls, PMC main effects and constructed IV×PMC inputs. These are full-sample model attributions, not causal effects or SHAP interaction values."),
        ("Figure 6 data", "shap_figure6_composition.csv", "Feature-class composition of total mean absolute SHAP used in Figure 6 panel (c)."),
        ("Table A1", "table_A1_city_corpus.csv", "41-city corpus list and policy coverage source."),
        ("Table A2", "table_A2_frozen_rf_contract.csv", "Frozen M0-only tuning contract and parameter hash."),
        ("Table A3", "table_A3_oof_dispersion.csv", "Fold-level OOF R² dispersion."),
        ("Table A4", "table_A4_targeted_ablations.csv", "Targeted diagnostic ablations."),
        ("Table A5", "table_A5_lpr_sensitivity.csv", "LPR multiplier sensitivity and linearity check."),
        ("Figure A1", "fig_A1_corr_heatmap.png", "Caption: (a) Spearman rank correlations between Dep_reg and the retained analysis variables (N = 41), with significance stars in the upper triangle and numeric coefficients in the lower triangle (*** p<0.001, ** p<0.01, * p<0.10); (b) absolute difference between Pearson and Spearman coefficients, with stars in the upper triangle referring to the underlying Spearman tests and numeric differences in the lower triangle. Right-panel stars are white; larger differences indicate rank–moment divergence driven by skewness. Panel labels are placed below the two plots."),
        ("Figure A1 data", "corr_spearman_7vars.csv", "Seven-variable Spearman coefficient matrix used by Figure A1."),
        ("Figure A1 audit", "corr_spearman_pvalues_7vars.csv", "Pairwise Spearman p-values retained for audit and displayed as upper-triangle stars in both panels."),
        ("Figure A1 data", "corr_abs_pearson_minus_spearman_7vars.csv", "Absolute Pearson-Spearman difference matrix used by Figure A1 right panel."),
        ("Figure A2", "fig_A2_shapley_r2.png", "PMC R² Shapley decomposition; exploratory attribution only."),
        ("H1 audit artifact", "h1_dep_expert_mapping_template.csv", "41-city template for confirming Dep stage semantics, common denominators, expert source evidence and final H1 values; not a manuscript result table."),
    ]
    path = config.OUTPUT_DIR / "Section5_assets_manifest.md"
    lines = ["# Section 5 assets generated by Code 10.4", "", "Section 5 figures use the Viridis colour family where a sequential scale is appropriate; Figure A1 uses a zero-centred diverging scale for signed correlations and Viridis for rank-moment divergence. Captions and interpretive boundary are recorded here; figures contain no internal titles.", "", "| Asset | File | Where to find it | Description |", "|---|---|---|---|"]
    for asset, filename, description in assets:
        file_path = config.FIGURE_DIR / filename if filename.endswith((".png", ".pdf")) else config.TABLE_DIR / filename
        lines.append(f"| {asset} | `{filename}` | `{config.display_path(file_path)}` | {description} |")
    lines += ["", "## Reproduction entry points", "", f"- Main pipeline: `{config.display_path(ROOT / 'run_pipeline.py')}`", f"- Section 5 asset assembly: `{config.display_path(ROOT / 'run_16_section5_assets.py')}`", f"- Strategy input: `{config.display_path(config.REVISED_SECTIONS_PATH)}`", f"- Manual rule coding input: `{config.display_path(config.RULE_ATTRIBUTES_PATH)}`", "", "## Evidence boundary", "", "The RF results are predictive comparisons. SHAP and the R² Shapley decomposition are exploratory model-attribution quantities; they are not effect estimates, mechanisms, significance tests, or causal policy claims. B2 products are constructed features, not SHAP interaction values."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run() -> dict:
    config.ensure_directories()
    df = load_combined()
    parallel_values, block_values = _save_paired_value_tables()
    table9 = _table9(parallel_values)
    table10 = _table10(block_values)
    _figure4(parallel_values, block_values)
    _figure5()
    _figure6()
    _appendices(df)
    manifest = _manifest()
    return {"table9_rows": len(table9), "table10_rows": len(table10), "manifest": config.display_path(manifest), "figures": sorted(path.name for path in config.FIGURE_DIR.glob("*.png"))}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
