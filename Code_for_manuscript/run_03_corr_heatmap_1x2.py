"""Generate Figure A1: retained-variable rank correlations and rank-moment divergence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config
from src.data_io import load_combined


# The formal dependent variable is displayed consistently as Dep_reg. The
# frequency-based text representation is displayed as PMC_F while its source
# column retains the historical implementation name PMC_Frequency.
VARIABLES = ["Dep_reg", "IV1", "IV2", "PMC_T", "PMC_I", "PMC_F", "PMC_BERT"]
SOURCE_COLUMNS = {
    "Dep_reg": "Dep_reg",
    "IV1": "IV1",
    "IV2": "IV2",
    "PMC_T": "PMC_T",
    "PMC_I": "PMC_I",
    "PMC_F": "PMC_Frequency",
    "PMC_BERT": "PMC_BERT",
}


def _pairwise_correlations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return Spearman rho, Pearson r, and Spearman p-value matrices."""
    rho = pd.DataFrame(np.nan, index=VARIABLES, columns=VARIABLES, dtype=float)
    pearson = rho.copy()
    pvalues = rho.copy()
    for left in VARIABLES:
        for right in VARIABLES:
            pair = frame[[SOURCE_COLUMNS[left], SOURCE_COLUMNS[right]]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if left == right:
                rho.loc[left, right] = 1.0
                pearson.loc[left, right] = 1.0
                pvalues.loc[left, right] = 0.0
            elif len(pair) >= 3 and all(pair[column].nunique() > 1 for column in pair.columns):
                s_result = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                p_result = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                rho.loc[left, right] = float(s_result.statistic)
                pearson.loc[left, right] = float(p_result.statistic)
                pvalues.loc[left, right] = float(s_result.pvalue)
    return rho, pearson, pvalues


def _diverging_cmap() -> LinearSegmentedColormap:
    """Use a print-friendly blue-white-red diverging scale centred at zero."""
    # This is the matplotlib equivalent of a ``vlag``-style diverging scale.
    return LinearSegmentedColormap.from_list(
        "vlag_like", ["#2166ac", "#f7f7f7", "#b2182b"], N=256
    )


def _significance_stars(pvalue: float) -> str:
    """Return the requested unadjusted two-sided significance notation."""
    if pd.isna(pvalue):
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def _draw_upper_triangle(
    ax: plt.Axes,
    values: pd.DataFrame,
    labels: list[str],
    *,
    cmap,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    bold_threshold: float,
    signed: bool,
    pvalues: pd.DataFrame | None = None,
    show_stars: bool = False,
    star_color: str | None = None,
):
    """Draw an upper-triangle heatmap with square cell frames and annotations."""
    matrix = values.to_numpy(dtype=float)
    display = np.ma.masked_where(~np.triu(np.ones(matrix.shape, dtype=bool), k=1), matrix)
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    cmap.set_bad("white")
    image = ax.imshow(display, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0, labelsize=8.2)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_facecolor("white")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#b8b8b8",
                    linewidth=0.55,
                    zorder=3,
                )
            )
            if pd.isna(matrix[i, j]):
                continue
            value = float(matrix[i, j])
            if i < j:
                if not show_stars or pvalues is None:
                    continue
                stars = _significance_stars(float(pvalues.iloc[i, j]))
                if stars:
                    if star_color is not None:
                        text_color = star_color
                    elif signed:
                        text_color = "white" if abs(value) >= bold_threshold else "#262626"
                    else:
                        text_color = "white" if value >= bold_threshold else "#262626"
                    ax.text(
                        j,
                        i,
                        stars,
                        ha="center",
                        va="center",
                        fontsize=8.8,
                        fontweight="bold",
                        color=text_color,
                        zorder=4,
                    )
            elif i > j:
                if signed:
                    text_color = "white" if abs(value) >= bold_threshold else "#262626"
                else:
                    text_color = "white" if value >= bold_threshold else "#262626"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    fontweight="bold" if (abs(value) if signed else value) >= bold_threshold else "normal",
                    color=text_color,
                    zorder=4,
                )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _write_outputs(rho: pd.DataFrame, pearson: pd.DataFrame, pvalues: pd.DataFrame, n: int) -> None:
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rho.rename_axis("variable").to_csv(
        config.TABLE_DIR / "corr_spearman_7vars.csv", encoding="utf-8-sig"
    )
    pearson.rename_axis("variable").to_csv(
        config.TABLE_DIR / "corr_pearson_7vars.csv", encoding="utf-8-sig"
    )
    pvalues.rename_axis("variable").to_csv(
        config.TABLE_DIR / "corr_spearman_pvalues_7vars.csv", encoding="utf-8-sig"
    )
    divergence = (pearson - rho).abs()
    divergence.rename_axis("variable").to_csv(
        config.TABLE_DIR / "corr_abs_pearson_minus_spearman_7vars.csv", encoding="utf-8-sig"
    )
    summary = {
        "n": int(n),
        "variables_displayed": VARIABLES,
        "source_columns": SOURCE_COLUMNS,
        "spearman_dep_reg_iv1": float(rho.loc["Dep_reg", "IV1"]),
        "spearman_dep_reg_iv2": float(rho.loc["Dep_reg", "IV2"]),
        "spearman_dep_reg_pmc_f": float(rho.loc["Dep_reg", "PMC_F"]),
        "spearman_dep_reg_pmc_all": {
            key: float(rho.loc["Dep_reg", key]) for key in ["PMC_T", "PMC_I", "PMC_F", "PMC_BERT"]
        },
        "pmc_f_is_highest_absolute_dep_correlation": bool(
            abs(rho.loc["Dep_reg", "PMC_F"])
            == max(abs(rho.loc["Dep_reg", key]) for key in ["PMC_T", "PMC_I", "PMC_F", "PMC_BERT"])
        ),
        "significance_stars": True,
        "significance_thresholds": {"***": "p<0.001", "**": "p<0.01", "*": "p<0.10"},
        "significance_note": "Unadjusted two-sided Spearman p-values; stars occupy upper-triangle cells in both panels, while numeric coefficients occupy lower-triangle cells. Panel (b) stars refer to the underlying Spearman test, not to the Pearson-Spearman difference itself.",
        "divergence_definition": "abs(Pearson - Spearman)",
    }
    (config.LOG_DIR / "corr_heatmap_7vars_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.LOG_DIR / "figure_A1_section5_4_insertion.md").write_text(
        "# Figure A1 insertion note\n\n"
        "Figure A1 reports Spearman rank correlations and the absolute Pearson--Spearman divergence for seven retained analysis variables (N = 41). "
        "Stars in both panels refer to the underlying unadjusted two-sided Spearman tests; panel (b) is a descriptive divergence display, not a significance test of the difference.\n",
        encoding="utf-8",
    )


def main() -> None:
    config.ensure_directories()
    frame = load_combined()
    required = list(SOURCE_COLUMNS.values())
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Correlation figure data are missing columns: {missing}")
    rho, pearson, pvalues = _pairwise_correlations(frame)
    _write_outputs(rho, pearson, pvalues, len(frame))

    divergence = (pearson - rho).abs()
    labels = VARIABLES
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.45),
        gridspec_kw={"wspace": 0.42},
        constrained_layout=False,
    )
    left_image = _draw_upper_triangle(
        axes[0],
        rho,
        labels,
        cmap=_diverging_cmap(),
        vmin=-1,
        vmax=1,
        colorbar_label="Spearman $\\rho$",
        bold_threshold=0.5,
        signed=True,
        pvalues=pvalues,
        show_stars=True,
        star_color=None,
    )
    right_image = _draw_upper_triangle(
        axes[1],
        divergence,
        labels,
        cmap=plt.get_cmap("viridis"),
        vmin=0,
        vmax=1,
        colorbar_label="$|$Pearson $-$ Spearman$|$",
        bold_threshold=0.5,
        signed=False,
        pvalues=pvalues,
        show_stars=True,
        star_color="white",
    )
    axes[0].text(0.5, -0.30, "(a)", transform=axes[0].transAxes, ha="center", va="top", fontsize=10, fontweight="bold")
    axes[1].text(0.5, -0.30, "(b)", transform=axes[1].transAxes, ha="center", va="top", fontsize=10, fontweight="bold")
    for ax, image, label in [
        (axes[0], left_image, "Spearman $\\rho$"),
        (axes[1], right_image, "$|$Pearson $-$ Spearman$|$"),
    ]:
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, shrink=0.82)
        cbar.set_label(label, fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5, length=2)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.31, top=0.94, wspace=0.38)
    output = config.FIGURE_DIR / "fig_corr_heatmap.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"N: {len(frame)}")
    print("Variables:", VARIABLES)
    print(f"Spearman Dep_reg-IV1: {rho.loc['Dep_reg', 'IV1']:.6f}")
    print(f"Spearman Dep_reg-IV2: {rho.loc['Dep_reg', 'IV2']:.6f}")
    print(f"Spearman Dep_reg-PMC_F: {rho.loc['Dep_reg', 'PMC_F']:.6f}")
    for key in ["PMC_T", "PMC_I", "PMC_F", "PMC_BERT"]:
        print(f"Spearman Dep_reg-{key}: {rho.loc['Dep_reg', key]:.6f}")


if __name__ == "__main__":
    main()
