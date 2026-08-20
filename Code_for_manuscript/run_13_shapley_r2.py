"""模块 C：四种 PMC 对预测 R² 的顺序无关 Shapley 分解。"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import config
from src.plot_style import colors, save
from src.data_io import load_combined
from src.evaluate import repeated_cv
from src.analysis_utils import get_manifest
from src.features import build_custom


def run():
    config.ensure_directories(); df = load_combined(); y = pd.to_numeric(df[config.TARGET], errors="raise"); manifest = get_manifest(df)
    values, rows = {}, []
    for size in range(len(config.MODERATORS) + 1):
        for subset in itertools.combinations(config.MODERATORS, size):
            key = "|".join(subset) if subset else "EMPTY"
            features = [*config.M0_FEATURES, *subset]
            X, _ = build_custom(df, features)
            summary, folds, predictions, _ = repeated_cv(X, y, manifest=manifest, city_keys=df.city_key.tolist(), model_id=f"C_{key}", return_predictions=True)
            values[frozenset(subset)] = float(summary["R2_mean"])
            rows.append({"subset": key, "subset_size": len(subset), "features": ";".join(features), "n_features": len(features), "rf_param_hash": summary["rf_param_hash"], "R2_mean": summary["R2_mean"], "R2_sd": summary["R2_sd"], "R2_min": summary["R2_min"], "R2_max": summary["R2_max"], "cv_protocol": f"RepeatedKFold({config.CV_SPLITS}x{config.N_REPEATS})", "seed": config.SEED, "interpretation_scope": "order-independent predictive decomposition; not causal"})
    subset_frame = pd.DataFrame(rows)
    contributions = []
    n = len(config.MODERATORS)
    for feature in config.MODERATORS:
        total = 0.0
        terms = 0
        for size in range(n):
            for subset in itertools.combinations([x for x in config.MODERATORS if x != feature], size):
                s = frozenset(subset); marginal = values[s | {feature}] - values[s]; weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n); total += weight * marginal; terms += 1
        contributions.append({"PMC": feature, "shapley_delta_R2": total, "n_marginal_terms": terms, "game_baseline_R2": values[frozenset()], "game_full_R2": values[frozenset(config.MODERATORS)], "sum_check_target_delta_R2": values[frozenset(config.MODERATORS)] - values[frozenset()]})
    contribution_frame = pd.DataFrame(contributions); contribution_frame["sum_check_error"] = contribution_frame.shapley_delta_R2.sum() - (values[frozenset(config.MODERATORS)] - values[frozenset()])
    subset_frame.to_csv(config.TABLE_DIR / "shapley_subset_results.csv", index=False, encoding="utf-8-sig"); contribution_frame.to_csv(config.TABLE_DIR / "shapley_r2_values.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.3, 3.7)); bar_colors = colors(len(contribution_frame)); ax.bar(contribution_frame.PMC, contribution_frame.shapley_delta_R2, color=bar_colors); ax.axhline(0, color="#555", lw=.7); ax.set_ylabel("Shapley contribution to Δ OOF R²"); fig.tight_layout(); save(fig, config.FIGURE_DIR / "fig_shapley_r2.png")
    (config.LOG_DIR / "shapley_scope.md").write_text("Shapley values decompose the predictive R² difference among correlated PMC representations. They are attribution quantities, not causal effects or hypothesis-test p-values. The contributions must sum to R²(full PMC block)−R²(M0).\n", encoding="utf-8")
    return contribution_frame, subset_frame


if __name__ == "__main__": print(run()[0].to_string(index=False))
