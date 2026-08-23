"""H2 模块 A：M0 与四种文本表征的并行、非嵌套 RF 比较。"""
from __future__ import annotations

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
from src.evaluate import repeated_cv
from src.features import build_spec
from src.analysis_utils import get_manifest, paired_delta, kendall_bootstrap
from src.plot_style import colors, save

MODEL_LABELS = {"M0": "M0: context + controls", "M1": "M1: M0 + PMC-T", "M2": "M2: M0 + PMC-I", "M3": "M3: M0 + PMC-Frequency", "M4": "M4: M0 + PMC-BERT"}


def run():
    config.ensure_directories()
    df = load_combined()
    y = pd.to_numeric(df[config.TARGET], errors="raise")
    manifest = get_manifest(df)
    all_results, all_folds, all_predictions, full_rank_rows = [], [], [], []
    prediction_map = {}
    for name, label in MODEL_LABELS.items():
        X, features = build_spec(df, name)
        summary, folds, predictions, repeat_predictions = repeated_cv(X, y, manifest=manifest, city_keys=df.city_key.tolist(), model_id=name, return_predictions=True)
        folds["feature_set"] = ";".join(features)
        all_folds.append(folds)
        predictions["feature_set"] = ";".join(features)
        all_predictions.append(predictions)
        prediction_map[name] = predictions
        from src.evaluate import fit_full
        full_model = fit_full(X, y)
        full_pred = full_model.predict(X)
        rank = pd.DataFrame({"city_key": df.city_key, "representation": name, "predicted_Dep_reg": full_pred})
        rank["pred_rank"] = rank.predicted_Dep_reg.rank(ascending=False, method="min")
        full_rank_rows.append(rank)
        all_results.append({"model": name, "representation": label, "features": ";".join(features), "n_features": len(features), "fixed_controls": ";".join(config.CONTROLS), "target": config.TARGET, "cv_protocol": f"RepeatedKFold({config.CV_SPLITS}x{config.N_REPEATS})", "training_test_split": "shared split_manifest; repeated OOF", "seed": config.SEED, "tuned": "M0-only Optuna; frozen across all specifications", "interpretation_scope": "predictive comparison only; no causal interpretation", **summary})
    results = pd.DataFrame(all_results)
    folds = pd.concat(all_folds, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    ranks = pd.concat(full_rank_rows, ignore_index=True)
    deltas = []
    for name in list(MODEL_LABELS)[1:]:
        deltas.append(paired_delta(prediction_map[name], prediction_map["M0"], f"{name} - M0"))
    delta_frame = pd.DataFrame(deltas)
    rank_wide = ranks.pivot(index="city_key", columns="representation", values="pred_rank")
    pair_rows = []
    for i, a in enumerate(rank_wide.columns):
        for b in rank_wide.columns[i + 1:]:
            boot = kendall_bootstrap(rank_wide[a].to_numpy(), rank_wide[b].to_numpy())
            pair_rows.append({"representation_a": a, "representation_b": b, **boot})
    pair = pd.DataFrame(pair_rows)
    displacement = []
    for name in rank_wide.columns:
        d = (rank_wide[name] - rank_wide["M0"]).abs()
        displacement.append({"representation": name, "mean_abs_rank_displacement_vs_M0": d.mean(), "median_abs_rank_displacement_vs_M0": d.median(), "max_abs_rank_displacement_vs_M0": d.max()})
    pd.DataFrame(displacement).to_csv(config.TABLE_DIR / "rank_displacement.csv", index=False, encoding="utf-8-sig")
    results.to_csv(config.TABLE_DIR / "canonical_results.csv", index=False, encoding="utf-8-sig")
    results.to_csv(config.TABLE_DIR / "representation_results.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(config.TABLE_DIR / "canonical_folds.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(config.TABLE_DIR / "canonical_oof_predictions.csv", index=False, encoding="utf-8-sig")
    ranks.to_csv(config.TABLE_DIR / "city_representation_ranks.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(config.TABLE_DIR / "pairwise_rank_agreement.csv", index=False, encoding="utf-8-sig")
    delta_frame.to_csv(config.TABLE_DIR / "paired_delta_r2.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"model": name, "feature_order": i + 1, "feature": feature, "label": config.LABELS.get(feature, feature), "is_control": feature in config.CONTROLS, "is_interaction": "_x_" in feature} for name in MODEL_LABELS for i, feature in enumerate(build_spec(df, name)[1])]).to_csv(config.TABLE_DIR / "model_feature_sets.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(results.model, results.R2_mean, color=colors(len(results)))
    ax.errorbar(np.arange(len(results)), results.R2_mean, yerr=results.R2_sd, fmt="none", ecolor="#222222", capsize=3)
    ax.axhline(0, color="#555", lw=.7, ls="--")
    ax.set_ylabel("Repeat-level OOF R²")
    ax.set_title("H2: parallel RF comparison of text representations")
    save(fig, config.FIGURE_DIR / "fig_cv_distribution.png")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar([f"{x} vs M0" for x in delta_frame.comparison], delta_frame.median_delta_R2, color=colors(len(delta_frame)))
    ax.axhline(0, color="#555", lw=.7)
    ax.set_ylabel("Median paired Δ OOF R²")
    ax.tick_params(axis="x", rotation=25)
    save(fig, config.FIGURE_DIR / "fig_h2_delta_r2.png")
    (config.LOG_DIR / "h2_scope.md").write_text("Module A uses parallel, non-nested M0 plus one PMC representation at a time. Rank changes and OOF diagnostics are empirical predictive comparisons, not causal policy effects.\n", encoding="utf-8")
    return results, folds


if __name__ == "__main__":
    print(run()[0].to_string(index=False))
