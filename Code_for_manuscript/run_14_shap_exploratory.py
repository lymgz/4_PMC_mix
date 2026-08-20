"""模块 D：探索性 SHAP 可视化与特征重要性排序。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import config
from src.tuning import rf_param_hash
from src.data_io import load_combined
from src.features import build_spec
from src.evaluate import fit_full
from src.plot_style import colors, save


def run():
    config.ensure_directories(); df = load_combined(); y = pd.to_numeric(df[config.TARGET], errors="raise")
    try:
        import shap
    except Exception as exc:
        payload = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}", "exploratory_only": True}
        (config.LOG_DIR / "shap_provenance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    all_rows, tier_rows, statuses = [], [], []
    matrices = {}
    for model_name in ("B1", "B2"):
        X, features = build_spec(df, model_name); model = fit_full(X, y)
        try:
            values = np.asarray(shap.TreeExplainer(model)(X).values, dtype=float)
            if values.ndim == 3: values = values[:, :, 0]
        except Exception as exc:
            statuses.append({"model": model_name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}); continue
        matrices[model_name] = values
        mean_abs = np.abs(values).mean(axis=0); signed = values.mean(axis=0); importance = model.feature_importances_
        table = pd.DataFrame({"model": model_name, "feature": features, "label": [config.LABELS.get(f, f.replace("_x_", " × ")) for f in features], "mean_abs_SHAP": mean_abs, "signed_mean_SHAP": signed, "RF_impurity_importance": importance, "rank_by_mean_abs_SHAP": pd.Series(mean_abs).rank(method="min", ascending=False).astype(int), "fit_scope": "full sample", "exploratory_only": True, "interpretation_scope": "model attribution visualization; not causal; no hypothesis test"}).sort_values(["rank_by_mean_abs_SHAP", "feature"])
        all_rows.extend(table.to_dict("records")); table.to_csv(config.TABLE_DIR / f"shap_{model_name}_global.csv", index=False, encoding="utf-8-sig")
        for tier, group in df.assign(_tier=df.Region)[["_tier"]].groupby("_tier"):
            idx = group.index.to_numpy(); positions = idx
            for feature_pos, feature in enumerate(features):
                subset = values[positions, feature_pos]
                tier_rows.append({"model": model_name, "city_tier": tier, "feature": feature, "mean_abs_SHAP": np.abs(subset).mean(), "signed_mean_SHAP": subset.mean(), "n_cities": len(subset), "exploratory_only": True})
        display = table.head(min(15, len(table))).sort_values("mean_abs_SHAP")
        fig, ax = plt.subplots(figsize=(7.0, 4.6)); ax.barh(display.label, display.mean_abs_SHAP, color=colors(1)[0]); ax.set_xlabel("Mean |SHAP|"); fig.tight_layout(); save(fig, config.FIGURE_DIR / f"fig_shap_global_{model_name}.png")
        shap.summary_plot(values, X, feature_names=[config.LABELS.get(f, f) for f in features], show=False, max_display=min(15, len(features)), cmap="viridis"); plt.tight_layout(); plt.savefig(config.FIGURE_DIR / f"fig_shap_summary_{model_name}.png", dpi=300, bbox_inches="tight"); plt.close()
        statuses.append({"model": model_name, "status": "ok", "n_features": len(features)})
    global_table = pd.DataFrame(all_rows); tier_table = pd.DataFrame(tier_rows); global_table.to_csv(config.TABLE_DIR / "shap_exploratory_global.csv", index=False, encoding="utf-8-sig"); tier_table.to_csv(config.TABLE_DIR / "shap_exploratory_city_tier.csv", index=False, encoding="utf-8-sig")
    if not global_table.empty:
        best = global_table[global_table.model == "B2"].sort_values("mean_abs_SHAP").tail(10); fig, ax = plt.subplots(figsize=(7.0, 4.4)); ax.barh(best.label, best.mean_abs_SHAP, color=colors(1)[0]); ax.set_xlabel("Mean |SHAP|"); fig.tight_layout(); save(fig, config.FIGURE_DIR / "fig_shap_exploratory_bar.png")
    payload = {"status": "ok" if not global_table.empty else "error", "models": statuses, "exploratory_only": True, "n_cities": len(df), "seed": config.SEED, "n_estimators": config.RF_COMMON["n_estimators"], "rf_param_hash": rf_param_hash(config.RF_COMMON), "tuning_scope": "M0_only_frozen_all_models", "causal_interpretation": False, "shap_interaction_values_used": False, "outputs": ["shap_exploratory_global.csv", "shap_exploratory_city_tier.csv"]}
    (config.LOG_DIR / "shap_provenance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.LOG_DIR / "shap_audit.md").write_text("SHAP 在 10.3 中仅用于探索性模型归因可视化和全局重要性排序。signed SHAP 与 mean |SHAP| 分开保存；没有 SHAP interaction values、p 值或因果解释。\n", encoding="utf-8")
    return payload


if __name__ == "__main__": print(run())
