"""模块 B：预先指定的增量块比较 B0 → B1 → B2。"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import config
from src.data_io import load_combined
from src.evaluate import repeated_cv
from src.features import build_spec
from src.analysis_utils import get_manifest, paired_delta
from src.plot_style import colors, save


def _summary_from_predictions(pred):
    rows = []
    for repeat, group in pred.groupby("repeat"):
        rows.append({"repeat": repeat, "R2": r2_score(group.y_true, group.y_pred)})
    r = pd.DataFrame(rows).R2
    hashes = pred.rf_param_hash.dropna().unique() if "rf_param_hash" in pred else []
    return {"rf_param_hash": hashes[0] if len(hashes) == 1 else None, "R2_mean": float(r.mean()), "R2_sd": float(r.std(ddof=1)), "R2_min": float(r.min()), "R2_max": float(r.max()), "n_repeats": len(r), "R2_fold_mean": np.nan, "R2_fold_sd": np.nan, "RMSE_mean": float(np.sqrt(mean_squared_error(pred.y_true, pred.y_pred)))}


def run():
    config.ensure_directories(); df = load_combined(); y = pd.to_numeric(df[config.TARGET], errors="raise"); manifest = get_manifest(df)
    names = ["B0", "B1", "B2"]; predictions = {}; folds = []; results = []
    canonical_pred_path = config.TABLE_DIR / "canonical_oof_predictions.csv"
    from src.tuning import rf_param_hash
    runtime_hash = rf_param_hash(config.RF_COMMON)
    if canonical_pred_path.exists():
        canonical = pd.read_csv(canonical_pred_path)
        hash_ok = "rf_param_hash" in canonical and set(canonical.rf_param_hash.dropna()) == {runtime_hash}
        b0 = canonical[canonical.model == "M0"].copy() if hash_ok else pd.DataFrame()
    else:
        b0 = pd.DataFrame()
    if not b0.empty:
        b0["model"] = "B0"; predictions["B0"] = b0; results.append({"model": "B0", "feature_set": ";".join(config.MODEL_FEATURES["B0"]), "n_features": len(config.MODEL_FEATURES["B0"]), "added_block": "none; context covariates + fixed controls", "cv_protocol": f"RepeatedKFold({config.CV_SPLITS}x{config.N_REPEATS})", "seed": config.SEED, "interpretation_scope": "predictive comparison; no causal interpretation", **_summary_from_predictions(b0)})
    else:
        X, features = build_spec(df, "B0"); summary, fold, pred, _ = repeated_cv(X, y, manifest=manifest, city_keys=df.city_key.tolist(), model_id="B0", return_predictions=True); predictions["B0"] = pred; folds.append(fold); results.append({"model": "B0", "feature_set": ";".join(features), "n_features": len(features), "added_block": "none", "cv_protocol": f"RepeatedKFold({config.CV_SPLITS}x{config.N_REPEATS})", "seed": config.SEED, "interpretation_scope": "predictive comparison; no causal interpretation", **summary})
    for name, added in [("B1", "four PMC block"), ("B2", "predeclared 8 IV×PMC products; 16 total features")]:
        X, features = build_spec(df, name); summary, fold, pred, _ = repeated_cv(X, y, manifest=manifest, city_keys=df.city_key.tolist(), model_id=name, return_predictions=True); predictions[name] = pred; folds.append(fold); results.append({"model": name, "feature_set": ";".join(features), "n_features": len(features), "added_block": added, "cv_protocol": f"RepeatedKFold({config.CV_SPLITS}x{config.N_REPEATS})", "seed": config.SEED, "interaction_scope": "constructed IV×PMC products; not SHAP interaction values", "interpretation_scope": "predictive comparison; no causal interpretation", **summary})
    result_frame = pd.DataFrame(results).drop_duplicates("model"); fold_frame = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame(); pred_frame = pd.concat([p for p in predictions.values()], ignore_index=True)
    delta_rows = [paired_delta(predictions["B1"], predictions["B0"], "B1 - B0"), paired_delta(predictions["B2"], predictions["B1"], "B2 - B1"), paired_delta(predictions["B2"], predictions["B0"], "B2 - B0")]
    deltas = pd.DataFrame(delta_rows)
    result_frame.to_csv(config.TABLE_DIR / "block_results.csv", index=False, encoding="utf-8-sig"); fold_frame.to_csv(config.TABLE_DIR / "block_folds.csv", index=False, encoding="utf-8-sig"); pred_frame.to_csv(config.TABLE_DIR / "block_oof_predictions.csv", index=False, encoding="utf-8-sig"); deltas.to_csv(config.TABLE_DIR / "block_paired_delta_r2.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(5.8, 3.5)); ax.bar(result_frame.model, result_frame.R2_mean, color=colors(len(result_frame))); ax.set_ylabel("Repeat-level OOF R²"); ax.set_title("Incremental specification: B0, B1, B2"); save(fig, config.FIGURE_DIR / "fig_block_oof_r2.png")
    fig, ax = plt.subplots(figsize=(5.8, 3.5)); ax.bar(deltas.comparison, deltas.median_delta_R2, color=colors(len(deltas))); ax.axhline(0, color="#555", lw=.7); ax.set_ylabel("Median paired Δ OOF R²"); ax.tick_params(axis="x", rotation=20); save(fig, config.FIGURE_DIR / "fig_block_delta_r2.png")
    (config.LOG_DIR / "block_scope.md").write_text("B0→B1→B2 is an incremental specification sequence. The IV×PMC products are constructed feature interactions; they do not establish causal moderation or a mechanism. The key complexity criterion is the paired B2−B1 distribution.\n", encoding="utf-8")
    return result_frame, fold_frame


if __name__ == "__main__": print(run()[0].to_string(index=False))
