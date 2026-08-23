"""针对新 RQ/P1/H1/H2 的消融：删去上下文变量、单项 PMC 和交互块。"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import config
from src.data_io import load_combined
from src.evaluate import repeated_cv
from src.features import build_custom
from src.analysis_utils import get_manifest, paired_delta
from src.plot_style import colors, save


REGISTRY = [
    {"model": "M0_full", "reference": "M0_full", "features": config.M0_FEATURES, "what_it_tests": "baseline anchor", "expected_if_component_matters": "reference only"},
    {"model": "M0_no_IV1", "reference": "M0_full", "features": ["IV2", *config.CONTROLS], "what_it_tests": "housing-price context contribution", "expected_if_component_matters": "dropping IV1 changes paired OOF R²"},
    {"model": "M0_no_IV2", "reference": "M0_full", "features": ["IV1", *config.CONTROLS], "what_it_tests": "absorption-period context contribution", "expected_if_component_matters": "dropping IV2 changes paired OOF R²"},
    {"model": "M0_controls_only", "reference": "M0_full", "features": config.CONTROLS, "what_it_tests": "market-context block versus fixed controls", "expected_if_component_matters": "context-only difference is descriptive"},
    {"model": "B1_full", "reference": "B1_full", "features": config.MODEL_FEATURES["B1"], "what_it_tests": "full PMC block anchor", "expected_if_component_matters": "reference only"},
]
for pmc in config.MODERATORS:
    REGISTRY.append({"model": f"B1_no_{pmc}", "reference": "B1_full", "features": [f for f in config.MODEL_FEATURES["B1"] if f != pmc], "what_it_tests": f"single PMC removal: {pmc}", "expected_if_component_matters": f"removing {pmc} changes the block-level OOF distribution"})
REGISTRY.extend([
    {"model": "B2_full", "reference": "B2_full", "features": config.MODEL_FEATURES["B2"], "what_it_tests": "full predeclared interaction block anchor", "expected_if_component_matters": "reference only"},
    {"model": "B2_no_interactions", "reference": "B2_full", "features": config.MODEL_FEATURES["B1"], "what_it_tests": "all IV×PMC interaction block", "expected_if_component_matters": "B2 should differ from B1 only if interaction block adds predictive information"},
    {"model": "B2_no_IV1_interactions", "reference": "B2_full", "features": [f for f in config.MODEL_FEATURES["B2"] if not f.startswith("IV1_x_")], "what_it_tests": "housing-price interaction subset", "expected_if_component_matters": "deleting IV1 interactions changes B2 OOF output"},
    {"model": "B2_no_IV2_interactions", "reference": "B2_full", "features": [f for f in config.MODEL_FEATURES["B2"] if not f.startswith("IV2_x_")], "what_it_tests": "absorption-period interaction subset", "expected_if_component_matters": "deleting IV2 interactions changes B2 OOF output"},
])


def run():
    config.ensure_directories(); df = load_combined(); y = pd.to_numeric(df[config.TARGET], errors="raise"); manifest = get_manifest(df); pred_map = {}; rows = []; fold_rows = []
    for item in REGISTRY:
        name, features = item["model"], item["features"]
        X, _ = build_custom(df, features)
        summary, folds, pred, _ = repeated_cv(X, y, manifest=manifest, city_keys=df.city_key.tolist(), model_id=name, return_predictions=True)
        pred_map[name] = pred; fold_rows.append(folds)
        rows.append({**item, "features": ";".join(features), "n_features": len(features), "cv_protocol": f"RepeatedKFold({config.CV_SPLITS}x{config.N_REPEATS})", "seed": config.SEED, "interpretation_scope": "targeted predictive ablation; no causal interpretation", **summary})
    result = pd.DataFrame(rows); deltas = []
    for item in REGISTRY:
        if item["model"] == item["reference"]:
            continue
        deltas.append(paired_delta(pred_map[item["model"]], pred_map[item["reference"]], f"{item['model']} - {item['reference']}"))
    delta_frame = pd.DataFrame(deltas)
    result.to_csv(config.TABLE_DIR / "targeted_ablation_results.csv", index=False, encoding="utf-8-sig"); result[["model", "reference", "what_it_tests", "expected_if_component_matters", "features"]].to_csv(config.TABLE_DIR / "targeted_ablation_registry.csv", index=False, encoding="utf-8-sig"); pd.concat(fold_rows, ignore_index=True).to_csv(config.TABLE_DIR / "targeted_ablation_folds.csv", index=False, encoding="utf-8-sig"); delta_frame.to_csv(config.TABLE_DIR / "targeted_ablation_paired_delta_r2.csv", index=False, encoding="utf-8-sig")
    plot = result[result.model != result.reference].copy(); fig, ax = plt.subplots(figsize=(8.6, 4.5)); ax.bar(np.arange(len(plot)), plot.R2_mean, color=colors(len(plot))); ax.set_xticks(np.arange(len(plot)), plot.model, rotation=55, ha="right", fontsize=7); ax.set_ylabel("Repeat-level OOF R²"); ax.set_title("Targeted ablations for RQ/P1/H1/H2 design"); save(fig, config.FIGURE_DIR / "fig_targeted_ablations.png")
    (config.LOG_DIR / "ablation_scope.md").write_text("消融只回答被删除组件是否改变预测诊断；不把性能差异解释为政策因果效应。IV3/IV4 没有进入任何消融特征集，因为它们分别是近常量描述量和结果侧机械重叠量。\n", encoding="utf-8")
    return result, delta_frame


if __name__ == "__main__": print(run()[0].to_string(index=False))
