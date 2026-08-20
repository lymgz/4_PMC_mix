"""Code 10.4 正式结果与 Section 5 资产总入口。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import config
from src.data_io import load_combined
from src.tuning import rf_param_hash
from run_00_m0_optuna import run as run_m0_tuning
from run_01_descriptives import run as run_descriptives
from run_02_canonical import run as run_canonical
from run_03_overlap import run as run_p1
from run_03_corr_heatmap_1x2 import main as run_corr_heatmap
from run_04_rate_sensitivity import run as run_rate
from run_06_reconciliation import run as run_reconciliation
from run_07_export_latex import run as run_export
from run_08_reviewer_tables import run as run_reviewer_tables
from run_09_claim_scan import run as run_claim_scan
from run_10_requirements_audit import run as run_requirements_audit
from run_11_pca_h1 import run as run_h1
from run_12_incremental_blocks import run as run_blocks
from run_13_shapley_r2 import run as run_shapley
from run_14_shap_exploratory import run as run_shap
from run_15_targeted_ablations import run as run_ablations
from run_16_section5_assets import run as run_section5_assets


def main():
    parser = argparse.ArgumentParser(description="Code 10.4 formal evidence and Section 5 asset pipeline")
    parser.add_argument("--force-rebuild", action="store_true", help="重新从本地源工作簿重建41城数据")
    parser.add_argument("--skip-shap", action="store_true", help="仅调试时跳过探索性SHAP；正式结果不应使用")
    parser.add_argument("--skip-shapley", action="store_true", help="仅调试时跳过16子集Shapley")
    parser.add_argument("--skip-ablations", action="store_true", help="仅调试时跳过针对性消融")
    parser.add_argument("--skip-notebook-check", action="store_true", help="10.3不依赖旧notebook")
    parser.add_argument("--retune-m0", action="store_true", help="正式重跑M0-only Optuna并覆盖既有冻结参数")
    parser.add_argument("--optuna-trials", type=int, default=None, help="覆盖M0 Optuna trials；正式默认40")
    args = parser.parse_args(); started = time.time(); config.ensure_directories()
    df = load_combined(force_rebuild=args.force_rebuild)
    tuning = run_m0_tuning(df, retune=args.retune_m0, n_trials=args.optuna_trials)
    summary = {"version": config.VERSION, "seed": config.SEED, "n_cities": len(df), "target": config.TARGET, "n_estimators": config.RF_COMMON["n_estimators"], "rf_tuning": tuning, "rf_params": dict(config.RF_COMMON), "rf_param_hash": rf_param_hash(config.RF_COMMON), "cv_protocol": f"RepeatedKFold({config.CV_SPLITS}x{config.N_REPEATS})", "data_mode": "raw_rebuild" if args.force_rebuild else "cached_local_output", "stages": {}}
    summary["stages"]["descriptives"] = run_descriptives()
    canonical, canonical_folds = run_canonical(); summary["stages"]["canonical_models"] = len(canonical); summary["stages"]["canonical_folds"] = len(canonical_folds)
    h1_rules, h1_components = run_h1(); summary["stages"]["h1_components"] = len(h1_components)
    p1_pairs, p1_summary = run_p1(); summary["stages"]["p1_pairs"] = len(p1_pairs)
    blocks, block_folds = run_blocks(); summary["stages"]["block_models"] = len(blocks); summary["stages"]["block_folds"] = len(block_folds)
    if args.skip_shapley: summary["stages"]["shapley"] = {"status": "skipped"}
    else:
        shapley, shapley_subsets = run_shapley(); summary["stages"]["shapley_subsets"] = len(shapley_subsets)
    if args.skip_ablations: summary["stages"]["ablations"] = {"status": "skipped"}
    else:
        ablation, ablation_delta = run_ablations(); summary["stages"]["ablation_models"] = len(ablation)
    summary["stages"]["rate_sensitivity"] = len(run_rate()[0])
    try:
        run_corr_heatmap(); summary["stages"]["corr_heatmap"] = "ok"
    except Exception as exc:
        summary["stages"]["corr_heatmap"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    summary["stages"]["shap"] = {"status": "skipped"} if args.skip_shap else run_shap()
    summary["stages"]["reconciliation_rows"] = len(run_reconciliation()); summary["stages"]["latex_files"] = len(run_export()); summary["stages"]["reviewer_tables"] = run_reviewer_tables(); summary["stages"]["claim_scan"] = run_claim_scan()
    summary["stages"]["section5_assets"] = run_section5_assets(); summary["stages"]["requirements_audit"] = run_requirements_audit(); summary["elapsed_seconds"] = time.time() - started
    (config.LOG_DIR / "run_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return summary


if __name__ == "__main__": main()
