"""只在M0上执行Optuna，并将同一RF参数集冻结到全部后续模型。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.features import build_spec
from src.tuning import apply_frozen_payload, load_frozen_payload, save_tuning_outputs, tune_m0


def run(df: pd.DataFrame, *, retune: bool = False, n_trials: int | None = None) -> dict:
    config.ensure_directories()
    n_trials = config.OPTUNA_TRIALS if n_trials is None else int(n_trials)
    if config.FROZEN_RF_PATH.exists() and not retune:
        payload = load_frozen_payload(config.FROZEN_RF_PATH)
        apply_frozen_payload(payload)
        return {
            "status": "reused",
            "tuning_scope": payload["tuning_scope"],
            "rf_param_hash": payload["rf_param_hash"],
            "n_trials": payload.get("tuning_metadata", {}).get("n_trials"),
            "frozen_params_path": config.display_path(config.FROZEN_RF_PATH),
        }

    X, features = build_spec(df, "M0")
    if features != config.M0_FEATURES:
        raise AssertionError(f"M0调参特征漂移: {features}")
    y = pd.to_numeric(df[config.TARGET], errors="raise")
    payload, trials = tune_m0(
        X,
        y,
        city_keys=df.city_key.tolist(),
        n_trials=n_trials,
        cv_splits=config.TUNING_CV_SPLITS,
        cv_repeats=config.TUNING_CV_REPEATS,
        tuning_n_estimators=config.TUNING_N_ESTIMATORS,
        seed=config.TUNING_SEED,
    )
    paths = save_tuning_outputs(payload, trials, log_dir=config.LOG_DIR, table_dir=config.TABLE_DIR)
    return {
        "status": "retuned" if retune else "created",
        "tuning_scope": payload["tuning_scope"],
        "rf_param_hash": payload["rf_param_hash"],
        "n_trials": n_trials,
        "best_value_tuning_cv": payload["tuning_metadata"]["best_value_tuning_cv"],
        "baseline_value_tuning_cv": payload["tuning_metadata"]["baseline_value_tuning_cv"],
        "delta_vs_default_tuning_cv": payload["tuning_metadata"]["delta_vs_default_tuning_cv"],
        "frozen_params_path": config.display_path(paths["frozen_params"]),
        "trials_path": config.display_path(paths["trials"]),
    }


if __name__ == "__main__":
    from src.data_io import load_combined

    print(run(load_combined(), retune=True))
