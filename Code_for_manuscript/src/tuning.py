"""M0-only Optuna tuning and one-parameter-set freezing contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import config
import pandas as pd


TUNABLE_KEYS = {
    "criterion",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "ccp_alpha",
    "max_samples",
}


def rf_param_hash(params: dict[str, Any]) -> str:
    encoded = json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_frozen_payload(best_params: dict[str, Any], tuning_metadata: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(best_params) - TUNABLE_KEYS)
    if unknown:
        raise ValueError(f"Optuna返回未授权参数: {unknown}")
    frozen = dict(config.RF_DEFAULT)
    frozen.update(best_params)
    frozen["n_estimators"] = config.N_ESTIMATORS
    frozen["random_state"] = config.SEED
    frozen["n_jobs"] = config.N_JOBS
    scope = tuning_metadata.get("scope")
    return {
        "schema_version": 1,
        "tuning_scope": scope,
        "study_name": tuning_metadata.get("study_name", "m0_rf_optuna"),
        "frozen_params": frozen,
        "rf_param_hash": rf_param_hash(frozen),
        "tuning_metadata": dict(tuning_metadata),
    }


def load_frozen_payload(path: Path | str = config.FROZEN_RF_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_frozen_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("tuning_scope") != "M0_only":
        raise ValueError("冻结参数必须来自 M0_only 调参")
    frozen = payload.get("frozen_params")
    if not isinstance(frozen, dict):
        raise ValueError("冻结参数文件缺少 frozen_params")
    expected = rf_param_hash(frozen)
    if payload.get("rf_param_hash") != expected:
        raise ValueError("冻结参数哈希校验失败")
    if frozen.get("n_estimators") != config.N_ESTIMATORS:
        raise ValueError("正式 n_estimators 与冻结参数不一致")
    if frozen.get("random_state") != config.SEED:
        raise ValueError("正式 random_state 与冻结参数不一致")
    if frozen.get("n_jobs") != config.N_JOBS:
        raise ValueError("正式 n_jobs 与冻结参数不一致")
    config.RF_COMMON.clear()
    config.RF_COMMON.update(frozen)
    return dict(config.RF_COMMON)


def tune_m0(
    X: pd.DataFrame,
    y,
    *,
    city_keys: list[str],
    n_trials: int,
    cv_splits: int,
    cv_repeats: int,
    tuning_n_estimators: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """仅用M0输入搜索RF正则化参数，随后恢复正式树数并冻结。"""
    import optuna
    from src.evaluate import make_split_manifest, repeated_cv

    if list(X.columns) != list(config.M0_FEATURES):
        raise ValueError(f"Optuna只能接收M0特征，收到: {list(X.columns)}")
    if n_trials < 1:
        raise ValueError("n_trials 必须大于0")
    manifest = make_split_manifest(
        len(y),
        city_keys=city_keys,
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        seed=seed,
    )
    baseline_params = dict(config.RF_DEFAULT)
    baseline_params["n_estimators"] = tuning_n_estimators
    baseline_summary, _ = repeated_cv(
        X,
        y,
        params=baseline_params,
        manifest=manifest,
        city_keys=city_keys,
        model_id="M0_tuning_default",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="m0_rf_optuna")

    def objective(trial) -> float:
        params = dict(config.RF_DEFAULT)
        params.update(
            {
                "n_estimators": tuning_n_estimators,
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse"]),
                "max_depth": trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5, 6, 8, 10]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", [0.4, 0.5, 0.65, 0.8, 1.0, "sqrt"]),
                "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.05),
                "max_samples": trial.suggest_categorical("max_samples", [None, 0.7, 0.85, 1.0]),
                "random_state": config.SEED,
                "n_jobs": config.N_JOBS,
            }
        )
        summary, _ = repeated_cv(X, y, params=params, manifest=manifest, city_keys=city_keys, model_id="M0_tuning")
        return float(summary["R2_mean"])

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    metadata = {
        "scope": "M0_only",
        "study_name": study.study_name,
        "feature_set": list(config.M0_FEATURES),
        "objective": "mean repeat-level OOF R2",
        "direction": "maximize",
        "n_trials": int(n_trials),
        "best_value_tuning_cv": float(study.best_value),
        "baseline_value_tuning_cv": float(baseline_summary["R2_mean"]),
        "delta_vs_default_tuning_cv": float(study.best_value - baseline_summary["R2_mean"]),
        "tuning_cv_protocol": f"RepeatedKFold({cv_splits}x{cv_repeats})",
        "tuning_n_estimators": int(tuning_n_estimators),
        "formal_n_estimators": int(config.N_ESTIMATORS),
        "seed": int(seed),
        "optimism_boundary": "M0 tuning and formal OOF evaluation use the same 41-city development sample; formal OOF is not external validation",
    }
    payload = build_frozen_payload(dict(study.best_params), metadata)
    apply_frozen_payload(payload)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    return payload, trials


def save_tuning_outputs(
    payload: dict[str, Any],
    trials: pd.DataFrame,
    *,
    log_dir: Path,
    table_dir: Path,
) -> dict[str, Path]:
    log_dir = Path(log_dir)
    table_dir = Path(table_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = log_dir / "m0_frozen_rf_params.json"
    trials_path = table_dir / "m0_optuna_trials.csv"
    summary_path = table_dir / "m0_optuna_summary.csv"
    frozen_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    metadata = payload.get("tuning_metadata", {})
    pd.DataFrame(
        [
            {
                "tuning_scope": payload.get("tuning_scope"),
                "study_name": payload.get("study_name"),
                "rf_param_hash": payload.get("rf_param_hash"),
                "best_value_tuning_cv": metadata.get("best_value_tuning_cv"),
                "baseline_value_tuning_cv": metadata.get("baseline_value_tuning_cv"),
                "delta_vs_default_tuning_cv": metadata.get("delta_vs_default_tuning_cv"),
                "n_trials": metadata.get("n_trials"),
                "tuning_cv_protocol": metadata.get("tuning_cv_protocol"),
                "tuning_n_estimators": metadata.get("tuning_n_estimators"),
                "formal_n_estimators": metadata.get("formal_n_estimators"),
                "feature_set": ";".join(metadata.get("feature_set", [])),
                "optimism_boundary": metadata.get("optimism_boundary"),
            }
        ]
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {"frozen_params": frozen_path, "trials": trials_path, "summary": summary_path}
