"""共享 split manifest 的 RandomForestRegressor 重复 OOF 评估。"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
import config
from src.tuning import rf_param_hash


def make_split_manifest(n: int, *, city_keys=None, n_splits=None, n_repeats=None, seed=None) -> pd.DataFrame:
    n_splits = config.CV_SPLITS if n_splits is None else n_splits
    n_repeats = config.N_REPEATS if n_repeats is None else n_repeats
    seed = config.SEED if seed is None else seed
    if city_keys is None:
        city_keys = [str(i) for i in range(n)]
    city_keys = list(city_keys)
    if len(city_keys) != n:
        raise ValueError("city_keys 长度与样本数不一致")
    rows = []
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for split_id, (train_idx, valid_idx) in enumerate(splitter.split(np.arange(n))):
        repeat = split_id // n_splits + 1
        fold = split_id % n_splits + 1
        valid_set = set(valid_idx.tolist())
        for pos in valid_idx:
            rows.append({"repeat": repeat, "fold": fold, "split_id": split_id, "city_position": int(pos), "city_key": city_keys[int(pos)], "n_train": int(len(train_idx)), "n_valid": int(len(valid_idx))})
        if valid_set & set(train_idx.tolist()):
            raise AssertionError("split manifest 训练集与验证集重叠")
    return pd.DataFrame(rows)


def save_split_manifest(manifest: pd.DataFrame) -> None:
    config.ensure_directories()
    manifest.to_csv(config.TABLE_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")
    audit = {"n_rows": int(len(manifest)), "n_splits": int(manifest.split_id.nunique()), "n_repeats": int(manifest.repeat.nunique()), "city_keys": sorted(manifest.city_key.unique()), "pairing_key": "repeat+fold+split_id+city_key"}
    (config.LOG_DIR / "split_manifest_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def _groups(manifest: pd.DataFrame):
    for (repeat, fold, split_id), group in manifest.groupby(["repeat", "fold", "split_id"], sort=True):
        yield int(repeat), int(fold), int(split_id), group


def paired_bootstrap(values, seed=None, n_boot=None) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"median": np.nan, "mean": np.nan, "q025": np.nan, "q975": np.nan, "n": 0}
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n_boot = config.BOOTSTRAP_N if n_boot is None else n_boot
    boot = np.array([np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    return {"median": float(np.median(values)), "mean": float(np.mean(values)), "q025": float(np.quantile(boot, .025)), "q975": float(np.quantile(boot, .975)), "n": int(len(values))}


def repeated_cv(X, y, params=None, manifest=None, city_keys=None, model_id=None, return_predictions=False):
    params = dict(config.RF_COMMON if params is None else params)
    params_hash = rf_param_hash(params)
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    if manifest is None:
        manifest = make_split_manifest(len(y), city_keys=city_keys)
    if len(manifest.city_key.unique()) != len(y):
        raise ValueError("manifest 城市数与 y 不一致")
    rows, pred_rows, repeat_predictions = [], [], {}
    positions = set(range(len(y)))
    for repeat, fold, split_id, group in _groups(manifest):
        valid_idx = group.city_position.astype(int).to_numpy()
        train_idx = np.array(sorted(positions - set(valid_idx)), dtype=int)
        model = RandomForestRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        rows.append({"model": model_id, "repeat": repeat, "fold": fold, "split_id": split_id, "n_train": len(train_idx), "n_valid": len(valid_idx), "rf_param_hash": params_hash, "R2": r2_score(y.iloc[valid_idx], pred), "RMSE": np.sqrt(mean_squared_error(y.iloc[valid_idx], pred)), "MSE": mean_squared_error(y.iloc[valid_idx], pred)})
        repeat_predictions.setdefault(repeat, np.full(len(y), np.nan))[valid_idx] = pred
        if return_predictions:
            for pos, value in zip(valid_idx, pred):
                pred_rows.append({"model": model_id, "repeat": repeat, "fold": fold, "split_id": split_id, "city_position": int(pos), "city_key": group.loc[group.city_position == pos, "city_key"].iloc[0], "rf_param_hash": params_hash, "y_true": float(y.iloc[pos]), "y_pred": float(value)})
    fold_frame = pd.DataFrame(rows)
    repeat_r2 = np.array([r2_score(y, p) for p in repeat_predictions.values()])
    summary = {"rf_param_hash": params_hash, "R2_mean": float(repeat_r2.mean()), "R2_sd": float(repeat_r2.std(ddof=1)), "R2_min": float(repeat_r2.min()), "R2_max": float(repeat_r2.max()), "R2_oof_mean": float(repeat_r2.mean()), "R2_oof_sd": float(repeat_r2.std(ddof=1)), "R2_fold_mean": float(fold_frame.R2.mean()), "R2_fold_sd": float(fold_frame.R2.std(ddof=1)), "R2_fold_min": float(fold_frame.R2.min()), "R2_fold_max": float(fold_frame.R2.max()), "RMSE_mean": float(fold_frame.RMSE.mean()), "RMSE_sd": float(fold_frame.RMSE.std(ddof=1)), "MSE_mean": float(fold_frame.MSE.mean()), "n_repeats": int(len(repeat_r2)), "n_splits": int(len(fold_frame))}
    if return_predictions:
        return summary, fold_frame, pd.DataFrame(pred_rows), repeat_predictions
    return summary, fold_frame


def fit_full(X, y, params=None):
    model = RandomForestRegressor(**dict(config.RF_COMMON if params is None else params))
    model.fit(X, y)
    return model


def repeat_level_r2(y, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        rows.append({"repeat": int(repeat), "R2": r2_score(group.y_true, group.y_pred)})
    return pd.DataFrame(rows)
