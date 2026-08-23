"""跨模块共用的小型审计与比较函数。"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import config
from .evaluate import paired_bootstrap


def get_manifest(df: pd.DataFrame) -> pd.DataFrame:
    path = config.TABLE_DIR / "split_manifest.csv"
    if path.exists():
        manifest = pd.read_csv(path)
        expected = set(df.city_key)
        expected_splits = config.CV_SPLITS * config.N_REPEATS
        expected_rows = len(df) * config.N_REPEATS
        if (
            set(manifest.city_key) == expected
            and manifest.city_key.nunique() == len(df)
            and len(manifest) == expected_rows
            and manifest.split_id.nunique() == expected_splits
            and manifest.repeat.nunique() == config.N_REPEATS
        ):
            return manifest
    from .evaluate import make_split_manifest, save_split_manifest
    manifest = make_split_manifest(len(df), city_keys=df.city_key.tolist())
    save_split_manifest(manifest)
    return manifest


def paired_delta(model_a: pd.DataFrame, model_b: pd.DataFrame, label: str) -> dict:
    a = model_a.groupby("repeat").apply(lambda g: np.nan, include_groups=False) if False else None
    keys = ["repeat", "city_key"]
    left = model_a.rename(columns={"y_pred": "pred_a"})[keys + ["y_true", "pred_a"]]
    right = model_b.rename(columns={"y_pred": "pred_b"})[keys + ["pred_b"]]
    merged = left.merge(right, on=keys, validate="one_to_one")
    values = []
    for repeat, group in merged.groupby("repeat"):
        from sklearn.metrics import r2_score
        values.append(float(r2_score(group.y_true, group.pred_a) - r2_score(group.y_true, group.pred_b)))
    summary = paired_bootstrap(values)
    return {"comparison": label, "median_delta_R2": summary["median"], "mean_delta_R2": summary["mean"], "q025": summary["q025"], "q975": summary["q975"], "n_repeats": summary["n"]}


def rank_table(df: pd.DataFrame, target: str, model_name: str) -> pd.DataFrame:
    return pd.DataFrame({"city_key": df.city_key, "predicted_target": target, "representation": model_name})


def kendall_bootstrap(x, y, *, seed=None, n_boot=None):
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n_boot = config.BOOTSTRAP_N if n_boot is None else n_boot
    point = kendalltau(x, y, nan_policy="omit").statistic
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        values.append(kendalltau(x[idx], y[idx], nan_policy="omit").statistic)
    return {"kendall": float(point), "q025": float(np.nanquantile(values, .025)), "q975": float(np.nanquantile(values, .975)), "n_boot": int(n_boot)}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
