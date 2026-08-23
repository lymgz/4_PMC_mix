"""10.3 特征白名单；IV3/IV4 永远不参与预测侧派生。"""
from __future__ import annotations

from collections.abc import Iterable
import pandas as pd
import config


def _kept(values: Iterable[str], drop: Iterable[str] = ()) -> list[str]:
    dropped = set(drop)
    return [v for v in values if v not in dropped and not ("_x_" in v and any(part in dropped for part in v.split("_x_")))]


def _construct(df: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    work = df.copy()
    for feature in requested:
        if "_x_" in feature:
            left, right = feature.split("_x_", 1)
            if left not in work.columns or right not in work.columns:
                raise KeyError(f"无法构造 {feature}: 缺少 {left}/{right}")
            work[feature] = pd.to_numeric(work[left], errors="raise") * pd.to_numeric(work[right], errors="raise")
    missing = [f for f in requested if f not in work.columns]
    if missing:
        raise KeyError(f"数据缺少特征: {missing}")
    forbidden = [f for f in requested if f in {"IV3", "IV4", "Dep", "Dep_reg"} or "IV3" in f or "IV4" in f]
    if forbidden:
        raise AssertionError(f"结果侧/近常量变量回流到预测特征：{forbidden}")
    return work[requested].copy()


def build_spec(df: pd.DataFrame, name: str, drop: Iterable[str] = ()):
    if name not in config.MODEL_FEATURES:
        raise KeyError(f"未知模型规格: {name}")
    requested = _kept(config.MODEL_FEATURES[name], drop)
    return _construct(df, requested), requested


def build_parallel_spec(df: pd.DataFrame, name: str, drop: Iterable[str] = ()):
    if name not in config.PARALLEL_SPECS:
        raise KeyError(name)
    requested = _kept(config.PARALLEL_SPECS[name], drop)
    return _construct(df, requested), requested


def build_custom(df: pd.DataFrame, features: Iterable[str]):
    requested = list(features)
    return _construct(df, requested), requested
