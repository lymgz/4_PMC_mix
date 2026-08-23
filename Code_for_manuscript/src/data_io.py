"""从本地工作簿重建 41 城 keyed dataset；10.2 输入保持只读。"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd

import config
from .dep_reg import build_regulated_outcomes, load_engine, write_audit

REQUIRED = [*config.BASELINE, *config.CONTROLS, *config.MODERATORS, config.TARGET]
ENGLISH_CITY_ALIASES = {
    "baisha": "白沙", "beihai": "北海", "beijing": "北京", "chengmai": "澄迈", "danzhou": "儋州", "dingwan": "定安", "dingan": "定安", "dongfang": "东方", "dongguan": "东莞", "dongying": "东营", "foshan": "佛山", "guangzhou": "广州", "haikou": "海口", "hefei": "合肥", "huizhou": "惠州", "jiujiang": "九江", "lanzhou": "兰州", "ledong": "乐东", "leshan": "乐山", "lishui": "丽水", "lingao": "临高", "lingshui": "陵水", "luan": "六安", "qingyuan": "清远", "qionghai": "琼海", "qiongzhong": "琼中", "shantou": "汕头", "shanghai": "上海", "shangrao": "上饶", "shaoguan": "韶关", "shenzhen": "深圳", "wanning": "万宁", "wenchang": "文昌", "wuzhishan": "五指山", "xinyang": "信阳", "yiwu": "义乌", "zhaoqing": "肇庆", "zhongshan": "中山", "zhongwei": "中卫", "zhuhai": "珠海", "tunchang": "屯昌", "changjiang": "昌江",
}


def city_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).strip())
    if not text:
        return ""
    if re.search(r"[一-鿿]", text):
        return re.sub(r"[市县]$", "", text)
    lowered = text.lower().replace("’", "'")
    lowered = re.sub(r"\s+(?:city|county)$", "", lowered)
    return ENGLISH_CITY_ALIASES.get(lowered, lowered)


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    if "Dep" not in df.columns and "Dep_engine_original" in df.columns:
        df = df.copy()
        df["Dep"] = pd.to_numeric(df["Dep_engine_original"], errors="coerce")
    if len(df) != config.N_CITIES:
        raise ValueError(f"样本数应为41，实际为 {len(df)}。")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"合并数据缺少字段：{missing}")
    numeric = df[REQUIRED].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.isna().sum()
        raise ValueError(f"模型字段存在缺失或非数值：{bad[bad > 0].to_dict()}")
    if df["city_key"].duplicated().any() or df["city_key"].eq("").any():
        raise ValueError("合并数据存在空或重复城市键。")
    return df.reset_index(drop=True)


def _normalize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="raise")
    span = float(values.max() - values.min())
    return pd.Series(0.0, index=values.index) if span == 0 else 10.0 * (values - values.min()) / span


def _write_provenance(mode: str, frame: pd.DataFrame, dep_audit: dict) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in config.source_files():
        record = {"path": config.display_path(path), "exists": path.exists()}
        if path.is_file():
            record.update({"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        records.append(record)
    payload = {"mode": mode, "n_cities": len(frame), "n_unique_city_keys": frame.city_key.nunique(), "required_columns": REQUIRED, "source_files": records, "target_combined_data": config.display_path(config.COMBINED_DATA_PATH), "dep_reg_audit": dep_audit}
    (config.LOG_DIR / "data_provenance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    frame[["city", "city_key"]].assign(status="通过").to_csv(config.TABLE_DIR / "data_merge_audit.csv", index=False, encoding="utf-8-sig")


def _policy_features() -> pd.DataFrame:
    policy = pd.read_excel(config.POLICIES_PATH, sheet_name="Sheet1")
    if len(policy) != config.N_CITIES:
        raise ValueError("Policies_text.xlsx 不是41行。")
    import jieba
    import torch
    from transformers import AutoModel, AutoTokenizer

    stopwords = {"的", "了", "和", "是", "在", "以", "对", "等", "与", "中", "将", "为"}

    def preprocess(value):
        text = value if isinstance(value, str) else ""
        cleaned = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
        return " ".join(w for w in jieba.cut(cleaned) if len(w.strip()) > 1 and w not in stopwords)

    policy["city_key"] = policy["city"].map(city_key)
    policy["processed_text"] = policy["content"].map(preprocess)
    raw_frequency = policy.processed_text.map(lambda text: len(text.split()) / max(1, len(set(text.split()))) * np.log1p(len(text.split())))
    policy["PMC_Frequency"] = _normalize(raw_frequency)
    if not config.BERT_MODEL_PATH.exists():
        raise FileNotFoundError(f"本地 BERT 模型不存在：{config.BERT_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(config.BERT_MODEL_PATH), local_files_only=True)
    model = AutoModel.from_pretrained(str(config.BERT_MODEL_PATH), local_files_only=True)
    model.eval()

    # Batch all sentence chunks once. This preserves the 10.2 CLS/cosine
    # definition while avoiding one transformer forward pass per city.
    parts_by_city = []
    flat_parts = []
    for text in policy.content.fillna("").astype(str):
        parts = [part.strip() for part in re.split(r"[。！？]", text) if part.strip()][:32]
        parts_by_city.append(parts)
        flat_parts.extend(parts)
    flat_vectors = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(flat_parts), batch_size):
            batch = flat_parts[start : start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", truncation=True, max_length=128, padding=True)
            flat_vectors.append(model(**encoded).last_hidden_state[:, 0, :].numpy())
    flat_vectors = np.concatenate(flat_vectors, axis=0) if flat_vectors else np.empty((0, 1))
    scores, offset = [], 0
    for parts in parts_by_city:
        if len(parts) < 2:
            scores.append(0.0); continue
        vectors = np.asarray(flat_vectors[offset : offset + len(parts)], dtype=float); offset += len(parts)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        values = (vectors @ vectors.T)[np.triu_indices(len(vectors), 1)]
        scores.append(float(values.mean()) if len(values) else 0.0)
    policy["PMC_BERT"] = _normalize(pd.Series(scores, index=policy.index))
    source = pd.read_excel(config.VALIDITY_PATH, sheet_name="machine_learning", header=1)
    source.columns = [str(c).strip() for c in source.columns]
    source = source[["PMCcode", "City", "PMC-I", "PMC-T"]].dropna(subset=["PMCcode", "City"]).copy()
    expected = {f"P{i}" for i in range(1, config.N_CITIES + 1)}
    if len(source) != config.N_CITIES or set(source.PMCcode) != expected or source.PMCcode.duplicated().any():
        raise ValueError("machine_learning 有效 PMC 行必须是唯一的 P1-P41。")
    source["city_key"] = source.City.map(city_key)
    source["PMC_T"] = pd.to_numeric(source["PMC-T"], errors="raise")
    source["PMC_I"] = pd.to_numeric(source["PMC-I"], errors="raise")
    if source.city_key.duplicated().any() or source.city_key.eq("").any():
        raise ValueError("machine_learning 城市键为空或重复。")
    return policy.merge(source[["city_key", "PMC_T", "PMC_I"]], on="city_key", validate="one_to_one")


def _rebuild_from_local_sources() -> pd.DataFrame:
    if config.DEP_ENGINE_PATH is None or not config.DEP_ENGINE_PATH.exists():
        raise RuntimeError(
            "Raw rebuild requires the separately published Modelling Dep engine. "
            "Set JHBE_DEP_DIR to its 41_Cities_Dep directory, or omit --force-rebuild "
            "to use the packaged 41-city combined-data cache."
        )
    policy = _policy_features()
    engine = load_engine()
    catalog = engine.extract_catalog(config.PMC_DATA_PATH)
    from . import dep_reg
    dep_frame, dep_audit = build_regulated_outcomes(catalog, lpr_multiplier=config.BASE_LPR_MULTIPLIER, engine=engine)
    market_rows = []
    for city in catalog["city_names"]:
        item = catalog["cities"][city]
        market_rows.append({"city_key": city_key(city), "IV1": item["average_price"], "IV2": item["absorption_months"], "IV3": item["timing_parameters"].get("completion_months"), "IV4": item.get("peak_exposure_baseline")})
    market = pd.DataFrame(market_rows)
    dep_for_merge = dep_frame.drop(columns=["city"], errors="ignore")
    result = policy.merge(market, on="city_key", validate="one_to_one").merge(dep_for_merge, on="city_key", validate="one_to_one")
    result["Region_1"] = result["Region"].map({"一线": 1, "二线": 2, "三线": 3}).fillna(pd.to_numeric(result["Region"], errors="coerce"))
    result["year_1"] = pd.to_numeric(result["year"], errors="raise") - 2018
    result = _validate(result)
    config.ensure_directories()
    result.to_excel(config.COMBINED_DATA_PATH, index=False)
    dep_reg.write_audit(dep_audit)
    _write_provenance("raw_rebuild", result, dep_audit)
    return result


def _load_existing() -> pd.DataFrame | None:
    if not config.COMBINED_DATA_PATH.exists():
        return None
    try:
        frame = pd.read_excel(config.COMBINED_DATA_PATH)
        frame = _validate(frame)
        provenance_path = config.LOG_DIR / "data_provenance.json"
        if not provenance_path.exists():
            _write_provenance(
                "cached_local_output",
                frame,
                {
                    "definition": "packaged 41-city combined-data cache",
                    "source": config.display_path(config.COMBINED_DATA_PATH),
                    "engine_path": config.display_path(config.DEP_ENGINE_PATH),
                    "cache_mode": True,
                },
            )
        return frame
    except (ValueError, FileNotFoundError):
        return None


def load_combined(force_rebuild: bool = False) -> pd.DataFrame:
    if not force_rebuild:
        existing = _load_existing()
        if existing is not None:
            return existing
    return _rebuild_from_local_sources()
