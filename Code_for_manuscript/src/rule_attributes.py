"""Load and audit the 41-city clause-level release-attribute coding.

The categorical labels come from the manually reviewed ``City_audit`` sheet.
Quantitative ranking fields are populated only when the evidence excerpt
contains an explicit, parseable clause.  Missing values remain missing; they
are never replaced by a city-model stage count or an imputed policy value.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from .data_io import city_key


AUDIT_SHEET = "City_audit"
CATALOG_COLUMNS = [
    "basis_category",
    "release_trigger_category",
    "schedule_form_category",
    "intermediate_steps_category",
    "terminal_condition_category",
]
QUANT_COLUMNS = ["n_release_steps", "share_drawable_at_topout", "final_retained_share"]


def _yes(value: object) -> bool:
    return str(value).strip() == "是"


def _parse_numbered_steps(text: str, trigger: str) -> tuple[float, str]:
    if trigger == "no intermediate steps":
        return 0.0, "explicit no-intermediate-steps label"
    release_chunks = [
        chunk
        for chunk in re.split(r"[。；;\n|]", text)
        if any(token in chunk for token in ("监管额度", "提取", "拨付", "留存", "释放", "节点", "使用比例", "累计申请"))
    ]
    release_text = " ".join(release_chunks)
    numbers = [int(value) for value in re.findall(r"(?:\(|（)(\d{1,2})(?:\)|）)", release_text)]
    if len(numbers) >= 2:
        return float(max(numbers)), "numbered clauses in evidence excerpt"
    chinese = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    matches = re.findall(r"([一二三四五六七八九])个(?:时间节点|节点|阶段|环节)", release_text)
    if matches:
        return float(max(chinese[value] for value in matches)), "explicit Chinese count in evidence excerpt"
    return np.nan, "not explicitly countable from retained excerpt"


def _percent_range(text: str, start: int, stop: int) -> tuple[float, float] | None:
    window = text[start:stop]
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:[-—至]\s*(\d+(?:\.\d+)?)\s*%)?",
        window,
    )
    if not match:
        return None
    left = float(match.group(1)) / 100.0
    right = float(match.group(2)) / 100.0 if match.group(2) else left
    return left, right


def _parse_topout_drawable(text: str, schedule: str) -> tuple[float, str]:
    anchors = ["主体结构封顶", "主体结构验收", "主体结构", "封顶"]
    for anchor in anchors:
        start = text.find(anchor)
        if start < 0:
            continue
        parsed = _percent_range(text, start, min(len(text), start + 360))
        if parsed is None:
            continue
        low, high = parsed
        midpoint = (low + high) / 2.0
        if schedule == "retained balance":
            midpoint = 1.0 - midpoint
        return midpoint, f"explicit percentage near {anchor}; midpoint used for a stated range"
    return np.nan, "not explicitly parseable from retained excerpt"


def _parse_final_retained(text: str, flag: object) -> tuple[float, str]:
    if _yes(flag):
        return 0.05, "manual 5% retained flag"
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        window = text[max(0, match.start() - 45) : min(len(text), match.end() + 65)]
        value = float(match.group(1)) / 100.0
        if value <= 1.0 and any(token in window for token in ("留存", "预留", "保留")) and any(token in window for token in ("最终", "最后", "末期", "登记前", "备案前")):
            return value, "explicit terminal retention percentage in source text"
    return np.nan, "not explicitly parseable from retained excerpt"


def _basis(value: object) -> str | float:
    text = str(value).strip()
    if text.startswith("approved construction cost"):
        return "approved construction cost"
    if text.startswith("presale or registered sales proceeds"):
        return "presale proceeds"
    return np.nan


def load_city_audit() -> pd.DataFrame:
    if not config.RULE_ATTRIBUTES_PATH.exists():
        raise FileNotFoundError(f"缺少人工条款编码源：{config.RULE_ATTRIBUTES_PATH}")
    frame = pd.read_excel(config.RULE_ATTRIBUTES_PATH, sheet_name=AUDIT_SHEET)
    required = {
        "city",
        "base_basis",
        "multiplier_or_floor",
        "release_trigger_primary",
        "schedule_form_primary",
        "final_retained_5pct",
        "extra_condition_after_first_registration",
        "evidence_excerpt",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"City_audit 缺少字段：{missing}")
    frame["city_key"] = frame["city"].map(city_key)
    if len(frame) != config.N_CITIES or frame.city_key.nunique() != config.N_CITIES:
        raise ValueError("City_audit 必须包含41个唯一城市键")
    policy = pd.read_excel(config.POLICIES_PATH, sheet_name="Sheet1")
    policy["city_key"] = policy["city"].map(city_key)
    if policy.city_key.duplicated().any():
        raise ValueError("Policies_text.xlsx 城市键重复，不能用于量化条款取证")
    frame = frame.merge(policy[["city_key", "content"]].rename(columns={"content": "full_policy_text"}), on="city_key", how="left", validate="one_to_one")
    if frame.full_policy_text.isna().any():
        raise ValueError("人工编码城市未能与 Policies_text.xlsx 完整对齐")
    return frame


def build_rule_attributes() -> tuple[pd.DataFrame, dict]:
    source = load_city_audit().copy()
    rows = []
    for _, item in source.iterrows():
        city = str(item["city"])
        evidence = str(item.get("evidence_excerpt", ""))
        full_policy_text = str(item.get("full_policy_text", ""))
        coding_text = f"{evidence} {full_policy_text}"
        trigger_raw = str(item["release_trigger_primary"]).strip()
        trigger = "no intermediate steps" if trigger_raw == "no intermediate steps" else trigger_raw
        steps, steps_status = _parse_numbered_steps(coding_text, trigger)
        topout, topout_status = _parse_topout_drawable(coding_text, str(item["schedule_form_primary"]).strip())
        final_share, final_status = _parse_final_retained(coding_text, item["final_retained_5pct"])
        if trigger == "no intermediate steps":
            intermediate = "none specified"
        elif pd.notna(steps):
            intermediate = "specified"
        else:
            intermediate = "specified but count unavailable"
        terminal = (
            "condition beyond first registration"
            if _yes(item["extra_condition_after_first_registration"])
            else "5% retained"
            if _yes(item["final_retained_5pct"])
            else "other / unclear"
        )
        rows.append(
            {
                "city_key": item["city_key"],
                "city": city,
                "source_id": item.get("id", ""),
                "basis_category": _basis(item["base_basis"]),
                "multiplier_or_floor": "with multiplier or floor" if _yes(item["multiplier_or_floor"]) else np.nan,
                "release_trigger_category": trigger,
                "schedule_form_category": str(item["schedule_form_primary"]).strip(),
                "intermediate_steps_category": intermediate,
                "terminal_condition_category": terminal,
                "final_retained_5pct": _yes(item["final_retained_5pct"]),
                "extra_condition_after_first_registration": _yes(item["extra_condition_after_first_registration"]),
                "n_release_steps": steps,
                "share_drawable_at_topout": topout,
                "final_retained_share": final_share,
                "n_release_steps_status": steps_status,
                "share_drawable_at_topout_status": topout_status,
                "final_retained_share_status": final_status,
                "evidence_excerpt": evidence,
                "full_policy_text_used": True,
                "source_file": config.RULE_ATTRIBUTES_PATH.name,
                "source_sheet": AUDIT_SHEET,
            }
        )
    result = pd.DataFrame(rows)
    result = result.sort_values("city_key").reset_index(drop=True)
    audit = {
        "source_file": str(config.RULE_ATTRIBUTES_PATH),
        "source_sheet": AUDIT_SHEET,
        "n_cities": int(len(result)),
        "categorical_components": CATALOG_COLUMNS,
        "quantitative_ranking_fields": QUANT_COLUMNS,
        "missing_counts": {column: int(result[column].isna().sum()) for column in QUANT_COLUMNS},
        "quantitative_coding_rule": "Only explicit percentages or explicit numbered release clauses in the retained evidence excerpt plus full Policies_text.xlsx text are used; values above 100% are rejected and missing remains missing.",
        "distance_components": [
            "basis_category",
            "release_trigger_category",
            "schedule_form_category",
            "n_release_steps",
            "final_retained_share",
        ],
        "distance_note": "Gower uses three categorical components plus two quantitative components, matching Revised Sections 3 and 4; share_drawable_at_topout is reserved for H1 ranking concordance.",
    }
    return result, audit


def attribute_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(attribute: str, category: str, mask: pd.Series, *, absent: int = 0, note: str = ""):
        n = int(mask.sum())
        rows.append(
            {
                "attribute": attribute,
                "category": category,
                "n_cities": n,
                "share": n / config.N_CITIES,
                "n_absent": int(absent),
                "note": note,
            }
        )

    add("Basis of the supervised amount", "approved construction cost", frame.basis_category.eq("approved construction cost"), absent=int(frame.basis_category.isna().sum()))
    add("Basis of the supervised amount", "presale proceeds", frame.basis_category.eq("presale proceeds"), absent=int(frame.basis_category.isna().sum()))
    add("Basis of the supervised amount", "with multiplier or floor", frame.multiplier_or_floor.eq("with multiplier or floor"), note="non-exclusive flag")
    trigger_categories = [
        "verified physical milestones",
        "named administrative certificates",
        "successive inspection stages",
        "unspecified construction progress",
        "no intermediate steps",
    ]
    for category in trigger_categories:
        add("Class of release trigger", category, frame.release_trigger_category.eq(category), absent=0)
    add("Form of the schedule", "retained balance", frame.schedule_form_category.eq("retained balance"), absent=int(frame.schedule_form_category.str.contains("找不到", na=False).sum()))
    add("Form of the schedule", "drawdown ceiling", frame.schedule_form_category.eq("drawdown ceiling"), absent=int(frame.schedule_form_category.str.contains("找不到", na=False).sum()))
    add("Intermediate steps", "specified", frame.intermediate_steps_category.eq("specified"), absent=0)
    add("Intermediate steps", "none specified", frame.intermediate_steps_category.eq("none specified"), absent=0)
    add("Intermediate steps", "specified but count unavailable", frame.intermediate_steps_category.eq("specified but count unavailable"), absent=0)
    add("Terminal condition", "5% retained", frame.final_retained_5pct, note="non-exclusive flag")
    add("Terminal condition", "other", ~frame.final_retained_5pct, note="non-exclusive flag")
    add("Terminal condition", "condition beyond first registration", frame.extra_condition_after_first_registration, note="non-exclusive flag")
    return pd.DataFrame(rows)
