"""Code 10.4 的可复现研究设计与 Section 5 资产输出契约。

本版本按 2026-08-11 Revised Sections 3 and 4 v2 重建：Dep_reg 是规则化监管资金暴露的确定性代理；
RF 只用于预测比较和探索性模型归因，不构成因果识别。
"""
from __future__ import annotations

import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
POLICIES_PATH = Path(os.environ.get("JHBE_POLICIES_PATH", str(DATA_DIR / "policies_text.xlsx")))
VALIDITY_PATH = Path(os.environ.get("JHBE_VALIDITY_PATH", str(DATA_DIR / "validity_data.xlsx")))
PMC_DATA_PATH = Path(os.environ.get("JHBE_PMC_DATA_PATH", str(DATA_DIR / "pmc_data.xlsx")))
RULE_ATTRIBUTES_PATH = Path(os.environ.get("JHBE_RULE_ATTRIBUTES_PATH", str(DATA_DIR / "policy_identification_filled.xlsx")))
BERT_MODEL_PATH = Path(os.environ.get("JHBE_BERT_MODEL", str(DATA_DIR / "bert-base-chinese")))
REVISED_SECTIONS_PATH = Path(
    os.environ.get("JHBE_REVISED_SECTIONS", str(PROJECT_ROOT / "docs" / "revised_sections_3_4_v2.md"))
)

# The Dep engine is intentionally not vendored here: it is maintained and
# published in the separate Modelling Dep project.  Set JHBE_DEP_DIR when a
# raw rebuild is required.  Normal cached reproduction does not need it.
DEP_DIR = Path(os.environ["JHBE_DEP_DIR"]) if os.environ.get("JHBE_DEP_DIR") else None
DEP_ENGINE_PATH = DEP_DIR / "multi_city_dep_model.py" if DEP_DIR else None
DEP_CATALOG_PATH = DEP_DIR / "city_catalog.json" if DEP_DIR else None
DEP_OPERATIONAL_BRIDGE_PATH = Path(
    os.environ.get("JHBE_DEP_BRIDGE_PATH", str(DATA_DIR / "dep_operational_attributes_41cities.csv"))
)
DEP_OPERATIONAL_AUDIT_PATH = DATA_DIR / "dep_attribute_audit.json"
REPRESENTATIVE_RELEASE_PATH = DATA_DIR / "representative_release_milestones.csv"
RATE_SENSITIVITY_PATH = DATA_DIR / "rate_sensitivity.csv"
RATE_SENSITIVITY_CITY_PATH = DATA_DIR / "rate_sensitivity_city_level.csv"
RATE_SENSITIVITY_RATIO_PATH = DATA_DIR / "rate_sensitivity_ratio_check.csv"
DEP_PIPELINE_AUDIT_PATH = DATA_DIR / "dep_pipeline_audit.json"
DEP_FIXED_INPUT_AUDIT_PATH = DATA_DIR / "dep_reg_fixed_input_audit.json"
RATE_JUSTIFICATION_PATH = DATA_DIR / "rate_justification.md"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
TEX_DIR = OUTPUT_DIR / "tex"
COMBINED_DATA_PATH = OUTPUT_DIR / "combined_data_10.4.xlsx"
VERSION = "10.4"

SEED = 42
CV_SPLITS = int(os.environ.get("JHBE_CV_SPLITS", "5"))
N_REPEATS = int(os.environ.get("JHBE_N_REPEATS", "20"))
N_ESTIMATORS = int(os.environ.get("JHBE_N_ESTIMATORS", "1000"))
N_JOBS = int(os.environ.get("JHBE_N_JOBS", "1"))
BASE_LPR_ANNUAL = 0.0345
BASE_LPR_MULTIPLIER = 3.0
RATE_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0)
N_CITIES = 41
DEP_TOLERANCE = 1e-8
BOOTSTRAP_N = int(os.environ.get("JHBE_BOOTSTRAP_N", "2000"))
H1_BOOTSTRAP_N = int(os.environ.get("JHBE_H1_BOOTSTRAP_N", "1000"))
P1_CHANCE_BENCHMARK = 51
OPTUNA_TRIALS = int(os.environ.get("JHBE_OPTUNA_TRIALS", "40"))
TUNING_CV_SPLITS = int(os.environ.get("JHBE_TUNING_CV_SPLITS", "5"))
TUNING_CV_REPEATS = int(os.environ.get("JHBE_TUNING_CV_REPEATS", "5"))
TUNING_N_ESTIMATORS = int(os.environ.get("JHBE_TUNING_N_ESTIMATORS", "300"))
TUNING_SEED = int(os.environ.get("JHBE_TUNING_SEED", str(SEED)))
FROZEN_RF_PATH = LOG_DIR / "m0_frozen_rf_params.json"

# IV3/IV4 remain descriptive audit fields only. They are forbidden in every RF
# predictor, rank input, SHAP input, and constructed interaction.
BASELINE = ["IV1", "IV2"]
CONTROLS = ["year_1", "Region_1"]
MODERATORS = ["PMC_T", "PMC_I", "PMC_Frequency", "PMC_BERT"]
TARGET = "Dep_reg"

M0_FEATURES = [*BASELINE, *CONTROLS]
MODEL_FEATURES = {
    "M0": M0_FEATURES,
    "M1": [*M0_FEATURES, "PMC_T"],
    "M2": [*M0_FEATURES, "PMC_I"],
    "M3": [*M0_FEATURES, "PMC_Frequency"],
    "M4": [*M0_FEATURES, "PMC_BERT"],
    "B0": M0_FEATURES,
    "B1": [*M0_FEATURES, *MODERATORS],
}
INTERACTION_FEATURES = [f"{iv}_x_{pmc}" for iv in BASELINE for pmc in MODERATORS]
MODEL_FEATURES["B2"] = [*MODEL_FEATURES["B1"], *INTERACTION_FEATURES]
PARALLEL_SPECS = {"M0": MODEL_FEATURES["M0"], "M1": MODEL_FEATURES["M1"], "M2": MODEL_FEATURES["M2"], "M3": MODEL_FEATURES["M3"], "M4": MODEL_FEATURES["M4"]}

LABELS = {
    "IV1": "Housing price",
    "IV2": "Absorption period",
    "IV3": "Construction period (descriptive only)",
    "IV4": "Peak supervised-fund exposure (descriptive only)",
    "year_1": "Policy year (control)",
    "Region_1": "City tier (control)",
    "PMC_T": "PMC-T",
    "PMC_I": "PMC-I",
    "PMC_Frequency": "PMC-Frequency",
    "PMC_BERT": "PMC-BERT",
    TARGET: "Dep_reg (rule-based proxy)",
}

RF_DEFAULT = {
    "n_estimators": N_ESTIMATORS,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": 0.5,
    "ccp_alpha": 0.0,
    "random_state": SEED,
    "n_jobs": N_JOBS,
}
RF_COMMON = dict(RF_DEFAULT)
FROZEN_LOAD_STATUS = "not_found"


def display_path(path: Path | str | None) -> str | None:
    """Return a portable project-relative path for provenance and manifests."""
    if path is None:
        return None
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_existing_frozen_params() -> None:
    """让单独运行的分析脚本也自动复用正式冻结参数。"""
    global FROZEN_LOAD_STATUS
    if not FROZEN_RF_PATH.exists():
        return
    payload = json.loads(FROZEN_RF_PATH.read_text(encoding="utf-8"))
    if payload.get("tuning_scope") != "M0_only":
        raise ValueError("冻结参数文件的 tuning_scope 必须为 M0_only")
    frozen = payload.get("frozen_params")
    if not isinstance(frozen, dict):
        raise ValueError("冻结参数文件缺少 frozen_params")
    if frozen.get("n_estimators") != N_ESTIMATORS or frozen.get("n_jobs") != N_JOBS:
        FROZEN_LOAD_STATUS = "runtime_mismatch_requires_retune"
        return
    RF_COMMON.clear()
    RF_COMMON.update(frozen)
    FROZEN_LOAD_STATUS = "loaded"

def ensure_directories() -> None:
    for path in (OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, TEX_DIR):
        path.mkdir(parents=True, exist_ok=True)

def source_files() -> list[Path]:
    files = [
        POLICIES_PATH,
        VALIDITY_PATH,
        PMC_DATA_PATH,
        RULE_ATTRIBUTES_PATH,
        REVISED_SECTIONS_PATH,
        BERT_MODEL_PATH,
    ]
    if DEP_ENGINE_PATH is not None:
        files.append(DEP_ENGINE_PATH)
    if DEP_CATALOG_PATH is not None:
        files.append(DEP_CATALOG_PATH)
    return files


_load_existing_frozen_params()
