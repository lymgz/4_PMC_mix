"""把 10.3 的主线结果集中成可供论文核对的 reconciliation 表。"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import config


def run():
    config.ensure_directories(); rows = []
    for filename, record_type in [("canonical_results.csv", "parallel_representation"), ("block_results.csv", "incremental_block"), ("targeted_ablation_results.csv", "targeted_ablation")]:
        path = config.TABLE_DIR / filename
        if not path.exists(): continue
        frame = pd.read_csv(path)
        model_col = "model" if "model" in frame else None
        if model_col:
            for _, item in frame.iterrows():
                rows.append({"record_type": record_type, "source_file": filename, "model": item.get(model_col), "target": config.TARGET, "n_features": item.get("n_features"), "R2_mean": item.get("R2_mean"), "R2_sd": item.get("R2_sd"), "cv_protocol": item.get("cv_protocol"), "interpretation_scope": item.get("interpretation_scope", "predictive only")})
    result = pd.DataFrame(rows); result.to_csv(config.TABLE_DIR / "reconciliation.csv", index=False, encoding="utf-8-sig"); return result


if __name__ == "__main__": print(run().to_string(index=False))
