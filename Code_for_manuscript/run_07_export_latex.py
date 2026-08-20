"""Export canonical tables as optional LaTeX fragments."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config


def run():
    config.ensure_directories()
    outputs = []
    for filename in ("canonical_results.csv", "descriptives.csv", "reconciliation.csv"):
        source = config.TABLE_DIR / filename
        if not source.exists():
            continue
        target = config.TEX_DIR / filename.replace(".csv", ".tex")
        pd.read_csv(source).to_latex(target, index=False, float_format="%.3f", escape=False)
        outputs.append(str(target))
    return outputs


if __name__ == "__main__":
    print("\n".join(run()))

