"""Scan manuscript R-squared claims against the canonical result tables."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config


R2_PATTERN = re.compile(r"(?:R\^2|R²|R\{\^2\})\s*=?\s*([01](?:\.\d+)?)")


def _manuscript_candidates() -> list[Path]:
    # A public release must scan only files shipped in this project.  The
    # original workspace contains manuscript drafts outside the release root.
    search_root = config.PROJECT_ROOT
    return sorted(
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.name in {"Manuscript_V9_2.tex", "Appendix_V2.tex"}
    )


def run() -> dict:
    config.ensure_directories()
    canonical_path = config.TABLE_DIR / "canonical_results.csv"
    reconciliation_path = config.TABLE_DIR / "reconciliation.csv"
    canonical = pd.read_csv(canonical_path) if canonical_path.exists() else pd.DataFrame()
    reconciliation = pd.read_csv(reconciliation_path) if reconciliation_path.exists() else pd.DataFrame()
    expected = []
    if not canonical.empty:
        for column in ("R2_oof_mean", "R2_mean"):
            if column in canonical.columns:
                expected.extend(pd.to_numeric(canonical[column], errors="coerce").dropna().tolist())
    if not reconciliation.empty:
        for column in ("old_value", "canonical_new_value", "new_value"):
            if column in reconciliation:
                expected.extend(pd.to_numeric(reconciliation[column], errors="coerce").dropna().tolist())

    rows = []
    files = _manuscript_candidates()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in R2_PATTERN.finditer(text):
            value = float(match.group(1))
            nearest = min(expected, key=lambda item: abs(item - value), default=None)
            matched = nearest is not None and abs(nearest - value) <= 0.0011
            rows.append(
                {
                    "file": str(path),
                    "line": text.count("\n", 0, match.start()) + 1,
                    "claimed_r2": value,
                    "nearest_authoritative_value": nearest,
                    "matched_within_0.0011": matched,
                    "context": text[max(0, match.start() - 60) : match.end() + 60].replace("\n", " "),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(
            [
                {
                    "file": "",
                    "line": "",
                    "claimed_r2": "",
                    "nearest_authoritative_value": "",
                    "matched_within_0.0011": "",
                    "context": "No Manuscript_V9_2.tex or Appendix_V2.tex found under the project root; scan is ready but not executable yet.",
                }
            ]
        )
    result.to_csv(config.TABLE_DIR / "r2_claim_scan.csv", index=False, encoding="utf-8-sig")
    unmatched = int((result["matched_within_0.0011"] == False).sum()) if "matched_within_0.0011" in result else 0
    note = (
        f"files_scanned={len(files)}\n"
        f"claims_found={0 if not rows else len(rows)}\n"
        f"unmatched_claims={unmatched}\n"
        "The scan is read-only and does not modify manuscript TeX.\n"
    )
    (config.LOG_DIR / "r2_claim_scan.md").write_text(note, encoding="utf-8")
    return {"files_scanned": len(files), "claims_found": len(rows), "unmatched_claims": unmatched}


if __name__ == "__main__":
    print(run())




