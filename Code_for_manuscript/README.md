# Code 10.4: JHBE Evidence Pipeline

This repository contains the reproducible analysis pipeline for the Code 10.4 evidence bundle accompanying the revised *Journal of Housing and the Built Environment* manuscript. It evaluates policy-text representations and rule-based operational proxies for 41 cities using predictive comparisons, measurement-validity diagnostics, and exploratory model attribution.

The results are not causal policy effects. Random-forest scores, SHAP values, R² Shapley decompositions, rank changes, and discordant-pair counts must be interpreted within their stated predictive or descriptive scope.

## What is included

- Python scripts for the full Code 10.4 pipeline and individual analysis modules.
- The 41-city input workbooks required by the included analysis.
- A validated `combined_data_10.4.xlsx` cache for immediate reproduction.
- The audited Dep operational bridge and rate-sensitivity cache.
- Unit tests under `tests/`.

The separate `Modelling Dep` project is intentionally not copied here because it has already been published separately. This release contains only the audited bridge and cached results needed for the default reproduction path.

## Requirements

- Python 3.10 or newer; Python 3.13 was used for local verification.
- A working `pip` installation.
- Windows, macOS, and Linux are supported for the cached reproduction path.
- The optional raw-rebuild path additionally requires the separately published Dep engine and a local Chinese BERT model.

Install the pinned dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

Run these commands from the repository root:

```bash
python -m pytest -q
python -B run_pipeline.py
```

The first command runs the test suite. The second command runs the complete cached reproduction pipeline. It reads the packaged 41-city cache and writes regenerated tables, figures, logs, and LaTeX fragments under `outputs/`.

For a fast smoke run:

```bash
python -B run_pipeline.py --skip-shap --skip-shapley --skip-ablations
```

The smoke run is for debugging only and is not expected to satisfy the formal requirements audit because several evidence modules are intentionally skipped.

## Raw rebuild from source inputs

The default command does not need `Modelling Dep` or the BERT model. To rebuild `combined_data_10.4.xlsx` from the local source workbooks, provide the separately published `41_Cities_Dep` directory and a local `bert-base-chinese` model:

Windows PowerShell:

```powershell
$env:JHBE_DEP_DIR = "C:\path\to\Modelling Dep\41_Cities_Dep"
$env:JHBE_BERT_MODEL = "C:\path\to\bert-base-chinese"
python -B run_pipeline.py --force-rebuild --retune-m0 --optuna-trials 40
```

`JHBE_DEP_DIR` must point directly to the directory containing `multi_city_dep_model.py` and `city_catalog.json`. The BERT model is not included in this repository because its model weights are approximately 412 MB; use a model directory compatible with the original local-files-only loading code.

## Main entry points

| File | Purpose |
|---|---|
| `run_pipeline.py` | Runs the complete Code 10.4 evidence pipeline |
| `run_00_m0_optuna.py` | Tunes M0 only and freezes one RF parameter set |
| `run_02_canonical.py` | Compares four PMC representations against M0 |
| `run_03_overlap.py` | Computes P1 rule/text distance diagnostics |
| `run_11_pca_h1.py` | Historical filename; runs the current H1 attribute bridge without PCA |
| `run_14_shap_exploratory.py` | Generates exploratory SHAP summaries |
| `run_16_section5_assets.py` | Assembles Section 5 tables, figures, and the asset manifest |

## Reproducibility contract

- 41-city cross-sectional sample.
- `seed=42`.
- Random forest with repeated cross-validation; the exact runtime contract is recorded in `outputs/logs/m0_frozen_rf_params.json` after the first run.
- Optuna tuning is restricted to M0; the frozen parameter set is reused for later specifications.
- `IV3` is descriptive only and `IV4` is reserved for the mechanical-overlap audit.
- SHAP and R² Shapley products are exploratory attribution quantities, not effect estimates or significance tests.

## Data and licensing

The package includes policy text, manually coded policy attributes, and derived 41-city analysis data. Check the rights and redistribution conditions of the source materials before pushing this directory to a public repository. No license is declared automatically here; add a code/data license appropriate to the source materials before public release.

For the Chinese instructions, see [`README.zh-CN.md`](README.zh-CN.md).
