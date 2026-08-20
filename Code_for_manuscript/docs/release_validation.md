# Release validation

Validated locally on 2026-08-20 in the `lym313` environment.

- Unit tests: `9 passed`.
- Cached full pipeline: completed successfully.
- Formal requirements audit: `19/19` checks passed.
- Cached mode was exercised without `Modelling Dep` and without a local BERT model.
- Absolute paths from the original private workspace were removed from the text-based release files.
- The optional `--force-rebuild` path was not executed in this release check because it requires the separately published Dep engine and an external BERT model.

The generated `outputs/tables`, `outputs/figures`, and `outputs/tex` directories are present only as local validation artifacts and are ignored by `.gitignore`; they can be omitted from the GitHub commit. Run `python -B run_pipeline.py` to regenerate them.
