# Supplement README

This directory is the staged source for the anonymized NeurIPS 2026 E&D supplement bundle.

The bundle is intentionally narrow and supports artifact-backed verification of the paper's main claims:

- live `main.tex` paper source
- exact commands for rebuilding the PDF and regenerating the listed command-backed claim-bearing paper-facing figures and tables from saved artifacts
- minimal Python dependencies for those paper-facing scripts
- claim-bearing CSV artifacts used by the live paper
- a compact selector-comparison artifact suite for comparing alternative checkpoint selectors on the released runs without retraining
- Camelyon17 ERM selector-dose, validation-accuracy selector-sensitivity, and leave-one-hospital-out selector-only diagnostic summaries generated from saved artifacts
- `ASSET_TERMS.md`, which credits the external datasets, evaluation suites, and code assets used by the submission and summarizes their verified governing terms
- no raw third-party benchmark data
- no exploratory rebuttal-only outputs

The final packaged archive for submission is the anonymized supplement ZIP uploaded alongside the PDF.
Internal labels of the form `rcgdro_*` are legacy run-family identifiers for Camelyon selector/finetune artifacts; they are not a separate method claim and do not identify authors or institutions.

Expected archive-root layout:

- `supplement/`: top-level copy of this supplement documentation for quick reviewer access
- `paper/neurips2026_selection_risk/`: LaTeX source, tables, figures, and this supplement documentation
- `src/scripts/`: artifact-backed scripts used by `SUPPLEMENT_COMMANDS.md`
- `src/utils/`: small utilities needed by those scripts
- `configs/`: YAML configs referenced by the paper-facing regeneration path
- `artifacts/metrics/`: saved CSV/JSON summaries needed by the bundled scripts
- `artifacts/partitions_eval/`: compact fixed-bank metadata and cell assignments for the teacher-defined hard-example CVaR diagnostic
- `figures/`: small root-level CSV summaries used by the mechanism figure script

The supplement is designed to regenerate the listed command-backed claim-bearing paper-facing outputs from saved repository artifacts, not to rerun full model training or every static support/audit table. Benchmark datasets must be obtained separately under their original terms, and the bundle does not redistribute any third-party raw images or benchmark shards.
The supported review entry points are the commands in `SUPPLEMENT_COMMANDS.md`; additional source/config files are included only to make the artifact provenance inspectable and are not separate guaranteed command-line entry points.

See `MANIFEST.md` for the archive layout, `DATA_ARTIFACT_LEDGER.csv` for the claim-bearing artifact map, `SUPPLEMENT_COMMANDS.md` for the exact commands, `STATIC_SUPPORT_TABLES.md` for static support/audit table provenance, `requirements.txt` for the minimal Python environment used by the bundled artifact-backed scripts, and `ASSET_TERMS.md` for third-party asset notes.
