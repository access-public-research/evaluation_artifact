# Supplement Manifest

This supplement is an artifact-backed regeneration package for the listed command-backed claim-bearing paper-facing outputs. It is not a full training-rerun package and does not redistribute raw benchmark data, pretrained weights, or third-party benchmark shards.

Archive-root directories:

- `supplement/`: top-level copy of the supplement documentation for quick reviewer access.
- `paper/neurips2026_selection_risk/`: LaTeX source, bibliography, checklist, figures, tables, and supplement documentation.
- `src/`: source scripts and utilities needed by `SUPPLEMENT_COMMANDS.md`.
- `configs/`: YAML configs referenced by the bundled regeneration path.
- `artifacts/metrics/`: saved CSV/JSON summaries needed by the listed paper-facing scripts.
- `artifacts/metrics/camelyon_erm_selector_dose_*`: saved Camelyon17 ERM selector-dose rows and generated summaries.
- `artifacts/metrics/camelyon_valacc_selector_sensitivity_*`: saved Camelyon17 ERM validation-accuracy selector sensitivity rows and generated summaries.
- `artifacts/metrics/camelyon_loo_selector_standard_metrics_*`: saved leave-one-hospital-out Camelyon selector-only rows and generated summaries.
- `artifacts/metrics/camelyon17_bootstrap_selector_*`: saved hard-bootstrap selector summaries supporting the Appendix A.16 corroboration table.
- `artifacts/metrics/camelyon*_finetune*_objfam*`: saved and derived Camelyon17 finetune-control summaries backing the method-ranking rows in Table 3 and Appendix Table 24.
- `artifacts/metrics/acs_income/`: saved ACSIncome regression summaries.
- `artifacts/partitions_eval/`: compact fixed-bank metadata and cell-assignment arrays for the teacher-defined hard-example CVaR diagnostic.
- `figures/`: small root-level CSV summaries used by the mechanism-figure script.

`DATA_ARTIFACT_LEDGER.csv` maps the main included CSV/JSON/PDF artifacts to the paper-facing scripts or tables they support.
`STATIC_SUPPORT_TABLES.md` documents supporting static/audit tables that are not separate command-line entry points.

Reviewers should run commands from `SUPPLEMENT_COMMANDS.md` from the extracted archive root.
