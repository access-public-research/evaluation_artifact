## NeurIPS 2026 Selection-Risk Paper Workspace

This directory is a clean manuscript workspace for the rebuilt NeurIPS 2026 submission.

Reviewer note: this internal workspace README is provenance-only, not the review workflow.
For the submitted artifact, use the extracted archive's `supplement/README.md` and
`supplement/SUPPLEMENT_COMMANDS.md`.

Goals:
- build the paper from current evidence rather than legacy draft structure
- keep only paper-facing assets here
- avoid copying core training/evaluation code into the paper folder
- stage only the tables, figures, and prose that survive the final acceptance-focused filter

Principles:
- in the full development workspace, `src/`, `configs/`, `runs/`, `artifacts/`, and repo-level `figures/` were the source of truth; the review ZIP intentionally includes only the anonymized saved artifacts and scripts documented in `supplement/`
- this folder contains the manuscript, final exported tables/figures, and compact notes only
- no historical audit clutter, old PDFs, or dead appendix material should be carried forward by default

Expected workflow:
1. Run experiments and analysis from the repo root.
2. Export only accepted paper assets into this folder's `tables/` and `figures/`.
3. Write live main-text sections in `sections_ed/`; shared appendix and reproducibility sections remain in `sections/`.
4. Build review PDFs into `build/`.
