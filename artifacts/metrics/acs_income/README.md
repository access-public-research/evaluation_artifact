# ACSIncome Claim-Bearing Artifacts

These files were imported from the completed ACSIncome continuation workspace so that the paper's regression claims are fully backed by in-repo artifacts.
In these artifacts, `test_tail_mse` is raw MSE on held-out examples whose un-winsorized target lies above the training-set P95 target threshold, matching the winsorization cutoff used for the suppressive regression proxy.

## Main paper provenance

- `phase3_selector_winsorized_p95.csv`
  - backs the Table 2 ACSIncome selector-comparison row
- `phase3_rw_summary.json`
  - backs the reported `R_w=0.41` winsorization and `R_w=0.53` Huber values
- `phase2_all_history.csv`
  - backs the fixed-horizon ACSIncome entry in the persistence-support table

## Supporting provenance

- `phase3_selector_huber_p95.csv`
  - appendix Huber dose-response corroboration

The live paper uses these copied files stored under this in-repository artifact directory.
