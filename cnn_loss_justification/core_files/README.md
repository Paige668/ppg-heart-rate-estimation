# CNN Penalty Search Notes

This folder stores the retained CNN shared-penalty search that compared `cfg_a`, `cfg_b`, and `cfg_c` for the current `cnn_unified` route.

Current interpretation:

- `cfg_a` (`bias_penalty=0.40`, `over_penalty=0.18`) is the retained shared baseline because it gives the best cross-case balance.
- `cfg_b` and `cfg_c` improved the no-pre case but degraded the full-pipeline case too much to replace the shared baseline.

Key seed-42 outcomes from `search_results.json`:

- `cfg_a`: No-Prep `17.0237`, Full Pipeline `17.4716`
- `cfg_b`: No-Prep `16.3872`, Full Pipeline `18.6359`
- `cfg_c`: No-Prep `15.9164`, Full Pipeline `18.8997`

These artifacts support the retained CNN baseline rationale, but they are not the main final comparison tables. For the official retained comparison, use `MODEL_COMPARISON_AND_FINAL_SELECTION.md` and `general_model_restart/cnn_unified/MODEL_SELECTION_RATIONALE.md`.
