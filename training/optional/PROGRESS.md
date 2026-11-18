# Optional experiments - progress

See optional_tracker.csv for the full run table with all numbers.

## Status summary
- variant_dryrun_TEST: DONE (smoke test, 24,436,369 params)
- variant_mass_seed42: DONE (val 0.6620; test mass 0.6808 - beats baseline 0.6665)
- variant_whu_seed42: DONE (val 0.8951; test whu 0.8593)
- variant_inria_seed42: DONE (val 0.7400; test inria 0.7224)
- focal_mass_seed42: INVALID (loss bug; kept on disk)
- focal_whu_seed42: INVALID (loss bug; kept on disk)
- focal_inria_seed42: INVALID (loss bug; kept on disk)
- FOCALFIX_mass_seed42: DONE (val 0.6550; test mass 0.6683 - beats baseline 0.6665)
- FOCALFIX_whu_seed42: DONE (val 0.8664; test whu 0.8509 vs baseline 0.8624)
- FOCALFIX_inria_seed42: PENDING (awaiting GO)

## Notes / incidents
- LOSS BUG FIXED in FocalDice: Dice smoothing eps=1.0 leaked into the focal
  log terms (log(p+1.0)) making the focal term negative. Fixed: log terms
  now clamp at 1e-7; dice_loss untouched (eps=1.0 correct there).
- Sanity check PASSED: perfect -> focal 0.000000; wrong -> 8.79; 200 random
  batches never negative. NaN/inf guard added to the training loop.
- Original FOCAL_* runs INVALID (kept on disk, marked in tracker).
- FOCALFIX_* re-runs replace them.
- FOCALFIX_inria_seed42: DONE (best ep 37, val 0.7003; test inria 0.6859
  vs baseline 0.7169; whu 0.7102, mass 0.3035). Cross-domain clearly
  better than baseline (whu +0.31, mass +0.10).
- All 3 FOCALFIX re-runs complete; focal matrix rebuilt from fixed runs.
