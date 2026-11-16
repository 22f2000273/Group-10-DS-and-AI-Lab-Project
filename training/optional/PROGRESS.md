# Optional experiments — progress

Label rule: everything in Phase 1 carries VARIANT (ResNet-34 pre-trained
encoder) — comparison only, never mixed into the main baseline matrix.

## Setup
- smp version installed: **0.6.0** (pip install --break-system-packages)
- Variant dry-run test: PASSED (full flow, params 24,436,369, finite loss)

## Run table

| # | Phase | run_id | Status |
|---|-------|--------|--------|
| 0 | 0 | variant_dryrun_TEST | DONE |
| 1 | 1 | variant_mass_seed42 | DONE (best epoch 42, val IoU 0.6620; test: mass 0.6808, whu 0.5960, inria 0.2169; 6 min) |
| 2 | 1 | variant_whu_seed42 | DONE (best epoch 49, val IoU 0.8951; test: whu 0.8593, inria 0.4229, mass 0.3610; 40 min) |
| 3 | 1 | variant_inria_seed42 | DONE (best epoch 49, val IoU 0.7400; test: inria 0.7224, whu 0.6656, mass 0.3246; 40 min) |
| 4 | 2 | focal_mass_seed42 | DONE (best epoch 11/early-stop@21, val IoU 0.5698; test: mass 0.5279, whu 0.4266, inria 0.1563; 8 min) |
| 5 | 2 | focal_whu_seed42 | PENDING |
| 6 | 2 | focal_inria_seed42 | DONE (best epoch 48, val IoU 0.5780; test: inria 0.5633, whu 0.4725, mass 0.2738; 5.5 h) |

## Notes / incidents
- (empty)
