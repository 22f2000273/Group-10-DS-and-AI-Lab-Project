# M4 Training Progress

All runs are launched with `nohup` (log per run in `training/logs/`) and linger
is enabled for this user, so runs survive dropped connections. One run at a
time. After every run: verify outputs, update `training/results_tracker.csv`,
mark DONE here, write `training/summaries/<run_id>.txt`. Run outputs are never
deleted; checkpoints stay on the server (gitignored).

## Run table

| # | Phase | run_id | Description | Status |
|---|-------|--------|-------------|--------|
| 0 | 1 | probe/timing_probe | timing probe (63 steps, DONE) | DONE |
| 1 | 2 | baseline/unet_mass_seed42 | Protocol A baseline, Mass | DONE (best epoch 43, val IoU 0.6569; test: mass 0.6665, whu 0.5687, inria 0.1995; 20 min) |
| 2 | 2 | baseline/unet_whu_seed42 | Protocol A baseline, WHU | DONE (best epoch 39, val IoU 0.8851; test: whu 0.8624, inria 0.3677, mass 0.3959; 2.7 h) |
| 3 | 2 | baseline/unet_inria_seed42 | Protocol A baseline, Inria | DONE (best epoch 44, val IoU 0.7293; test: inria 0.7169, whu 0.6968, mass 0.3052; 5.6 h) |
| 4 | 3 | seeds/unet_mass_seed43 | seed repeat | DONE (best epoch 41, val IoU 0.6543; test: mass 0.6640, whu 0.5943, inria 0.2090; 18 min) |
| 5 | 3 | seeds/unet_whu_seed43 | seed repeat | DONE (best epoch 44, val IoU 0.8868; test: whu 0.8681, inria 0.4033, mass 0.3847; 2.7 h) |
| 6 | 3 | seeds/unet_inria_seed43 | seed repeat | DONE (best epoch 48, val IoU 0.7290; test: inria 0.7168, whu 0.6971, mass 0.3294; 5.6 h) |
| 7 | 3 | seeds/unet_mass_seed44 | seed repeat | DONE (best epoch 50, val IoU 0.6516; test: mass 0.6498, whu 0.5728, inria 0.2053; 18 min) |
| 8 | 3 | seeds/unet_whu_seed44 | seed repeat | DONE (best epoch 24/early-stop@34, val IoU 0.8709; test: whu 0.8453, inria 0.2757, mass 0.3560; 1.7 h) |
| 9 | 3 | seeds/unet_inria_seed44 | seed repeat | DONE (best epoch 40, val IoU 0.7318; test: inria 0.7169, whu 0.6650, mass 0.3499; 5.6 h) |
| 10 | 4 | equal_budget/unet_whu_stepsB | --max_steps 3450 | DONE (best epoch 5, val IoU 0.8173; test: whu 0.8015, inria 0.2453, mass 0.3279; 19 min) |
| 11 | 4 | equal_budget/unet_inria_stepsB | --max_steps 3450 | DONE (best epoch 3, val IoU 0.6461; test: inria 0.6314, whu 0.5782, mass 0.3411; 19 min) |
| 12 | 5 | subset/unet_inria_subset548_s1 | 548-tile subset control | DONE (best epoch 46, val IoU 0.6193; test: inria 0.6078, whu 0.5314, mass 0.2913; 31 min) |
| 13 | 5 | subset/unet_inria_subset548_s2 | 548-tile subset control | DONE (best epoch 49, val IoU 0.5737; test: inria 0.5580, whu 0.4318, mass 0.2624; 31 min) |
| 14 | 5 | subset/unet_inria_subset548_s3 | 548-tile subset control | DONE (best epoch 46, val IoU 0.5962; test: inria 0.5871, whu 0.5236, mass 0.3093; 31 min) |
| 15 | 5 | subset/unet_whu_subset548_s1 | 548-tile subset control | DONE (best epoch 14/early-stop@24, val IoU 0.7593; test: whu 0.7363, inria 0.1771, mass 0.2669; 13 min) |
| 16 | 5 | subset/unet_whu_subset548_s2 | 548-tile subset control | DONE (best epoch 48, val IoU 0.8317; test: whu 0.8230, inria 0.2390, mass 0.3679; 25 min) |
| 17 | 5 | subset/unet_whu_subset548_s3 | 548-tile subset control | DONE (best epoch 49, val IoU 0.8221; test: whu 0.8010, inria 0.2504, mass 0.3234; 25 min) |

## Baseline matrix (Phase 2, seed 42) — training/matrices/baseline_seed42/

IoU:
| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.8624 | 0.3677 | 0.3959 |
| inria | 0.6968 | 0.7169 | 0.3052 |
| mass | 0.5687 | 0.1995 | 0.6665 |

RTD %:
| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.0000 | 57.36 | 54.10 |
| inria | 2.79 | 0.0000 | 57.42 |
| mass | 14.67 | 70.07 | 0.0000 |

## Equal-budget matrix (Phase 4, Protocol B, B=3450 steps) — training/matrices/equal_budget/

IoU:
| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.8015 | 0.2453 | 0.3279 |
| inria | 0.5782 | 0.6314 | 0.3411 |
| mass | 0.5687 | 0.1995 | 0.6665 |

RTD %:
| train\test | whu | inria | mass |
|---|---|---|---|
| whu | 0.0000 | 69.40 | 59.09 |
| inria | 8.43 | 0.0000 | 45.98 |
| mass | 14.67 | 70.07 | 0.0000 |

## Notes / incidents
- baseline_mass_seed42: first launch (PID 504015) died at start of epoch 2 —
  killed by the tool-call timeout because it was launched inside a shell call
  that also slept. Epoch 1 completed and last.pt was saved. Relaunched with
  setsid + --resume runs/baseline/unet_mass_seed42/last.pt (epoch 2).
