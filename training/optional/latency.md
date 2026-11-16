# Inference latency measurement (no training)

GPU: NVIDIA L40S  |  batch size: 1  |  tile size: 512x512
Timed tiles per dataset: up to 100 (Mass test split has only 40 tiles, so 30 timed)
Warm-up tiles: 10  |  measured: forward pass + 0.5 threshold per tile

| dataset | checkpoints used | tiles timed | mean ms/tile | std ms/tile | tiles/sec |
|---|---|---|---:|---:|---:|
| whu | baseline unet seed-42 best.pt | 100 | 7.90 | 0.02 | 126.7 |
| inria | baseline unet seed-42 best.pt | 100 | 7.91 | 0.01 | 126.5 |
| mass | baseline unet seed-42 best.pt | 30 | 7.93 | 0.02 | 126.1 |

Interpretation: per-tile forward+threshold on a single 512x512 tile on one L40S GPU.
