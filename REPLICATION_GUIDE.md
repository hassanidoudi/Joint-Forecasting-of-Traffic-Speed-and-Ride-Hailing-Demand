# Chicago Loop STT-CTA — Standalone Replicability Package

A fully self-contained package for reproducing the results in "Learning Urban
Dynamics via Joint Forecasting of Traffic Speed and Ride-Hailing Demand" (hEART
short paper, accepted 2026) and the related ITSC 2026 paper. **No dependency on
any other Drive folder** — everything the pipeline needs lives inside this one.

## Folder structure (final, paper-matching pipeline)

```
chicago-stt-cta-replicability/
├── README.md                    <- project README (architecture, usage)
├── REPLICATION_GUIDE.md          <- this file
├── DATA_LICENSE.md                <- data/model licensing detail
├── LICENSE                          <- MIT, code only
├── config.py                          <- hyperparameters, paths, 6x6 grid config
├── data_preprocessing.py                <- raw CSVs -> fused (T,H,W,F) tensor
├── fetch_weather.py                       <- pulls weather_loop_2025.csv from Open-Meteo
├── dataset.py                               <- PyTorch Dataset + train/val/test split
├── dataset_v2.py                              <- lag-feature dataloaders for the final joint model
├── model.py                                     <- shared transformer building blocks
├── model_v4.py                                    <- STT-CTA final architecture (lag-mixture + 3-seed ensemble = "joint")
├── baselines.py                                     <- HistoricalAverage, SingleTask, IndependentDual
├── train_v8.py                                        <- training loop (--model {name}), final pipeline
├── evaluate_v8.py                                       <- test-set metrics + all figures, final pipeline
├── evaluate_ha_vs_joint.py                                <- focused HA-vs-joint comparison (headline claim) — needs re-run, see README's "Open items"
├── Boundaries_-_Community_Areas_20260209.csv                <- Loop polygon (raw data)
├── Chicago_Traffic_Tracker.csv                                <- traffic sensor readings (raw data)
├── Transportation_Network_Providers_-_Trips_(2025-)_20260209.csv <- TNP trips (raw data)
├── weather_loop_2025.csv                                        <- hourly temp/precip (raw data)
├── output/
│   ├── evaluation_results_v8.csv    <- final metrics table (all 4 models x all segments) — this is the authoritative one
│   ├── checkpoints/                   <- trained weights + loss histories (joint_best.pt = model_v4 ensemble)
│   └── figures/
├── notebooks/
│   └── 08_regenerate_all_models_v8.ipynb  <- drives the whole pipeline end-to-end (executed, output-bearing copy)
└── archive/                                 <- superseded script/notebook versions (model.py-era joint, train_v2..v7,
                                                 evaluate.py/evaluate_v2.py, dev/diagnostic scripts, old results). Kept
                                                 for provenance only — not part of the reproduction path.
```

This mirrors what `config.py` expects out of the box: `DATA_DIR` = the
script's own folder, `CHECKPOINT_DIR` = `output/checkpoints`, `FIGURE_DIR` =
`output/figures`. No path edits needed anywhere.

## How to run

1. Open `notebooks/08_regenerate_all_models_v8.ipynb` in Colab.
2. Set `PROJECT_DIR` to wherever you've placed this folder in your Drive.
3. Run top to bottom:
   - Skips `data_preprocessing.py` if processed tensors already exist
   - Trains `speed_only`, `demand_only`, `independent_dual` via `baselines.py` (unchanged), and `joint` via `model_v4.py`'s 3-member lag-mixture ensemble
   - Runs `evaluate_v8.py`, writing `output/evaluation_results_v8.csv` and the figures
   - Displays the regenerated figures inline

To just inspect results without running anything: open `output/evaluation_results_v8.csv`
and `output/figures/*.png` directly.

## Sharing this package

Because everything is self-contained and flat (aside from `archive/`, which
can be dropped for a cleaner mirror), you can zip this whole folder, hand it
to a collaborator, or move it to another Drive account, and it will work
unmodified as long as the relative structure above is preserved.

## Provenance

- Final joint model (`model_v4`, 3-seed lag-mixture ensemble): trained 2026-08-06.
- `output/evaluation_results_v8.csv` last regenerated: 2026-08-06.
- Package assembled: 2026-07-26, from the original working project.
- Package reorganized around the final `model_v4`/`train_v8`/`evaluate_v8` pipeline and superseded versions archived: 2026-08-26.

## Known open items

See the "Open items" section of `README.md` — the HA-vs-joint headline comparison needs re-running against the current checkpoint, the duplicate `ckpt/` folder needs consolidating with `output/checkpoints/`, and `paper.tex` is not yet in this folder.
