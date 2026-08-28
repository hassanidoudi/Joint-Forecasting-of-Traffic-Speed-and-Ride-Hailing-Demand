# Joint-Forecasting-of-Traffic-Speed-and-Ride-Hailing-Demand
the paper's code
# chicago-stt-cta-replicability

Replication code and results for:

**Learning Urban Dynamics via Joint Forecasting of Traffic Speed and Ride-Hailing Demand** — Spatial-Temporal Transformer with Cross-Task Attention (STT-CTA) — Chicago Loop case study (hEART short paper, accepted 2026)

This repository reproduces every number and figure reported in the paper by re-running the training and evaluation pipeline end-to-end:
`data_preprocessing.py` → `train_v8.py` (4 models) → `evaluate_v8.py` → comparison tables.

## Contents (final, paper-matching pipeline)

```
config.py                  # model / training hyperparameters
dataset.py                 # Chicago Loop dataset construction (traffic + TNP trip data)
dataset_v2.py               # lag-feature dataloaders used by the final joint model
model.py                   # shared transformer building blocks (used by baselines.py)
model_v4.py                 # STT-CTA final architecture: lag-mixture + 3-seed ensemble (this is "joint")
baselines.py                # speed_only, demand_only, independent_dual, historical_average
train_v8.py                 # training entry point — the exact pipeline used for the paper's numbers
evaluate_v8.py               # generates the metrics tables and figures below
evaluate_ha_vs_joint.py       # focused HA-vs-joint comparison (headline claim) — see note below
data_preprocessing.py        # raw CSV -> processed tensors
fetch_weather.py             # pulls weather_loop_2025.csv from Open-Meteo
output/                     # final metrics CSVs + figures + checkpoints
notebooks/                  # end-to-end Colab notebook
archive/                    # superseded script/notebook versions, kept for provenance only — not part of the reproduction path
```

## Models reproduced

| Model | Description |
|---|---|
| `historical_average` | Non-parametric seasonal-average baseline |
| `speed_only` | Single-task model, speed head only |
| `demand_only` | Single-task model, demand head only |
| `independent_dual` | Two single-task models trained independently, no shared representation |
| `joint` (STT-CTA) | Proposed model — `model_v4.py`, lag-mixture decoder + 3-seed ensemble, shared encoder with cross-task attention |

## Reproducing the results

```bash
python data_preprocessing.py           # builds processed_data.npz (skips if already present)

python train_v8.py --model speed_only
python train_v8.py --model demand_only
python train_v8.py --model independent_dual
python train_v8.py --model joint        # trains the 3-member lag-mixture ensemble

python evaluate_v8.py                   # writes output/evaluation_results_v8.csv and figures
```

Or run `notebooks/08_regenerate_all_models_v8.ipynb` in Colab, which drives the same scripts end-to-end on a GPU runtime (T4 or better) and is the executed, output-bearing copy of the notebook.

## A note on reproducibility

Re-running training will **not** reproduce the paper's numbers bit-for-bit. Small differences (typically <2% relative on MAE/RMSE/MAPE across all models and horizons) are expected and are due to **random seed variance** in weight initialization, data loader shuffling, and dropout sampling. This is normal for stochastic gradient-based training and does not indicate an error in the pipeline or a discrepancy with the reported method.

## Open items (not yet resolved — see repository owner before treating as final)

- `evaluate_ha_vs_joint.py`'s last output (now in `archive/ha_vs_joint_results.csv`) was generated against the earlier `model.py`/`model_v2.py` joint checkpoint, predating the final `model_v4` ensemble. It needs to be re-run against the current `output/checkpoints/joint_best.pt` (architecture `model_v4`) before the headline HA-vs-joint comparison number can be cited.
- There are two checkpoint locations at the moment — a root-level `ckpt/` folder and `output/checkpoints/`. These should be consolidated to avoid ambiguity about which checkpoints are canonical.
- `paper.tex` is referenced in `REPLICATION_GUIDE.md`'s folder tree but is not currently in this folder.

## Citation

If you use this code, please cite the paper (citation to be added upon publication).

## License

Code: MIT (see LICENSE). Data and derived artifacts: see `DATA_LICENSE.md`.