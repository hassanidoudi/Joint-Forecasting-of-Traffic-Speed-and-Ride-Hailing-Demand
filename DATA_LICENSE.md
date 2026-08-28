# Data & Model Licensing

This project mixes three categories of content with three different licensing
situations. `LICENSE` at the repo root (MIT) covers **code only**. This file
documents everything else.

## 1. Raw third-party data (NOT relicensed — original terms apply)

These raw CSVs are included in this folder purely for end-to-end
replicability (so `data_preprocessing.py` runs without external downloads).
They are **not owned by this project** and remain governed by their original
publishers' terms — nothing here grants any additional rights beyond those
terms, and this project makes no license claim over their content.

| File | Source | Terms |
|---|---|---|
| `Chicago_Traffic_Tracker.csv` | City of Chicago Data Portal | [Chicago Data Portal Terms of Use](https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html) — public, no formal open-data license asserted by the publisher; provided "as is," and the City reserves the right to require removal of the data from any downstream use. |
| `Transportation_Network_Providers_-_Trips_(2025-)_*.csv` | City of Chicago Data Portal | Same as above. |
| `Boundaries_-_Community_Areas_*.csv` | City of Chicago Data Portal | Same as above. |
| `weather_loop_2025.csv` | [Open-Meteo](https://open-meteo.com/) (via `fetch_weather.py`) | [CC BY 4.0](https://open-meteo.com/en/license) — redistribution is permitted with attribution. **Attribution:** Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed under CC BY 4.0. |

**Practical implications:**
- If you redistribute this repository (e.g. a public GitHub mirror), you may
  need to re-verify the Chicago Traffic Tracker / TNP Trips / Community Areas
  datasets are still permitted for redistribution at that time, since the
  City's terms allow them to require removal and do not assert a fixed open
  license (e.g. CC-BY, ODbL, PDDL). If in doubt, point collaborators to the
  City's dataset pages instead of shipping the CSVs directly.
- The `weather_loop_2025.csv` file can be freely redistributed as long as the
  Open-Meteo attribution above stays attached to it.

## 2. Code (MIT — see `LICENSE`)

All `.py` scripts, notebooks (`.ipynb`), and markdown documentation authored
for this project (`config.py`, `dataset.py`, `model.py`, `baselines.py`,
`train*.py`, `evaluate*.py`, everything under `notebooks/`, this file,
`README.md`, `REPLICATION_GUIDE.md`) are MIT licensed. Use, modify, and
redistribute freely with attribution.

## 3. Trained model checkpoints, figures, and result CSVs

Everything under `output/` (`checkpoints/*.pt`, `figures/*.png`,
`evaluation_results*.csv`, `ha_vs_joint_results.csv`) is a derived artifact
produced by running the MIT-licensed code against the raw data above. These
are released under **CC BY 4.0** — free to use, share, and adapt with
attribution to this repository and the associated paper (citation in
`README.md`).

Note that the checkpoints were trained on data covered by the City of
Chicago's terms in §1; if the City of Chicago dataset terms change in a way
that affects downstream derived works, that could in principle affect reuse
of the checkpoints too. This is a standard caveat for any model trained on
municipal open data and is not a known current restriction.

## Questions

For anything not covered here, or for uses outside these terms, contact the
authors (see `README.md`).
