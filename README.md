# NeuralTFR

**Authors:** Daniel Ciganda, Facundo Morini, Francisco Piriz, Henrik-Alexander Schubert,
Ugofilippo Basellini and Mikko Myrskylä

Code, data and trained models for **"Anticipating Continued Global Fertility
Decline via Neural Forecasting"**. NeuralTFR is a recurrent
encoder-decoder network that forecasts the total fertility rate (TFR) fifteen
years ahead, as five predictive quantiles, from twelve-year histories pooled
across 230 countries and territories.

The repository rebuilds the fertility panel from its raw inputs; fits
NeuralTFR, a random walk with drift and two pooled quantile regressions at the
historical origins 1995, 2000 and 2006 and from 2023; re-estimates BayesTFR at
the historical origins; and recomputes, from the recorded forecasts, every
score, paired comparison and figure table the manuscript reports. The recorded
forecasts and the trained models are included, so the evaluation can be
checked without refitting anything.

## Setup

Python 3.12 and the packages in `requirements.txt`, on CPU:

```bash
conda create -y -n neuraltfr python=3.12
conda activate neuraltfr
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Re-estimating BayesTFR also needs R with the `bayesTFR` package (the recorded
runs used bayesTFR 7.4.4; see `bayestfr/README.md`).

## Reproducing the results

Commands run from the repository root. The model runners are resumable: they
skip the runs already recorded in their output folder, so a full reproduction
writes to a new folder with `--out` and leaves `results/` as recorded.

| step | command | time |
|---|---|---|
| panel | `python data/build_panel.py` | seconds |
| NeuralTFR, historical origins and 2023 | `python -m neuraltfr.runner --stage all --jobs 10 --out <folder>` | 40 runs on one core each: about 5-9 minutes per historical run, 10-23 minutes per 2023 run |
| drift | `python -m baselines.drift --out <folder>` | seconds |
| spline regression | `python -m baselines.regression --model spline --stage all --jobs 4 --out <folder>` | 37 fits, 1-2 minutes each |
| linear regression | `python -m baselines.regression --model linear --stage select --out <folder>`, then `--stage eval` | 12 fits, about 1 minute each |
| BayesTFR | from `bayestfr/`: `Rscript run.R 1995`, `Rscript run.R 2000 13830` (phase III extended by 13,830 iterations to converge) and `Rscript run.R 2006`; then `python -m bayestfr.tidy` | about 17 hours per origin on three cores |
| scores and comparisons | `python evaluation.py` | seconds |
| figure and text tables | `python figures.py` | seconds |
| visualizer data | `python docs/prepare_data.py` | seconds |

`evaluation.py` and `figures.py` read the forecasts in `results/` and
reproduce the recorded scores and tables exactly. Refitting reproduces the
drift exactly. The regressions and the network are deterministic for a given
machine and environment, and the regressions also depend on the number of
torch threads, which `baselines/regression.py` fixes to the recorded runs'
setting: on the machine that recorded them, the runs regenerate bit for bit;
fits made elsewhere differ by up to about 1e-3 births per woman per fit, and
by about 1e-4 in the network's ten-fit ensemble. `results/README.md` says
which runs come from which machine.

## Data

`data/raw/` holds the external inputs, unchanged:

- `un_empirical_portal/EMPIRICAL_TFR.csv`: the UN Population Division's
  compilation of empirical fertility estimates;
- `new_sources/NEW_SOURCES_TFR.csv`: the estimates added from national
  statistical offices, Eurostat and the Economic Commission for Latin America
  and the Caribbean;
- `wpp2024/`: the World Population Prospects 2024 projections and 2023
  population estimates;
- `gbd2021/`: the GBD 2021 fertility forecasts.

## License

The code is released under the [Apache License 2.0](LICENSE). Data files
derived from external sources are redistributed only for reproducibility and
remain subject to the terms of their original providers.
