# Results

The recorded forecasts, trained models and scores the manuscript reports, in
the layout the code writes. Commands run from the repository root.

## Scores and comparisons

`python evaluation.py` writes:

- `metrics_countries.csv`: the scores of every model's forecasts, per origin
  and population.
- `paired_comparisons.csv`: the paired comparisons between models on the 144
  populations with complete outcomes at every origin (NeuralTFR against each
  benchmark, with Holm's adjustment, and the spline against BayesTFR).
- `group_comparisons.csv`: the same comparisons within the fertility groups at
  each origin.

`figures/` holds the tables behind the figures and the figures as published
(see its README).

## neuraltfr/

The network's runs, written by `python -m neuraltfr.runner`: thirty historical
runs (origins 1995, 2000 and 2006, ten seeds each) and ten from 2023. For each
stage, the test forecasts (`<stage>_predictions.parquet`), the selection curves
(`<stage>_curve.csv`), one record per run with its epoch, sizes and time
(`<stage>_runs.csv`), and the trained models (`models/`).

Training durations are selected on validation histories that end in a reported
value. Two 1995 runs (seeds 1236 and 1237) and the ten 2023 runs were
fitted on a second machine (Windows, torch 2.10); the others on the main one
(Windows 11, torch 2.14). There the runner reproduces its own runs bit for bit
and picks the same epochs for the others, whose forecasts it matches to a few
thousandths of a birth per woman per fit and about 1e-4 in the ten-fit
ensemble.

## baselines/

- `drift/`: `python -m baselines.drift`; deterministic.
- `linear/` and `spline/`: `python -m baselines.regression`; the selection of
  k and of the penalty (`selection.csv`), the refits at each origin (`eval_*`),
  the projection from 2023 (`forecast_*`) and the fitted parameters (`models/`).
  The fits depend on the number of torch threads, which the runner fixes to the
  recorded runs' setting; with it, the main machine reproduces them bit for
  bit. The linear regression's 2023 projection was fitted on the second machine
  and its parameters were not saved.

## bayestfr/

BayesTFR re-estimated at each historical origin (inputs and configuration in
`../bayestfr/README.md`): the package's projection summaries
(`summary_<origin>.csv`, kept to fifteen years), its convergence diagnostics,
and `eval_predictions.parquet`, the summaries in the layout of the other
forecasts (`python -m bayestfr.tidy`). The MCMC runs are not repeated here.
