# NeuralTFR Usage Guide

This is a short guide to the main workflow of the project.

## Main entry point

The main script is:

`run_neuraltfr.py`

It supports three modes:

- `--mode eval`: train on historical data and evaluate on a holdout period.
- `--mode forecast`: train and generate future forecasts.
- `--mode all`: run evaluation and forecast in one execution.

## Recommended workflow

1. Make sure the main dataset is available at `data/final/tfr_smooth.csv`.
2. Run evaluation first to check historical performance.
3. Review the saved metrics and prediction files.
4. Run forecast once the setup looks correct.
5. Optionally explore a few hyperparameter configurations.

## Basic commands

Run historical evaluation:

```bash
python run_neuraltfr.py --mode eval
```

Run future forecasting:

```bash
python run_neuraltfr.py --mode forecast
```

Run both in one call:

```bash
python run_neuraltfr.py --mode all
```

## What evaluation does

Evaluation uses data before `eval_split_year` for training and compares predictions against the later observed period.

Main outputs:

- `results/evaluation/metrics/metrics.csv`
- `results/evaluation/predictions/predictions.csv`
- `results/evaluation/predictions/plot_evaluation.pdf`

## What forecast does

Forecasting trains the model on the available history and produces forward projections.

Main outputs:

- `results/forecast/predictions/predictions.csv`
- `results/forecast/predictions/plot_forecast.pdf`

## Hyperparameter search

For a very simple hyperparameter search, use:

`common/run_optuna.py`

Monte Carlo example:

```bash
python common/run_optuna.py --search-mode montecarlo --n-trials 5
```

Optuna minimize example:

```bash
python common/run_optuna.py --search-mode minimize --n-trials 10
```

By default, the search evaluates configurations using `loss` on the holdout period.

Main outputs:

- `results/hyperparam_search/<study_name>/trials.csv`
- `results/hyperparam_search/<study_name>/best_config.json`

## Main files to know

- `run_neuraltfr.py`: main script for training, evaluation, and forecasting.
- `neuraltfr.py`: core model wrapper and training pipeline.
- `common/run_optuna.py`: simple hyperparameter search script.
- `common/evaluation.py`: evaluation metrics and plots.

## Minimal starting point

If you only want one command to begin with, use:

```bash
python run_neuraltfr.py --mode eval
```
