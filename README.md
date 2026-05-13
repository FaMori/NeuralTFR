# NeuralTFR

**WORKING IN PROGRESS**

Reproducibility code for the paper **"NeuralTFR: A neural ensemble approach to global Total Fertility Rate forecasting"**. The repository ships the model, the harmonized dataset, the scripts that produce every figure and metric in the manuscript, and an interactive visualizer.

NeuralTFR is an **encoder-decoder GRU ensemble** trained with a multi-quantile loss to forecast country-level TFR series. It is evaluated against a holdout period and benchmarked against established projections (WPP, WCDE, bayesTFR).

---

## Table of Contents

1. [Motivation](#motivation)
2. [Key Features](#key-features)
3. [Repository Structure](#repository-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Quick-Start Tutorial](#quick-start-tutorial)
6. [Interactive Visualizer](#interactive-visualizer)
7. [Citing this Work](#citing-this-work)
8. [License](#license)

---

## Motivation

Total Fertility Rate (TFR) projections underpin population forecasts that guide decades of demographic, public health, and economic policy. Established approaches — from the UN's probabilistic Bayesian methods to expert-based scenario models — either impose strong parametric assumptions about convergence dynamics or lack formal uncertainty quantification. At the same time, the empirical record of fertility transitions has grown rich enough to support purely data-driven learning.

NeuralTFR addresses this gap by training directly on a harmonized global panel of TFR series and learning temporal dynamics without committing to a pre-specified functional form for the demographic transition. By combining an encoder-decoder GRU ensemble with a multi-quantile loss, the framework produces calibrated prediction intervals that capture the heterogeneity of national trajectories. Systematic benchmarking against WPP, WCDE, and bayesTFR projections positions NeuralTFR as a principled, data-driven complement to established demographic methods.

---

## Key Features

- **Encoder-decoder GRU ensemble** with categorical embeddings for country/region identity.
- **Multi-quantile loss** (q = 0.05, 0.10, 0.50, 0.90, 0.95) producing calibrated prediction intervals.
- **Data augmentation** by oversampling recent windows of low-TFR series.
- **Two execution modes**: historical evaluation against a holdout, and future forecasting up to user-defined horizons.
- **Baseline comparison** against WPP, WCDE, bayesTFR and a naive-drift benchmark.
- **Interactive visualizer** ([docs/](docs/)) deployable as a GitHub Pages site.

---

## Repository Structure

```
NeuralTFR/
├── compute/                   # Reproducibility code
│   ├── run_neuraltfr.py       # CLI entry-point (eval / forecast / all)
│   ├── neuraltfr.py           # Model wrapper and training pipeline
│   ├── common/                # Dataset, preprocessing, trainer, losses, evaluation, Optuna
│   ├── models/                # ENC_DEC_GRU and NAIVE_DRIFT baselines
│   ├── data/                  # Raw and harmonized TFR series (WPP, GBD, empirical)
│   ├── results/               # Outputs reproducing the figures and metrics of the paper
│   └── GUIDE.md               # Short usage guide
├── docs/                      # Interactive visualizer
├── paper/                     # Manuscript: LaTeX source, bibliography, figures, compiled PDF
├── environment.yml            # Conda environment specification
├── LICENSE                    # Apache 2.0
└── README.md
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/NeuralTFR.git
cd NeuralTFR
```

### 2. Create the Conda environment

The environment is fully specified in [environment.yml](environment.yml) and pins Python 3.12 with PyTorch, scikit-learn, scipy, matplotlib, seaborn, joblib, tqdm and optuna.

```bash
conda env create -f environment.yml
conda activate neuraltfr
```

### 3. Verify the install

From the repository root, run a short sanity check:

```bash
python -c "import torch, sklearn, optuna; print('torch', torch.__version__)"
```

> **GPU note:** the default `pytorch` build from the `pytorch` channel installs the CPU version. The CLI runs comfortably on CPU; if you want CUDA, replace the `pytorch` dependency in `environment.yml` with the appropriate `pytorch` / `pytorch-cuda` combination for your driver (see [pytorch.org](https://pytorch.org/get-started/locally/)).

---

## Quick-Start Tutorial

All commands are run from the [compute/](compute/) directory:

```bash
cd compute
```

### Run a historical evaluation

Trains the ensemble on data up to `--eval-split-year` (default 2009) and scores predictions against the observed holdout.

```bash
python run_neuraltfr.py --mode eval
```

Outputs:

- `results/evaluation/metrics/metrics.csv` — per-series scores (RMSE, RMSSE, sMAPE, CRPS, 90% coverage and MPIW).
- `results/evaluation/predictions/predictions.csv` — point and quantile predictions.
- `results/evaluation/predictions/plot_evaluation.pdf` — per-country prediction plots.

### Run a forecast to the future

Trains on all available history and projects forward.

```bash
python run_neuraltfr.py --mode forecast
```

Outputs land in `results/forecast/predictions/`.

### Run both in a single call

```bash
python run_neuraltfr.py --mode all
```

### Hyperparameter search (optional)

```bash
python common/run_optuna.py --search-mode minimize --n-trials 10
```

For the full list of CLI flags and a more detailed walkthrough, see [compute/GUIDE.md](compute/GUIDE.md).

---

## Interactive Visualizer

**WORKING IN PROGRESS**

---

## Citing this Work

**WORKING IN PROGRESS**

---

## License

This project is released under the [Apache License 2.0](LICENSE).