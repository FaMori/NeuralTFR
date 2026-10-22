"""Pooled quantile regressions: one map from recent history to future fertility.

Both read the level at the end of each of the network's training windows and
its k most recent annual changes, and predict the five quantiles of the next
HORIZON years with coefficients shared across countries: the median directly,
the gaps between quantiles through softplus, so that they cannot cross. The
linear regression uses the predictors as they are; the spline turns each into
a linear spline with knots at its training deciles and penalises the second
differences of its coefficients across forecast years. Both minimise the
network's quantile loss with full-batch Adam.

    python -m baselines.regression --model spline --stage all --jobs 4    # from the repository root
    python -m baselines.regression --model linear --stage eval --out <folder>
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from neuraltfr.preprocess import annual_panel, to_tfr
from neuraltfr.runner import (DATA, ENC_LEN, FORECAST_ORIGIN, HORIZON, MIN_TARGET, ORIGINS, ROOT,
                              VALID_GAP, predict, prediction_frame, validation_loss)
from neuraltfr.training import QUANTILES, quantile_loss
from neuraltfr.windows import Windows, build_fold, validation_fold

RESULTS = ROOT / "results" / "baselines"
STEPS, LR = 4000, 0.05
THREADS = {"linear": 4, "spline": 2}                     # the recorded runs' settings
KNOTS = {"linear": 0, "spline": 9}
K_GRID = {"linear": (0, 1, 3), "spline": (0, 1, 2, 3, 11)}
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
K, LAMBDA = 3, {"linear": 0.0, "spline": 0.1}          # read from selection.csv, by hand
MEDIAN = QUANTILES.index(0.5)
KEYS = ["period", "k", "lam"]


# --- the model ---------------------------------------------------------------

def predictors(history: np.ndarray, k: int) -> np.ndarray:
    """(n, k + 1): the level at the end of each history and its k most recent annual changes."""
    history = history.astype(np.float64)
    columns = [history[:, -1]] + [history[:, -1 - lag] - history[:, -2 - lag] for lag in range(k)]
    return np.column_stack(columns)


def expand(predictors: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Each predictor followed by its hinges (x - knot)+, in order."""
    blocks = [np.column_stack([column[:, None], np.maximum(column[:, None] - row[None, :], 0.0)])
              for column, row in zip(predictors.T, knots)]
    return np.column_stack(blocks)


class Basis:
    """Each predictor as a linear spline, centred and scaled; ``knots`` is (predictors, knots per predictor).

    With no knots the terms are the predictors themselves.
    """

    def __init__(self, knots: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> None:
        self.knots, self.mean, self.sd = knots, mean, sd

    @classmethod
    def fit(cls, predictors: np.ndarray, n_knots: int) -> Basis:
        """Knots at the deciles 0.1 ... 0.9 of each predictor; moments of the resulting terms."""
        knots = np.quantile(predictors, np.linspace(0.1, 0.9, n_knots), axis=0).T
        terms = expand(predictors, knots)
        sd = terms.std(0)
        sd[sd < 1e-6] = 1.0
        return cls(knots, terms.mean(0), sd)

    def __call__(self, predictors: np.ndarray) -> np.ndarray:
        return (expand(predictors, self.knots) - self.mean) / self.sd

    @property
    def width(self) -> int:
        return self.knots.size + self.knots.shape[0]


def ordered(raw: torch.Tensor) -> torch.Tensor:
    """Per year, the first output is the median; the others, through softplus, the gaps outward.

    The upper chain is built before the lower one: autograd accumulates the
    median's gradient in construction order, and this order reproduces the recorded fits.
    """
    median, gaps = raw[..., 0], nn.functional.softplus(raw[..., 1:]) + 1e-4
    upper, lower = [median], [median]
    for step in range(len(QUANTILES) - 1 - MEDIAN):
        upper.append(upper[-1] + gaps[..., MEDIAN + step])
    for step in range(MEDIAN):
        lower.append(lower[-1] - gaps[..., MEDIAN - 1 - step])
    return torch.stack(lower[:0:-1] + upper, dim=-1)


class QuantileRegression(nn.Module):
    """One linear layer from the basis to the HORIZON x 5 outputs, quantiles ordered by construction."""

    def __init__(self, basis: Basis, k: int, horizon: int = HORIZON) -> None:
        super().__init__()
        self.basis, self.k, self.horizon = basis, k, horizon
        self.linear = nn.Linear(basis.width, horizon * len(QUANTILES))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def design(self, history: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.basis(predictors(history.numpy(), self.k))).float()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return ordered(self.linear(self.design(history)).view(-1, self.horizon, len(QUANTILES)))


def roughness(model: QuantileRegression) -> torch.Tensor:
    """Sum of squared second differences, across adjacent years, of each output's coefficients."""
    coefficients = torch.cat([model.linear.weight, model.linear.bias[:, None]], dim=1)
    coefficients = coefficients.view(model.horizon, len(QUANTILES), -1)
    return (coefficients[2:] - 2 * coefficients[1:-1] + coefficients[:-2]).pow(2).sum()


def fit(model: QuantileRegression, windows: Windows, lam: float = 0.0) -> float:
    """Full-batch Adam from zero coefficients; returns the final quantile loss, without the penalty."""
    design = model.design(torch.from_numpy(windows.encoder))
    target, mask = torch.from_numpy(windows.target), torch.from_numpy(windows.mask)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=STEPS)
    for _ in range(STEPS):
        optimiser.zero_grad()
        prediction = ordered(model.linear(design).view(-1, model.horizon, len(QUANTILES)))
        loss = quantile_loss(prediction, target, mask)
        objective = loss + lam * roughness(model) if lam else loss
        objective.backward()
        optimiser.step()
        schedule.step()
    return loss.item()


def save(model: QuantileRegression, path: Path) -> None:
    np.savez_compressed(path, weight=model.linear.weight.detach().numpy(),
                        bias=model.linear.bias.detach().numpy(), mean=model.basis.mean,
                        sd=model.basis.sd, knots=model.basis.knots, horizon=model.horizon,
                        k=model.k, quantiles=np.asarray(QUANTILES))


def load(path: Path) -> QuantileRegression:
    saved = np.load(path)
    model = QuantileRegression(Basis(saved["knots"], saved["mean"], saved["sd"]),
                               int(saved["k"]), int(saved["horizon"]))
    with torch.no_grad():
        model.linear.weight.copy_(torch.from_numpy(saved["weight"]))
        model.linear.bias.copy_(torch.from_numpy(saved["bias"]))
    return model


# --- the stages --------------------------------------------------------------

def run(task: dict) -> dict:
    """One fit: scored on its validation windows (select) or forecasting the test cohort."""
    torch.set_num_threads(task["threads"])
    started = time.time()
    stage, origin, name, k, lam = task["stage"], task["origin"], task["model"], task["k"], task["lam"]

    observations = pd.read_csv(DATA)
    truth = annual_panel(observations, interpolate=False)
    panel = annual_panel(observations[observations["year"] <= origin])
    settings = dict(enc_len=ENC_LEN, horizon=HORIZON, min_target=MIN_TARGET)
    fold = build_fold(panel, truth, origin, valid_gap=0, balanced=False, **settings)
    balanced = fold.test.ids[fold.test.mask.all(axis=1)]
    if stage == "select":
        cut_panel = annual_panel(observations[observations["year"] <= origin - VALID_GAP])
        train, valid = validation_fold(cut_panel, truth, origin, balanced, valid_gap=VALID_GAP, **settings)
    else:
        train, valid = fold.train, None

    model = QuantileRegression(Basis.fit(predictors(train.encoder, k), KNOTS[name]), k)
    loss = fit(model, train, lam)
    row = {"period": origin, "k": k, "lam": lam, "n_train": len(train),
           "n_params": sum(p.numel() for p in model.parameters()), "train_loss": loss,
           "threads": task["threads"]}
    if stage == "select":
        row |= {"n_valid": len(valid), "valid_pinball": validation_loss(model, valid)}
        frames = {}
    else:
        paths = to_tfr(predict(model, fold.test))
        frames = {"predictions": prediction_frame(fold.test, paths, balanced, "test", 0)
                  .assign(period=origin, k=k, lam=lam)}
        out = Path(task["out"])
        (out / "models").mkdir(parents=True, exist_ok=True)
        save(model, out / "models" / f"{stage}_{origin}.npz")
    row |= {"wall_s": round(time.time() - started, 1)}
    return frames | {"runs": pd.DataFrame([row])}


def tasks(stage: str, name: str, k: int, lam: float, threads: int, out: Path) -> list:
    if stage == "select":
        grid = [(k_, 0.0) for k_ in K_GRID[name]]
        if name == "spline":
            grid += [(K, lam_) for lam_ in LAMBDAS]
        origins = ORIGINS
    else:
        grid = [(k, lam)]
        origins = ORIGINS if stage == "eval" else (FORECAST_ORIGIN,)
    return [dict(stage=stage, origin=origin, model=name, k=k_, lam=lam_, threads=threads, out=str(out))
            for origin in origins for k_, lam_ in grid]


def file_path(stage: str, name: str, out: Path) -> Path:
    if stage == "select":
        return out / "selection.csv"
    return out / f"{stage}_{name}.{'parquet' if name == 'predictions' else 'csv'}"


def record(stage: str, frames: dict, out: Path) -> None:
    """Add a fit's frames to the stage files, replacing any earlier rows of the same fit."""
    for name, frame in frames.items():
        path = file_path(stage, name, out)
        if path.exists():
            # round_trip: the default parser can move a float's last digit, and the file is rewritten
            old = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, float_precision="round_trip")
            incoming = set(map(tuple, frame[KEYS].to_numpy()))
            old = old[[key not in incoming for key in map(tuple, old[KEYS].to_numpy())]]
            frame = pd.concat([old, frame], ignore_index=True)
        if path.suffix == ".parquet":
            frame.to_parquet(path, compression="zstd", index=False)
        else:
            frame.to_csv(path, index=False)


def finished(stage: str, out: Path) -> set:
    path = file_path(stage, "runs", out)
    return set(map(tuple, pd.read_csv(path)[KEYS].to_numpy())) if path.exists() else set()


def run_stage(stage: str, todo: list, jobs: int, out: Path) -> None:
    done = finished(stage, out)
    todo = [task for task in todo if (task["origin"], task["k"], task["lam"]) not in done]
    print(f"[{stage}] {len(todo)} fits to do, {len(done)} already recorded in {out}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    if todo:
        with get_context("spawn").Pool(min(jobs, len(todo))) as pool:
            for result in pool.imap_unordered(run, todo):
                record(stage, result, out)
                row = result["runs"].astype(object).iloc[0]   # keeps the integer columns integers
                valid = f", valid {row.valid_pinball:.5f}" if "valid_pinball" in row else ""
                print(f"[{stage}] {row.period} k={row.k} lam={row.lam:g}: "
                      f"train {row.train_loss:.5f}{valid}, {row.wall_s:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(KNOTS), required=True)
    parser.add_argument("--stage", choices=("select", "eval", "forecast", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--lam", type=float, help="penalty for eval and forecast; default LAMBDA[model]")
    parser.add_argument("--threads", type=int, help="torch threads per fit; default THREADS[model]")
    parser.add_argument("--out", type=Path, help="default results/baselines/<model>/")
    args = parser.parse_args()
    lam = LAMBDA[args.model] if args.lam is None else args.lam
    threads = THREADS[args.model] if args.threads is None else args.threads
    out = args.out or RESULTS / args.model
    stages = ("select", "eval", "forecast") if args.stage == "all" else (args.stage,)
    for stage in stages:
        run_stage(stage, tasks(stage, args.model, args.k, lam, threads, out), args.jobs, out)


if __name__ == "__main__":
    main()
