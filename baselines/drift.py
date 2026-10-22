"""Random walk with drift, the local benchmark.

Each country is forecast on the log TFR scale from its own history up to the
origin: the mean annual change of its last k years is carried forward, with k
between 5 and 10 chosen per country by how well it would have forecast those
years. Intervals add the quantiles of the rule's backtest errors, pooled
across countries; at a 15-year horizon each history allows one backtest. 

    python -m baselines.drift                      # from the repository root; writes results/baselines/drift/
    python -m baselines.drift --origins 2006 --out <folder>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from neuraltfr.preprocess import LOCATION, SCALE, annual_panel
from neuraltfr.runner import DATA, ENC_LEN, HORIZON, MIN_TARGET, ORIGINS, ROOT, prediction_frame
from neuraltfr.training import QUANTILES
from neuraltfr.windows import build_fold

OUT = ROOT / "results" / "baselines" / "drift"
K_RANGE = range(5, 11)
INNER_ORIGINS, INNER_STEP, MIN_HISTORY = 5, 3, 25


def point_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Continue the average annual change of the last k years, k chosen in sample."""
    best_k, best_error = K_RANGE[0], np.inf
    for k in K_RANGE:
        if history.size < 2 * k + 1:
            continue
        slope = (history[-k - 1] - history[-2 * k - 1]) / k
        error = np.abs(history[-k:] - (history[-k - 1] + slope * np.arange(1, k + 1))).mean()
        if error < best_error:
            best_k, best_error = k, error
    if history.size < best_k + 1:
        return np.full(horizon, history[-1])
    slope = (history[-1] - history[-best_k - 1]) / best_k
    return history[-1] + slope * np.arange(1, horizon + 1)


def error_quantiles(histories: list, horizon: int) -> np.ndarray:
    """(horizon, quantiles): quantiles of the rule's own errors at each step ahead, pooled."""
    errors = []
    for history in histories:
        for back in range(1, INNER_ORIGINS + 1):
            cut = history.size - back * INNER_STEP
            if MIN_HISTORY <= cut <= history.size - horizon:
                errors.append(history[cut:cut + horizon] - point_forecast(history[:cut], horizon))
    return np.quantile(np.array(errors), QUANTILES, axis=0).T


def forecast(panel: pd.DataFrame, truth: pd.DataFrame, origin: int, horizon: int = HORIZON) -> pd.DataFrame:
    """Quantile paths in TFR for the origin's cohort, with the reported outcomes beside them."""
    windows = build_fold(panel, truth, origin, ENC_LEN, horizon, MIN_TARGET, valid_gap=0, balanced=False).test
    log_tfr = {country: group["y"].to_numpy(np.float64) * SCALE + LOCATION
               for country, group in panel.groupby("id")}
    histories = [log_tfr[country] for country in windows.ids]
    spread = error_quantiles(histories, horizon)
    paths = np.exp(np.stack([point_forecast(history, horizon)[:, None] + spread for history in histories]))
    balanced = windows.ids[windows.mask.all(axis=1)]
    return prediction_frame(windows, paths, balanced, "test", 0).assign(period=origin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origins", type=int, nargs="+", default=list(ORIGINS))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    observations = pd.read_csv(DATA)
    truth = annual_panel(observations, interpolate=False)
    frames = []
    for origin in args.origins:
        panel = annual_panel(observations[observations["year"] <= origin])
        frames.append(forecast(panel, truth, origin))
        print(f"origin {origin}: {frames[-1]['id'].nunique()} countries")
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(args.out / "eval_predictions.parquet",
                                                    compression="zstd", index=False)
    print(f"written {args.out / 'eval_predictions.parquet'}")


if __name__ == "__main__":
    main()
