"""Backtests and projections of NeuralTFR: the paper's recorded runs.

One run is one (origin, seed) pair and follows a two-fit protocol:

    A. Selection. Training targets stop VALID_GAP years before the origin and
       the gap is forecast as nested validation, from the histories that end
       in a reported value at the cut, so that none has been interpolated
       toward an observation inside the gap. Every CHECK_EVERY epochs the
       validation forecast is scored by its mean quantile loss on the TFR
       scale; the epoch kept is the first that the next PATIENCE checkpoints
       fail to improve on. The test period never enters this choice.
    B. Deliverable. A fresh model with the same seed is trained on everything
       up to the origin for exactly the chosen number of epochs; its forecast
       from the origin is what gets recorded.

Stage ``eval`` runs the historical origins against the reported outcomes;
stage ``forecast`` runs the projection from FORECAST_ORIGIN. Each stage writes
to the output folder:

    <stage>_predictions.parquet   quantile paths in TFR with the reported
                                  value beside them: the validation forecast
                                  of model A (row_set valid) and the test
                                  forecast of model B (row_set test)
    <stage>_curve.csv             the selection curve of each run
    <stage>_runs.csv              one row per run: chosen epoch, sizes, time
    models/<stage>_<origin>_s<seed>.pt   model B

Runs already in <stage>_runs.csv are skipped, so a stage can be resumed.

    python -m neuraltfr.runner --stage all --jobs 10      # from the repository root
"""
from __future__ import annotations

import argparse
import time
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .model import NeuralTFR, save
from .preprocess import annual_panel, to_tfr
from .training import COLUMNS, QUANTILES, train
from .windows import Windows, build_fold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "empirical_tfr.csv"
RESULTS = ROOT / "results" / "neuraltfr"

HIDDEN_SIZE, ENC_LEN, HORIZON, MIN_TARGET = 16, 12, 15, 5
LR, WEIGHT_DECAY, BATCH_SIZE = 3e-4, 1e-4, 64
BUDGET, CHECK_EVERY, PATIENCE, VALID_GAP = 200, 10, 5, 10
ORIGINS, FORECAST_ORIGIN, SEEDS = (1995, 2000, 2006), 2023, range(1234, 1244)
KEYS = ["period", "seed"]


def run(task: dict) -> dict:
    """One (origin, seed) run of the two-fit protocol; returns the frames to record."""
    torch.set_num_threads(1)
    started = time.time()
    stage, origin, seed, budget = task["stage"], task["origin"], task["seed"], task["budget"]

    observations = pd.read_csv(DATA)
    panel = annual_panel(observations[observations["year"] <= origin])
    truth = annual_panel(observations, interpolate=False)
    settings = dict(enc_len=ENC_LEN, horizon=HORIZON, min_target=MIN_TARGET)
    fold = build_fold(panel, truth, origin, valid_gap=VALID_GAP, balanced=stage == "eval", **settings)
    refit_fold = build_fold(panel, truth, origin, valid_gap=0, balanced=False, **settings)

    # A. selection: train for the whole budget, score every CHECK_EVERY epochs
    model = build_model(fold.train, seed)
    curve, checkpoints = {}, {}

    def score(epoch, network):
        if epoch % CHECK_EVERY == 0:
            curve[epoch] = validation_loss(network, fold.valid)
            checkpoints[epoch] = deepcopy(network.state_dict())

    train(model, fold.train.loader(BATCH_SIZE, generator(seed)), budget, LR, WEIGHT_DECAY, score)
    epoch = chosen_epoch(curve)
    model.load_state_dict(checkpoints[epoch])

    # B. deliverable: same seed, all data up to the origin, the chosen epochs
    deliverable = build_model(refit_fold.train, seed)
    train(deliverable, refit_fold.train.loader(BATCH_SIZE, generator(seed)), epoch, LR, WEIGHT_DECAY)

    balanced = refit_fold.test.ids[refit_fold.test.mask.all(axis=1)]
    valid_paths = to_tfr(predict(model, fold.valid)).astype(np.float32)        # recorded in single precision
    test_paths = to_tfr(predict(deliverable, refit_fold.test)).astype(np.float32)
    predictions = pd.concat([prediction_frame(fold.valid, valid_paths, balanced, "valid", VALID_GAP),
                             prediction_frame(refit_fold.test, test_paths, balanced, "test", 0)],
                            ignore_index=True)
    out = Path(task["out"])
    (out / "models").mkdir(parents=True, exist_ok=True)
    save(deliverable, out / "models" / f"{stage}_{origin}_s{seed}.pt")

    keys = {"period": origin, "seed": seed}
    return {"predictions": predictions.assign(**keys),
            "curve": pd.DataFrame({**keys, "checkpoint": list(curve), "valid_pinball": list(curve.values())}),
            "runs": pd.DataFrame([{**keys, "epoch": epoch, "at_boundary": int(epoch == max(curve)),
                                   "budget": budget, "n_train_windows": len(refit_fold.train),
                                   "n_test_countries": len(refit_fold.test),
                                   "n_params": sum(p.numel() for p in deliverable.parameters()),
                                   "wall_s": round(time.time() - started, 1)}])}


def build_model(training: Windows, seed: int) -> NeuralTFR:
    torch.manual_seed(seed)
    nn.GRU(1, HIDDEN_SIZE, batch_first=True)   # the recorded fits drew their initial weights after this layer
    return NeuralTFR(HORIZON, HIDDEN_SIZE, float(np.diff(training.encoder, axis=1).std()))


def generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def chosen_epoch(curve: dict) -> int:
    """First checkpoint that the following PATIENCE checkpoints fail to improve on."""
    best = None
    for index, (epoch, value) in enumerate(sorted(curve.items())):
        if best is None or value < curve[best]:
            best, best_at = epoch, index
        elif index - best_at >= PATIENCE:
            break
    return best


def predict(model: NeuralTFR, windows: Windows) -> np.ndarray:
    """Quantile paths on the model scale, cut to the windows' target length."""
    model.eval()
    with torch.no_grad():
        paths = model(torch.from_numpy(windows.encoder)).numpy()
    return paths[:, :windows.target.shape[1]]


def validation_loss(model: NeuralTFR, windows: Windows) -> float:
    """Mean quantile loss on the TFR scale over the reported cells."""
    error = to_tfr(predict(model, windows)) - to_tfr(windows.target)[..., None]
    quantiles = np.asarray(QUANTILES)
    loss = np.maximum(quantiles * -error, (1.0 - quantiles) * error)
    return float(loss[windows.mask > 0].mean())


def prediction_frame(windows: Windows, paths: np.ndarray, balanced: np.ndarray,
                     row_set: str, gap: int) -> pd.DataFrame:
    """One row per window and target year: the reported value and the quantile paths in TFR."""
    truth = np.where(windows.mask > 0, to_tfr(windows.target), np.nan)
    count, width = truth.shape
    frame = pd.DataFrame({"id": np.repeat(windows.ids, width),
                          "year": windows.years.reshape(-1),
                          "horizon": np.tile(np.arange(1, width + 1), count),
                          "in_balanced": np.repeat(np.isin(windows.ids, balanced), width),
                          "y_true": truth.reshape(-1).astype(np.float32)})
    for index, column in enumerate(COLUMNS):
        frame[column] = paths[..., index].reshape(-1)
    return frame.assign(row_set=row_set, gap=gap)


def record(stage: str, frames: dict, out: Path) -> None:
    """Add a run's frames to the stage files, replacing any earlier rows of the same run."""
    for name, frame in frames.items():
        path = out / f"{stage}_{name}.{'parquet' if name == 'predictions' else 'csv'}"
        if path.exists():
            # round_trip: the default parser can move a float's last digit, and the file is rewritten
            old = pd.read_parquet(path) if name == "predictions" else pd.read_csv(path, float_precision="round_trip")
            incoming = set(map(tuple, frame[KEYS].to_numpy()))
            old = old[[key not in incoming for key in map(tuple, old[KEYS].to_numpy())]]
            frame = pd.concat([old, frame], ignore_index=True)
        if name == "predictions":
            frame.to_parquet(path, compression="zstd", index=False)
        else:
            frame.to_csv(path, index=False)


def finished(stage: str, out: Path) -> set:
    path = out / f"{stage}_runs.csv"
    return set(map(tuple, pd.read_csv(path)[KEYS].to_numpy())) if path.exists() else set()


def run_stage(stage: str, tasks: list, jobs: int, out: Path) -> None:
    done = finished(stage, out)
    tasks = [task for task in tasks if (task["origin"], task["seed"]) not in done]
    print(f"[{stage}] {len(tasks)} runs to do, {len(done)} already recorded in {out}")
    if not tasks:
        return
    out.mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(min(jobs, len(tasks))) as pool:
        for result in pool.imap_unordered(run, tasks):
            record(stage, result, out)
            row = result["runs"].astype(object).iloc[0]   # keeps the integer columns integers
            print(f"[{stage}] {row.period} s{row.seed}: epoch {row.epoch}, {row.wall_s:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("eval", "forecast", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--origins", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    for stage in ("eval", "forecast") if args.stage == "all" else (args.stage,):
        origins = args.origins or (ORIGINS if stage == "eval" else (FORECAST_ORIGIN,))
        tasks = [{"stage": stage, "origin": origin, "seed": seed, "budget": args.budget,
                  "out": str(args.out)} for origin in origins for seed in args.seeds]
        run_stage(stage, tasks, args.jobs, args.out)


if __name__ == "__main__":
    main()
