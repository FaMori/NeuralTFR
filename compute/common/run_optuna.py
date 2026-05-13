from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.evaluation import eval_models
from neuraltfr import NeuralTFR

import optuna


QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95]
SEARCH_SPACE = {
    "enc_len": [15, 20, 25],
    "hidden_size": [8, 12, 16, 24],
    "dim_embedding": [4, 6, 8],
    "batch_size": [8, 16, 32],
    "dropout": [0.0, 0.1, 0.2],
    "lr": (1e-4, 1e-3),
    "weight_decay": (1e-6, 1e-4),
}

QUANTILE_COLUMNS = [(q, f"y_hat_{int(q * 100):02d}") for q in QUANTILES]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Basic NeuralTFR hyperparameter search using Optuna minimization or Monte Carlo sampling."
    )

    parser.add_argument(
        "--search-mode",
        type=str,
        choices=["minimize", "montecarlo"],
        default="montecarlo",
        help="Search mode: 'minimize' uses Optuna, 'montecarlo' samples random configurations.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="neuraltfr_search_demo",
        help="Name used to save search artifacts.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/final/tfr_smooth.csv",
        help="Path to the main TFR dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/hyperparam_search",
        help="Directory where the search results will be saved.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["loss", "rmse", "smape", "rmsse", "crps"],
        default="loss",
        help="Metric to minimize. By default it uses the same quantile loss used in training.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=8,
        help="Number of configurations to evaluate.",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        nargs="+",
        default=[8, 8],
        help="Prediction horizons used during evaluation.",
    )
    parser.add_argument(
        "--eval-split-year",
        type=int,
        default=2000,
        help="Train with years < eval_split_year and evaluate on later years.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Training epochs per trial.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional fixed batch size. If omitted, it is sampled from the coarse search space.",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=2,
        help="Number of ensemble members per trial.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs used inside NeuralTFR.fit.",
    )
    parser.add_argument(
        "--stop-patience",
        type=int,
        default=4,
        help="Early stopping patience passed to NeuralTFR.fit.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional fixed learning rate. If omitted, it is sampled from the coarse search space.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Optional fixed weight decay. If omitted, it is sampled from the coarse search space.",
    )
    parser.add_argument(
        "--augment-tfr-level",
        type=float,
        default=1.3,
        help="Augmentation threshold reused from run_neuraltfr.py.",
    )
    parser.add_argument(
        "--augment-n-windows",
        type=int,
        default=10,
        help="Number of recent windows duplicated during augmentation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable the most reproducible execution mode possible.",
    )
    parser.add_argument(
        "--log-training",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Print full NeuralTFR training logs for every trial.",
    )

    return parser.parse_args()

def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")

def get_output_dir(args) -> Path:
    return Path(args.out_dir) / args.study_name

def format_params(params: dict) -> str:
    ordered_keys = [
        "enc_len",
        "hidden_size",
        "dim_embedding",
        "batch_size",
        "dropout",
        "lr",
        "weight_decay",
    ]
    chunks = []
    for key in ordered_keys:
        value = params.get(key)
        if isinstance(value, float):
            value = f"{value:.6g}"
        chunks.append(f"{key}={value}")
    return ", ".join(chunks)


def sample_loguniform(rng: random.Random, low: float, high: float) -> float:
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def sample_montecarlo_params(args, rng: random.Random) -> dict:
    params = {
        "enc_len": rng.choice(SEARCH_SPACE["enc_len"]),
        "hidden_size": rng.choice(SEARCH_SPACE["hidden_size"]),
        "dim_embedding": rng.choice(SEARCH_SPACE["dim_embedding"]),
        "batch_size": rng.choice(SEARCH_SPACE["batch_size"]),
        "dropout": rng.choice(SEARCH_SPACE["dropout"]),
        "lr": sample_loguniform(rng, *SEARCH_SPACE["lr"]),
        "weight_decay": sample_loguniform(rng, *SEARCH_SPACE["weight_decay"]),
    }

    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.lr is not None:
        params["lr"] = args.lr
    if args.weight_decay is not None:
        params["weight_decay"] = args.weight_decay

    return params


def sample_optuna_params(args, trial) -> dict:
    return {
        "enc_len": trial.suggest_categorical("enc_len", SEARCH_SPACE["enc_len"]),
        "hidden_size": trial.suggest_categorical("hidden_size", SEARCH_SPACE["hidden_size"]),
        "dim_embedding": trial.suggest_categorical("dim_embedding", SEARCH_SPACE["dim_embedding"]),
        "batch_size": args.batch_size
        if args.batch_size is not None
        else trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
        "dropout": trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"]),
        "lr": args.lr if args.lr is not None else trial.suggest_float("lr", *SEARCH_SPACE["lr"], log=True),
        "weight_decay": args.weight_decay
        if args.weight_decay is not None
        else trial.suggest_float("weight_decay", *SEARCH_SPACE["weight_decay"], log=True),
    }


def build_model(args, params: dict) -> NeuralTFR:
    try:
        from common.losses import MQLoss
        from neuraltfr import NeuralTFR
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "NeuralTFR dependencies are not available in this Python environment. "
            "Make sure torch and the project requirements are installed before running the search."
        ) from exc

    mq_loss = MQLoss(quantiles=QUANTILES)
    return NeuralTFR(
        enc_len=params["enc_len"],
        model_name="NeuralTFR_search",
        pred_len=args.pred_len,
        cat_features=["id_serie"],
        lags_features=[2, 4, 6],
        preprocess_config={
            "target": "TFR",
            "time_col": "year",
            "id_col": "id",
            "apply_log": True,
            "scaler": "Standard",
        },
        model_config={
            "enc_hidden_size": params["hidden_size"],
            "dec_hidden_size": params["hidden_size"],
            "dim_embedding": [params["dim_embedding"]],
            "dropout_enc_feat": params["dropout"],
            "dropout_dec_feat": params["dropout"],
            "tf_config": {"tf_type": "linear", "tf_max_rate": 0.3, "tf_decay": 0.035},
        },
        augment_tfr_level=args.augment_tfr_level,
        augment_n_windows=args.augment_n_windows,
        loss_function=mq_loss,
        n_models=args.n_models,
        seed=args.seed,
        deterministic=args.deterministic,
        log=args.log_training,
    )


def compute_eval_loss(real_df: pd.DataFrame, predicts_df: pd.DataFrame) -> pd.DataFrame:
    eval_loss_df = (
        real_df[["id", "year", "TFR"]]
        .merge(predicts_df, on=["id", "year"], how="inner")
        .sort_values(["id", "year"])
    )

    rows = []
    for (serie_id, model_name), group_df in eval_loss_df.groupby(["id", "model"], dropna=True):
        group_losses = []
        target = group_df["TFR"].to_numpy(dtype=float)

        for quantile, pred_col in QUANTILE_COLUMNS:
            if pred_col not in group_df.columns:
                continue

            pred = group_df[pred_col].to_numpy(dtype=float)
            valid_mask = ~np.isnan(target) & ~np.isnan(pred)
            if not valid_mask.any():
                continue

            errors = pred[valid_mask] - target[valid_mask]
            quantile_loss = np.maximum(quantile * -errors, (1 - quantile) * errors)
            group_losses.append(quantile_loss)

        if not group_losses:
            continue

        rows.append(
            {
                "id": int(serie_id),
                "model": model_name,
                "loss": float(np.concatenate(group_losses).mean()),
            }
        )

    return pd.DataFrame(rows)


def evaluate_configuration(args, tfr_data: pd.DataFrame, params: dict) -> tuple[float, dict]:
    train_df = tfr_data[tfr_data["year"] < args.eval_split_year].copy()
    train_df["id_serie"] = train_df["id"]

    model = build_model(args, params)
    model.fit(
        train_df,
        epochs=args.epochs,
        batch_size=params["batch_size"],
        stop_patience=args.stop_patience,
        optimizer_config={
            "lr": params["lr"],
            "weight_decay": params["weight_decay"],
            "type_scheduler": "Exponential",
            "gamma": 0.99,
        },
        valid_epochs=0,
        n_jobs=args.n_jobs,
    )

    preds_df = model.predict(train_df)
    max_pred_len = max(args.pred_len)
    valid_ids = (
        tfr_data[tfr_data["year"] >= args.eval_split_year]
        .groupby("id")
        .size()
        .loc[lambda series: series >= max_pred_len]
        .index
    )
    preds_df = preds_df[preds_df["id"].isin(valid_ids)].reset_index(drop=True)

    if preds_df.empty:
        raise ValueError("No evaluation predictions were generated for the requested holdout period.")

    eval_df = eval_models(
        real_df=tfr_data,
        target_col="TFR",
        time_col="year",
        id_col="id",
        reg_col="id_reg",
        predicts_df=preds_df,
    )
    loss_df = compute_eval_loss(real_df=tfr_data, predicts_df=preds_df)
    if not loss_df.empty:
        eval_df = eval_df.merge(loss_df, on=["id", "model"], how="left")

    metric_values = eval_df[args.metric].dropna()
    if metric_values.empty:
        raise ValueError(f"Metric '{args.metric}' could not be computed for this configuration.")

    summary = {
        "n_eval_ids": int(eval_df["id"].nunique()),
        "n_eval_rows": int(len(eval_df)),
        "median_loss": float(eval_df["loss"].median()) if "loss" in eval_df.columns else float("nan"),
        "median_rmse": float(eval_df["rmse"].median()),
        "median_smape": float(eval_df["smape"].median()),
        "median_rmsse": float(eval_df["rmsse"].median()),
        "median_crps": float(eval_df["crps"].median()),
        "mean_last_epoch": float(np.mean(model.training_summary["last_epochs"])),
        "mean_last_step": float(np.mean(model.training_summary["last_steps"])),
    }

    score = float(metric_values.median())
    return score, summary


def evaluate_trial(args, tfr_data: pd.DataFrame, params: dict, trial_number: int) -> dict:
    start_time = time.perf_counter()
    record = {
        "trial": trial_number,
        "search_mode": args.search_mode,
        "metric": args.metric,
        "status": "ok",
        "error": "",
        **params,
    }

    try:
        score, summary = evaluate_configuration(args, tfr_data, params)
        record["score"] = score
        record.update(summary)
    except Exception as exc:  # pragma: no cover - defensive path for long runs
        record["status"] = "error"
        record["error"] = str(exc)
        record["score"] = float("inf")

    record["duration_seconds"] = round(time.perf_counter() - start_time, 3)
    return record


def save_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(records)
    results_df.to_csv(output_dir / "trials.csv", index=False)

    valid_records = [record for record in records if record["status"] == "ok"]
    if not valid_records:
        return

    best_record = min(valid_records, key=lambda item: item["score"])
    with (output_dir / "best_config.json").open("w", encoding="utf-8") as file:
        json.dump(best_record, file, indent=2)


def print_trial_update(record: dict, best_score: float, total_trials: int) -> None:
    score_text = f"{record['score']:.6f}" if np.isfinite(record["score"]) else "inf"
    best_text = f"{best_score:.6f}" if np.isfinite(best_score) else "inf"
    print(
        f"Trial {record['trial'] + 1}/{total_trials} | "
        f"status={record['status']} | {record['metric']}={score_text} | best={best_text}"
    )
    print(f"  {format_params(record)}")
    if record["status"] != "ok":
        print(f"  error={record['error']}")


def run_montecarlo_search(args, tfr_data: pd.DataFrame) -> tuple[list[dict], dict | None]:
    rng = random.Random(args.seed)
    records = []
    best_record = None

    for trial_number in range(args.n_trials):
        params = sample_montecarlo_params(args, rng)
        record = evaluate_trial(args, tfr_data, params, trial_number)
        records.append(record)
        save_outputs(records, get_output_dir(args))

        if record["status"] == "ok" and (best_record is None or record["score"] < best_record["score"]):
            best_record = record

        current_best = best_record["score"] if best_record is not None else float("inf")
        print_trial_update(record, current_best, args.n_trials)

    return records, best_record


def run_optuna_search(args, tfr_data: pd.DataFrame) -> tuple[list[dict], dict | None]:
    if optuna is None:
        raise SystemExit(
            "Optuna is not installed in this environment. Install it or run with --search-mode montecarlo."
        )

    records: list[dict] = []
    best_record = None

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", study_name=args.study_name, sampler=sampler)

    def objective(trial):
        params = sample_optuna_params(args, trial)
        record = evaluate_trial(args, tfr_data, params, trial.number)
        records.append(record)
        trial.set_user_attr("status", record["status"])
        if record["error"]:
            trial.set_user_attr("error", record["error"])
        return record["score"]

    def callback(study, trial):
        nonlocal best_record
        save_outputs(records, get_output_dir(args))

        latest_record = next((record for record in reversed(records) if record["trial"] == trial.number), None)
        if latest_record is None:
            return

        if latest_record["status"] == "ok" and (
            best_record is None or latest_record["score"] < best_record["score"]
        ):
            best_record = latest_record

        current_best = best_record["score"] if best_record is not None else float("inf")
        print_trial_update(latest_record, current_best, args.n_trials)

    study.optimize(objective, n_trials=args.n_trials, callbacks=[callback])
    return records, best_record


def print_summary(best_record: dict | None, output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("Hyperparameter search finished")
    print("=" * 60)
    print(f"Saved artifacts to: {output_dir}")

    if best_record is None:
        print("No valid configuration finished successfully.")
        return

    print(f"Best metric ({best_record['metric']}): {best_record['score']:.6f}")
    print("Best configuration:")
    print(f"  {format_params(best_record)}")


def main():
    args = parse_args()
    output_dir = get_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"NeuralTFR Hyperparameter Search - Mode: {args.search_mode.upper()}")
    print("=" * 60)
    print("Coarse search space:")
    print(f"  enc_len        -> {SEARCH_SPACE['enc_len']}")
    print(f"  hidden_size    -> {SEARCH_SPACE['hidden_size']}")
    print(f"  dim_embedding  -> {SEARCH_SPACE['dim_embedding']}")
    print(f"  batch_size     -> {SEARCH_SPACE['batch_size']}")
    print(f"  dropout        -> {SEARCH_SPACE['dropout']}")
    print(f"  lr             -> {SEARCH_SPACE['lr']}")
    print(f"  weight_decay   -> {SEARCH_SPACE['weight_decay']}")

    tfr_data = pd.read_csv(args.data_path)

    if args.search_mode == "minimize":
        _, best_record = run_optuna_search(args, tfr_data)
    else:
        _, best_record = run_montecarlo_search(args, tfr_data)

    print_summary(best_record, output_dir)


if __name__ == "__main__":
    main()
