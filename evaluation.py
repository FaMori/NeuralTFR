"""Scores of the recorded forecasts and the paired comparisons between models.

Each forecast year is scored against the reported value on the TFR scale: the
error and absolute error of the median, sMAPE, the quantile loss, and the
coverage, width and Winkler score of the 80% and 95% intervals. The network's
ten fits are first combined into the median of each quantile. Models are
compared on the populations with complete outcomes at every origin: each
population's scores are averaged over origins, and the paired differences are
summarised by the Hodges-Lehmann estimate, a bootstrap interval over
populations and a Wilcoxon test with Holm's adjustment. Within the fertility
groups at the origin the same comparison runs without tests.

    python evaluation.py      # from the repository root; writes results/metrics_countries.csv,
                              # paired_comparisons.csv and group_comparisons.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from neuraltfr.runner import DATA, ORIGINS, ROOT
from neuraltfr.training import COLUMNS, QUANTILES

RESULTS = ROOT / "results"
MODELS = {"neuraltfr": RESULTS / "neuraltfr" / "eval_predictions.parquet",
          "spline": RESULTS / "baselines" / "spline" / "eval_predictions.parquet",
          "linear": RESULTS / "baselines" / "linear" / "eval_predictions.parquet",
          "bayestfr": RESULTS / "bayestfr" / "eval_predictions.parquet",
          "drift": RESULTS / "baselines" / "drift" / "eval_predictions.parquet"}
KEYS = ["period", "id", "year", "horizon"]
INTERVALS = {"80": ("y_hat_100", "y_hat_900"), "95": ("y_hat_025", "y_hat_975")}
METRICS = ["mae", "bias", "smape", "pinball"] + [f"{name}_{level}" for level in INTERVALS
                                                 for name in ("coverage", "width", "winkler")]
COMPARED = ["smape", "winkler_80", "winkler_95", "mae", "pinball"]
SEED, RESAMPLES = 2026, 2000
GROUPS = ("very_low", "moderately_low", "above_replacement")
THRESHOLDS = (1.5, 2.1)
GROUP_MEASURES = ["smape", "winkler_80", "winkler_95"]
GROUP_PAIRS = [("neuraltfr", "bayestfr"), ("neuraltfr", "spline"), ("spline", "bayestfr")]   # (reference, benchmark)


def ensemble(predictions: pd.DataFrame) -> pd.DataFrame:
    """One forecast per population and year: the median across seeds of each quantile."""
    aggregations = {column: "median" for column in COLUMNS} | {"y_true": "first", "in_balanced": "first"}
    return predictions.groupby(KEYS).agg(aggregations).reset_index()


def scores(paths: pd.DataFrame) -> pd.DataFrame:
    """One row per forecast year with a reported value: the scores of that year's forecast."""
    scored = paths.dropna(subset=["y_true"]).astype({c: float for c in ("y_true", *COLUMNS)})
    truth, median = scored["y_true"], scored["y_hat_500"]
    error = median - truth
    out = scored[["model", *KEYS, "in_balanced"]].copy()
    out["mae"], out["bias"] = error.abs(), error
    out["smape"] = 200.0 * error.abs() / (median.abs() + truth.abs())
    out["pinball"] = sum(np.maximum(q * (truth - scored[column]), (q - 1.0) * (truth - scored[column]))
                         for column, q in zip(COLUMNS, QUANTILES)) / len(QUANTILES)
    for level, (low, high) in INTERVALS.items():
        lower, upper, alpha = scored[low], scored[high], 1.0 - int(level) / 100.0
        out[f"coverage_{level}"] = ((truth >= lower) & (truth <= upper)).astype(float)
        out[f"width_{level}"] = upper - lower
        out[f"winkler_{level}"] = ((upper - lower) + (2.0 / alpha) * (lower - truth).clip(lower=0.0)
                                   + (2.0 / alpha) * (truth - upper).clip(lower=0.0))
    return out


def country_metrics(paths: pd.DataFrame) -> pd.DataFrame:
    """Mean scores per model, origin and population over the years with a reported value."""
    cells = scores(paths)
    metrics = cells.groupby(["model", "period", "id"]).agg(
        n_years=("year", "size"), in_balanced=("in_balanced", "first"),
        **{metric: (metric, "mean") for metric in METRICS})
    return metrics.reset_index()


def hodges_lehmann(differences: np.ndarray) -> float:
    """The median of the pairwise averages: the location the Wilcoxon test targets."""
    i, j = np.triu_indices(len(differences))
    return float(np.median((differences[i] + differences[j]) / 2))


def holm(pvalues: list) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty_like(pvalues), 0.0
    for rank, index in enumerate(np.argsort(pvalues)):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def difference_summary(pooled: pd.DataFrame, by_origin: pd.DataFrame, metric: str, reference: str,
                       benchmark: str, rng: np.random.Generator) -> tuple:
    """Benchmark minus reference per population: the estimate, its resampled interval, the
    estimate at each origin, the share where the reference scores better and both medians.

    ``pooled`` holds the scores per model and population averaged over origins, ``by_origin``
    the scores per model, origin and population. Returns the row, the differences and the draws.
    """
    wide = pooled[metric].unstack("model")
    difference = (wide[benchmark] - wide[reference]).to_numpy()
    draws = np.array([hodges_lehmann(rng.choice(difference, len(difference))) for _ in range(RESAMPLES)])
    low, high = np.percentile(draws, [2.5, 97.5])
    row = {"metric": metric, "reference": reference, "benchmark": benchmark,
           "estimate": hodges_lehmann(difference), "ci_low": low, "ci_high": high,
           **{f"estimate_{origin}": hodges_lehmann((by_origin.loc[(benchmark, origin), metric]
                                                    - by_origin.loc[(reference, origin), metric]).to_numpy())
              for origin in ORIGINS},
           "share_reference_better": float((difference > 0).mean()),
           "reference_median": float(wide[reference].median()),
           "benchmark_median": float(wide[benchmark].median()),
           "n_populations": len(difference)}
    return row, difference, draws


def paired_differences(metrics: pd.DataFrame, reference: str, benchmarks: list) -> tuple:
    """Benchmark minus reference, per population averaged over origins; positive favours the reference.

    Returns the table and, for each metric and benchmark, the resampled estimates.
    """
    rng = np.random.default_rng(SEED)
    pooled = metrics.groupby(["model", "id"])[COMPARED].mean()
    by_origin = metrics.set_index(["model", "period", "id"]).sort_index()
    rows, draws = [], {}
    for metric in COMPARED:
        block = []
        for benchmark in benchmarks:
            row, difference, draws[metric, benchmark] = difference_summary(pooled, by_origin, metric, reference,
                                                                           benchmark, rng)
            block.append({**row, "p_raw": float(wilcoxon(difference).pvalue)})
        rows += [{**row, "p_holm": p} for row, p in zip(block, holm([row["p_raw"] for row in block]))]
    return pd.DataFrame(rows), draws


def group_differences(metrics: pd.DataFrame, reference: str, benchmark: str) -> pd.DataFrame:
    """Benchmark minus reference within each fertility group at the origin; positive favours the reference."""
    groups = group_of(level_at(annual_medians(), metrics["id"], metrics["period"]))
    rows = []
    for group in GROUPS:
        part = metrics[groups == group]
        pooled = part.groupby(["model", "id"])[GROUP_MEASURES].mean()   # over the origins in the group
        by_origin = part.set_index(["model", "period", "id"]).sort_index()
        for metric in GROUP_MEASURES:
            row, _, _ = difference_summary(pooled, by_origin, metric, reference, benchmark,
                                           np.random.default_rng(SEED))
            rows.append({"group": group, **row, "n_population_origins": int((part["model"] == reference).sum())})
    return pd.DataFrame(rows)


def annual_medians() -> pd.Series:
    """The reported annual median TFR of each population and year."""
    return pd.read_csv(DATA).groupby(["id", "year"])["TFR"].median()


def group_of(level) -> np.ndarray:
    return np.array(GROUPS)[np.searchsorted(THRESHOLDS, np.asarray(level), side="right")]


def level_at(annual: pd.Series, ids, years) -> np.ndarray:
    return annual.reindex(pd.MultiIndex.from_arrays([np.asarray(ids), np.asarray(years)])).to_numpy()


def forecasts() -> pd.DataFrame:
    """Every model's recorded test forecasts in one frame, the network's as the seed ensemble."""
    frames = []
    for model, path in MODELS.items():
        predictions = pd.read_parquet(path)
        predictions = predictions[predictions["row_set"] == "test"]
        frames.append((ensemble(predictions) if model == "neuraltfr" else predictions).assign(model=model))
    return pd.concat(frames, ignore_index=True)


def cohort(metrics: pd.DataFrame) -> set:
    """The populations with complete outcomes at every origin in every model."""
    complete = metrics[metrics["in_balanced"]]
    return set.intersection(*(set(part["id"]) for _, part in complete.groupby(["model", "period"])))


def main() -> None:
    metrics = country_metrics(forecasts())
    metrics.to_csv(RESULTS / "metrics_countries.csv", index=False)

    ids = cohort(metrics)
    compared = metrics[metrics["in_balanced"] & metrics["id"].isin(ids)]
    tables = [paired_differences(compared, "neuraltfr", ["spline", "linear", "bayestfr", "drift"])[0],
              paired_differences(compared, "spline", ["bayestfr"])[0]]
    pd.concat(tables, ignore_index=True).to_csv(RESULTS / "paired_comparisons.csv", index=False)
    groups = [group_differences(compared, reference, benchmark) for reference, benchmark in GROUP_PAIRS]
    pd.concat(groups, ignore_index=True).to_csv(RESULTS / "group_comparisons.csv", index=False)
    print(f"{len(metrics):,} population scores written; {len(ids)} populations compared at every origin")


if __name__ == "__main__":
    main()
