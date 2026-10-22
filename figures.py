"""The numbers behind the manuscript's figures, appendix analyses and text.

Each function returns the tables a figure, an appendix passage or a passage
of the text reports, computed from the recorded forecasts, their scores and
the panel; ``main`` writes them to ``results/figures/``. Nothing here draws:
the figures as published are kept beside the tables, and
``results/figures/README.md`` says which table each one shows.

Populations are compared on the cohort of ``evaluation``: the populations
with complete outcomes at every origin in every model. Fertility groups at an
origin use the reported annual median of that year: very low below 1.5,
moderately low from 1.5 to 2.1, at or above replacement from 2.1.

    python figures.py      # from the repository root, after evaluation.py
"""
import re
import unicodedata

import numpy as np
import pandas as pd
import torch

from evaluation import (COMPARED, GROUPS, INTERVALS, RESULTS, THRESHOLDS, annual_medians, cohort, ensemble,
                        forecasts, group_of, level_at, scores)
from neuraltfr.model import load
from neuraltfr.preprocess import annual_panel, to_model_scale, to_tfr
from neuraltfr.runner import DATA, FORECAST_ORIGIN, ORIGINS, ROOT, SEEDS, predict
from neuraltfr.training import COLUMNS, QUANTILES
from neuraltfr.windows import build_fold

OUT = RESULTS / "figures"
RAW = ROOT / "data" / "raw"
BANDS = ("1-5", "6-10", "11-15")
MODELS = ["neuraltfr", "spline", "linear", "bayestfr", "drift"]
SEED, RESAMPLES = 2026, 2000

# Figures 5, G1 and G2: populations chosen to illustrate recurring trajectory
# patterns, the first of each pattern in the main figure, the others in the appendix.
CASES = [("Limited recovery at very low fertility", [392, 620, 616]),
         ("Renewed decline", [840, 56, 124]),
         ("Further falls at very low fertility", [410, 158, 344]),
         ("Recovery from very low fertility", [642, 276, 705]),
         ("Persistent or rising fertility at higher levels", [398, 704, 376, 860, 496])]
TRAJECTORY_ORIGIN, TRAJECTORY_YEARS = 2006, (1995, 2021)

# Figure 4: ranges of starting fertility, each shown at its mean starting level;
# its population bootstrap draws more samples than the paired comparisons.
RANGES = (0.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 3.0, 4.0, np.inf)
CHANGE_RESAMPLES = 5000

# Figure 6: GBD location names that differ from the panel's after normalisation.
GBD_NAMES = {180: "Democratic Republic of the Congo", 410: "Republic of Korea",
             498: "Republic of Moldova", 528: "Netherlands", 826: "United Kingdom",
             850: "United States Virgin Islands"}
FORWARD_YEARS = list(range(FORECAST_ORIGIN + 1, FORECAST_ORIGIN + 16))
FORWARD_THRESHOLDS = (1.3, 1.5, 2.1)

# Sections 3.1-3.2 and the Introduction: a strong rise by the fifteenth forecast year, and
# the panel's experience of very low fertility by the last historical origin.
STRONG_RISE = 0.15
RECOVERY_CUTOFF, RECOVERY_RISE = 2006, 0.2

# Appendix C: simulated histories ending at a level after a constant annual change.
LEVELS = (0.7, 0.9, 1.2, 1.6, 2.1, 3.0, 4.5)
TRENDS = (-0.08, -0.04, 0.0, 0.04)
GRID_HORIZONS = (1, 5, 10, 15)


# --- figures -----------------------------------------------------------------

def composition() -> pd.DataFrame:
    """Figure B1: populations with a reported value each year, by fertility group."""
    annual = annual_medians().reset_index()
    annual["group"] = group_of(annual["TFR"])
    years = pd.RangeIndex(annual["year"].min(), annual["year"].max() + 1, name="year")
    counts = (annual.groupby(["year", "group"]).size().unstack(fill_value=0)
              .reindex(index=years, columns=GROUPS, fill_value=0))
    table = pd.DataFrame({"reported_populations": counts.sum(axis=1)})
    for group in GROUPS:
        table[f"count_{group}"] = counts[group]
        table[f"share_{group}"] = counts[group] / table["reported_populations"]
    return table.reset_index()


def historical_performance(metrics: pd.DataFrame) -> tuple:
    """Figure F1: each population's scores averaged over origins, and their medians pooled and by origin."""
    populations = metrics.groupby(["model", "id"])[COMPARED].mean().reset_index()
    pooled = populations.groupby("model")[COMPARED].median().reset_index().assign(period="pooled")
    by_origin = metrics.groupby(["model", "period"])[COMPARED].median().reset_index()
    return populations, pd.concat([pooled, by_origin.astype({"period": str})], ignore_index=True)


def trajectories(paths: pd.DataFrame) -> tuple:
    """Figures 5, G1 and G2: forecasts from 2006 and reported values for the illustrated populations."""
    names = pd.read_csv(DATA).drop_duplicates("id").set_index("id")["name"]
    cases = pd.DataFrame([{"id": population, "name": names[population], "pattern": pattern,
                           "display": "main" if index == 0 else "appendix"}
                          for pattern, populations in CASES for index, population in enumerate(populations)])
    forecast = paths[(paths["period"] == TRAJECTORY_ORIGIN) & paths["id"].isin(cases["id"])]
    forecast = cases.merge(forecast[["model", "id", "year", "horizon", "y_true", *COLUMNS]], on="id")
    observed = annual_medians().reset_index()
    observed = observed[observed["id"].isin(cases["id"]) & observed["year"].between(*TRAJECTORY_YEARS)]
    return forecast, cases[["id", "name"]].merge(observed, on="id")


def interval_groups(paths: pd.DataFrame) -> tuple:
    """Figure E1 and Appendix D: interval scores by fertility group at the origin and horizon band, and where the misses fall."""
    annual = annual_medians()
    cells = scores(paths)
    cells["group"] = group_of(level_at(annual, cells["id"], cells["period"]))
    cells["band"] = np.array(BANDS)[(cells["horizon"] - 1) // 5]
    measures = [f"{name}_{level}" for level in INTERVALS for name in ("coverage", "width", "winkler")]
    pairs = cells.groupby(["model", "group", "period", "id", "band"])[measures].mean().reset_index()
    groups = pairs.groupby(["model", "group", "band"])
    by_band = groups[measures].mean()
    by_band["n_population_origins"] = groups.size()
    by_band["n_populations"] = groups["id"].nunique()

    misses = paths[["model", "period", "id"]].copy()
    misses["group"] = group_of(level_at(annual, paths["id"], paths["period"]))
    for level, (low, high) in INTERVALS.items():
        misses[f"below_{level}"] = (paths["y_true"] < paths[low]).astype(float)
        misses[f"above_{level}"] = (paths["y_true"] > paths[high]).astype(float)
    misses["error"] = paths["y_hat_500"] - paths["y_true"]
    columns = [c for c in misses.columns if c.startswith(("below", "above"))] + ["error"]
    combined = pd.concat([misses.assign(group="all"), misses])
    groups = combined.groupby(["group", "model"])
    direction = groups[columns].mean()
    direction["n_forecast_values"] = groups.size()
    direction["n_populations"] = groups["id"].nunique()
    return by_band.reset_index(), direction.reset_index()


def fertility_change(paths: pd.DataFrame) -> tuple:
    """Figure 4: fifteen-year change, observed and forecast, by starting level; the level where mean change turns negative."""
    annual = annual_medians()
    models = MODELS[:4]
    ends = paths[(paths["horizon"] == 15) & paths["model"].isin(models)].copy()
    ends["origin_tfr"] = level_at(annual, ends["id"], ends["period"])
    ends["observed_change"] = level_at(annual, ends["id"], ends["year"]) - ends["origin_tfr"]
    ends["predicted_change"] = ends["y_hat_500"] - ends["origin_tfr"]
    wide = ends.pivot_table(index=["period", "id"], columns="model", values="predicted_change")
    first = ends[ends["model"] == models[0]].set_index(["period", "id"])
    wide = wide.join(first[["origin_tfr", "observed_change"]]).reset_index()
    series = ["observed_change", *models]
    ranges = pd.cut(wide["origin_tfr"], RANGES, right=False)
    codes = ranges.cat.codes.to_numpy()
    values = wide[["origin_tfr", *series]].to_numpy(float)
    populations, member = np.unique(wide["id"], return_inverse=True)

    def means(weights):
        out = np.full((len(weights), len(ranges.cat.categories), values.shape[1]), np.nan)
        for r in range(out.shape[1]):
            w = weights[:, codes == r]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, r] = (w @ values[codes == r]) / w.sum(axis=1)[:, None]
        return out

    def crossing(x, d):
        """Level of the first change from positive to negative, linearly interpolated."""
        for i in range(len(x) - 1):
            if d[i] >= 0 > d[i + 1] or d[i] > 0 >= d[i + 1]:
                return float(x[i] + (x[i + 1] - x[i]) * d[i] / (d[i] - d[i + 1]))
        return np.nan

    def crossings(table):
        keep = np.isfinite(table[:, 0])
        return [crossing(table[keep, 0], table[keep, j + 1]) for j in range(len(series))]

    point = means(np.ones((1, len(wide))))[0]
    rng = np.random.default_rng(SEED)      # populations resampled with their three origins together
    sampled = rng.integers(0, len(populations), size=(CHANGE_RESAMPLES, len(populations)))
    counts = np.stack([np.bincount(row, minlength=len(populations)) for row in sampled])
    draws = means(counts[:, member].astype(float))
    low, high = np.nanquantile(draws[:, :, 1], [0.025, 0.975], axis=0)

    table = pd.DataFrame(point, columns=["mean_origin_tfr", *series])
    table.insert(0, "range", [str(c) for c in ranges.cat.categories])
    table.insert(1, "n_population_origins", np.bincount(codes, minlength=len(table)))
    table["observed_ci_low"], table["observed_ci_high"] = low, high
    crossing_draws = np.array([crossings(draw) for draw in draws])
    bounds = [np.nanquantile(column, [0.025, 0.975]) if np.isfinite(column).any() else (np.nan, np.nan)
              for column in crossing_draws.T]
    zero = pd.DataFrame({"series": series, "crossing": crossings(point),
                         "ci_low": [b[0] for b in bounds], "ci_high": [b[1] for b in bounds],
                         "draws_without_crossing": np.isnan(crossing_draws).sum(axis=0)})
    return table, zero


# --- text: very low fertility and what followed --------------------------------

def very_low_pairs(paths: pd.DataFrame) -> pd.DataFrame:
    """Sections 3.1-3.2 and Appendix D: each population-origin pair that started below 1.5, its later
    fertility, and the mean scores over the fifteen years of NeuralTFR, the spline and BayesTFR."""
    annual = annual_medians()
    cells = scores(paths[paths["model"].isin(["neuraltfr", "spline", "bayestfr"])])
    cells = cells[group_of(level_at(annual, cells["id"], cells["period"])) == "very_low"]
    cells["bias_11_15"] = cells["bias"].where(cells["horizon"] > 10)
    pairs = cells.groupby(["period", "id", "model"])[["smape", "bias", "bias_11_15", "winkler_95"]].mean().unstack("model")
    pairs.columns = [f"{measure}_{model}" for measure, model in pairs.columns]
    pairs = pairs.reset_index()
    ids, periods = pairs["id"].to_numpy(), pairs["period"].to_numpy()
    series = {population: annual.xs(population, level="id") for population in np.unique(ids)}
    names = pd.read_csv(DATA).drop_duplicates("id").set_index("id")["name"]
    later = pd.DataFrame({"name": names.reindex(ids).to_numpy(),
                          "origin_tfr": level_at(annual, ids, periods),
                          "tfr_15": level_at(annual, ids, periods + 15),
                          "latest_year": [series[i].index[-1] for i in ids],
                          "latest_tfr": [series[i].iloc[-1] for i in ids],
                          "peak_year_after_origin": [series[i][series[i].index > p].idxmax() for i, p in zip(ids, periods)]})
    later.insert(3, "change_15", later["tfr_15"] - later["origin_tfr"])
    return pd.concat([pairs[["period", "id"]], later, pairs.drop(columns=["period", "id"])], axis=1)


def recovery_panel() -> pd.DataFrame:
    """Introduction: every population of the panel below 1.5 by RECOVERY_CUTOFF, its lowest level by
    then, its highest level after that low, its latest value and its 2023 population."""
    annual = annual_medians()
    names = pd.read_csv(DATA).drop_duplicates("id").set_index("id")["name"]
    weights = pd.read_csv(RAW / "wpp2024" / "pop_2023.csv").set_index("id")["Pop"]
    rows = []
    for population, series in annual.groupby(level="id"):
        series = series.droplevel("id")
        before = series[series.index <= RECOVERY_CUTOFF]
        if before.empty or before.min() >= THRESHOLDS[0]:
            continue
        after = series[series.index > before.idxmin()]
        rows.append({"id": population, "name": names[population], "low": before.min(), "low_year": before.idxmin(),
                     "peak_after_low": after.max(), "peak_year": after.idxmax() if len(after) else np.nan,
                     "latest_year": series.index[-1], "latest_tfr": series.iloc[-1],
                     "population_2023": weights.get(population, np.nan)})
    return pd.DataFrame(rows)


def very_low_counts(pairs: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """The counts and means the text cites from ``very_low_pairs`` and ``recovery_panel``, one row each."""
    bayes_better = pairs["smape_bayestfr"] < pairs["smape_neuraltfr"]
    fell, rose, strong = pairs["change_15"] < 0, pairs["change_15"] > 0, pairs["change_15"] > STRONG_RISE
    wins = pairs[strong & bayes_better]
    won = wins.drop_duplicates("id")
    penalty = (pairs["winkler_95_spline"] - pairs["winkler_95_neuraltfr"]).sort_values().head(3)
    recovered = panel[panel["peak_after_low"] - panel["low"] >= RECOVERY_RISE]
    above = panel[panel["latest_tfr"] >= THRESHOLDS[0]]
    total = pd.read_csv(RAW / "wpp2024" / "pop_2023.csv")["Pop"].sum()
    models = ("neuraltfr", "spline", "bayestfr")
    rows = [("3.1", "populations that started a forecast period below 1.5", pairs["id"].nunique()),
            ("3.1", "population-origin pairs", len(pairs)),
            *[("3.1", f"mean error of the median, {model}", pairs[f"bias_{model}"].mean()) for model in models],
            *[("3.1", f"mean error of the median in years 11-15, {model}", pairs[f"bias_11_15_{model}"].mean())
              for model in models],
            ("3.1", "pairs lower at year 15 than at the origin", int(fell.sum())),
            ("3.1", "of which NeuralTFR has the lower sMAPE", int((fell & ~bayes_better).sum())),
            ("3.1", f"pairs more than {STRONG_RISE} higher at year 15 than at the origin", int(strong.sum())),
            ("3.1", "of which BayesTFR has the lower sMAPE", int(len(wins))),
            ("3.2", "peak years after the origin of those pairs", "; ".join(map(str, sorted(wins["peak_year_after_origin"])))),
            ("Discussion", "populations of those pairs", len(won)),
            ("Discussion", "of which latest value below 1.5", int((won["latest_tfr"] < THRESHOLDS[0]).sum())),
            ("Discussion", "of which latest value at or above 1.5",
             "; ".join(sorted(won.loc[won["latest_tfr"] >= THRESHOLDS[0], "name"]))),
            ("3.2", "pairs higher at year 15 than at the origin", int(rose.sum())),
            ("3.2", "of which latest value below the year-15 value", int((rose & (pairs["latest_tfr"] < pairs["tfr_15"])).sum())),
            ("3.2", "of which latest value below the origin value", int((rose & (pairs["latest_tfr"] < pairs["origin_tfr"])).sum())),
            ("3.2", "years of the latest values", f"{pairs['latest_year'].min()}-{pairs['latest_year'].max()}"),
            *[("D", f"mean Winkler score of the 95% intervals, {model}", pairs[f"winkler_95_{model}"].mean())
              for model in models[:2]],
            *[("D", f"spline minus NeuralTFR Winkler 95, most negative {rank}: {pairs.at[i, 'name']} {pairs.at[i, 'period']}", value)
              for rank, (i, value) in enumerate(penalty.items(), start=1)],
            ("Introduction", f"populations below 1.5 by {RECOVERY_CUTOFF}", len(panel)),
            ("Introduction", f"of which later at least {RECOVERY_RISE} above their lowest level", len(recovered)),
            ("Introduction", "of which latest value below 1.5", int((recovered["latest_tfr"] < THRESHOLDS[0]).sum())),
            ("Introduction", f"latest value at or below the lowest level by {RECOVERY_CUTOFF}",
             int((panel["latest_tfr"] <= panel["low"]).sum())),
            ("Introduction", "latest value at or above 1.5", "; ".join(sorted(above["name"]))),
            ("Introduction", "their 2023 population, millions", above["population_2023"].sum() / 1e6),
            ("Introduction", "their share of the 2023 population of pop_2023.csv", above["population_2023"].sum() / total)]
    return pd.DataFrame(rows, columns=["passage", "quantity", "value"])


def forward() -> tuple:
    """Figure 6: projections to 2038 by NeuralTFR, the regressions, UN WPP 2024 and GBD 2021, for the populations covered by all."""
    net = pd.read_parquet(RESULTS / "neuraltfr" / "forecast_predictions.parquet")
    net = net[(net["row_set"] == "test") & (net["period"] == FORECAST_ORIGIN)]
    systems = [ensemble(net).assign(system="NeuralTFR")]
    for model, system in (("spline", "Spline"), ("linear", "Linear")):
        part = pd.read_parquet(RESULTS / "baselines" / model / "forecast_predictions.parquet")
        systems.append(part[part["row_set"] == "test"].assign(system=system))
    wpp = pd.read_csv(RAW / "wpp2024" / "WPP_2024_2045_TFR.csv")
    wpp = wpp[wpp["Variant"] == "Median"]
    systems.append(pd.DataFrame({"id": wpp["LocationId"], "year": wpp["Time"].astype(int),
                                 "y_hat_500": wpp["Value"], "system": "WPP2024"}))
    names = pd.read_csv(DATA).drop_duplicates("id").set_index("id")["name"]
    gbd = pd.read_csv(RAW / "gbd2021" / "GBD_1950_2100_TFR.CSV")
    gbd = gbd[gbd["scenario_name"] == "Reference"]

    def normalise(name):
        return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower())

    lookup = {normalise(name): name for name in gbd["location_name"].unique()}
    matched = {i: GBD_NAMES.get(i, lookup.get(normalise(names[i]))) for i in names.index}
    matched = {i: n for i, n in matched.items() if n}
    gbd = gbd[gbd["location_name"].isin(matched.values())]
    systems.append(pd.DataFrame({"id": gbd["location_name"].map({n: i for i, n in matched.items()}),
                                 "year": gbd["year_id"], "y_hat_500": gbd["val"], "system": "GBD"}))

    frame = pd.concat(systems, ignore_index=True)
    frame = frame[frame["year"].isin(FORWARD_YEARS)]
    common = set.intersection(*(set(part.groupby("id")["year"].nunique().loc[lambda s: s == len(FORWARD_YEARS)].index)
                                for _, part in frame.groupby("system")))
    frame = frame[frame["id"].isin(common)]
    weights = pd.read_csv(RAW / "wpp2024" / "pop_2023.csv").set_index("id")["Pop"]
    at_origin = annual_medians().xs(FORECAST_ORIGIN, level="year").reindex(sorted(common))
    group = pd.Series(group_of(at_origin), index=at_origin.index)
    levels = np.asarray(QUANTILES)

    rows = [{"system": f"Observed {FORECAST_ORIGIN}", "year": FORECAST_ORIGIN, "n_populations": len(at_origin),
             "mean": at_origin.mean(), "population_weighted_mean": np.average(at_origin, weights=weights.reindex(at_origin.index)),
             **{f"count_below_{t}": int((at_origin < t).sum()) for t in FORWARD_THRESHOLDS}}]
    for (system, year), part in frame.groupby(["system", "year"]):
        part = part.set_index("id").sort_index()
        median = part["y_hat_500"]
        row = {"system": system, "year": year, "n_populations": len(part), "mean": median.mean(),
               "population_weighted_mean": np.average(median, weights=weights.reindex(median.index))}
        for t in FORWARD_THRESHOLDS:
            row[f"count_below_{t}"] = int((median < t).sum())
            if part[list(COLUMNS)].notna().all().all():    # expected count: P(TFR < t) interpolated between the quantiles
                row[f"expected_below_{t}"] = sum(np.interp(t, q, levels, left=0.0, right=1.0)
                                                 for q in part[list(COLUMNS)].to_numpy(float))
        rows.append(row)
    by_year = pd.DataFrame(rows)

    end = frame[frame["year"] == FORWARD_YEARS[-1]].pivot(index="id", columns="system", values="y_hat_500")
    by_group = []
    for name in GROUPS:
        ids = group.index[group == name]
        row = {"group": name, "n_populations": len(ids), "mean_tfr_2023": at_origin[ids].mean()}
        for system in end.columns:
            row[f"mean_change_{system}"] = (end.loc[ids, system] - at_origin[ids]).mean()
            row[f"count_below_1.5_{system}"] = int((end.loc[ids, system] < 1.5).sum())
        by_group.append(row)
    end.insert(0, "group_2023", group)
    end.insert(0, "tfr_2023", at_origin)
    end.insert(0, "name", names.reindex(end.index))
    return by_year, pd.DataFrame(by_group), end.reset_index()


# --- appendix: what the network responds to ----------------------------------

def grid_responses() -> pd.DataFrame:
    """Appendix C: the median forecast for simulated histories ending at a level after a constant annual change."""
    grid = [(level, trend) for level in LEVELS for trend in TRENDS]
    histories = np.stack([np.maximum(0.15, level - trend * np.arange(11, -1, -1)) for level, trend in grid])
    inputs = torch.from_numpy(to_model_scale(histories))
    rows = []
    for origin in ORIGINS:
        for seed in SEEDS:
            model = load(RESULTS / "neuraltfr" / "models" / f"eval_{origin}_s{seed}.pt")
            with torch.no_grad():
                responses = to_tfr(model(inputs).numpy()[:, :, QUANTILES.index(0.5)])
            rows += [{"period": origin, "seed": seed, "level": level, "trend": trend, "horizon": horizon,
                      "predicted_median_tfr": float(responses[i, horizon - 1])}
                     for i, (level, trend) in enumerate(grid) for horizon in GRID_HORIZONS]
    by_seed = pd.DataFrame(rows)
    table = by_seed.groupby(["period", "level", "trend", "horizon"], as_index=False).agg(
        predicted_median_tfr=("predicted_median_tfr", "median"),
        minimum_across_seeds=("predicted_median_tfr", "min"),
        maximum_across_seeds=("predicted_median_tfr", "max"))
    table["predicted_change"] = table["predicted_median_tfr"] - table["level"]
    return table


def flat_history(ids: set) -> tuple:
    """Appendix C: forecast loss when each history is replaced by its final value, against the observed-history forecasts."""
    observations = pd.read_csv(DATA)
    truth = annual_panel(observations, interpolate=False)
    annual = annual_medians()
    frames = {"original": [], "flat": []}
    for origin in ORIGINS:
        panel = annual_panel(observations[observations["year"] <= origin])
        windows = build_fold(panel, truth, origin, valid_gap=0, balanced=False).test
        windows = windows.subset(np.isin(windows.ids, list(ids)))
        actual = torch.from_numpy(windows.encoder)
        flat = actual[:, -1:].expand_as(actual).contiguous()
        count, width = windows.target.shape
        keys = pd.DataFrame({"period": origin, "id": np.repeat(windows.ids, width), "year": windows.years.reshape(-1),
                             "horizon": np.tile(np.arange(1, width + 1), count), "in_balanced": True,
                             "y_true": np.where(windows.mask > 0, to_tfr(windows.target), np.nan).reshape(-1).astype(np.float32)})
        for seed in SEEDS:
            model = load(RESULTS / "neuraltfr" / "models" / f"eval_{origin}_s{seed}.pt")
            with torch.no_grad():
                paths = {"original": to_tfr(predict(model, windows)), "flat": to_tfr(model(flat).numpy())}
            for label, values in paths.items():
                frame = keys.assign(seed=seed)
                for index, column in enumerate(COLUMNS):
                    frame[column] = values[..., index].reshape(-1)
                frames[label].append(frame)
    cells = {}
    for label, parts in frames.items():
        by_seed = pd.concat(parts, ignore_index=True)
        cells[label] = scores(by_seed.assign(model=label))[["period", "id", "year", "horizon", "pinball"]].assign(seed=by_seed["seed"].to_numpy())
        cells[label + "_ensemble"] = scores(ensemble(by_seed).assign(model=label))[["period", "id", "year", "horizon", "pinball"]]
    paired = cells["original_ensemble"].merge(cells["flat_ensemble"], on=["period", "id", "year", "horizon"], suffixes=("_original", "_flat"))
    paired["group"] = group_of(level_at(annual, paired["id"], paired["period"]))
    by_seed = cells["original"].merge(cells["flat"], on=["period", "seed", "id", "year", "horizon"], suffixes=("_original", "_flat"))

    def summary(frame, **labels):
        original, flat = frame["pinball_original"].mean(), frame["pinball_flat"].mean()
        return {**labels, "n_populations": frame["id"].nunique(), "original_loss": original, "flat_loss": flat,
                "difference": flat - original, "percent_increase": 100 * (flat / original - 1)}

    rows, rng = [], np.random.default_rng(5502026)
    for period in ("all", *ORIGINS):
        part = paired if period == "all" else paired[paired["period"] == period]
        for group in ("all", *GROUPS):
            rows.append(summary(part if group == "all" else part[part["group"] == group], period=period, group=group))
        per_population = part.groupby("id")[["pinball_original", "pinball_flat"]].mean().to_numpy()
        draws = per_population[rng.integers(0, len(per_population), size=(5000, len(per_population)))].mean(axis=1)
        rows[-len(GROUPS) - 1].update({
            "difference_ci_low": np.quantile(draws[:, 1] - draws[:, 0], 0.025),
            "difference_ci_high": np.quantile(draws[:, 1] - draws[:, 0], 0.975),
            "percent_increase_ci_low": np.quantile(100 * (draws[:, 1] / draws[:, 0] - 1), 0.025),
            "percent_increase_ci_high": np.quantile(100 * (draws[:, 1] / draws[:, 0] - 1), 0.975),
            "n_populations_worse_flat": int((per_population[:, 1] > per_population[:, 0]).sum())})
    seeds = pd.DataFrame([summary(part, period=period, seed=seed) for (period, seed), part in by_seed.groupby(["period", "seed"])])
    horizons = pd.DataFrame([summary(part, period=period, horizon=horizon) for (period, horizon), part in paired.groupby(["period", "horizon"])]
                            + [summary(part, period="all", horizon=horizon) for horizon, part in paired.groupby("horizon")])
    return pd.DataFrame(rows), seeds, horizons


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "metrics_countries.csv")
    ids = cohort(metrics)
    compared = metrics[metrics["in_balanced"] & metrics["id"].isin(ids)]
    paths = forecasts()
    paths = paths[paths["id"].isin(ids) & paths["period"].isin(ORIGINS)]

    tables = {"figB1_composition": composition()}
    tables["figF1_populations"], tables["figF1_medians"] = historical_performance(compared)
    tables["fig5_trajectories_forecasts"], tables["fig5_trajectories_observed"] = trajectories(paths)
    tables["figE1_interval_groups"], tables["appendixD_miss_direction"] = interval_groups(paths)
    tables["fig4_change_by_range"], tables["fig4_zero_crossings"] = fertility_change(paths)
    tables["text_very_low_pairs"], tables["text_recovery_panel"] = very_low_pairs(paths), recovery_panel()
    tables["text_very_low_counts"] = very_low_counts(tables["text_very_low_pairs"], tables["text_recovery_panel"])
    tables["fig6_forward_by_year"], tables["fig6_forward_by_group"], tables["fig6_forward_2038"] = forward()
    tables["appendixC_grid_responses"] = grid_responses()
    (tables["appendixC_flat_history"], tables["appendixC_flat_history_by_seed"],
     tables["appendixC_flat_history_by_horizon"]) = flat_history(ids)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(table):,} rows")


if __name__ == "__main__":
    main()
