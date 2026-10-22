"""The data of the interactive visualizer (index.html), written to data.js.

Reads the repository's own files, so the page shows what the manuscript reports:
    ../data/empirical_tfr.csv                              reported estimates and their annual median
    ../results/neuraltfr/forecast_predictions.parquet      NeuralTFR from 2023, median of the ten fits
    ../data/raw/wpp2024/WPP_2024_2045_TFR.csv              UN WPP 2024, median and 95% bounds
    ../data/raw/gbd2021/GBD_1950_2100_TFR.CSV              GBD 2021 reference scenario, 95% interval
The manuscript projects from 2023 the 149 populations with a reported value that
year. For the page only, the other populations are projected from their latest
reported year with the same fitted networks, without refitting: the ten 2023
fits of NeuralTFR (median of their quantiles), applied to the twelve-year
history ending at that year. These projections are not part of
the manuscript; each country records the year its projection starts from.
Every system is shown up to 2038 with its median and 95% interval; countries carry
their ISO3 (map) and ISO2 (flag) codes from WPP 2024. data.js holds
one global variable, so the page also works when opened from the file system.

    python docs/prepare_data.py      # from the repository root
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from evaluation import annual_medians, ensemble  # noqa: E402
from figures import FORWARD_YEARS, GBD_NAMES  # noqa: E402
from neuraltfr.model import load  # noqa: E402
from neuraltfr.preprocess import annual_panel, to_tfr  # noqa: E402
from neuraltfr.runner import ENC_LEN, FORECAST_ORIGIN, HORIZON, SEEDS, predict  # noqa: E402
from neuraltfr.training import COLUMNS  # noqa: E402
from neuraltfr.windows import _series, forecast_windows  # noqa: E402

DATA, RESULTS, RAW = ROOT / "data", ROOT / "results", ROOT / "data" / "raw"
OUTPUT = HERE / "data.js"
DATASETS = {"un_portal": "UN compilation", "national_sources": "national sources", "eurostat": "Eurostat"}
FIRST, LAST = FORWARD_YEARS[0], FORWARD_YEARS[-1]
DIGITS = 4


def forecast_rows(frame: pd.DataFrame, median: str, lower: str, upper: str) -> dict:
    """{id: [{year, median, lower, upper}]} up to LAST."""
    frame = frame[frame["year"] <= LAST].sort_values(["id", "year"])
    return {int(i): [{"year": int(r.year), "median": round(float(getattr(r, median)), DIGITS),
                      "lower": round(float(getattr(r, lower)), DIGITS), "upper": round(float(getattr(r, upper)), DIGITS)}
                     for r in part.itertuples()]
            for i, part in frame.groupby("id")}


def from_latest_year(observations: pd.DataFrame, ids: list) -> tuple:
    """NeuralTFR, as fitted for 2023, projecting each population in ``ids`` from its latest
    reported year. Returns the forecasts and the year each one starts from."""
    series = _series(annual_panel(observations))
    reported = _series(annual_panel(observations, interpolate=False))
    networks = [load(RESULTS / "neuraltfr" / "models" / f"forecast_{FORECAST_ORIGIN}_s{seed}.pt") for seed in SEEDS]
    frames, origins = [], {}
    for population in ids:
        first, values, _ = series[population]
        origin = origins[population] = first + len(values) - 1
        windows = forecast_windows({population: series[population]}, {population: reported[population]},
                                   origin, ENC_LEN, HORIZON)
        path = np.median([to_tfr(predict(network, windows))[0] for network in networks], axis=0)
        frames.append(pd.DataFrame({"id": population, "year": origin + 1 + np.arange(HORIZON),
                                    **{column: path[:, j] for j, column in enumerate(COLUMNS)}}))
    return pd.concat(frames, ignore_index=True), origins


def normalise(name: str) -> str:
    """Location names compared without accents, case or punctuation (as in figures.forward)."""
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower())


def main() -> None:
    panel = pd.read_csv(DATA / "empirical_tfr.csv")
    names = panel.drop_duplicates("id").set_index("id")["name"]
    ids = sorted(names.index)

    net = pd.read_parquet(RESULTS / "neuraltfr" / "forecast_predictions.parquet")
    net = ensemble(net[net["row_set"] == "test"])
    origins = {int(i): FORECAST_ORIGIN for i in net["id"].unique()}
    extra, latest = from_latest_year(panel, [i for i in ids if i not in origins])
    origins |= latest
    net = pd.concat([net, extra], ignore_index=True)

    wpp = pd.read_csv(RAW / "wpp2024" / "WPP_2024_2045_TFR.csv", keep_default_na=False)   # Namibia's ISO2 is "NA"
    codes = wpp.drop_duplicates("LocationId").set_index("LocationId")[["Iso2", "Iso3"]]
    wpp = (wpp[wpp["LocationId"].isin(ids)].pivot_table(index=["LocationId", "Time"], columns="Variant", values="Value")
           .reset_index().rename(columns={"LocationId": "id", "Time": "year", "Median": "median",
                                          "95% lower bound": "lower", "95% upper bound": "upper"}))

    gbd = pd.read_csv(RAW / "gbd2021" / "GBD_1950_2100_TFR.CSV")
    gbd = gbd[(gbd["scenario_name"] == "Reference") & (gbd["year_id"] >= FIRST)]
    lookup = {normalise(name): name for name in gbd["location_name"].unique()}
    matched = {i: GBD_NAMES.get(i, lookup.get(normalise(names[i]))) for i in ids}
    by_name = {name: i for i, name in matched.items() if name}
    gbd = gbd[gbd["location_name"].isin(by_name)].assign(id=lambda f: f["location_name"].map(by_name),
                                                          year=lambda f: f["year_id"])

    medians = annual_medians().reset_index()
    output = {
        "countries": sorted(({"id": int(i), "name": names[i], "origin": origins[i],
                              "iso3": codes["Iso3"].get(i) or None,                 # matches the map's shapes
                              "flag": (codes["Iso2"].get(i) or "").lower() or None}  # flag-icons code
                             for i in ids),
                            key=lambda c: c["name"]),
        "projectionOrigin": FORECAST_ORIGIN,
        "historical_mean": {int(i): [{"year": int(r.year), "tfr": round(float(r.TFR), DIGITS)} for r in part.itertuples()]
                            for i, part in medians.groupby("id")},
        "historical_points": {int(i): [{"year": int(r.year), "tfr": round(float(r.TFR), DIGITS), "source": r.source,
                                        "method": DATASETS[r.dataset]} for r in part.itertuples()]
                              for i, part in panel.sort_values(["id", "year", "TFR"]).groupby("id")},
        "forecast": {"NeuralTFR": forecast_rows(net, "y_hat_500", "y_hat_025", "y_hat_975"),
                     "WPP": forecast_rows(wpp, "median", "lower", "upper"),
                     "GBD": forecast_rows(gbd, "val", "lower", "upper")},
    }
    output["modelYearRanges"] = {model: {"min": min(d["year"] for rows in data.values() for d in rows),
                                         "max": max(d["year"] for rows in data.values() for d in rows)}
                                 for model, data in output["forecast"].items()}
    OUTPUT.write_text("window.NEURALTFR_DATA = " + json.dumps(output, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"{OUTPUT.name}: {len(ids)} countries ({sum(o == FORECAST_ORIGIN for o in origins.values())} projected from "
          f"{FORECAST_ORIGIN} as in the manuscript, {len(latest)} from their latest reported year), "
          f"{OUTPUT.stat().st_size / 1e6:.1f} MB; populations per system: "
          + ", ".join(f"{model} {len(data)}" for model, data in output["forecast"].items()))


if __name__ == "__main__":
    main()
