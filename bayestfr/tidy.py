"""The recorded BayesTFR summaries in the layout of the other models' forecasts.

Reads ``results/bayestfr/summary_<origin>.csv`` for the historical origins:
the package's projection summary, one row per population and variant, one
column per year. Keeps the median and the 80% and 95% bounds over the fifteen
years after the origin, for the populations of the network's test cohort at
that origin, and writes ``results/bayestfr/eval_predictions.parquet`` with the
reported outcomes beside the forecasts.

    python -m bayestfr.tidy      # from the repository root
"""
import numpy as np
import pandas as pd

from neuraltfr.preprocess import annual_panel
from neuraltfr.runner import DATA, ENC_LEN, HORIZON, MIN_TARGET, ORIGINS, ROOT, prediction_frame
from neuraltfr.training import COLUMNS
from neuraltfr.windows import build_fold

RESULTS = ROOT / "results" / "bayestfr"
VARIANTS = {"lower 95": "y_hat_025", "lower 80": "y_hat_100", "median": "y_hat_500",
            "upper 80": "y_hat_900", "upper 95": "y_hat_975"}


def main() -> None:
    observations = pd.read_csv(DATA)
    truth = annual_panel(observations, interpolate=False)
    frames = []
    for origin in ORIGINS:
        panel = annual_panel(observations[observations["year"] <= origin])
        windows = build_fold(panel, truth, origin, ENC_LEN, HORIZON, MIN_TARGET, valid_gap=0, balanced=False).test
        balanced = windows.ids[windows.mask.all(axis=1)]
        years = [str(origin + step) for step in range(1, HORIZON + 1)]

        summary = pd.read_csv(RESULTS / f"summary_{origin}.csv")
        paths = np.full((len(windows), HORIZON, len(COLUMNS)), np.nan)
        for index, column in enumerate(COLUMNS):
            variant = next(name for name, target in VARIANTS.items() if target == column)
            table = summary[summary["variant"] == variant].set_index("country_code")[years]
            paths[..., index] = table.reindex(windows.ids).to_numpy()

        frame = prediction_frame(windows, paths, balanced, "test", 0).assign(period=origin)
        frames.append(frame[frame["y_hat_500"].notna()])
        print(f"origin {origin}: {frames[-1]['id'].nunique()} populations in both the summary and the test cohort")
    pd.concat(frames, ignore_index=True).to_parquet(RESULTS / "eval_predictions.parquet",
                                                    compression="zstd", index=False)


if __name__ == "__main__":
    main()
