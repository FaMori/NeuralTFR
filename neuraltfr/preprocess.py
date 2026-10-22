"""Annual panel and model scale.

The panel holds one row per reported estimate (``id``, ``year``, ``TFR``).
Each country becomes one annual series: the median of the estimates reported
for a country-year, on a consecutive grid from the country's first to its last
reported year, with interior gaps filled by linear interpolation of TFR.
The model works on z = (log TFR - LOCATION) / SCALE, in single precision.
"""
import numpy as np
import pandas as pd

LOCATION, SCALE = 1.3, 0.55


def annual_panel(observations: pd.DataFrame, interpolate: bool = True) -> pd.DataFrame:
    """One row per country-year: ``id, year, TFR, y, is_observed``.

    ``TFR`` is the annual median where the year has estimates and NaN
    otherwise; ``is_observed`` marks those years. ``y`` is the series on the
    model scale, with interior gaps filled when ``interpolate`` is set.
    """
    annual = observations.groupby(["id", "year"], as_index=False)["TFR"].median()
    frames = []
    for country, group in annual.groupby("id"):
        years = np.arange(group["year"].min(), group["year"].max() + 1)
        series = group.set_index("year")["TFR"].reindex(years)
        frames.append(pd.DataFrame({"id": country, "year": years, "TFR": series.to_numpy()}))
    panel = pd.concat(frames, ignore_index=True)

    tfr = panel["TFR"]
    if interpolate:
        tfr = tfr.groupby(panel["id"]).transform(lambda s: s.interpolate(limit_area="inside"))
    panel["y"] = to_model_scale(tfr)
    panel["is_observed"] = panel["TFR"].notna().astype("float32")
    return panel


def to_model_scale(tfr) -> np.ndarray:
    return ((np.log(np.asarray(tfr, dtype=np.float64)) - LOCATION) / SCALE).astype(np.float32)


def to_tfr(values) -> np.ndarray:
    return np.exp(np.asarray(values, dtype=np.float64) * SCALE + LOCATION)
