"""Training windows and forecast windows for one origin.

A window pairs ``enc_len`` consecutive years of a country's series (the
encoder history) with the ``horizon`` years that follow (the target). Both
come from the annual panel on the model scale.

Training windows slide one year at a time through every country whose series
is long enough. Their targets are masked where the year has no reported value
or lies beyond the training limit, so the most recent windows keep a partial
target: a window is admitted while at least ``min_target`` of its target years
fall inside the limit, and kept if any of them is reported.

Forecast windows are one per country: the history ending at the origin, which
must be complete and end in a reported value, and the ``horizon`` reported
values that follow, taken from the uninterpolated panel so that forecasts are
only scored against reported values.

``build_fold`` is the network's fold, built from the panel cut at the origin.
Its validation forecasts start ``valid_gap`` years earlier, where a history can
end in values interpolated toward an observation inside the validation period;
requiring a reported final value leaves those histories out. Its training
windows are admitted on the panel cut at the origin, which adds a few windows
whose existence, though not their values, depends on later observations. ``validation_fold`` is the regressions' fold: it
takes the panel cut at the validation cut, so that nothing after it is read.
For the same cohort both folds give the same validation windows.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Windows:
    """Aligned arrays, one row per window.

    ``encoder`` and ``target`` are on the model scale; ``target`` is zero and
    ``mask`` is zero wherever the target year is not a reported value.
    """
    ids: np.ndarray       # country of each window
    years: np.ndarray     # (n, horizon) target years
    encoder: np.ndarray   # (n, enc_len) float32
    target: np.ndarray    # (n, horizon) float32
    mask: np.ndarray      # (n, horizon) float32

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, keep: np.ndarray) -> "Windows":
        return Windows(*(array[keep] for array in self.arrays()))

    def arrays(self) -> tuple:
        return self.ids, self.years, self.encoder, self.target, self.mask

    def loader(self, batch_size: int, generator: torch.Generator | None = None) -> DataLoader:
        """Shuffled batches of (encoder, target, mask); ``generator`` fixes the order."""
        tensors = (torch.from_numpy(array) for array in (self.encoder, self.target, self.mask))
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True,
                          generator=generator)


@dataclass(frozen=True)
class Fold:
    train: Windows
    valid: Windows | None   # None when the fold has no validation gap
    test: Windows


def build_fold(panel: pd.DataFrame, truth: pd.DataFrame, origin: int, enc_len: int = 12,
               horizon: int = 15, min_target: int = 5, valid_gap: int = 10,
               balanced: bool = True) -> Fold:
    """Windows for one origin from the panel cut at that origin.

    Training targets stop ``valid_gap`` years before the origin and the gap is
    forecast as nested validation, from the countries of the test set whose
    history ends in a reported value at the cut.
    The test set holds one forecast per country from the origin; ``balanced``
    keeps only countries with a reported value in every target year.
    ``truth`` is the uninterpolated panel of all observations.
    """
    series = _series(panel)
    reported = _series(truth)
    limit = origin - valid_gap

    train = training_windows(series, limit, enc_len, horizon, min_target)
    test = forecast_windows(series, reported, origin, enc_len, horizon)
    if balanced:
        test = test.subset(test.mask.all(axis=1))
    valid = None
    if valid_gap:
        valid = forecast_windows(series, reported, limit, enc_len, valid_gap)
        valid = valid.subset(np.isin(valid.ids, test.ids))
    return Fold(train, valid, test)


def validation_fold(panel: pd.DataFrame, truth: pd.DataFrame, origin: int, cohort: np.ndarray,
                    enc_len: int = 12, horizon: int = 15, min_target: int = 5,
                    valid_gap: int = 10) -> tuple[Windows, Windows]:
    """Nested validation that reads nothing from after the cut.

    ``panel`` must be the panel cut ``valid_gap`` years before the origin, so
    that no value in a window, interpolated or not, depends on the validation
    years. Training targets stop at the cut; the validation set forecasts the
    gap for the countries of ``cohort`` whose history is complete at the cut.
    Returns the training and the validation windows.
    """
    series = _series(panel)
    cut = origin - valid_gap
    train = training_windows(series, cut, enc_len, horizon, min_target)
    valid = forecast_windows(series, _series(truth), cut, enc_len, valid_gap)
    return train, valid.subset(np.isin(valid.ids, cohort))


def training_windows(series: dict, limit: int, enc_len: int, horizon: int,
                     min_target: int) -> Windows:
    parts = []
    for country, (first, values, observed) in series.items():
        last = first + len(values) - 1
        count = min(last, limit) - min_target - (first + enc_len - 1) + 1
        if count <= 0:
            continue
        padded = np.concatenate([values, np.full(horizon, np.nan)])
        padded_mask = np.concatenate([observed, np.zeros(horizon)])
        window = sliding_window_view(padded, enc_len + horizon)[:count]
        years = first + enc_len + np.arange(count)[:, None] + np.arange(horizon)
        mask = sliding_window_view(padded_mask, enc_len + horizon)[:count, enc_len:] * (years <= limit)
        parts.append((np.full(count, country), years, window[:, :enc_len], window[:, enc_len:], mask))
    windows = _stack(parts, enc_len, horizon)
    return windows.subset(windows.mask.any(axis=1))


def forecast_windows(series: dict, reported: dict, origin: int, enc_len: int,
                     horizon: int) -> Windows:
    parts = []
    years = origin + 1 + np.arange(horizon)
    for country, (first, values, history_observed) in series.items():
        start = origin - enc_len + 1 - first
        if start < 0 or origin - first >= len(values) or not history_observed[origin - first]:
            continue
        truth_first, truth, observed = reported[country]
        position = years - truth_first
        inside = (position >= 0) & (position < len(truth))
        clipped = np.clip(position, 0, len(truth) - 1)
        mask = np.where(inside, observed[clipped], 0.0)
        target = np.where(inside, truth[clipped], np.nan)
        parts.append((np.array([country]), years[None], values[None, start:start + enc_len],
                      target[None], mask[None]))
    return _stack(parts, enc_len, horizon)


def _series(panel: pd.DataFrame) -> dict:
    """Per country: first year, model-scale values, and the reported-year mask."""
    return {country: (int(group["year"].iloc[0]),
                      group["y"].to_numpy(np.float64),
                      group["is_observed"].to_numpy(np.float64))
            for country, group in panel.groupby("id")}


def _stack(parts: list, enc_len: int, horizon: int) -> Windows:
    if not parts:
        return Windows(np.zeros(0, dtype=np.int64), np.zeros((0, horizon), dtype=np.int64),
                       np.zeros((0, enc_len), np.float32), np.zeros((0, horizon), np.float32),
                       np.zeros((0, horizon), np.float32))
    ids, years, encoder, target, mask = (np.concatenate(column) for column in zip(*parts))
    mask = mask.astype(np.float32)
    return Windows(ids, years.astype(np.int64), encoder.astype(np.float32),
                   np.where(mask > 0, target, 0.0).astype(np.float32), mask)
