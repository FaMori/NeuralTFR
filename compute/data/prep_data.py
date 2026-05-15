from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_EMPIRICAL_PATH = Path("data/raw/EMPIRICAL_TFR.csv")
DEFAULT_INCLUDE_PATH = Path("data/include_2024.txt")

SMOOTH_FENCE_K = 1.5
SMOOTH_CONSISTENT_QUANTILE = 0.75
SMOOTH_MIN_CONSISTENT_FLAGS = 2
SMOOTH_EWM_SPAN = 5


@dataclass(frozen=True)
class BuildResult:
    data: pd.DataFrame
    quality: pd.DataFrame
    smooth_ids: set[int]
    output_path: Path
    version_name: str


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected True/False, got: {value}")


def default_output_path(smooth: bool) -> Path:
    filename = "tfr_smooth.csv" if smooth else "tfr_no_smooth.csv"
    return ROOT_DIR / "data" / "final" / filename


def load_empirical_panel(
    empirical_path: str | Path = DEFAULT_EMPIRICAL_PATH,
    include_path: str | Path = DEFAULT_INCLUDE_PATH,
) -> pd.DataFrame:
    empirical_df = pd.read_csv(resolve_path(empirical_path))
    include_df = pd.read_table(resolve_path(include_path))
    keep_ids = set(include_df.loc[include_df["include_code"] == 2, "country_code"])
    return empirical_df[empirical_df["id"].isin(keep_ids)].copy().reset_index(drop=True)


def build_yearly_medians(empirical_df: pd.DataFrame) -> pd.DataFrame:
    return (
        empirical_df.groupby(["name", "id", "year", "id_reg", "id_sreg"], as_index=False)["TFR"]
        .median()
        .sort_values(["id", "year"])
        .reset_index(drop=True)
    )


def linear_interpolate_by_series(medians_df: pd.DataFrame) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []

    for id_value, group in medians_df.groupby("id", sort=True):
        group = group.sort_values("year").reset_index(drop=True)
        full_years = pd.DataFrame({"year": np.arange(int(group["year"].min()), int(group["year"].max()) + 1)})
        full_panel = full_years.merge(group, on="year", how="left")
        full_panel["id"] = int(id_value)
        full_panel[["name", "id_reg", "id_sreg"]] = full_panel[["name", "id_reg", "id_sreg"]].ffill().bfill()

        observed_mask = full_panel["TFR"].notna()
        full_panel["TFR"] = full_panel["TFR"].interpolate(method="linear", limit_area="inside")
        full_panel["source"] = np.where(observed_mask, "empirical", "Linear interpolation")

        groups.append(full_panel[["id", "year", "name", "id_reg", "id_sreg", "TFR", "source"]])

    return pd.concat(groups, ignore_index=True).sort_values(["id", "year"]).reset_index(drop=True)


def safe_descending_score(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    if series.isna().all():
        return pd.Series(1.0, index=series.index)

    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or np.isclose(min_val, max_val):
        return pd.Series(1.0, index=series.index)

    return 1 - (series - min_val) / (max_val - min_val)


def eval_quality_series(
    df: pd.DataFrame,
    target_col: str = "TFR",
    time_col: str = "year",
    id_col: str = "id",
) -> pd.DataFrame:
    eval_df = df.copy()
    min_time = int(eval_df[time_col].min())
    max_time = int(eval_df[time_col].max())
    range_time = max_time - min_time + 1

    quality_df = eval_df.groupby(id_col).agg(
        unique_periods=(time_col, "nunique"),
        name=("name", "first"),
        start_year=(time_col, "min"),
        end_year=(time_col, "max"),
    )
    quality_df["cover_score"] = quality_df["unique_periods"] / range_time
    quality_df["span_years"] = quality_df["end_year"] - quality_df["start_year"] + 1
    quality_df["observed_gap_years"] = quality_df["span_years"] - quality_df["unique_periods"]
    quality_df["gap_ratio"] = np.where(
        quality_df["span_years"] > 1,
        quality_df["observed_gap_years"] / quality_df["span_years"],
        0.0,
    )

    def inter_rmad(group: pd.Series) -> float:
        values = group.to_numpy(dtype=float)
        if values.size < 2:
            return np.nan
        median_val = np.median(values)
        return float(np.median(np.abs(values - median_val)))

    def intra_rmad(group: pd.DataFrame) -> float:
        median_series = group.groupby(time_col)[target_col].median().sort_index()
        diff1 = median_series.diff().dropna()
        if diff1.empty:
            return np.nan
        return float(np.median(np.abs(diff1 - np.median(diff1))))

    inter_series = (
        eval_df.groupby([id_col, time_col])[target_col]
        .apply(inter_rmad)
        .groupby(level=0)
        .median()
    )
    intra_series = eval_df.groupby(id_col)[[time_col, target_col]].apply(intra_rmad)

    quality_df["inter_rmad_raw"] = inter_series
    quality_df["intra_rmad_raw"] = intra_series
    quality_df["inter_score"] = safe_descending_score(inter_series)
    quality_df["intra_score"] = safe_descending_score(intra_series)
    quality_df["score"] = (
        quality_df[["cover_score", "inter_score", "intra_score"]]
        .mean(axis=1)
        .fillna(quality_df[["cover_score", "inter_score", "intra_score"]].mean(axis=1).mean())
    )

    return quality_df.reset_index()


def tukey_upper_fence(series: pd.Series, k: float = SMOOTH_FENCE_K) -> float:
    values = series.dropna().astype(float)
    if values.empty:
        return np.nan
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    return q3 if np.isclose(iqr, 0.0) else q3 + k * iqr


def select_smooth_ids(quality_df: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    result = quality_df.copy()
    fence_cols = ["gap_ratio", "inter_rmad_raw", "intra_rmad_raw"]

    fences = {col: tukey_upper_fence(result[col], k=SMOOTH_FENCE_K) for col in fence_cols}
    consistent_cutoffs = {
        col: float(result[col].dropna().quantile(SMOOTH_CONSISTENT_QUANTILE))
        if not result[col].dropna().empty
        else np.nan
        for col in fence_cols
    }

    for col in fence_cols:
        fence = fences[col]
        cutoff = consistent_cutoffs[col]
        result[f"{col}_flag"] = False if pd.isna(fence) else result[col] > fence
        result[f"{col}_consistent_flag"] = False if pd.isna(cutoff) else result[col] > cutoff

    result["need_outlier_flag"] = result[[f"{col}_flag" for col in fence_cols]].any(axis=1)
    result["need_consistent_flag_count"] = result[[f"{col}_consistent_flag" for col in fence_cols]].sum(axis=1)
    result["need_consistent_flag"] = result["need_consistent_flag_count"] >= SMOOTH_MIN_CONSISTENT_FLAGS
    result["smooth"] = result["need_outlier_flag"] | result["need_consistent_flag"]

    def describe_reason(row: pd.Series) -> str:
        reasons = []
        if row["gap_ratio_flag"]:
            reasons.append("high_gap_ratio")
        if row["inter_rmad_raw_flag"]:
            reasons.append("high_inter_dispersion")
        if row["intra_rmad_raw_flag"]:
            reasons.append("high_intra_irregularity")
        if row["need_consistent_flag"] and not reasons:
            reasons.append("consistent_multi_metric_need")
        return ",".join(reasons)

    result["smooth_reason"] = result.apply(describe_reason, axis=1)
    smooth_ids = set(result.loc[result["smooth"], "id"].astype(int))

    return result, smooth_ids


def bidirectional_ewm(values: np.ndarray, span: int = SMOOTH_EWM_SPAN) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    forward = series.ewm(span=span, adjust=False).mean()
    backward = series.iloc[::-1].ewm(span=span, adjust=False).mean().iloc[::-1]
    return np.maximum(((forward + backward) / 2).to_numpy(), 0.05)


def apply_smoothing(base_df: pd.DataFrame, smooth_ids: set[int]) -> pd.DataFrame:
    if not smooth_ids:
        return base_df.copy()

    result = base_df.copy().sort_values(["id", "year"]).reset_index(drop=True)
    for id_value, group in result[result["id"].isin(smooth_ids)].groupby("id", sort=False):
        ordered = group.sort_values("year")
        result.loc[ordered.index, "TFR"] = bidirectional_ewm(ordered["TFR"].to_numpy(dtype=float))
    return result


def build_dataset(
    smooth: bool = True,
    empirical_path: str | Path = DEFAULT_EMPIRICAL_PATH,
    include_path: str | Path = DEFAULT_INCLUDE_PATH,
    output_path: str | Path | None = None,
) -> BuildResult:
    empirical_df = load_empirical_panel(empirical_path=empirical_path, include_path=include_path)
    base_df = linear_interpolate_by_series(build_yearly_medians(empirical_df))
    quality_df = eval_quality_series(empirical_df)
    quality_df, smooth_ids = select_smooth_ids(quality_df)

    final_df = apply_smoothing(base_df, smooth_ids) if smooth else base_df.copy()
    final_df = final_df.sort_values(["id", "year"]).reset_index(drop=True)
    final_df["id"] = final_df["id"].astype(int)
    final_df["year"] = final_df["year"].astype(int)

    version_name = "smooth" if smooth else "no_smooth"
    target_path = default_output_path(smooth) if output_path is None else resolve_path(output_path)

    return BuildResult(
        data=final_df,
        quality=quality_df,
        smooth_ids=smooth_ids if smooth else set(),
        output_path=target_path,
        version_name=version_name,
    )


def prep_data(
    empirical_path: str | Path = DEFAULT_EMPIRICAL_PATH,
    include_path: str | Path = DEFAULT_INCLUDE_PATH,
    output_path: str | Path | None = None,
    smooth: bool = True,
) -> pd.DataFrame:
    result = build_dataset(
        smooth=smooth,
        empirical_path=empirical_path,
        include_path=include_path,
        output_path=output_path,
    )
    result.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(result.output_path, index=False)
    return result.data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final TFR panel, with optional smoothing only for the series that show the strongest quality problems."
    )
    parser.add_argument(
        "--smooth",
        type=parse_bool,
        default=True,
        help="True smooths only the series with the clearest quality issues; False leaves all series unchanged.",
    )
    parser.add_argument(
        "--empirical-path",
        type=str,
        default=str(DEFAULT_EMPIRICAL_PATH),
        help="Empirical input file.",
    )
    parser.add_argument(
        "--include-path",
        type=str,
        default=str(DEFAULT_INCLUDE_PATH),
        help="Country universe include file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit output CSV path. Defaults to data/final/tfr_smooth.csv or data/final/tfr_no_smooth.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset(
        smooth=args.smooth,
        empirical_path=args.empirical_path,
        include_path=args.include_path,
        output_path=args.output_path,
    )

    result.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(result.output_path, index=False)

    print(f"Saved {result.version_name} version to: {result.output_path}")
    print(f"Rows: {len(result.data):,}")
    print(f"Countries: {result.data['id'].nunique()}")
    print(f"Smoothed ids: {len(result.smooth_ids)}")


if __name__ == "__main__":
    main()
