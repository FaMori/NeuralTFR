from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FORECAST_DIR = DATA_DIR / "final" / "preds" / "forecast"
OUT_DIR = ROOT / "results" / "plots"


PALETTE = {
    "NeuralTFR": "#FF3400",
    "BayesTFR": "#99FF00",
    "GBD": "#C97B00",
}

CONTOUR_PALETTE = {
    "NeuralTFR": "#FC5000",
    "BayesTFR": "#00C745",
    "GBD": "#9A5E00",
}


def _load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    hist = pd.read_csv(DATA_DIR / "final" / "tfr.csv")
    pop = pd.read_csv(DATA_DIR / "pop.csv")

    forecast_map = {
        "NeuralTFR": pd.read_csv(FORECAST_DIR / "NeuralTFR.csv"),
        "BayesTFR": pd.read_csv(FORECAST_DIR / "wpp.csv"),
        "GBD": pd.read_csv(FORECAST_DIR / "gbd.csv"),
    }
    return hist, {"pop": pop, **forecast_map}


def _shared_country_sample(forecasts: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    ntfr = forecasts["NeuralTFR"].copy()
    ntfr["year"] = ntfr["year"].astype(int)

    min_year = ntfr.groupby("id")["year"].min()
    valid_ntfr_ids = set(min_year[min_year >= 2020].index.astype(str))

    shared_ids = valid_ntfr_ids.copy()
    processed: dict[str, pd.DataFrame] = {}
    for label, df in forecasts.items():
        clean = df.copy()
        clean["id"] = clean["id"].astype(str)
        clean["year"] = clean["year"].astype(int)
        clean = clean[clean["year"] <= 2042].copy()
        clean = clean[clean["year"] >= 2024].copy()
        processed[label] = clean
        shared_ids &= set(clean["id"].unique())

    shared = sorted(shared_ids)
    for label, df in processed.items():
        processed[label] = df[df["id"].isin(shared)].copy()

    return processed, shared


def _weighted_global_series(
    forecasts: dict[str, pd.DataFrame], pop: pd.DataFrame, ids: list[str]
) -> pd.DataFrame:
    pop = pop.copy()
    pop["id"] = pop["id"].astype(str)
    pop = pop[pop["id"].isin(ids)][["id", "Pop"]].copy()

    rows = []
    for label, df in forecasts.items():
        merged = df.merge(pop, on="id", how="inner")
        weighted = (
            merged.assign(weighted=lambda x: x["y_hat_50"] * x["Pop"])
            .groupby("year", as_index=False)
            .agg(weighted=("weighted", "sum"), pop=("Pop", "sum"))
        )
        weighted["value"] = weighted["weighted"] / weighted["pop"]
        weighted["model"] = label
        rows.append(weighted[["year", "value", "model"]])

    return pd.concat(rows, ignore_index=True)


def _weighted_history(hist: pd.DataFrame, pop: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    hist = hist.copy()
    hist["id"] = hist["id"].astype(str)
    pop = pop.copy()
    pop["id"] = pop["id"].astype(str)

    merged = hist[hist["id"].isin(ids)].merge(pop[["id", "Pop"]], on="id", how="inner")
    merged = merged[merged["year"] >= 2008].copy()
    coverage = merged.groupby("year")["id"].nunique().reset_index(name="n_ids")
    valid_years = coverage.loc[coverage["n_ids"] >= int(0.8 * len(ids)), "year"]
    merged = merged[merged["year"].isin(valid_years)].copy()
    weighted = (
        merged.assign(weighted=lambda x: x["TFR"] * x["Pop"])
        .groupby("year", as_index=False)
        .agg(weighted=("weighted", "sum"), pop=("Pop", "sum"))
    )
    weighted["value"] = weighted["weighted"] / weighted["pop"]
    weighted = weighted[weighted["year"] <= 2021].copy()
    return weighted[["year", "value"]]


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_trajectory_plot(history: pd.DataFrame, global_series: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.titlesize": 17,
            "axes.labelsize": 14.4,
            "xtick.labelsize": 13.4,
            "ytick.labelsize": 13.4,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    forecast_order = ["NeuralTFR", "BayesTFR", "GBD"]

    for label in forecast_order:
        df = global_series[global_series["model"] == label].sort_values("year")
        ax.plot(
            df["year"],
            df["value"],
            color=PALETTE[label],
            linewidth=0.6,
            linestyle=(0, (10.0, 5.0)),
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            df["year"],
            df["value"],
            s=37,
            facecolors=mcolors.to_rgba(PALETTE[label], 0.20),
            edgecolors=mcolors.to_rgba(CONTOUR_PALETTE[label], 0.92),
            linewidths=0.34,
            zorder=4,
        )

    legend_handles = []
    for label in forecast_order:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=PALETTE[label],
                lw=0.6,
                linestyle=(0, (10.0, 5.0)),
                marker="o",
                markersize=6.8,
                markerfacecolor=mcolors.to_rgba(PALETTE[label], 0.20),
                markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE[label], 0.92),
                markeredgewidth=0.54,
                label=label,
            )
        )

    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.72,
        labelspacing=0.56,
        handlelength=2.2,
        handletextpad=0.78,
        prop={"size": 13.8},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Population-weighted TFR")
    ax.set_xlim(2023.4, 2043.0)
    ax.set_ylim(1.40, 1.75)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.grid(axis="x", which="major", color="#D7DEE7", linewidth=0.8, alpha=0.70)
    ax.grid(axis="y", which="major", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.tick_params(axis="both", length=0, colors="#475569")
    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)
    ax.spines["bottom"].set_color("#9AA4B2")
    ax.spines["bottom"].set_linewidth(0.8)

    _save(fig, "forecast_weighted_trajectory")


def make_relative_decline_plot(global_series: pd.DataFrame) -> None:
    base = global_series[global_series["year"] >= 2024].copy()
    pivot = base.pivot(index="year", columns="model", values="value")
    delta = pivot.subtract(pivot.loc[2024], axis=1)
    spread = delta.subtract(delta["BayesTFR"], axis=0)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 14.4,
            "xtick.labelsize": 13.4,
            "ytick.labelsize": 13.4,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    for label in ["NeuralTFR", "GBD"]:
        ax.plot(
            spread.index,
            spread[label],
            color=PALETTE[label],
            linewidth=0.6,
            linestyle=(0, (10.0, 5.0)),
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            spread.index,
            spread[label],
            s=37,
            facecolors=mcolors.to_rgba(PALETTE[label], 0.20),
            edgecolors=mcolors.to_rgba(CONTOUR_PALETTE[label], 0.92),
            linewidths=0.34,
            zorder=4,
        )

    ax.axhline(
        0,
        color="#2F343B",
        linewidth=0.82,
        linestyle=(0, (10.0, 5.0)),
        zorder=2,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#2F343B",
            lw=0.82,
            linestyle=(0, (10.0, 5.0)),
            label="BayesTFR (baseline)",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["NeuralTFR"],
            lw=0.6,
            linestyle=(0, (10.0, 5.0)),
            marker="o",
            markersize=6.8,
            markerfacecolor=mcolors.to_rgba(PALETTE["NeuralTFR"], 0.20),
            markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE["NeuralTFR"], 0.92),
            markeredgewidth=0.54,
            label="NeuralTFR",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["GBD"],
            lw=0.6,
            linestyle=(0, (10.0, 5.0)),
            marker="o",
            markersize=6.8,
            markerfacecolor=mcolors.to_rgba(PALETTE["GBD"], 0.20),
            markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE["GBD"], 0.92),
            markeredgewidth=0.54,
            label="GBD",
        ),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.035),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.72,
        labelspacing=0.56,
        handlelength=2.2,
        handletextpad=0.78,
        prop={"size": 13.8},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.set_xlim(2023.4, 2043.0)
    ax.set_ylim(-0.16, 0.008)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.grid(axis="x", which="major", color="#D7DEE7", linewidth=0.8, alpha=0.70)
    ax.grid(axis="y", which="major", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.tick_params(axis="both", length=0, colors="#475569")
    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)
    ax.spines["bottom"].set_color("#9AA4B2")
    ax.spines["bottom"].set_linewidth(0.8)

    _save(fig, "forecast_relative_decline")


def make_distribution_plot(forecasts: dict[str, pd.DataFrame]) -> None:
    year = 2040
    frames = []
    for label, df in forecasts.items():
        curr = df[df["year"] == year][["id", "y_hat_50"]].copy()
        curr["model"] = label
        frames.append(curr)
    dist = pd.concat(frames, ignore_index=True)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11.8, 7.0))

    order = ["NeuralTFR", "BayesTFR", "GBD"]
    sns.violinplot(
        data=dist,
        x="model",
        hue="model",
        y="y_hat_50",
        order=order,
        palette=[PALETTE[m] for m in order],
        legend=False,
        inner=None,
        cut=0,
        linewidth=1.0,
        ax=ax,
    )
    sns.boxplot(
        data=dist,
        x="model",
        y="y_hat_50",
        order=order,
        width=0.16,
        showcaps=True,
        boxprops={"facecolor": "white", "zorder": 3},
        whiskerprops={"linewidth": 1.2},
        medianprops={"color": "#222222", "linewidth": 1.6},
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=dist,
        x="model",
        y="y_hat_50",
        order=order,
        color="#111111",
        alpha=0.22,
        size=2.8,
        jitter=0.18,
        ax=ax,
    )

    ax.axhline(1.5, color="#444444", linestyle="--", linewidth=1.1)
    ax.text(3.45, 1.53, "Low-fertility threshold", ha="right", va="bottom", fontsize=11, color="#444444")
    ax.set_title("Distribution of Country-Level TFR Projections in 2040", pad=16, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Median projected TFR")
    ax.set_ylim(0.65, 4.15)
    ax.tick_params(axis="x", rotation=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    sns.despine(left=False, bottom=False)

    _save(fig, "forecast_distribution_2040")


def make_threshold_curve_plot(forecasts: dict[str, pd.DataFrame]) -> None:
    year = 2040
    thresholds = [round(x, 2) for x in list(pd.Series(range(105, 221, 5)) / 100)]
    rows = []
    for label, df in forecasts.items():
        vals = df[df["year"] == year]["y_hat_50"].astype(float)
        n = len(vals)
        for threshold in thresholds:
            share = 100.0 * (vals < threshold).sum() / n
            rows.append({"model": label, "threshold": threshold, "share": share})
    curve = pd.DataFrame(rows)

    highlight_thresholds = [2.1, 1.5, 1.3]

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )
    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    order = ["NeuralTFR", "BayesTFR", "GBD"]
    handles = []
    for label in order:
        sub = curve[curve["model"] == label].sort_values("threshold")
        ax.plot(
            sub["threshold"],
            sub["share"],
            color=PALETTE[label],
            linewidth=0.72,
            linestyle=(0, (10.0, 5.0)),
            solid_capstyle="round",
            zorder=3,
        )
        high = sub[sub["threshold"].isin(highlight_thresholds)]
        ax.scatter(
            high["threshold"],
            high["share"],
            s=26,
            facecolors=mcolors.to_rgba(PALETTE[label], 0.20),
            edgecolors=mcolors.to_rgba(CONTOUR_PALETTE[label], 0.92),
            linewidths=0.34,
            zorder=4,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=PALETTE[label],
                lw=0.72,
                linestyle=(0, (10.0, 5.0)),
                marker="o",
                markersize=4.8,
                markerfacecolor=mcolors.to_rgba(PALETTE[label], 0.20),
                markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE[label], 0.92),
                markeredgewidth=0.34,
                label=label,
            )
        )

    for t, lbl in [(2.1, "Replacement"), (1.5, "Very low"), (1.3, "Ultra-low")]:
        ax.axvline(t, color="#8E98A4", linewidth=0.6, linestyle=":", zorder=1)
        ax.text(
            t,
            101.5,
            lbl,
            ha="center",
            va="bottom",
            fontsize=10.2,
            color="#5C6673",
        )

    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.04),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.65,
        labelspacing=0.5,
        handlelength=2.0,
        handletextpad=0.7,
        prop={"size": 10.6},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    ax.set_xlabel("TFR threshold in 2040")
    ax.set_ylabel("Countries below threshold (%)")
    ax.set_xlim(1.02, 2.23)
    ax.set_ylim(0, 103)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="x", which="major", color="#D7DEE7", linewidth=0.8, alpha=0.70)
    ax.grid(axis="y", which="major", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.tick_params(axis="both", length=0, colors="#475569")
    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)
    ax.spines["bottom"].set_color("#9AA4B2")
    ax.spines["bottom"].set_linewidth(0.8)

    _save(fig, "forecast_threshold_curve_2040")


def make_country_matrix_plot(hist: pd.DataFrame) -> None:
    year = 2042
    base = DATA_DIR / "final" / "preds" / "forecast"
    nt = pd.read_csv(base / "NeuralTFR.csv")
    bt = pd.read_csv(base / "wpp.csv")
    codes = pd.read_csv(DATA_DIR / "countries_codes.csv", encoding="utf-8")

    for df in (nt, bt):
        df["id"] = df["id"].astype(str)
        df["year"] = df["year"].astype(int)

    min_year = nt.groupby("id")["year"].min()
    valid_ids = set(min_year[min_year >= 2020].index)
    bt_ids = set(bt[bt["year"] <= year]["id"].unique())
    shared_ids = sorted(valid_ids & bt_ids)

    names = (
        hist[["id", "name"]]
        .drop_duplicates()
        .assign(id=lambda x: x["id"].astype(str))
        .set_index("id")["name"]
        .to_dict()
    )

    codes["id"] = codes["country-code"].astype(str).str.lstrip("0")
    codes.loc[codes["id"] == "", "id"] = "0"
    region_map = codes.set_index("id")["region"].to_dict()

    nt_y = nt[(nt["id"].isin(shared_ids)) & (nt["year"] == year)][["id", "y_hat_50"]].rename(columns={"y_hat_50": "NeuralTFR"})
    bt_y = bt[(bt["id"].isin(shared_ids)) & (bt["year"] == year)][["id", "y_hat_50"]].rename(columns={"y_hat_50": "BayesTFR"})
    comp = nt_y.merge(bt_y, on="id", how="inner")
    comp["country"] = comp["id"].map(names)
    comp["region"] = comp["id"].map(region_map)
    comp["lowest_model"] = comp[["BayesTFR", "NeuralTFR"]].idxmin(axis=1)
    comp["lowest_value"] = comp[["BayesTFR", "NeuralTFR"]].min(axis=1)
    comp["spread"] = (comp["NeuralTFR"] - comp["BayesTFR"]).abs()

    region_order = ["Africa", "Americas", "Asia", "Europe", "Oceania"]
    comp["region"] = pd.Categorical(comp["region"], categories=region_order, ordered=True)
    comp = comp.sort_values(["region", "lowest_model", "lowest_value", "spread", "country"]).reset_index(drop=True)

    grouped = {region: comp[comp["region"] == region].copy() for region in region_order}
    max_rows = max(len(df) for df in grouped.values())

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 12,
            "xtick.labelsize": 10.6,
            "ytick.labelsize": 7.0,
        }
    )

    fig, axes = plt.subplots(1, len(region_order), figsize=(14.5, 24), sharey=False)
    fig.patch.set_facecolor("#FFFFFF")

    for ax, region in zip(axes, region_order):
        ax.set_facecolor("#FBFCFE")
        df = grouped[region]
        ax.set_xlim(0, 1)
        ax.set_ylim(max_rows, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(region, fontsize=12.5, color="#475569", pad=12)

        for row_idx, (_, row) in enumerate(df.iterrows()):
            face = mcolors.to_rgba(PALETTE[row["lowest_model"]], 0.26)
            edge = mcolors.to_rgba(CONTOUR_PALETTE[row["lowest_model"]], 0.98)
            ax.text(
                0.02,
                row_idx + 0.5,
                row["country"],
                ha="left",
                va="center",
                fontsize=7.1,
                color="#475569",
            )
            ax.add_patch(
                Rectangle(
                    (0.72, row_idx + 0.14),
                    0.24,
                    0.72,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.7,
                )
            )
            ax.text(
                0.84,
                row_idx + 0.5,
                f"{row['lowest_value']:.2f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="#2F343B",
            )

        for y in range(max_rows + 1):
            ax.hlines(y, 0, 1, color="#E1E7EF", linewidth=0.45, zorder=0)

        for spine_name in ["top", "right", "left", "bottom"]:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color("#C9D2DE")
            ax.spines[spine_name].set_linewidth(0.55)

    legend_handles = [
        Patch(
            facecolor=mcolors.to_rgba(PALETTE[label], 0.25),
            edgecolor=mcolors.to_rgba(CONTOUR_PALETTE[label], 0.98),
            linewidth=0.85,
            label=f"Lowest: {label}",
        )
        for label in ["BayesTFR", "NeuralTFR"]
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.55,
        labelspacing=0.6,
        handlelength=1.8,
        handletextpad=0.6,
        prop={"size": 10.4},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    fig.subplots_adjust(left=0.04, right=0.985, top=0.965, bottom=0.045, wspace=0.08)

    _save(fig, "forecast_country_matrix_2042")


def _get_two_model_country_comparison(hist: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    year = 2042
    base = DATA_DIR / "final" / "preds" / "forecast"
    nt = pd.read_csv(base / "NeuralTFR.csv")
    bt = pd.read_csv(base / "wpp.csv")
    codes = pd.read_csv(DATA_DIR / "countries_codes.csv", encoding="utf-8")

    for df in (nt, bt):
        df["id"] = df["id"].astype(str)
        df["year"] = df["year"].astype(int)

    min_year = nt.groupby("id")["year"].min()
    valid_ids = set(min_year[min_year >= 2020].index)
    bt_ids = set(bt[bt["year"] <= year]["id"].unique())
    shared_ids = sorted(valid_ids & bt_ids)

    names = (
        hist[["id", "name"]]
        .drop_duplicates()
        .assign(id=lambda x: x["id"].astype(str))
        .set_index("id")["name"]
        .to_dict()
    )

    codes["id"] = codes["country-code"].astype(str).str.lstrip("0")
    codes.loc[codes["id"] == "", "id"] = "0"
    region_map = codes.set_index("id")["region"].to_dict()

    nt_y = nt[(nt["id"].isin(shared_ids)) & (nt["year"] == year)][["id", "y_hat_50"]].rename(columns={"y_hat_50": "NeuralTFR"})
    bt_y = bt[(bt["id"].isin(shared_ids)) & (bt["year"] == year)][["id", "y_hat_50"]].rename(columns={"y_hat_50": "BayesTFR"})
    comp = nt_y.merge(bt_y, on="id", how="inner")
    comp["country"] = comp["id"].map(names)
    comp["region"] = comp["id"].map(region_map)
    comp["lowest_model"] = comp[["BayesTFR", "NeuralTFR"]].idxmin(axis=1)
    comp["lowest_value"] = comp[["BayesTFR", "NeuralTFR"]].min(axis=1)
    comp["gap"] = comp["NeuralTFR"] - comp["BayesTFR"]
    comp["spread"] = comp["gap"].abs()

    region_order = ["Africa", "Americas", "Asia", "Europe", "Oceania"]
    comp["region"] = pd.Categorical(comp["region"], categories=region_order, ordered=True)
    comp = comp.sort_values(["region", "country"]).reset_index(drop=True)
    return comp, year, region_order


def make_dumbbell_variant_plot(hist: pd.DataFrame) -> None:
    comp, year, region_order = _get_two_model_country_comparison(hist)
    grouped = {region: comp[comp["region"] == region].sort_values("spread", ascending=False).reset_index(drop=True) for region in region_order}
    max_rows = max(len(df) for df in grouped.values())

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 6.6,
        }
    )
    fig, axes = plt.subplots(1, len(region_order), figsize=(14.8, 24), sharey=False)
    fig.patch.set_facecolor("#FFFFFF")

    for ax, region in zip(axes, region_order):
        ax.set_facecolor("#FBFCFE")
        df = grouped[region]
        ax.set_xlim(0.95, 2.35)
        ax.set_ylim(max_rows, 0)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["country"].tolist())
        ax.tick_params(axis="y", length=0, pad=4, colors="#475569")
        ax.tick_params(axis="x", length=0, colors="#475569")
        ax.set_title(region, fontsize=12.3, color="#475569", pad=12)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(axis="x", color="#E1E7EF", linewidth=0.55)
        ax.grid(axis="y", visible=False)

        for row_idx, (_, row) in enumerate(df.iterrows()):
            ax.plot([row["BayesTFR"], row["NeuralTFR"]], [row_idx, row_idx], color="#B7C1CC", linewidth=0.65, zorder=1)
            for label in ["BayesTFR", "NeuralTFR"]:
                ax.scatter(
                    row[label], row_idx, s=20,
                    facecolors=mcolors.to_rgba(PALETTE[label], 0.22),
                    edgecolors=mcolors.to_rgba(CONTOUR_PALETTE[label], 0.98),
                    linewidths=0.34, zorder=3,
                )

        for spine_name in ["top", "right", "left", "bottom"]:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color("#C9D2DE")
            ax.spines[spine_name].set_linewidth(0.55)

    legend_handles = [
        Line2D([0], [0], color=PALETTE[label], lw=0, marker="o", markersize=4.8,
               markerfacecolor=mcolors.to_rgba(PALETTE[label], 0.22),
               markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE[label], 0.98),
               markeredgewidth=0.34, label=label)
        for label in ["BayesTFR", "NeuralTFR"]
    ]
    legend = fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2,
                        frameon=True, fancybox=False, framealpha=1.0, borderpad=0.55, labelspacing=0.6,
                        handlelength=1.8, handletextpad=0.6, prop={"size": 10.4})
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.965, bottom=0.045, wspace=0.08)
    _save(fig, "forecast_variant_dumbbell_2042")


def make_gap_heatmap_variant_plot(hist: pd.DataFrame) -> None:
    comp, year, region_order = _get_two_model_country_comparison(hist)
    grouped = {region: comp[comp["region"] == region].sort_values("gap").reset_index(drop=True) for region in region_order}
    max_rows = max(len(df) for df in grouped.values())

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "ytick.labelsize": 6.9,
        }
    )
    fig, axes = plt.subplots(1, len(region_order), figsize=(12.8, 24), sharey=False)
    fig.patch.set_facecolor("#FFFFFF")
    cmap = sns.diverging_palette(145, 20, s=80, l=55, center="light", as_cmap=True)
    norm = mcolors.TwoSlopeNorm(vmin=-0.18, vcenter=0, vmax=0.18)

    for ax, region in zip(axes, region_order):
        ax.set_facecolor("#FBFCFE")
        df = grouped[region]
        ax.set_xlim(0, 1)
        ax.set_ylim(max_rows, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(region, fontsize=12.3, color="#475569", pad=12)
        for row_idx, (_, row) in enumerate(df.iterrows()):
            ax.text(0.02, row_idx + 0.5, row["country"], ha="left", va="center", fontsize=7.0, color="#475569")
            ax.add_patch(Rectangle((0.73, row_idx + 0.14), 0.22, 0.72, facecolor=cmap(norm(row["gap"])),
                                   edgecolor="#C9D2DE", linewidth=0.55))
            if abs(row["gap"]) >= 0.08:
                ax.text(0.84, row_idx + 0.5, f"{row['gap']:+.2f}", ha="center", va="center", fontsize=6.6, color="#2F343B")
        for y in range(max_rows + 1):
            ax.hlines(y, 0, 1, color="#E1E7EF", linewidth=0.45, zorder=0)
        for spine_name in ["top", "right", "left", "bottom"]:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color("#C9D2DE")
            ax.spines[spine_name].set_linewidth(0.55)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("NeuralTFR - BayesTFR in 2042", color="#475569")
    cbar.ax.tick_params(colors="#475569", length=0, labelsize=9.6)
    fig.subplots_adjust(left=0.05, right=0.91, top=0.965, bottom=0.03, wspace=0.08)
    _save(fig, "forecast_variant_gap_heatmap_2042")


def make_threshold_ladder_variant_plot(hist: pd.DataFrame) -> None:
    comp, year, region_order = _get_two_model_country_comparison(hist)
    grouped = {region: comp[comp["region"] == region].sort_values("lowest_value").reset_index(drop=True) for region in region_order}
    max_rows = max(len(df) for df in grouped.values())

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 6.6,
        }
    )
    fig, axes = plt.subplots(1, len(region_order), figsize=(14.8, 24), sharey=False)
    fig.patch.set_facecolor("#FFFFFF")

    for ax, region in zip(axes, region_order):
        ax.set_facecolor("#FBFCFE")
        df = grouped[region]
        ax.set_xlim(0.95, 2.35)
        ax.set_ylim(max_rows, 0)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["country"].tolist())
        ax.tick_params(axis="y", length=0, pad=4, colors="#475569")
        ax.tick_params(axis="x", length=0, colors="#475569")
        ax.set_title(region, fontsize=12.3, color="#475569", pad=12)
        for t in [2.1, 1.5, 1.3]:
            ax.axvline(t, color="#B8C2CC", linewidth=0.55, linestyle=":", zorder=0)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(axis="x", color="#E1E7EF", linewidth=0.55)
        ax.grid(axis="y", visible=False)
        for row_idx, (_, row) in enumerate(df.iterrows()):
            ax.scatter(row["BayesTFR"], row_idx, s=18, marker="s",
                       facecolors=mcolors.to_rgba(PALETTE["BayesTFR"], 0.22),
                       edgecolors=mcolors.to_rgba(CONTOUR_PALETTE["BayesTFR"], 0.98),
                       linewidths=0.34, zorder=3)
            ax.scatter(row["NeuralTFR"], row_idx, s=18, marker="o",
                       facecolors=mcolors.to_rgba(PALETTE["NeuralTFR"], 0.22),
                       edgecolors=mcolors.to_rgba(CONTOUR_PALETTE["NeuralTFR"], 0.98),
                       linewidths=0.34, zorder=3)
        for spine_name in ["top", "right", "left", "bottom"]:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color("#C9D2DE")
            ax.spines[spine_name].set_linewidth(0.55)

    legend_handles = [
        Line2D([0], [0], color=PALETTE["BayesTFR"], lw=0, marker="s", markersize=4.8,
               markerfacecolor=mcolors.to_rgba(PALETTE["BayesTFR"], 0.22),
               markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE["BayesTFR"], 0.98),
               markeredgewidth=0.34, label="BayesTFR"),
        Line2D([0], [0], color=PALETTE["NeuralTFR"], lw=0, marker="o", markersize=4.8,
               markerfacecolor=mcolors.to_rgba(PALETTE["NeuralTFR"], 0.22),
               markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE["NeuralTFR"], 0.98),
               markeredgewidth=0.34, label="NeuralTFR"),
    ]
    legend = fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2,
                        frameon=True, fancybox=False, framealpha=1.0, borderpad=0.55, labelspacing=0.6,
                        handlelength=1.8, handletextpad=0.6, prop={"size": 10.4})
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.965, bottom=0.045, wspace=0.08)
    _save(fig, "forecast_variant_threshold_ladder_2042")


def _short_country_name(name: str) -> str:
    replacements = {
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "Netherlands, Kingdom of the": "Netherlands",
        "Russian Federation": "Russia",
        "Venezuela, Bolivarian Republic of": "Venezuela",
        "Iran, Islamic Republic of": "Iran",
        "Moldova, Republic of": "Moldova",
        "Korea, Republic of": "South Korea",
        "Syrian Arab Republic": "Syria",
        "Lao People's Democratic Republic": "Laos",
        "Tanzania, United Republic of": "Tanzania",
        "Bolivia, Plurinational State of": "Bolivia",
        "Brunei Darussalam": "Brunei",
        "Micronesia, Federated States of": "Micronesia",
        "Viet Nam": "Vietnam",
        "Türkiye": "Turkey",
        "Czechia": "Czech Republic",
        "Czech Republic": "Czech Rep.",
        "North Macedonia": "Macedonia",
        "Bonaire, Sint Eustatius and Saba": "Bonaire/Sint Eustatius/Saba",
        "Saint Vincent and the Grenadines": "St. Vincent & Gren.",
        "Antigua and Barbuda": "Antigua and Barbuda",
        "Dominican Republic": "Dominican Rep.",
        "Bosnia and Herzegovina": "Bosnia-Herz.",
        "United States of America": "United States",
        "United Arab Emirates": "UAE",
        "Taiwan, Province of China": "Taiwan",
        "Sub-Saharan Africa": "Sub-Saharan Africa",
    }
    return replacements.get(name, name)


def _assign_geo_group(row: pd.Series) -> str:
    region = row.get("region")
    sub_region = row.get("sub_region")
    intermediate_region = row.get("intermediate_region")

    if region == "Europe":
        if sub_region in {"Northern Europe", "Western Europe"}:
            return "Northern & Western Europe"
        return "Southern & Eastern Europe"

    if region == "Asia":
        if sub_region in {"Eastern Asia", "South-eastern Asia"}:
            return "East & Southeast Asia"
        if sub_region in {"Southern Asia", "Central Asia"}:
            return "South & Central Asia"
        return "Middle East"

    if region == "Americas":
        if intermediate_region == "South America":
            return "South America"
        return "North/Central America & Caribbean"

    if region == "Africa":
        if sub_region == "Northern Africa":
            return "North Africa"
        return "Sub-Saharan Africa"

    return "Oceania"


def _get_three_model_country_comparison(hist: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    year = 2042
    base = DATA_DIR / "final" / "preds" / "forecast"
    nt = pd.read_csv(base / "NeuralTFR.csv")
    bt = pd.read_csv(base / "wpp.csv")
    gbd = pd.read_csv(base / "gbd.csv")
    codes = pd.read_csv(DATA_DIR / "countries_codes.csv", encoding="utf-8")

    for df in (nt, bt, gbd):
        df["id"] = df["id"].astype(str)
        df["year"] = df["year"].astype(int)

    min_year = nt.groupby("id")["year"].min()
    valid_ids = set(min_year[min_year >= 2020].index)
    gbd_ids = set(gbd[gbd["year"] <= year]["id"].unique())
    shared_ids = sorted(valid_ids & gbd_ids)

    names = (
        hist[["id", "name"]]
        .drop_duplicates()
        .assign(id=lambda x: x["id"].astype(str))
        .set_index("id")["name"]
        .to_dict()
    )

    codes["id"] = codes["country-code"].astype(str).str.lstrip("0")
    codes.loc[codes["id"] == "", "id"] = "0"
    meta = (
        codes.rename(
            columns={
                "sub-region": "sub_region",
                "intermediate-region": "intermediate_region",
            }
        )[["id", "region", "sub_region", "intermediate_region"]]
        .drop_duplicates()
    )

    frames = []
    for label, df in [("BayesTFR", bt), ("NeuralTFR", nt), ("GBD", gbd)]:
        curr = df[(df["id"].isin(shared_ids)) & (df["year"] == year)][["id", "y_hat_50"]].copy()
        curr = curr.rename(columns={"y_hat_50": label})
        frames.append(curr)

    comp = frames[0].merge(frames[1], on="id", how="inner").merge(frames[2], on="id", how="inner")
    comp["country"] = comp["id"].map(names)
    comp = comp.merge(meta, on="id", how="left")
    comp["country_label"] = comp["country"].map(_short_country_name)
    comp["geo_group"] = comp.apply(_assign_geo_group, axis=1)
    comp["min_value"] = comp[["BayesTFR", "NeuralTFR", "GBD"]].min(axis=1)
    comp["avg_value"] = comp[["BayesTFR", "NeuralTFR", "GBD"]].mean(axis=1)
    return comp, year


def _plot_threshold_ladder_panel_grid(
    df: pd.DataFrame,
    panel_rows: list[list[str | None]],
    filename: str,
    figsize: tuple[float, float],
) -> None:
    groups = [panel for row in panel_rows for panel in row if panel is not None]
    grouped = {
        group: df[df["geo_group"] == group].sort_values(["avg_value", "country_label"]).reset_index(drop=True)
        for group in groups
    }
    figure_vals = pd.concat([grouped[group][m] for group in groups for m in ["BayesTFR", "NeuralTFR", "GBD"]])
    shared_x_min = 0.6
    shared_x_max = min(4.2, math.ceil((figure_vals.max() + 0.25) / 0.1) * 0.1)
    if (shared_x_max - shared_x_min) < 1.8:
        shared_x_max = min(4.2, shared_x_min + 1.8)
    shared_max_label_len = max(grouped[group]["country_label"].str.len().max() for group in groups)
    shared_label_ratio = min(max(1.05, 0.09 * shared_max_label_len), 1.9)
    shared_plot_ratio = 4.75 - shared_label_ratio
    row_max_counts = [max(len(grouped[group]) for group in row if group is not None) for row in panel_rows]
    row_panel_bodies = [max(count, 4) * 0.24 for count in row_max_counts]
    row_heights = [panel_body + 0.22 for panel_body in row_panel_bodies]

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "xtick.labelsize": 12.2,
            "ytick.labelsize": 11.8,
        }
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        len(panel_rows),
        len(panel_rows[0]),
        height_ratios=row_heights,
        hspace=0.015,
        wspace=0.015,
    )
    fig.patch.set_facecolor("#FFFFFF")

    marker_map = {"BayesTFR": "s", "NeuralTFR": "o", "GBD": "^"}
    order = ["BayesTFR", "NeuralTFR", "GBD"]

    for r_idx, row in enumerate(panel_rows):
        row_max_count = row_max_counts[r_idx]
        row_panel_body = row_panel_bodies[r_idx]
        c_idx = 0
        while c_idx < len(row):
            group = row[c_idx]
            if group is None:
                ax = fig.add_subplot(gs[r_idx, c_idx])
                ax.axis("off")
                c_idx += 1
                continue

            sub = grouped[group]
            n = len(sub)
            if n == 1:
                y_positions = [row_panel_body / 2]
            else:
                edge_pad = min(row_panel_body * 0.28, 0.14 + 0.035 * max(row_max_count - n, 0))
                y_positions = np.linspace(edge_pad, row_panel_body - edge_pad, n).tolist()
            x_min = shared_x_min
            x_max = shared_x_max

            span = 1
            mirror_labels = c_idx == 1
            total_ratio = shared_label_ratio + shared_plot_ratio
            if mirror_labels:
                header_x = (shared_plot_ratio / 2) / total_ratio
            else:
                header_x = (shared_label_ratio + shared_plot_ratio / 2) / total_ratio

            panel_spec = gs[r_idx, c_idx : c_idx + span]
            cell = panel_spec.subgridspec(
                2,
                2,
                height_ratios=[0.22, row_panel_body],
                width_ratios=[shared_plot_ratio, shared_label_ratio] if mirror_labels else [shared_label_ratio, shared_plot_ratio],
                hspace=0.0,
                wspace=0.002,
            )
            header_ax = fig.add_subplot(cell[0, :])
            if mirror_labels:
                plot_ax = fig.add_subplot(cell[1, 0])
                label_ax = fig.add_subplot(cell[1, 1])
            else:
                label_ax = fig.add_subplot(cell[1, 0])
                plot_ax = fig.add_subplot(cell[1, 1])

            header_ax.axis("off")
            header_ax.text(
                header_x,
                0.04,
                group,
                transform=header_ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=15.0,
                fontweight="bold",
                color="#415264",
            )

            label_ax.set_ylim(row_panel_body + 0.06, -0.06)
            plot_ax.set_ylim(row_panel_body + 0.06, -0.06)

            label_ax.set_xlim(0, 1)
            label_ax.axis("off")

            for row_idx, (_, row_data) in enumerate(sub.iterrows()):
                label_ax.text(
                    0.01 if mirror_labels else 0.99,
                    y_positions[row_idx],
                    row_data["country_label"],
                    ha="left" if mirror_labels else "right",
                    va="center",
                    fontsize=13.8,
                    fontweight="bold",
                    color="#43515F",
                )

            plot_ax.set_facecolor("#FFFFFF")
            plot_ax.set_xlim(x_min, x_max)
            plot_ax.set_yticks([])
            plot_ax.tick_params(axis="x", length=0, colors="#4B5563", pad=2, labelsize=13.0)

            spans = [
                (x_min, 1.3, "#FFE8CC", 1.0),
                (1.3, 1.5, "#FFF1DE", 1.0),
                (1.5, 2.1, "#FFF8EF", 1.0),
            ]
            for left, right, color, alpha in spans:
                span_left = max(left, x_min)
                span_right = min(right, x_max)
                if span_right > span_left:
                    plot_ax.axvspan(span_left, span_right, color=color, alpha=alpha, zorder=0)

            visible_thresholds = [t for t in [1.3, 1.5, 2.1] if x_min <= t <= x_max]
            for t in visible_thresholds:
                plot_ax.axvline(t, color="#B6C0CB", linewidth=0.52, linestyle=":", zorder=1)

            tick_start = math.ceil(x_min)
            tick_end = math.floor(x_max)
            ticks = list(range(tick_start, tick_end + 1))
            plot_ax.set_xticks(ticks)
            plot_ax.grid(axis="x", color="#EDF2F7", linewidth=0.36)
            plot_ax.grid(axis="y", visible=False)

            for row_idx, (_, row_data) in enumerate(sub.iterrows()):
                for model in order:
                    plot_ax.scatter(
                        row_data[model],
                        y_positions[row_idx],
                        s=68,
                        marker=marker_map[model],
                        facecolors=mcolors.to_rgba(PALETTE[model], 0.24),
                        edgecolors=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.98),
                        linewidths=0.62,
                        zorder=3,
                    )

            for spine_name in ["top", "right", "left", "bottom"]:
                plot_ax.spines[spine_name].set_visible(True)
                plot_ax.spines[spine_name].set_color("#D2DAE4")
                plot_ax.spines[spine_name].set_linewidth(0.55)

            if r_idx < len(panel_rows) - 1:
                plot_ax.set_xticklabels([])
            else:
                tick_labels = [str(tick) for tick in ticks]
                plot_ax.set_xticklabels(tick_labels, color="#4B5563", fontweight="bold")

            c_idx += span

    legend_handles = [
        Line2D([0], [0], color=PALETTE[m], lw=0, marker=marker_map[m], markersize=10.4,
               markerfacecolor=mcolors.to_rgba(PALETTE[m], 0.24),
               markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE[m], 0.98),
               markeredgewidth=0.58, label=m)
        for m in order
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.42,
        labelspacing=0.50,
        handlelength=1.5,
        handletextpad=0.45,
        prop={"size": 14.0},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    fig.subplots_adjust(left=0.036, right=0.994, top=0.900, bottom=0.022)
    _save(fig, filename)


def make_threshold_ladder_regional_plots(hist: pd.DataFrame) -> None:
    comp, year = _get_three_model_country_comparison(hist)

    _plot_threshold_ladder_panel_grid(
        comp,
        [
            ["Northern & Western Europe", "Southern & Eastern Europe"],
            ["East & Southeast Asia", "South & Central Asia"],
        ],
        "forecast_threshold_ladder_regions_1_2042",
        (8.9, 7.6),
    )

    _plot_threshold_ladder_panel_grid(
        comp,
        [
            ["Middle East", "Oceania"],
            ["South America", "North/Central America & Caribbean"],
            ["North Africa", "Sub-Saharan Africa"],
        ],
        "forecast_threshold_ladder_regions_2_2042",
        (8.8, 9.2),
    )


def main() -> None:
    hist, loaded = _load_inputs()
    pop = loaded.pop("pop")
    forecasts, ids = _shared_country_sample(loaded)
    global_series = _weighted_global_series(forecasts, pop, ids)
    history = _weighted_history(hist, pop, ids)

    make_trajectory_plot(history, global_series)
    make_relative_decline_plot(global_series)
    make_threshold_ladder_regional_plots(hist)

    print(f"Saved figures to: {OUT_DIR}")
    print(f"Shared country sample: {len(ids)}")


if __name__ == "__main__":
    main()
