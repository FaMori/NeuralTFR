from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FORECAST_RESULTS_DIR = ROOT / "results" / "forecast" / "predictions"
OUT_DIR = ROOT / "results" / "plots"

PALETTE = {
    "NeuralTFR": "#FF3400",
    "WPP": "#99FF00",
    "GBD": "#C97B00",
}

CONTOUR_PALETTE = {
    "NeuralTFR": "#FC5000",
    "WPP": "#00C745",
    "GBD": "#9A5E00",
}

PERIOD_ORDER = ["2025-2030", "2030-2035", "2035-2040"]
PERIOD_LABELS = {
    "2025-2030": "2025-30",
    "2030-2035": "2030-35",
    "2035-2040": "2035-40",
}

DISPLAY_LABELS = {
    "NeuralTFR": "NeuralTFR",
    "WPP": "BayesTFR",
    "GBD": "GBD",
}

BAND_COLORS = {
    "TFR < 1.3": "#D85C41",
    "1.3 <= TFR < 1.5": "#F3A261",
    "1.5 <= TFR < 2.1": "#F7D58B",
    "TFR >= 2.1": "#D9E1EA",
}


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_weighted_periods() -> pd.DataFrame:
    df = pd.read_csv(IN_DIR / "five_years.csv")
    rows: list[dict[str, object]] = []
    for period in PERIOD_ORDER:
        grp = df[df["int"] == period].copy()
        pop = grp["Pop"].sum()
        rows.extend(
            [
                {
                    "period": period,
                    "model": "NeuralTFR",
                    "value": (grp["NeuralTFR"] * grp["Pop"]).sum() / pop,
                },
                {
                    "period": period,
                    "model": "WPP",
                    "value": (grp["WPP"] * grp["Pop"]).sum() / pop,
                },
                {
                    "period": period,
                    "model": "GBD",
                    "value": (grp["GBD"] * grp["Pop"]).sum() / pop,
                },
            ]
        )
    out = pd.DataFrame(rows)
    out["period"] = pd.Categorical(out["period"], categories=PERIOD_ORDER, ordered=True)
    return out.sort_values(["period", "model"]).reset_index(drop=True)


def _load_weighted_annual() -> pd.DataFrame:
    df = pd.read_csv(IN_DIR / "anual.csv")
    df = df[df["year"] >= 2024].copy()

    rows: list[dict[str, object]] = []
    for year, grp in df.groupby("year", sort=True):
        pop = grp["Pop"].sum()
        rows.extend(
            [
                {
                    "year": int(year),
                    "model": "NeuralTFR",
                    "value": (grp["NeuralTFR"] * grp["Pop"]).sum() / pop,
                },
                {
                    "year": int(year),
                    "model": "WPP",
                    "value": (grp["WPP"] * grp["Pop"]).sum() / pop,
                },
                {
                    "year": int(year),
                    "model": "GBD",
                    "value": (grp["GBD"] * grp["Pop"]).sum() / pop,
                },
            ]
        )

    return pd.DataFrame(rows).sort_values(["year", "model"]).reset_index(drop=True)


def _load_last_available_distribution() -> pd.DataFrame:
    df = pd.read_csv(IN_DIR / "anual.csv")
    df = df.sort_values(["id", "year"]).groupby("id", as_index=False).tail(1).copy()
    return df.reset_index(drop=True)


def make_weighted_trajectory_plot() -> None:
    weighted = _load_weighted_periods()

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.titlesize": 16,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.8,
            "ytick.labelsize": 12.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    x_positions = np.arange(len(PERIOD_ORDER))
    for model in ["NeuralTFR", "WPP", "GBD"]:
        current = weighted[weighted["model"] == model].copy()
        ax.plot(
            x_positions,
            current["value"],
            color=PALETTE[model],
            linewidth=1.1,
            linestyle=(0, (10.0, 5.0)),
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            x_positions,
            current["value"],
            s=50,
            facecolors=mcolors.to_rgba(PALETTE[model], 0.20),
            edgecolors=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.92),
            linewidths=0.40,
            zorder=4,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE[model],
            lw=1.1,
            linestyle=(0, (10.0, 5.0)),
            marker="o",
            markersize=7.2,
            markerfacecolor=mcolors.to_rgba(PALETTE[model], 0.20),
            markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.92),
            markeredgewidth=0.54,
            label=DISPLAY_LABELS[model],
        )
        for model in ["NeuralTFR", "WPP", "GBD"]
    ]
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
        prop={"size": 12.8},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    ax.set_xlabel("Projection interval")
    ax.set_ylabel("Population-weighted TFR")
    ax.set_xticks(x_positions, [PERIOD_LABELS[p] for p in PERIOD_ORDER])
    ax.set_ylim(1.70, 2.00)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="x", which="major", color="#D7DEE7", linewidth=0.8, alpha=0.70)
    ax.grid(axis="y", which="major", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.tick_params(axis="both", length=0, colors="#475569")

    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)
    ax.spines["bottom"].set_color("#9AA4B2")
    ax.spines["bottom"].set_linewidth(0.8)

    _save(fig, "forecast_weighted_trajectory_5year")


def make_weighted_trajectory_plot_annual() -> None:
    weighted = _load_weighted_annual()

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.titlesize": 16,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.8,
            "ytick.labelsize": 12.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    for model in ["NeuralTFR", "WPP", "GBD"]:
        current = weighted[weighted["model"] == model].copy()
        ax.plot(
            current["year"],
            current["value"],
            color=PALETTE[model],
            linewidth=1.0,
            linestyle=(0, (10.0, 5.0)),
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            current["year"],
            current["value"],
            s=34,
            facecolors=mcolors.to_rgba(PALETTE[model], 0.20),
            edgecolors=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.92),
            linewidths=0.40,
            zorder=4,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE[model],
            lw=1.0,
            linestyle=(0, (10.0, 5.0)),
            marker="o",
            markersize=6.6,
            markerfacecolor=mcolors.to_rgba(PALETTE[model], 0.20),
            markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.92),
            markeredgewidth=0.54,
            label=DISPLAY_LABELS[model],
        )
        for model in ["NeuralTFR", "WPP", "GBD"]
    ]
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
        prop={"size": 12.8},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    y_min = math.floor((weighted["value"].min() - 0.02) / 0.05) * 0.05
    y_max = math.ceil((weighted["value"].max() + 0.02) / 0.05) * 0.05

    ax.set_xlabel("Year")
    ax.set_ylabel("Population-weighted TFR")
    ax.set_xlim(weighted["year"].min() - 0.4, weighted["year"].max() + 0.4)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="x", which="major", color="#D7DEE7", linewidth=0.8, alpha=0.70)
    ax.grid(axis="y", which="major", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.tick_params(axis="both", length=0, colors="#475569")

    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)
    ax.spines["bottom"].set_color("#9AA4B2")
    ax.spines["bottom"].set_linewidth(0.8)

    _save(fig, "forecast_weighted_trajectory_annual")


def make_relative_decline_plot() -> None:
    weighted = _load_weighted_periods()
    pivot = weighted.pivot(index="period", columns="model", values="value").loc[PERIOD_ORDER]
    delta = pivot.subtract(pivot.iloc[0], axis=1)
    spread = delta.subtract(delta["WPP"], axis=0)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.8,
            "ytick.labelsize": 12.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    x_positions = np.arange(len(PERIOD_ORDER))

    for model in ["NeuralTFR", "GBD"]:
        ax.plot(
            x_positions,
            spread[model].values,
            color=PALETTE[model],
            linewidth=1.1,
            linestyle=(0, (10.0, 5.0)),
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            x_positions,
            spread[model].values,
            s=50,
            facecolors=mcolors.to_rgba(PALETTE[model], 0.20),
            edgecolors=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.92),
            linewidths=0.40,
            zorder=4,
        )

    ax.axhline(
        0,
        color="#2F343B",
        linewidth=1.0,
        linestyle=(0, (10.0, 5.0)),
        zorder=2,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#2F343B",
            lw=1.0,
            linestyle=(0, (10.0, 5.0)),
            label="BayesTFR (baseline)",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["NeuralTFR"],
            lw=1.1,
            linestyle=(0, (10.0, 5.0)),
            marker="o",
            markersize=7.2,
            markerfacecolor=mcolors.to_rgba(PALETTE["NeuralTFR"], 0.20),
            markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE["NeuralTFR"], 0.92),
            markeredgewidth=0.54,
            label="NeuralTFR",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["GBD"],
            lw=1.1,
            linestyle=(0, (10.0, 5.0)),
            marker="o",
            markersize=7.2,
            markerfacecolor=mcolors.to_rgba(PALETTE["GBD"], 0.20),
            markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE["GBD"], 0.92),
            markeredgewidth=0.54,
            label="GBD",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.05),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.72,
        labelspacing=0.56,
        handlelength=2.2,
        handletextpad=0.78,
        prop={"size": 12.8},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    ax.set_xlabel("Projection interval")
    ax.set_ylabel("")
    ax.set_xticks(x_positions, [PERIOD_LABELS[p] for p in PERIOD_ORDER])
    ax.set_ylim(-0.08, 0.005)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.grid(axis="x", which="major", color="#D7DEE7", linewidth=0.8, alpha=0.70)
    ax.grid(axis="y", which="major", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.tick_params(axis="both", length=0, colors="#475569")

    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)
    ax.spines["bottom"].set_color("#9AA4B2")
    ax.spines["bottom"].set_linewidth(0.8)

    _save(fig, "forecast_relative_decline_5year")


def make_distribution_bands_plot() -> None:
    df = _load_last_available_distribution()

    def _band(value: float) -> str:
        if value < 1.3:
            return "TFR < 1.3"
        if value < 1.5:
            return "1.3 <= TFR < 1.5"
        if value < 2.1:
            return "1.5 <= TFR < 2.1"
        return "TFR >= 2.1"

    band_order = ["TFR < 1.3", "1.3 <= TFR < 1.5", "1.5 <= TFR < 2.1", "TFR >= 2.1"]
    model_order = ["WPP", "NeuralTFR", "GBD"]

    rows: list[dict[str, object]] = []
    for model in model_order:
        bands = df[model].map(_band)
        shares = bands.value_counts(normalize=True).reindex(band_order, fill_value=0.0)
        for band, share in shares.items():
            rows.append({"model": model, "band": band, "share": float(share)})
    plot_df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.8,
            "ytick.labelsize": 13.0,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    y = np.arange(len(model_order))
    left = np.zeros(len(model_order))
    for band in band_order:
        shares = (
            plot_df[plot_df["band"] == band]
            .set_index("model")
            .reindex(model_order)["share"]
            .to_numpy()
        )
        bars = ax.barh(
            y,
            shares,
            left=left,
            height=0.56,
            color=BAND_COLORS[band],
            edgecolor="#FFFFFF",
            linewidth=1.0,
            label=band,
            zorder=3,
        )
        for idx, (bar, share) in enumerate(zip(bars, shares)):
            if share >= 0.09:
                ax.text(
                    left[idx] + share / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{share * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=11.0,
                    color="#334155",
                    zorder=4,
                )
        left += shares

    ax.set_yticks(y, [DISPLAY_LABELS[m] for m in model_order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of countries")
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(t * 100)}%" for t in np.linspace(0, 1, 6)])
    ax.grid(axis="x", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", length=0, colors="#475569")

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.55,
        labelspacing=0.55,
        handlelength=1.6,
        handletextpad=0.50,
        prop={"size": 11.8},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)

    _save(fig, "forecast_distribution_bands_last_available")


def make_distribution_ecdf_plot() -> None:
    df = _load_last_available_distribution()
    model_order = ["WPP", "NeuralTFR", "GBD"]

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.8,
            "ytick.labelsize": 12.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    for model in model_order:
        values = np.sort(df[model].to_numpy())
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(
            values,
            y,
            where="post",
            color=PALETTE[model],
            linewidth=1.5,
            label=DISPLAY_LABELS[model],
            zorder=3,
        )

    for threshold in [1.3, 1.5, 2.1]:
        ax.axvline(threshold, color="#B6C0CB", linewidth=0.70, linestyle=":", zorder=1)

    ax.set_xlabel("TFR at last available projected year")
    ax.set_ylabel("Cumulative share of countries")
    ax.set_xlim(0.7, max(4.2, float(df[["NeuralTFR", "WPP", "GBD"]].to_numpy().max()) + 0.1))
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(t * 100)}%" for t in np.linspace(0, 1, 6)])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(axis="x", color="#D7DEE7", linewidth=0.8, alpha=0.70)
    ax.grid(axis="y", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.tick_params(axis="both", length=0, colors="#475569")

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.62,
        labelspacing=0.56,
        handlelength=2.2,
        handletextpad=0.78,
        prop={"size": 12.8},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)

    _save(fig, "forecast_distribution_ecdf_last_available")


def make_distribution_histograms_plot() -> None:
    df = _load_last_available_distribution()
    model_order = ["WPP", "NeuralTFR", "GBD"]
    bins = np.arange(0.75, 6.55, 0.2)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 13.5,
            "xtick.labelsize": 12.2,
            "ytick.labelsize": 11.8,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.7), sharex=True, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    for ax, model in zip(axes, model_order):
        ax.set_facecolor("#FBFCFE")
        ax.hist(
            df[model],
            bins=bins,
            color=mcolors.to_rgba(PALETTE[model], 0.28),
            edgecolor=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.92),
            linewidth=0.55,
            zorder=3,
        )
        for threshold in [1.3, 1.5, 2.1]:
            ax.axvline(threshold, color="#B6C0CB", linewidth=0.75, linestyle=":", zorder=1)
        ax.text(
            0.985,
            0.88,
            DISPLAY_LABELS[model],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12.5,
            fontweight="bold",
            color="#475569",
        )
        ax.grid(axis="x", color="#D7DEE7", linewidth=0.8, alpha=0.70)
        ax.grid(axis="y", color="#E4EAF0", linewidth=0.8, alpha=0.95)
        ax.tick_params(axis="both", length=0, colors="#475569")
        for spine_name in ["top", "right", "left", "bottom"]:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color("#C9D2DE")
            ax.spines[spine_name].set_linewidth(0.55)

    axes[-1].set_xlabel("TFR at last available projected year")
    axes[1].set_ylabel("Number of countries")
    axes[-1].set_xlim(0.75, 6.45)
    axes[-1].xaxis.set_major_locator(MultipleLocator(0.5))

    fig.subplots_adjust(left=0.11, right=0.985, top=0.98, bottom=0.11, hspace=0.10)
    _save(fig, "forecast_distribution_histograms_last_available")


def make_distribution_frequency_polygon_plot() -> None:
    df = _load_last_available_distribution()
    model_order = ["WPP", "NeuralTFR", "GBD"]
    bins = np.arange(0.75, 6.55, 0.2)
    centers = (bins[:-1] + bins[1:]) / 2

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.8,
            "ytick.labelsize": 12.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    for model in model_order:
        counts, _ = np.histogram(df[model], bins=bins)
        ax.plot(
            centers,
            counts,
            color=PALETTE[model],
            linewidth=1.7,
            zorder=3,
            label=DISPLAY_LABELS[model],
        )
        ax.fill_between(
            centers,
            counts,
            color=mcolors.to_rgba(PALETTE[model], 0.10),
            alpha=1.0,
            zorder=2,
        )

    for threshold in [1.3, 1.5, 2.1]:
        ax.axvline(threshold, color="#B6C0CB", linewidth=0.75, linestyle=":", zorder=1)

    ax.set_xlabel("TFR at last available projected year")
    ax.set_ylabel("Number of countries")
    ax.set_xlim(0.75, 6.45)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(axis="x", color="#D7DEE7", linewidth=0.8, alpha=0.70)
    ax.grid(axis="y", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.tick_params(axis="both", length=0, colors="#475569")

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.62,
        labelspacing=0.56,
        handlelength=2.2,
        handletextpad=0.78,
        prop={"size": 12.8},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)

    _save(fig, "forecast_distribution_frequency_polygon_last_available")


def make_threshold_bars_plot() -> None:
    df = _load_last_available_distribution()
    model_order = ["WPP", "NeuralTFR", "GBD"]
    threshold_order = ["TFR >= 2.1", "TFR < 2.1", "TFR < 1.5", "TFR < 1.3"]
    threshold_values = {"TFR < 2.1": 2.1, "TFR < 1.5": 1.5, "TFR < 1.3": 1.3}

    rows: list[dict[str, object]] = []
    n = len(df)
    for thr_label in threshold_order:
        for model in model_order:
            if thr_label == "TFR >= 2.1":
                share = float((df[model] >= 2.1).sum() / n)
            else:
                thr = threshold_values[thr_label]
                share = float((df[model] < thr).sum() / n)
            rows.append({"threshold": thr_label, "model": model, "share": share})
    plot_df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.8,
            "ytick.labelsize": 13.0,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    group_y = np.arange(len(threshold_order))
    bar_h = 0.19
    offsets = np.array([-bar_h, 0.0, bar_h])

    for off, model in zip(offsets, model_order):
        vals = (
            plot_df[plot_df["model"] == model]
            .set_index("threshold")
            .reindex(threshold_order)["share"]
            .to_numpy()
        )
        bars = ax.barh(
            group_y + off,
            vals,
            height=bar_h * 0.92,
            color=mcolors.to_rgba(PALETTE[model], 0.82),
            edgecolor=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.96),
            linewidth=0.6,
            label=DISPLAY_LABELS[model],
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                min(val + 0.012, 0.98),
                bar.get_y() + bar.get_height() / 2,
                f"{val * 100:.1f}%",
                va="center",
                ha="left",
                fontsize=10.9,
                color="#334155",
                zorder=4,
            )

    ax.set_yticks(group_y, threshold_order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Share of countries")
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(t * 100)}%" for t in np.linspace(0, 1, 6)])
    ax.grid(axis="x", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", length=0, colors="#475569")

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.60,
        labelspacing=0.56,
        handlelength=1.8,
        handletextpad=0.60,
        prop={"size": 12.2},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)

    _save(fig, "forecast_threshold_bars_last_available")


def make_exclusive_category_bars_plot() -> None:
    df = _load_last_available_distribution()
    model_order = ["WPP", "NeuralTFR", "GBD"]
    category_order = ["TFR >= 2.1", "1.5 <= TFR < 2.1", "1.3 <= TFR < 1.5", "TFR < 1.3"]
    display_category_labels = [r"$\mathrm{TFR} \geq 2.1$", r"$1.5 \leq \mathrm{TFR} < 2.1$", r"$1.3 \leq \mathrm{TFR} < 1.5$", r"$\mathrm{TFR} < 1.3$"]

    rows: list[dict[str, object]] = []
    n = len(df)
    for model in model_order:
        series = df[model]
        shares = {
            "TFR >= 2.1": float((series >= 2.1).sum() / n),
            "1.5 <= TFR < 2.1": float(((series >= 1.5) & (series < 2.1)).sum() / n),
            "1.3 <= TFR < 1.5": float(((series >= 1.3) & (series < 1.5)).sum() / n),
            "TFR < 1.3": float((series < 1.3).sum() / n),
        }
        for cat in category_order:
            rows.append({"category": cat, "model": model, "share": shares[cat]})
    plot_df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.8,
            "ytick.labelsize": 13.0,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    group_y = np.arange(len(category_order))
    bar_h = 0.19
    offsets = np.array([-bar_h, 0.0, bar_h])

    for off, model in zip(offsets, model_order):
        vals = (
            plot_df[plot_df["model"] == model]
            .set_index("category")
            .reindex(category_order)["share"]
            .to_numpy()
        )
        bars = ax.barh(
            group_y + off,
            vals,
            height=bar_h * 0.92,
            color=mcolors.to_rgba(PALETTE[model], 0.82),
            edgecolor=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.96),
            linewidth=0.6,
            label=DISPLAY_LABELS[model],
            zorder=3,
        )
    ax.set_yticks(group_y, display_category_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("Share of countries")
    xticks = np.linspace(0, 0.8, 5)
    ax.set_xticks(xticks, [f"{int(t * 100)}%" for t in xticks])
    ax.grid(axis="x", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", length=0, colors="#475569")

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.60,
        labelspacing=0.56,
        handlelength=1.8,
        handletextpad=0.60,
        prop={"size": 11.6},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)

    fig.subplots_adjust(left=0.22, right=0.98, top=0.98, bottom=0.16)
    _save(fig, "forecast_exclusive_category_bars_last_available")


def make_relative_decline_plot_annual() -> None:
    weighted = _load_weighted_annual()
    pivot = weighted.pivot(index="year", columns="model", values="value").sort_index()
    delta = pivot.subtract(pivot.iloc[0], axis=1)
    spread = delta.subtract(delta["WPP"], axis=0)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.8,
            "ytick.labelsize": 12.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FBFCFE")

    for model in ["NeuralTFR", "GBD"]:
        ax.plot(
            spread.index,
            spread[model].values,
            color=PALETTE[model],
            linewidth=1.0,
            linestyle=(0, (10.0, 5.0)),
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            spread.index,
            spread[model].values,
            s=34,
            facecolors=mcolors.to_rgba(PALETTE[model], 0.20),
            edgecolors=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.92),
            linewidths=0.40,
            zorder=4,
        )

    ax.axhline(
        0,
        color="#2F343B",
        linewidth=1.0,
        linestyle=(0, (10.0, 5.0)),
        zorder=2,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#2F343B",
            lw=1.0,
            linestyle=(0, (10.0, 5.0)),
            label="BayesTFR (baseline)",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["NeuralTFR"],
            lw=1.0,
            linestyle=(0, (10.0, 5.0)),
            marker="o",
            markersize=6.6,
            markerfacecolor=mcolors.to_rgba(PALETTE["NeuralTFR"], 0.20),
            markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE["NeuralTFR"], 0.92),
            markeredgewidth=0.54,
            label="NeuralTFR",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["GBD"],
            lw=1.0,
            linestyle=(0, (10.0, 5.0)),
            marker="o",
            markersize=6.6,
            markerfacecolor=mcolors.to_rgba(PALETTE["GBD"], 0.20),
            markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE["GBD"], 0.92),
            markeredgewidth=0.54,
            label="GBD",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.05),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.72,
        labelspacing=0.56,
        handlelength=2.2,
        handletextpad=0.78,
        prop={"size": 12.8},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.7)

    y_min = math.floor((spread[["NeuralTFR", "GBD"]].min().min() - 0.01) / 0.02) * 0.02
    y_max = math.ceil((spread[["NeuralTFR", "GBD"]].max().max() + 0.01) / 0.02) * 0.02

    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.set_xlim(spread.index.min() - 0.4, spread.index.max() + 0.4)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.grid(axis="x", which="major", color="#D7DEE7", linewidth=0.8, alpha=0.70)
    ax.grid(axis="y", which="major", color="#E4EAF0", linewidth=0.8, alpha=0.95)
    ax.tick_params(axis="both", length=0, colors="#475569")

    for spine_name in ["top", "right", "left", "bottom"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#C9D2DE")
        ax.spines[spine_name].set_linewidth(0.55)
    ax.spines["bottom"].set_color("#9AA4B2")
    ax.spines["bottom"].set_linewidth(0.8)

    _save(fig, "forecast_relative_decline_annual")


def _short_country_name(name: str) -> str:
    replacements = {
        "Brunei Darussalam": "Brunei",
        "China, Hong Kong SAR": "Hong Kong",
        "China, Macao SAR": "Macao",
        "China, Taiwan Province of China": "Taiwan",
        "Taiwan, Province of China": "Taiwan",
        "Dem. People's Republic of Korea": "North Korea",
        "Korea, Democratic People's Republic of": "North Korea",
        "Korea, Republic of": "South Korea",
        "Iran (Islamic Republic of)": "Iran",
        "Iran, Islamic Republic of": "Iran",
        "Lao People's Democratic Republic": "Laos",
        "State of Palestine": "Palestine",
        "United Republic of Tanzania": "Tanzania",
        "Tanzania, United Republic of": "Tanzania",
        "United States of America": "United States",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        "Venezuela, Bolivarian Republic of": "Venezuela",
        "North Macedonia": "Macedonia",
        "Micronesia (Fed. States of)": "Micronesia",
        "Micronesia, Federated States of": "Micronesia",
        "Bosnia and Herzegovina": "Bosnia-Herzegovina",
        "Antigua and Barbuda": "Antigua-Barbuda",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Bolivia, Plurinational State of": "Bolivia",
        "Cabo Verde": "Cape Verde",
        "Congo": "Rep. Congo",
        "Central African Republic": "Central Afr. Rep.",
        "Czechia": "Czech Republic",
        "Democratic Republic of the Congo": "DR Congo",
        "Congo, Democratic Republic of the": "DR Congo",
        "Dominican Republic": "Dominican Rep.",
        "Moldova, Republic of": "Moldova",
        "Netherlands, Kingdom of the": "Netherlands",
        "Republic of Moldova": "Moldova",
        "Russian Federation": "Russia",
        "Sao Tome and Principe": "Sao Tome-Principe",
        "Saint Vincent and the Grenadines": "St. Vincent-Gren.",
        "Syrian Arab Republic": "Syria",
        "Trinidad and Tobago": "Trinidad-Tobago",
        "Türkiye": "Turkey",
        "United Arab Emirates": "UAE",
        "Papua New Guinea": "PNG",
        "Macedonia, FYR": "Macedonia",
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


def _load_threshold_comparison() -> pd.DataFrame:
    neural = (
        pd.read_csv(FORECAST_RESULTS_DIR / "predictions.csv")
        .query("model == 'NeuralTFR' and year >= 2024")[["id", "year", "y_hat_50"]]
        .rename(columns={"y_hat_50": "NeuralTFR"})
    )
    wpp = (
        pd.read_csv(FORECAST_RESULTS_DIR / "other models" / "wpp.csv")[["id", "year", "y_hat_50"]]
        .rename(columns={"y_hat_50": "WPP"})
    )
    gbd = (
        pd.read_csv(FORECAST_RESULTS_DIR / "other models" / "gbd.csv")[["id", "year", "y_hat_50"]]
        .rename(columns={"y_hat_50": "GBD"})
    )
    for frame in (neural, wpp, gbd):
        frame["id"] = frame["id"].astype(str)
        frame["year"] = frame["year"].astype(int)

    current = neural.merge(wpp, on=["id", "year"], how="inner").merge(gbd, on=["id", "year"], how="inner")
    current = current.sort_values(["id", "year"]).groupby("id", as_index=False).tail(1).copy()

    names = (
        pd.read_csv(DATA_DIR / "final" / "tfr_smooth.csv")[["id", "name"]]
        .drop_duplicates()
        .assign(id=lambda x: x["id"].astype(str))
    )

    codes = pd.read_csv(DATA_DIR / "countries_codes.csv", encoding="utf-8")
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

    comp = current.merge(names, on="id", how="left").merge(meta, on="id", how="left")
    comp["country_label"] = comp["name"].map(_short_country_name)
    comp["geo_group"] = comp.apply(_assign_geo_group, axis=1)
    comp["avg_value"] = comp[["WPP", "NeuralTFR", "GBD"]].mean(axis=1)
    return comp


def _plot_threshold_ladder_panel_grid(
    df: pd.DataFrame,
    panel_rows: list[list[str | None]],
    filename: str,
    figsize: tuple[float, float],
    row_body_scale: float = 0.275,
    x_max_cap: float | None = 4.2,
) -> None:
    groups = [panel for row in panel_rows for panel in row if panel is not None]
    grouped = {
        group: df[df["geo_group"] == group].sort_values(["avg_value", "country_label"]).reset_index(drop=True)
        for group in groups
    }
    figure_vals = pd.concat([grouped[group][m] for group in groups for m in ["WPP", "NeuralTFR", "GBD"]])
    shared_x_min = 0.6
    shared_x_max = math.ceil((figure_vals.max() + 0.25) / 0.1) * 0.1
    if x_max_cap is not None:
        shared_x_max = min(x_max_cap, shared_x_max)
    if (shared_x_max - shared_x_min) < 1.8:
        shared_x_max = shared_x_min + 1.8 if x_max_cap is None else min(x_max_cap, shared_x_min + 1.8)
    shared_max_label_len = max(grouped[group]["country_label"].str.len().max() for group in groups)
    shared_label_ratio = min(max(1.05, 0.09 * shared_max_label_len), 1.9)
    shared_plot_ratio = 4.75 - shared_label_ratio
    row_max_counts = [max(len(grouped[group]) for group in row if group is not None) for row in panel_rows]
    row_panel_bodies = [max(count, 4) * row_body_scale for count in row_max_counts]
    row_heights = [panel_body + 0.24 for panel_body in row_panel_bodies]

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

    marker_map = {"WPP": "s", "NeuralTFR": "o", "GBD": "^"}
    order = ["WPP", "NeuralTFR", "GBD"]

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
            group_max_label_len = sub["country_label"].astype(str).map(len).max()
            group_label_fontsize = 13.8
            if n >= 40:
                group_label_fontsize = 9.8
            elif n >= 34 or group_max_label_len >= 18:
                group_label_fontsize = 10.8
            elif n >= 28 or group_max_label_len >= 16:
                group_label_fontsize = 11.4
            elif n >= 22 or group_max_label_len >= 14:
                group_label_fontsize = 12.0
            elif n >= 16:
                group_label_fontsize = 12.8
            if n == 1:
                y_positions = [row_panel_body / 2]
            else:
                edge_pad = min(row_panel_body * 0.20, 0.18 + 0.028 * max(row_max_count - n, 0))
                y_positions = np.linspace(edge_pad, row_panel_body - edge_pad, n).tolist()
            x_min = shared_x_min
            x_max = shared_x_max

            mirror_labels = c_idx == 1
            total_ratio = shared_label_ratio + shared_plot_ratio
            header_x = (shared_plot_ratio / 2) / total_ratio if mirror_labels else (shared_label_ratio + shared_plot_ratio / 2) / total_ratio

            panel_spec = gs[r_idx, c_idx]
            cell = panel_spec.subgridspec(
                2,
                2,
                height_ratios=[0.24, row_panel_body],
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

            label_ax.set_ylim(row_panel_body + 0.10, -0.10)
            plot_ax.set_ylim(row_panel_body + 0.10, -0.10)

            label_ax.set_xlim(0, 1)
            label_ax.axis("off")

            for row_idx, (_, row_data) in enumerate(sub.iterrows()):
                label_ax.text(
                    0.01 if mirror_labels else 0.99,
                    y_positions[row_idx],
                    row_data["country_label"],
                    ha="left" if mirror_labels else "right",
                    va="center",
                    fontsize=group_label_fontsize,
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

            for threshold in [1.3, 1.5, 2.1]:
                if x_min <= threshold <= x_max:
                    plot_ax.axvline(threshold, color="#B6C0CB", linewidth=0.52, linestyle=":", zorder=1)

            ticks = list(range(math.ceil(x_min), math.floor(x_max) + 1))
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
                plot_ax.set_xticklabels([str(tick) for tick in ticks], color="#4B5563", fontweight="bold")

            c_idx += 1

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE[m],
            lw=0,
            marker=marker_map[m],
            markersize=10.4,
            markerfacecolor=mcolors.to_rgba(PALETTE[m], 0.24),
            markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE[m], 0.98),
            markeredgewidth=0.58,
            label=DISPLAY_LABELS[m],
        )
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


def make_threshold_ladder_regional_plots() -> None:
    comp = _load_threshold_comparison()

    _plot_threshold_ladder_panel_grid(
        comp,
        [
            ["Northern & Western Europe", "Southern & Eastern Europe"],
            ["East & Southeast Asia", "South & Central Asia"],
        ],
        "forecast_threshold_ladder_regions_1_last_available",
        (9.0, 8.2),
        row_body_scale=0.275,
        x_max_cap=4.2,
    )

    _plot_threshold_ladder_panel_grid(
        comp,
        [
            ["Middle East", "Oceania"],
            ["South America", "North Africa"],
            ["North/Central America & Caribbean", "Sub-Saharan Africa"],
        ],
        "forecast_threshold_ladder_regions_2_last_available",
        (10.4, 11.8),
        row_body_scale=0.33,
        x_max_cap=6.8,
    )


def main() -> None:
    make_weighted_trajectory_plot()
    make_weighted_trajectory_plot_annual()
    make_relative_decline_plot()
    make_relative_decline_plot_annual()
    make_threshold_bars_plot()
    make_exclusive_category_bars_plot()
    make_distribution_bands_plot()
    make_distribution_ecdf_plot()
    make_distribution_histograms_plot()
    make_distribution_frequency_polygon_plot()
    make_threshold_ladder_regional_plots()
    print(f"Saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
