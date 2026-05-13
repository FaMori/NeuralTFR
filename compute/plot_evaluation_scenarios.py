from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "plots"

PALETTE = {
    "NeuralTFR": "#FF3400",
    "BayesTFR": "#99FF00",
}

CONTOUR_PALETTE = {
    "NeuralTFR": "#FC5000",
    "BayesTFR": "#00C745",
}

OBSERVED_PRE = "#AEB7C2"
OBSERVED_POST = "#2B333C"
OBSERVED_PRE_EDGE = "#8F99A5"
OBSERVED_POST_EDGE = "#1F262D"

DISPLAY_NAMES = {
    "Korea, Republic of": "South Korea",
}

SCENARIO_COUNTRIES = {
    "fig5_scenarios_strengths": [
        "Italy",
        "Greece",
        "Canada",
        "Belgium",
    ],
    "fig5_scenarios_limitations": [
        "Korea, Republic of",
        "Kazakhstan",
        "Uzbekistan",
        "Romania",
    ],
    "fig4_nordics_us": [
        "Denmark",
        "Finland",
        "Iceland",
        "Norway",
        "Sweden",
        "United States of America",
    ],
    "fig4_nordics_us_subset": [
        "Denmark",
        "Sweden",
        "Norway",
        "United States of America",
    ],
}

FIXED_Y_LIMITS = {
    "Italy": (0.8, 2.0),
    "Greece": (0.8, 2.0),
    "Canada": (0.8, 2.0),
    "Romania": (0.8, 2.0),
    "Korea, Republic of": (None, 1.8),
    "Uzbekistan": (1.6, None),
}


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = pd.read_csv(DATA_DIR / "final" / "tfr_smooth.csv")
    preds = pd.read_csv(ROOT / "results" / "evaluation" / "predictions" / "predictions.csv")

    hist["id"] = hist["id"].astype(str)
    preds["id"] = preds["id"].astype(str)
    hist["year"] = hist["year"].astype(int)
    preds["year"] = preds["year"].astype(int)
    return hist, preds


def _country_id_map(hist: pd.DataFrame) -> dict[str, str]:
    return (
        hist[["id", "name"]]
        .drop_duplicates()
        .set_index("name")["id"]
        .to_dict()
    )


def _display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_group(hist: pd.DataFrame, preds: pd.DataFrame, countries: list[str], stem: str) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.titlesize": 13.8,
            "axes.labelsize": 11.2,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
        }
    )

    country_ids = _country_id_map(hist)
    n_countries = len(countries)
    n_cols = 2 if n_countries <= 4 else 3
    n_rows = math.ceil(n_countries / n_cols)
    figsize = (11.2, 7.7) if n_cols == 2 else (16.2, 7.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = list(axes.flatten())
    fig.patch.set_facecolor("#FFFFFF")

    holdout_start = 2009
    x_min = 2000
    x_max = 2023
    x_left_pad = 1.0
    x_right_pad = 0.8

    for ax, country in zip(axes, countries):
        country_id = country_ids[country]
        observed = (
            hist[(hist["id"] == country_id) & (hist["year"].between(x_min, x_max))]
            .sort_values("year")
            .copy()
        )
        pred = (
            preds[
                (preds["id"] == country_id)
                & (preds["model"].isin(["NeuralTFR", "BayesTFR"]))
                & (preds["year"].between(holdout_start, x_max))
            ]
            .sort_values(["model", "year"])
            .copy()
        )

        pre = observed[observed["year"] < holdout_start]
        post = observed[observed["year"] >= holdout_start]

        ax.set_facecolor("#FFFFFF")
        ax.axvspan(holdout_start, x_max + x_right_pad, color="#F6F8FB", zorder=0)
        ax.axvline(holdout_start, color="#B9C3CF", linewidth=0.75, linestyle=":", zorder=1)

        ax.plot(
            pre["year"],
            pre["TFR"],
            color=OBSERVED_PRE,
            linewidth=0.82,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            pre["year"],
            pre["TFR"],
            s=24,
            facecolors=mcolors.to_rgba(OBSERVED_PRE, 0.18),
            edgecolors=mcolors.to_rgba(OBSERVED_PRE_EDGE, 0.95),
            linewidths=0.56,
            zorder=3,
        )

        ax.plot(
            post["year"],
            post["TFR"],
            color=OBSERVED_POST,
            linewidth=0.90,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            post["year"],
            post["TFR"],
            s=26,
            facecolors=mcolors.to_rgba(OBSERVED_POST, 0.14),
            edgecolors=mcolors.to_rgba(OBSERVED_POST_EDGE, 0.98),
            linewidths=0.58,
            zorder=5,
        )

        for model in ["NeuralTFR", "BayesTFR"]:
            curr = pred[pred["model"] == model]
            ax.fill_between(
                curr["year"],
                curr["y_hat_05"],
                curr["y_hat_95"],
                color=PALETTE[model],
                alpha=0.08,
                linewidth=0,
                zorder=1.8,
            )
            ax.plot(
                curr["year"],
                curr["y_hat_50"],
                color=PALETTE[model],
                linewidth=0.82,
                linestyle=(0, (4.5, 2.2)),
                solid_capstyle="round",
                zorder=3,
            )
            ax.scatter(
                curr["year"],
                curr["y_hat_50"],
                s=24,
                facecolors=mcolors.to_rgba(PALETTE[model], 0.18),
                edgecolors=mcolors.to_rgba(CONTOUR_PALETTE[model], 0.98),
                linewidths=0.56,
                zorder=6,
            )

        y_min = min(observed["TFR"].min(), pred["y_hat_50"].min())
        y_max = max(observed["TFR"].max(), pred["y_hat_50"].max())
        fixed_limits = FIXED_Y_LIMITS.get(country)
        if fixed_limits is not None:
            lower, upper = fixed_limits
            if lower is None:
                lower = y_min - max(0.16, 0.24 * (y_max - y_min))
            if upper is None:
                upper = y_max + max(0.16, 0.24 * (y_max - y_min))
            ax.set_ylim(lower, upper)
        else:
            y_pad = max(0.16, 0.24 * (y_max - y_min))
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlim(x_min - x_left_pad, x_max + x_right_pad)
        ax.set_xticks([2000, 2004, 2008, 2012, 2016, 2020])
        ax.grid(axis="y", color="#E9EEF4", linewidth=0.7)
        ax.grid(axis="x", visible=False)
        ax.set_title(_display_name(country), pad=7, color="#415264", fontweight="bold")

        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("#D3DBE5")
            ax.spines[spine].set_linewidth(0.7)

    for idx, ax in enumerate(axes):
        if idx >= n_countries:
            ax.axis("off")
            continue
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        if col_idx == 0:
            ax.set_ylabel("TFR")
        if row_idx == n_rows - 1:
            ax.set_xlabel("Year")

    legend_handles = [
        Line2D([0], [0], color=OBSERVED_POST, lw=0.90, marker="o", markersize=6.8,
               markerfacecolor=mcolors.to_rgba(OBSERVED_POST, 0.14),
               markeredgecolor=mcolors.to_rgba(OBSERVED_POST_EDGE, 0.98), markeredgewidth=0.58,
               label="Observed"),
        Line2D([0], [0], color=PALETTE["NeuralTFR"], lw=0.82, linestyle=(0, (4.5, 2.2)),
               marker="o", markersize=6.8,
               markerfacecolor=mcolors.to_rgba(PALETTE["NeuralTFR"], 0.18),
               markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE["NeuralTFR"], 0.98),
               markeredgewidth=0.56, label="NeuralTFR"),
        Line2D([0], [0], color=PALETTE["BayesTFR"], lw=0.82, linestyle=(0, (4.5, 2.2)),
               marker="s", markersize=6.8,
               markerfacecolor=mcolors.to_rgba(PALETTE["BayesTFR"], 0.18),
               markeredgecolor=mcolors.to_rgba(CONTOUR_PALETTE["BayesTFR"], 0.98),
               markeredgewidth=0.56, label="BayesTFR"),
        Patch(facecolor=mcolors.to_rgba(PALETTE["NeuralTFR"], 0.08), edgecolor="none", label="90% interval"),
        Patch(facecolor="#F6F8FB", edgecolor="#D3DBE5", linewidth=0.7, label="2009--2023 test period"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.56,
        labelspacing=0.64,
        handlelength=2.1,
        handletextpad=0.62,
        prop={"size": 15.0},
    )
    legend.get_frame().set_facecolor("#FDFDFE")
    legend.get_frame().set_edgecolor("#D4DCE5")
    legend.get_frame().set_linewidth(0.72)

    if n_cols == 2:
        fig.subplots_adjust(left=0.075, right=0.992, top=0.82, bottom=0.1, wspace=0.16, hspace=0.24)
    else:
        fig.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.1, wspace=0.16, hspace=0.24)
    _save(fig, stem)


def main() -> None:
    hist, preds = _load_inputs()
    for stem, countries in SCENARIO_COUNTRIES.items():
        _plot_group(hist, preds, countries, stem)


if __name__ == "__main__":
    main()
