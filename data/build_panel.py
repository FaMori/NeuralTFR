"""Build the empirical TFR panel from the two raw compilations.

Inputs, both in ``raw/`` and in the UN extract's schema (``name, id, id_reg,
id_sreg, year, TFR, source``): ``un_empirical_portal/EMPIRICAL_TFR.csv`` and
``new_sources/NEW_SOURCES_TFR.csv``, with an ``is_projection`` flag.

Steps, in order:

1. Drop the rows of the additions flagged ``is_projection`` (values that
   national offices publish as projections or model outputs).
2. Stack the two files. A row's ``dataset`` is ``un_portal``, ``eurostat``
   (additions labelled ``Official statistics``) or ``national_sources``.
3. Keep one row per exact observation: rows with the same country, year and
   TFR (rounded to six decimals) are one observation. When the same value is
   reported by several datasets the row kept is, in order, Eurostat, national
   sources, UN portal. Different values for the same country-year are all
   kept; the model later takes their annual median. Two rows of one dataset
   that agree to six decimals keep the lower value.
4. Fill an empty ``source`` with ``Unknown``.

    python data/build_panel.py                       # from the repository root
    python data/build_panel.py --reference PATH      # also compare with a recorded panel
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
UN_EXTRACT = RAW / "un_empirical_portal" / "EMPIRICAL_TFR.csv"
NEW_SOURCES = RAW / "new_sources" / "NEW_SOURCES_TFR.csv"
OUTPUT = HERE / "empirical_tfr.csv"
REPORT = HERE / "build_report.md"

KEY = ["name", "id", "id_reg", "id_sreg", "year", "TFR", "source"]
PRIORITY = {"eurostat": 3, "national_sources": 2, "un_portal": 1}
EUROSTAT_LABEL = "Official statistics"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    un = pd.read_csv(UN_EXTRACT, usecols=KEY)
    new = pd.read_csv(NEW_SOURCES)
    for frame, path in ((un, UN_EXTRACT), (new, NEW_SOURCES)):
        if frame[["id", "year", "TFR"]].isna().any().any():
            raise ValueError(f"{path.name}: missing id, year or TFR")
    un["dataset"] = "un_portal"
    projections = new[new["is_projection"]]
    new = new[~new["is_projection"]].drop(columns="is_projection")
    new["dataset"] = new["source"].eq(EUROSTAT_LABEL).map(
        {True: "eurostat", False: "national_sources"})
    return pd.concat([un, new], ignore_index=True), projections


def combine(stacked: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One row per exact (country, year, TFR) observation, highest-priority dataset first."""
    frame = stacked.copy()
    frame["source"] = frame["source"].fillna("").astype(str).str.strip().replace("", "Unknown")
    frame["_key"] = frame["TFR"].round(6)
    frame["_priority"] = frame["dataset"].map(PRIORITY)
    # Within a dataset, two rows equal to six decimals keep the lower value.
    frame = frame.sort_values(["id", "year", "_key", "_priority", "TFR"],
                              ascending=[True, True, True, False, True], kind="stable")
    unique = frame.drop_duplicates(["id", "year", "_key"], keep="first")
    removed = len(frame) - len(unique)
    unique = (unique.drop(columns=["_key", "_priority"])
              .astype({"id": int, "id_reg": int, "id_sreg": int, "year": int})
              .sort_values(["id", "year", "TFR"], kind="stable")
              .reset_index(drop=True))
    return unique[KEY + ["dataset"]], removed


def compare(panel: pd.DataFrame, reference: Path) -> list[str]:
    """Content comparison with a recorded panel (its ``source_type`` is our ``source``)."""
    ref = pd.read_csv(reference, low_memory=False)
    if "source_type" in ref.columns:
        ref = ref.rename(columns={"source_type": "source"})
    lines = [f"Reference: `{reference.name}` (md5 `{md5(reference)}`), {len(ref):,} rows."]
    a = panel[KEY].reset_index(drop=True)
    b = ref[KEY].astype(a.dtypes.to_dict()).reset_index(drop=True)
    if len(a) != len(b):
        lines.append(f"- row count differs: {len(a):,} built vs {len(b):,} recorded")
        return lines
    for column in KEY:
        equal = (a[column] == b[column]) | (a[column].isna() & b[column].isna())
        n = int((~equal).sum())
        lines.append(f"- `{column}`: {'identical' if n == 0 else f'{n:,} rows differ'}")
    return lines


def report(panel: pd.DataFrame, stacked: pd.DataFrame, projections: pd.DataFrame,
           removed: int, checks: list[str]) -> str:
    counts = panel["dataset"].value_counts()
    shares = (panel["source"].value_counts(normalize=True) * 100).round(1)
    last = panel.sort_values(["id", "year"]).groupby("id").tail(1)
    recent = int(last["dataset"].ne("un_portal").sum())
    lines = [
        "# Empirical TFR panel: build report", "",
        f"Built {date.today().isoformat()} by `build_panel.py` from "
        f"`EMPIRICAL_TFR.csv` (md5 `{md5(UN_EXTRACT)}`) and "
        f"`NEW_SOURCES_TFR.csv` (md5 `{md5(NEW_SOURCES)}`). "
        f"Output `empirical_tfr.csv`, md5 `{md5(OUTPUT)}`.", "",
        "## Result", "",
        f"- Observations: **{len(panel):,}**; countries and territories: **{panel['id'].nunique()}**; "
        f"years {panel['year'].min()}-{panel['year'].max()}.",
        f"- Input rows: {len(stacked):,} after dropping {len(projections)} rows flagged as projections "
        f"({projections['id'].nunique()} entities); {removed} rows were the same value reported "
        f"by more than one dataset.",
        f"- By dataset: " + ", ".join(f"{k} {v:,} ({v / len(panel):.1%})" for k, v in counts.items()) + ".",
        f"- By source: " + ", ".join(f"{k} {v}%" for k, v in shares.items()) + ".",
        f"- Country-years with more than one value: "
        f"{int(panel.duplicated(['id', 'year'], keep=False).sum()):,} rows; the model uses the annual median.",
        f"- Populations whose most recent observation comes from the additions: **{recent}** of {panel['id'].nunique()}.",
        "", "## Comparison with the recorded panel", "",
        *(checks or ["Not run (no `--reference` given)."]), "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=None,
                        help="a recorded empirical_tfr.csv to compare the output with")
    args = parser.parse_args()

    stacked, projections = load_inputs()
    panel, removed = combine(stacked)
    panel.to_csv(OUTPUT, index=False, encoding="utf-8", lineterminator="\n")

    checks = compare(panel, args.reference.resolve()) if args.reference else []
    text = report(panel, stacked, projections, removed, checks)
    REPORT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
