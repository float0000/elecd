"""
data_loader.py
==============
Loads and aligns all three data sources into a single hourly DataFrame.

Key behaviours
--------------
1. Half-hourly (:30) timestamps are **weighted-averaged** with their
   preceding :00 reading before the series is collapsed to strict hourly.
   Evening-Peak flagged rows receive a higher weight on the :30 reading
   because they capture the true peak demand within that hour.

2. After resampling to a strict 1-hour grid, any remaining gaps are
   filled using **backward interpolation from previous observations**
   (forward-looking fill is avoided to prevent data leakage).

3. The weather file header is auto-detected — no hard-coded skip row
   needed. Columns are renamed using WEATHER_COLS from config.

4. Economic data is left in annual form here; the feature module handles
   spline interpolation to hourly frequency.
"""

import warnings
import pandas as pd
import numpy as np

from src.config import (
    FILE_PATHS, WEATHER_COLS,
    HALF_HOUR_WEIGHTS, HALF_HOUR_WEIGHTS_PEAK,
    MAX_INTERPOLATION_HOURS,
)
from src.utils import replace_spikes, fill_from_prior


# ─── helpers ─────────────────────────────────────────────────────────────────

def _detect_weather_header(path: str) -> int:
    """
    Scans the first 20 rows to find the row that contains the word 'time'
    in its cells, and returns the 0-based index to use as `header=` in
    pd.read_excel().  Falls back to row 0 if nothing is found.
    """
    probe = pd.read_excel(path, header=None, nrows=20)
    for i, row in probe.iterrows():
        if any(str(v).strip().lower() == "time" for v in row.values):
            return int(i)
    warnings.warn("Could not auto-detect weather header row; defaulting to 0.")
    return 0


def _aggregate_half_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a PGCB DataFrame (which may contain :30 readings) to a
    strict hourly series using weighted averages.

    Logic
    -----
    * Group rows by their floored-to-hour timestamp.
    * Within each group, separate :00 and :30 readings.
    * If **both** exist:
        - Use HALF_HOUR_WEIGHTS_PEAK when the :30 row is flagged as a
          peak (remarks contains 'Peak').
        - Otherwise use HALF_HOUR_WEIGHTS.
    * If only one reading exists, use it as-is.
    * Numeric columns are weighted; non-numeric columns take the value
      of the :00 row (or the only available row).
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    df = df.copy()
    df["_hour"] = df.index.floor("h")

    records = []
    for hour, grp in df.groupby("_hour"):
        on_hour = grp[grp.index.minute == 0]
        on_half = grp[grp.index.minute == 30]

        if on_hour.empty and on_half.empty:
            continue

        # Single reading — nothing to average
        if on_hour.empty or on_half.empty:
            chosen = on_half.iloc[0] if on_hour.empty else on_hour.iloc[0]
            row = chosen.to_dict()
            row["_hour"] = hour
            records.append(row)
            continue

        # Both exist — weighted average for numeric columns
        r0 = on_hour.iloc[0]
        r30 = on_half.iloc[0]

        is_peak = (
            "peak" in str(r30.get("remarks", "")).lower() or
            "peak" in str(r0.get("remarks", "")).lower()
        )
        w0, w30 = HALF_HOUR_WEIGHTS_PEAK if is_peak else HALF_HOUR_WEIGHTS

        row = {}
        for c in numeric_cols:
            v0  = r0[c]  if c in r0.index  else np.nan
            v30 = r30[c] if c in r30.index else np.nan
            if pd.notna(v0) and pd.notna(v30):
                row[c] = w0 * v0 + w30 * v30
            elif pd.notna(v0):
                row[c] = v0
            else:
                row[c] = v30

        # Non-numeric: prefer the :00 row's value
        for c in grp.columns:
            if c not in numeric_cols and c != "_hour":
                row[c] = r0[c] if c in r0.index else np.nan

        row["_hour"] = hour
        records.append(row)

    result = pd.DataFrame(records).set_index("_hour")
    result.index.name = None
    return result


# ─── main loader ─────────────────────────────────────────────────────────────

def load_pgcb() -> pd.DataFrame:
    """Load and clean the PGCB demand file."""
    df = pd.read_excel(FILE_PATHS["pgcb"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df             = df.sort_values("datetime").reset_index(drop=True)
    df             = df.set_index("datetime")

    # Keep only useful numeric columns — drop sparse/string cols such as
    # 'remarks' (93% NaN), 'nepal', 'india_adani' (mostly NaN too).
    KEEP_PGCB = ["demand_mw", "load_shedding", "gas", "coal",
                 "liquid_fuel", "hydro", "solar", "wind",
                 "generation_mw", "india_bheramara_hvdc", "india_tripura"]
    keep = [c for c in KEEP_PGCB if c in df.columns]
    df   = df[keep]

    # Coerce to numeric
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Aggregate :30 readings into their parent hour ─────────────────────
    df = _aggregate_half_hourly(df)

    # ── Resample to strict 1-hour grid ────────────────────────────────────
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df       = df.reindex(full_idx)

    # Fill newly created gaps using prior observations only
    for col in df.select_dtypes(include="number").columns:
        df[col] = fill_from_prior(df[col], MAX_INTERPOLATION_HOURS)

    # ── Spike removal on demand_mw ────────────────────────────────────────
    df["demand_mw"], n_spikes = replace_spikes(df["demand_mw"])
    print(f"  [PGCB]    {len(df):,} hourly rows | "
          f"{df.index.min().date()} → {df.index.max().date()} | "
          f"spikes replaced: {n_spikes}")
    return df


def load_weather() -> pd.DataFrame:
    """Load and clean the weather file with auto-detected header."""
    header_row = _detect_weather_header(FILE_PATHS["weather"])
    df         = pd.read_excel(FILE_PATHS["weather"], header=header_row)

    # Rename columns: handle both raw xlsx names and already-renamed names
    df.columns = [str(c).strip() for c in df.columns]
    df         = df.rename(columns=WEATHER_COLS)

    # The time column might be named 'time' after rename or still 'time'
    time_col = "time" if "time" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df           = df.dropna(subset=[time_col])
    df           = df.set_index(time_col).sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Resample to strict hourly and fill short gaps
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df       = df.reindex(full_idx)
    for col in df.columns:
        df[col] = fill_from_prior(df[col], max_hours=2)

    print(f"  [Weather] {len(df):,} hourly rows | "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


def load_econ() -> pd.DataFrame:
    """Load the economic file as-is (spline interpolation in features.py)."""
    df = pd.read_excel(FILE_PATHS["econ"])
    print(f"  [Econ]    {len(df)} indicators | "
          f"{sum(isinstance(c, int) for c in df.columns)} annual data points")
    return df


def load_and_align() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master loader: returns (merged_hourly_df, econ_df).

    merged_hourly_df has DatetimeIndex and contains:
      - PGCB columns: demand_mw, load_shedding, gas, coal, liquid_fuel, …
      - Weather columns (renamed via WEATHER_COLS)
    econ_df is the raw annual economic DataFrame (used by feature module).
    """
    print("=" * 65)
    print("LOADING & ALIGNING DATA")
    print("=" * 65)

    pgcb    = load_pgcb()
    weather = load_weather()
    econ    = load_econ()

    weather_keep = [c for c in WEATHER_COLS.values() if c in weather.columns]

    merged = pgcb.join(weather[weather_keep], how="left")

    # Fill any residual weather NaN (e.g. slight index misalignment)
    for col in weather_keep:
        if col in merged.columns:
            merged[col] = fill_from_prior(merged[col], max_hours=24)

    n_weather_nan = merged[weather_keep].isna().sum().sum()
    print(f"\n  Merged : {len(merged):,} rows | "
          f"{merged.index.min().date()} → {merged.index.max().date()}")
    print(f"  Weather NaN remaining : {n_weather_nan}")

    return merged, econ
