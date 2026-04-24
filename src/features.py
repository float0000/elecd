"""
features.py
===========
Builds the full feature matrix from the merged hourly DataFrame.

Features created
----------------
Target
  • target             — demand_mw at t+1 (shift −1)

Demand-side
  • total_requirement  — demand_mw + load_shedding  (true electricity need)
  • Lag features       — total_requirement at t−1, t−2, t−24, t−168
  • Rolling statistics — mean and std over 3h, 6h, 24h, 168h windows
                         (all computed on values ≤ t−1, zero leakage)

Temporal / cyclic
  • hour, dow, month, quarter, is_weekend, year_trend
  • sin/cos of hour, day-of-week, month
  • N=10 Fourier pairs for daily seasonality
  • N= 5 Fourier pairs for weekly seasonality

Weather physics
  • heat_index   — Rothfusz formula (valid for temp ≥ 27 °C)
  • cdh          — Cooling Degree Hours  (base CDH_BASE_TEMP °C)
  • hdh          — Heating Degree Hours
  • temp_sq      — T²  (nonlinear AC response)
  • temp_x_humid — T × RH  (apparent heat burden)

Economic trend
  • GDP_per_capita, Urban_pop_pct, Industry_VA_pct
    Smoothed from annual World Bank data via cubic spline → hourly.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.config import (
    ECON_INDICATORS, FOURIER_TERMS,
    LAG_HOURS, ROLLING_WINDOWS, CDH_BASE_TEMP,
)
from src.utils import add_fourier_terms


# ─── economic spline ─────────────────────────────────────────────────────────

def _spline_indicator(econ_df: pd.DataFrame, code: str,
                      target_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Fit a cubic spline to annual World Bank indicator values and evaluate
    it at every hour in target_index.  Uses decimal-year representation
    to avoid integer-overflow issues with nanosecond timestamps.
    """
    row  = econ_df[econ_df["Indicator Code"] == code]
    if row.empty:
        return np.full(len(target_index), np.nan)
    row  = row.iloc[0]

    year_cols = [c for c in econ_df.columns if isinstance(c, int)]
    vals      = pd.to_numeric(row[year_cols], errors="coerce")
    ok        = vals.notna()
    if ok.sum() < 2:
        return np.full(len(target_index), np.nan)

    years = np.array([c for c, flag in zip(year_cols, ok) if flag], dtype=float)
    vvals = vals.values[ok.values]

    cs  = CubicSpline(years, vvals, extrapolate=True)
    dec = target_index.year + (target_index.dayofyear - 1) / 365.25
    return cs(dec)


# ─── heat index ──────────────────────────────────────────────────────────────

def _heat_index(T: pd.Series, RH: pd.Series) -> pd.Series:
    """
    Rothfusz heat index regression (°C).
    Applied only where T ≥ 27 °C; otherwise returns T.
    """
    hi = (
        -8.78469475556
        + 1.61139411     * T
        + 2.33854883889  * RH
        - 0.14611605     * T  * RH
        - 0.012308094    * T  ** 2
        - 0.01642482778  * RH ** 2
        + 0.002211732    * T  ** 2 * RH
        + 0.00072546     * T  * RH ** 2
        - 0.000003582    * T  ** 2 * RH ** 2
    )
    return pd.Series(np.where(T >= 27, hi, T), index=T.index)


# ─── main feature builder ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, econ_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df       : Merged hourly DataFrame from data_loader.load_and_align().
    econ_df  : Raw annual economic DataFrame.

    Returns
    -------
    Feature-complete DataFrame with a 'target' column and no NaN rows.
    """
    print("=" * 65)
    print("FEATURE ENGINEERING")
    print("=" * 65)

    df = df.copy()

    # ── Target ──────────────────────────────────────────────────────────────
    df["target"] = df["demand_mw"].shift(-1)

    # ── Paradox variable: true electricity requirement ────────────────────
    ls_col = "load_shedding" if "load_shedding" in df.columns else None
    if ls_col:
        df["total_requirement"] = (
            df["demand_mw"].fillna(0) + df[ls_col].fillna(0)
        )
    else:
        df["total_requirement"] = df["demand_mw"]

    # ── Calendar features ────────────────────────────────────────────────
    df["hour"]        = df.index.hour
    df["dow"]         = df.index.dayofweek
    df["month"]       = df.index.month
    df["quarter"]     = df.index.quarter
    df["week_of_year"]= df.index.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)
    df["year_trend"]  = (df.index.year - df.index.year.min()).astype(float)

    # ── Cyclic encodings ─────────────────────────────────────────────────
    df["sin_hour"]    = np.sin(2 * np.pi * df["hour"]  / 24)
    df["cos_hour"]    = np.cos(2 * np.pi * df["hour"]  / 24)
    df["sin_dow"]     = np.sin(2 * np.pi * df["dow"]   / 7)
    df["cos_dow"]     = np.cos(2 * np.pi * df["dow"]   / 7)
    df["sin_month"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]   = np.cos(2 * np.pi * df["month"] / 12)

    # ── Fourier terms ─────────────────────────────────────────────────────
    df = add_fourier_terms(df, period=24,      n_terms=FOURIER_TERMS, col_name="daily")
    df = add_fourier_terms(df, period=24 * 7,  n_terms=5,             col_name="weekly")

    # ── Lag features (on total_requirement, shift so t uses only past) ───
    for lag in LAG_HOURS:
        df[f"lag_{lag}h"] = df["total_requirement"].shift(lag)

    # ── Rolling statistics (on t-1 to avoid leakage) ─────────────────────
    shifted = df["total_requirement"].shift(1)
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}h"] = shifted.rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}h"]  = shifted.rolling(w, min_periods=2).std()

    # Rolling min/max (useful for peak demand modelling)
    df["roll_min_24h"] = shifted.rolling(24, min_periods=1).min()
    df["roll_max_24h"] = shifted.rolling(24, min_periods=1).max()

    # ── Weather physics ───────────────────────────────────────────────────
    if "temp" in df.columns and "humidity" in df.columns:
        T  = df["temp"]
        RH = df["humidity"]
        df["heat_index"]    = _heat_index(T, RH)
        df["cdh"]           = (T - CDH_BASE_TEMP).clip(lower=0)
        df["hdh"]           = (CDH_BASE_TEMP - T).clip(lower=0)
        df["temp_sq"]       = T ** 2
        df["temp_x_humid"]  = T * RH

    if "apparent_temp" in df.columns:
        df["apparent_temp_lag1"] = df["apparent_temp"].shift(1)

    # ── Economic splines → hourly ─────────────────────────────────────────
    for feat_name, code in ECON_INDICATORS.items():
        df[feat_name] = _spline_indicator(econ_df, code, df.index)

    # Normalise economic features (removes scale sensitivity)
    for feat_name in ECON_INDICATORS:
        if feat_name in df.columns:
            mu  = df[feat_name].mean()
            sig = df[feat_name].std() + 1e-9
            df[f"{feat_name}_norm"] = (df[feat_name] - mu) / sig

    # ── Fill sparse generation columns before dropna ─────────────────────
    # Some PGCB generation columns (solar, wind, hydro) have early NaN years;
    # fill forward/back so they don't destroy rows.
    sparse_fill_cols = ["solar", "wind", "hydro", "generation_mw",
                        "india_bheramara_hvdc", "india_tripura"]
    for col in sparse_fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0)

    # ── Rolling std initialises to NaN for first (window-1) rows — fill 0 ─
    roll_std_cols = [c for c in df.columns if "roll_std" in c]
    df[roll_std_cols] = df[roll_std_cols].fillna(0)

    # ── Drop only rows missing the TARGET or the core lag features ─────────
    critical = ["target"] + [f"lag_{l}h" for l in LAG_HOURS]
    critical = [c for c in critical if c in df.columns]
    df.dropna(subset=critical, inplace=True)

    feature_cols = [c for c in df.columns if c not in ("target",)]
    print(f"  Features built   : {len(feature_cols)}")
    print(f"  Rows after dropna: {len(df):,}")
    return df
