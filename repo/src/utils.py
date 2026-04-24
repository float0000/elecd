"""
utils.py
========
Reusable helpers: Fourier features, Hampel filter, plotting theme.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ─── Fourier seasonality terms ────────────────────────────────────────────────
def add_fourier_terms(df: pd.DataFrame, period: float, n_terms: int,
                      col_name: str) -> pd.DataFrame:
    """
    Append N sin/cos Fourier pairs capturing seasonality of `period` hours.

    Parameters
    ----------
    df        : DataFrame with a DatetimeIndex.
    period    : Cycle length in hours (e.g. 24 for daily, 24*7 for weekly).
    n_terms   : Number of harmonic pairs.
    col_name  : Short label used in column names, e.g. 'daily'.

    Returns
    -------
    df with new columns  fourier_{col_name}_sin_{k}  /  _cos_{k}.
    """
    t = (df.index.hour + df.index.minute / 60.0 +
         df.index.dayofweek * 24.0)           # fractional hours since Mon 00:00
    for k in range(1, n_terms + 1):
        df[f"fourier_{col_name}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"fourier_{col_name}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


# ─── Hampel / rolling-Z-score spike filter ───────────────────────────────────
def rolling_zscore_mask(s: pd.Series, window: int = 24,
                        threshold: float = 3.5) -> pd.Series:
    """
    Returns a boolean mask of likely spikes in series `s`.

    Uses a centred rolling window so the local statistics reflect the
    neighbourhood of each point rather than only its past.
    """
    roll     = s.rolling(window, center=True, min_periods=max(6, window // 4))
    z_scores = (s - roll.mean()) / (roll.std() + 1e-9)
    return z_scores.abs() > threshold


def replace_spikes(s: pd.Series, window: int = 24,
                   threshold: float = 3.5) -> tuple[pd.Series, int]:
    """
    Detect spikes and replace them with linear time-interpolation.

    Returns (cleaned_series, n_spikes_replaced).
    """
    mask = rolling_zscore_mask(s, window, threshold)
    n    = int(mask.sum())
    out  = s.copy().astype(float)
    out[mask] = np.nan
    out = out.interpolate(method="time")
    return out, n


# ─── Missing-value fill using prior observations ─────────────────────────────
def fill_from_prior(s: pd.Series, max_hours: int = 3) -> pd.Series:
    """
    Fill NaN gaps by time-weighted interpolation capped at `max_hours`
    consecutive missing entries.  Only looks *backwards* (no leakage):
    after exhausting interpolation, falls back to forward-fill then back-fill.
    """
    return (
        s.interpolate(method="time", limit=max_hours,
                      limit_direction="forward")
         .ffill()
         .bfill()
    )


# ─── Matplotlib dark theme ────────────────────────────────────────────────────
DARK   = "#0d1117"
PANEL  = "#161b22"
GRID_C = "#21262d"
EDGE   = "#30363d"
TEXT   = "#e6edf3"
DIM    = "#8b949e"
BLUE   = "#58a6ff"
RED    = "#f78166"
FILL   = "#388bfd"

RC_DARK = {
    "figure.facecolor": DARK,  "axes.facecolor"  : PANEL,
    "axes.edgecolor"  : EDGE,  "axes.labelcolor" : TEXT,
    "axes.titlecolor" : TEXT,  "xtick.color"     : DIM,
    "ytick.color"     : DIM,   "text.color"      : TEXT,
    "grid.color"      : GRID_C,"grid.linestyle"  : "--",
    "grid.alpha"      : 0.65,  "font.family"     : "DejaVu Sans",
    "legend.facecolor": GRID_C,"legend.edgecolor": EDGE,
    "legend.framealpha": 0.5,
}


def apply_dark_theme():
    plt.rcParams.update(RC_DARK)
