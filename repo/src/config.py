"""
config.py
=========
Central configuration — edit this file to adapt the pipeline to new
data, indicators, or model hyperparameters without touching any logic.
"""

import os

# ─── File paths ──────────────────────────────────────────────────────────────
# Place all xlsx files inside the data/ directory.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FILE_PATHS = {
    "pgcb"   : os.path.join(DATA_DIR, "PGCB_date_power_demand.xlsx"),
    "weather": os.path.join(DATA_DIR, "weather_data.xlsx"),
    "econ"   : os.path.join(DATA_DIR, "economic_full_1.xlsx"),
}

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ─── Weather column rename map ────────────────────────────────────────────────
# Keys  = raw column names in the xlsx (after header auto-detection)
# Values = short names used throughout the pipeline
WEATHER_COLS = {
    "temperature_2m (°C)"           : "temp",
    "relative_humidity_2m (%)"      : "humidity",
    "apparent_temperature (°C)"     : "apparent_temp",
    "precipitation (mm)"            : "precipitation",
    "dew_point_2m (°C)"             : "dew_point",
    "soil_temperature_0_to_7cm (°C)": "soil_temp",
    "wind_direction_10m (°)"        : "wind_dir",
    "cloud_cover (%)"               : "cloud",
    "sunshine_duration (s)"         : "sunshine",
}

# ─── Economic indicators (Indicator Code → feature name) ─────────────────────
# Uses World Bank indicator codes — robust against minor name changes.
ECON_INDICATORS = {
    "GDP_per_capita"  : "NY.GDP.PCAP.CD",      # GDP per capita (current US$)
    "Urban_pop_pct"   : "SP.URB.TOTL.IN.ZS",   # Urban pop (% of total)
    "Industry_VA_pct" : "NV.IND.TOTL.ZS",       # Industry VA (% of GDP)
}

# ─── Half-hourly aggregation weights ─────────────────────────────────────────
# When a :30 reading exists alongside a :00 reading within the same hour,
# the final hourly value = w_hour * val(:00)  +  w_half * val(:30).
# Evening_Peak rows (annotated demand peak) are given more weight since
# they represent the actual observed peak for that hour.
HALF_HOUR_WEIGHTS       = (0.45, 0.55)   # (w_:00, w_:30) — general
HALF_HOUR_WEIGHTS_PEAK  = (0.35, 0.65)   # used when the :30 row is flagged as peak

# ─── Missing-timestamp fill ───────────────────────────────────────────────────
# Max consecutive hours to fill via time-weighted interpolation from
# previous observations (forward-looking fill is forbidden to avoid leakage).
MAX_INTERPOLATION_HOURS = 3

# ─── Spike detection ─────────────────────────────────────────────────────────
SPIKE_WINDOW    = 24     # rolling window in hours
SPIKE_THRESHOLD = 3.5    # |z-score| above this → spike

# ─── Feature engineering ─────────────────────────────────────────────────────
FOURIER_TERMS   = 10     # N Fourier pairs (sin+cos) for daily seasonality
LAG_HOURS       = [1, 2, 24, 168]          # lag features for demand
ROLLING_WINDOWS = [3, 6, 24, 168]          # rolling mean / std windows
CDH_BASE_TEMP   = 18.0   # °C base for Cooling Degree Hours

# ─── Train / Test split ───────────────────────────────────────────────────────
TRAIN_END = "2023-12-31 23:00"   # inclusive — everything up to this
TEST_START = "2024-01-01 00:00"
TEST_END   = "2024-12-31 23:00"

# ─── LightGBM hyper-parameters ───────────────────────────────────────────────
LGBM_PARAMS = {
    "objective"        : "mape",
    "metric"           : "mape",
    "n_estimators"     : 1500,
    "learning_rate"    : 0.04,
    "num_leaves"       : 63,
    "max_depth"        : -1,          # leaf-wise: no depth limit
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.2,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "verbose"          : -1,
}
EARLY_STOPPING_ROUNDS = 60
CV_FOLDS              = 5
