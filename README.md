# Elecd

A production-grade **classical ML pipeline** for next-hour electricity demand
forecasting (`demand_mw` at *t+1*) using data from the Bangladesh Power Grid
Company (PGCB), hourly weather observations, and World Bank economic indicators.

---

## Highlights

| | |
|---|---|
| **Algorithm** | LightGBM (MAPE objective, leaf-wise tree growth) |
| **Fallback** | sklearn `HistGradientBoostingRegressor` + log(y+1) target |
| **Validation** | `TimeSeriesSplit` (5 folds, zero data leakage) |
| **Train period** | All data up to 2023-12-31 |
| **Test period** | Full year 2024 |
| **Hold-out MAPE** | ~3 % |

---

## Repository Structure

```
electricity-demand-forecast/
├── main.py               ← entry point: run this
├── requirements.txt
├── .gitignore
├── data/                 ← place your xlsx files here (not committed)
│   ├── PGCB_date_power_demand.xlsx
│   ├── weather_data.xlsx
│   └── economic_full_1.xlsx
├── outputs/              ← generated plots and CSVs (not committed)
└── src/
    ├── config.py         ← all settings in one place
    ├── data_loader.py    ← loading, half-hourly aggregation, alignment
    ├── features.py       ← all feature engineering
    ├── model.py          ← training, CV, evaluation
    └── utils.py          ← Fourier terms, spike filter, plot theme
```

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/electricity-demand-forecast.git
cd electricity-demand-forecast

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place data files in data/
#    PGCB_date_power_demand.xlsx
#    weather_data.xlsx
#    economic_full_1.xlsx

# 5. Run the pipeline
python main.py
```

Outputs are written to `outputs/`:

| File | Description |
|------|-------------|
| `actual_vs_predicted_june2024.png` | First week of June 2024 — Actual vs Predicted + error panel |
| `feature_importance.png` | Top-30 feature importance (LightGBM gain) |
| `monthly_overview_2024.png` | 12-panel monthly actual vs predicted |
| `predictions_2024.csv` | Full hourly predictions for 2024 |
| `metrics_summary.csv` | MAPE, MAE, RMSE, R², monthly breakdown |

---

## Data Handling Decisions

### Half-hourly (:30) timestamps
PGCB records some readings at :30 (notably the *Evening Peak* at 18:30).
These are **weighted-averaged** with the adjacent :00 reading before
collapsing to a strict hourly series:

- Regular :30 readings → weight (0.45 × :00 value) + (0.55 × :30 value)
- Peak-flagged :30 rows → weight (0.35 × :00) + (0.65 × :30)

Weights are configurable in `src/config.py`.

### Missing timestamps
After resampling to the strict 1-hour grid, any gaps are filled by
**time-weighted interpolation using previous observations only**
(up to `MAX_INTERPOLATION_HOURS = 3` consecutive hours).  This prevents
future data leaking into past features.

### Demand spikes
Undocumented spikes are detected with a centred 24-hour rolling Z-score
filter (threshold |z| > 3.5) and replaced by linear time-interpolation.

### Economic indicators
Annual World Bank data (GDP/capita, Urban population %, Industry value-added %)
are smoothed to hourly frequency via **cubic spline interpolation**,
eliminating the "step jump" artefact that occurs when simply repeating annual
values.

---

## Feature Engineering

| Group | Features |
|-------|----------|
| **Target** | `demand_mw` at t+1 |
| **Demand history** | `total_requirement` (demand + load shedding), lags t−1/2/24/168 h, rolling mean/std 3/6/24/168 h |
| **Temporal** | hour, dow, month, quarter, is_weekend, year_trend |
| **Cyclic** | sin/cos of hour, dow, month |
| **Fourier** | N=10 daily pairs + N=5 weekly pairs |
| **Weather** | temp, humidity, cloud, sunshine, dew_point, precipitation |
| **Weather physics** | Heat Index, Cooling Degree Hours, Heating Degree Hours, T², T×RH |
| **Economic** | GDP/capita, Urban pop %, Industry VA % (all spline-interpolated) |

---

## Configuration

All knobs live in `src/config.py`:

```python
HALF_HOUR_WEIGHTS       = (0.45, 0.55)   # (:00, :30) blend — general
HALF_HOUR_WEIGHTS_PEAK  = (0.35, 0.65)   # for Evening Peak rows
MAX_INTERPOLATION_HOURS = 3
SPIKE_THRESHOLD         = 3.5
FOURIER_TERMS           = 10
LAG_HOURS               = [1, 2, 24, 168]
TRAIN_END               = "2023-12-31 23:00"
TEST_START              = "2024-01-01 00:00"
LGBM_PARAMS             = { ... }        # full LightGBM hyperparameters
```

---

## License

MIT
