"""
main.py
=======
Entry point — runs the full pipeline end-to-end and saves all outputs
to the outputs/ directory.

Usage
-----
    python main.py

Outputs
-------
  outputs/actual_vs_predicted_june2024.png  — Actual vs Predicted (1 week)
  outputs/feature_importance.png            — Top-30 feature importance
  outputs/monthly_overview_2024.png         — 12-panel monthly overview
  outputs/metrics_summary.csv              — All evaluation metrics
  outputs/predictions_2024.csv             — Full 2024 hourly predictions
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

from src.data_loader import load_and_align
from src.features    import build_features
from src.model       import train_and_evaluate, get_feature_importance
from src.config      import OUTPUTS_DIR, TEST_START, TEST_END
from src.utils       import (
    DARK, PANEL, GRID_C, EDGE, TEXT, DIM, BLUE, RED, FILL, RC_DARK
)
from sklearn.metrics import mean_absolute_percentage_error

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─── Pipeline ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # 1. Load & align
    df_raw, econ_df = load_and_align()

    # 2. Feature engineering
    df_final = build_features(df_raw, econ_df)

    # 3. Train & evaluate
    model, mape, metrics, predictions, feature_cols = train_and_evaluate(df_final)

    # 4. Save predictions CSV
    test_mask = (df_final.index >= TEST_START) & (df_final.index <= TEST_END)
    results   = pd.DataFrame({
        "actual"   : df_final.loc[test_mask, "target"],
        "predicted": predictions,
    })
    results["error_pct"] = 100 * (results["predicted"] - results["actual"]) / results["actual"].clip(lower=1)
    results.to_csv(os.path.join(OUTPUTS_DIR, "predictions_2024.csv"))

    # 5. Monthly breakdown
    print("\n  Monthly breakdown (2024):")
    print(f"  {'Month':<10} {'MAPE':>8}  {'MAE':>9}  {'n_hrs':>6}")
    print("  " + "─" * 42)
    y_true = results["actual"].values
    y_pred = results["predicted"].values
    for m in range(1, 13):
        mask_m = results.index.month == m
        if not mask_m.any():
            continue
        mm  = mean_absolute_percentage_error(y_true[mask_m], y_pred[mask_m]) * 100
        ma  = np.mean(np.abs(y_true[mask_m] - y_pred[mask_m]))
        mn  = pd.Timestamp(2024, m, 1).strftime("%b")
        print(f"  {mn:<10} {mm:>7.3f}%  {ma:>8.1f}  {mask_m.sum():>6,}")
        metrics[f"mape_{mn}"] = round(mm, 4)

    # 6. Save metrics CSV
    pd.Series(metrics).to_csv(
        os.path.join(OUTPUTS_DIR, "metrics_summary.csv"), header=["value"]
    )

    # 7. Plots
    _plot_week(results, metrics["mape"])
    _plot_feature_importance(model, feature_cols)
    _plot_monthly_overview(results)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  ★  Final Test MAPE = {mape:.3f}%")
    print(f"     Runtime         = {elapsed:.1f} s")
    print(f"     Outputs saved to: {os.path.abspath(OUTPUTS_DIR)}")
    print(f"{'='*65}")


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _plot_week(results: pd.DataFrame, full_mape: float):
    """Actual vs Predicted — first full week of June 2024 with error bar."""
    june_mask = (results.index >= "2024-06-01") & (results.index < "2024-06-08")
    t_idx     = results.index[june_mask]
    actual    = results.loc[june_mask, "actual"].values
    predicted = results.loc[june_mask, "predicted"].values
    w_mape    = mean_absolute_percentage_error(actual, predicted) * 100

    with plt.rc_context(RC_DARK):
        fig = plt.figure(figsize=(15, 9), facecolor=DARK)
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1], hspace=0.07)

        ax1 = fig.add_subplot(gs[0])
        ax1.fill_between(t_idx, actual, predicted, alpha=0.15, color=FILL)
        ax1.plot(t_idx, actual,    color=BLUE, lw=2.0, label="Actual",    zorder=3)
        ax1.plot(t_idx, predicted, color=RED,  lw=1.8, label="Predicted", zorder=3,
                 linestyle="--", dashes=(6, 2))
        ax1.set_ylabel("Demand  (MW)", fontsize=12, labelpad=8)
        ax1.set_xlim(t_idx[0], t_idx[-1])
        ax1.tick_params(labelbottom=False)
        ax1.grid(True, axis="y")
        ax1.legend(loc="upper right", fontsize=11)
        ax1.annotate(
            f"Week MAPE = {w_mape:.2f}%\n"
            f"MAE = {np.mean(np.abs(actual - predicted)):.0f} MW",
            xy=(0.02, 0.96), xycoords="axes fraction", fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc=GRID_C, ec=EDGE, alpha=0.9),
        )
        ax1.set_title(
            f"Bangladesh Electricity Demand Forecast — First Week of June 2024\n"
            f"Hold-Out Test  │  Full-Year 2024 MAPE = {full_mape:.3f}%",
            fontsize=13, fontweight="bold", pad=12,
        )

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        err = 100 * (predicted - actual) / np.maximum(actual, 1)
        ax2.bar(t_idx, err, width=0.038,
                color=[RED if v > 0 else BLUE for v in err], alpha=0.85)
        ax2.axhline(0, color=TEXT, lw=0.8, linestyle="--", alpha=0.5)
        ax2.set_ylabel("Error (%)", fontsize=10, labelpad=8)
        ax2.set_xlabel("Date", fontsize=11, labelpad=6)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
        ax2.xaxis.set_major_locator(mdates.DayLocator())
        ax2.grid(True, axis="y")

        path = os.path.join(OUTPUTS_DIR, "actual_vs_predicted_june2024.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
        plt.close()
        print(f"\n  ✓ Saved → {path}")


def _plot_feature_importance(model, feature_cols: list[str]):
    """Top-30 feature importance bar chart."""
    from src.model import get_feature_importance

    fi   = get_feature_importance(model, feature_cols)
    if fi.empty:
        print("  [skip] Feature importance not available for this backend.")
        return

    top  = fi.head(30).sort_values(ascending=True)
    cmap = plt.cm.plasma(np.linspace(0.15, 0.95, len(top)))

    with plt.rc_context(RC_DARK):
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=DARK)
        bars = ax.barh(top.index, top.values, color=cmap,
                       edgecolor="none", height=0.72)
        for bar, val in zip(bars, top.values):
            ax.text(bar.get_width() + top.values.max() * 0.008,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f}", va="center", ha="left",
                    fontsize=7.5, color=DIM)

        ax.set_title(
            "Feature Importance — Top 30 Drivers of Electricity Demand\n"
            "(LightGBM split-gain  │  Hold-Out 2024)",
            fontsize=12, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Importance Score (Gain)", fontsize=11, labelpad=8)
        ax.set_xlim(right=top.values.max() * 1.18)
        ax.grid(True, axis="x", alpha=0.5)
        plt.tight_layout()

        path = os.path.join(OUTPUTS_DIR, "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
        plt.close()
        print(f"  ✓ Saved → {path}")


def _plot_monthly_overview(results: pd.DataFrame):
    """3×4 grid of monthly Actual vs Predicted for 2024."""
    MNAMES = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    y_true = results["actual"].values
    y_pred = results["predicted"].values

    with plt.rc_context(RC_DARK):
        fig, axes = plt.subplots(3, 4, figsize=(20, 11), facecolor=DARK)
        fig.suptitle(
            f"2024 Hold-Out Forecast — Monthly Actual vs Predicted Demand  "
            f"(Full-Year MAPE = "
            f"{mean_absolute_percentage_error(y_true, y_pred)*100:.3f}%)",
            fontsize=14, fontweight="bold", y=1.01,
        )

        for ax, m, mn in zip(axes.flat, range(1, 13), MNAMES):
            mask_m = results.index.month == m
            if not mask_m.any():
                ax.set_visible(False)
                continue
            ti  = results.index[mask_m]
            at  = results.loc[mask_m, "actual"].values
            pr  = results.loc[mask_m, "predicted"].values
            mm  = mean_absolute_percentage_error(at, pr) * 100
            ma  = np.mean(np.abs(at - pr))

            ax.fill_between(ti, at, pr, alpha=0.18, color=FILL)
            ax.plot(ti, at, color=BLUE, lw=1.1, label="Actual")
            ax.plot(ti, pr, color=RED,  lw=0.9, linestyle="--", label="Pred")
            ax.set_title(f"{mn}  │  MAPE={mm:.2f}%  MAE={ma:.0f} MW",
                         fontsize=9)
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.45)

        axes[0][0].legend(fontsize=8, framealpha=0.4)
        plt.tight_layout()

        path = os.path.join(OUTPUTS_DIR, "monthly_overview_2024.png")
        plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK)
        plt.close()
        print(f"  ✓ Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
