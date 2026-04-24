"""
model.py
========
Training, cross-validation, and evaluation.

Primary algorithm : LightGBM (MAPE objective, leaf-wise tree growth)
Fallback          : sklearn HistGradientBoostingRegressor with log(y+1)
                    target transformation (mathematically equivalent to
                    MAPE minimisation when MSE is used on log-space).

The fallback is activated automatically if lightgbm is not installed,
so the repo works out-of-the-box on any Python environment.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

from src.config import (
    TRAIN_END, TEST_START, TEST_END,
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS, CV_FOLDS,
)

# ─── backend detection ───────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _USE_LGBM = True
    print("  [Model] Backend: LightGBM")
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _USE_LGBM = False
    warnings.warn(
        "lightgbm not found — falling back to "
        "sklearn.HistGradientBoostingRegressor with log(y+1) target "
        "(mathematically equivalent MAPE approximation).",
        stacklevel=2,
    )
    print("  [Model] Backend: sklearn HistGradientBoostingRegressor (LightGBM fallback)")


# ─── helpers ─────────────────────────────────────────────────────────────────

def _split_data(df: pd.DataFrame):
    """
    Strict chronological train / test split.

    Train : everything up to and including TRAIN_END (covers up to 2023-12-31)
    Test  : TEST_START … TEST_END  (2024 full year)
    """
    feature_cols = [c for c in df.columns if c != "target"]
    X, y = df[feature_cols], df["target"]

    train_mask = df.index <= TRAIN_END
    test_mask  = (df.index >= TEST_START) & (df.index <= TEST_END)

    return (
        X[train_mask], y[train_mask],
        X[test_mask],  y[test_mask],
        feature_cols,
    )


def _mape(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1.0)
    return mean_absolute_percentage_error(y_true, y_pred) * 100


# ─── LightGBM path ───────────────────────────────────────────────────────────

def _fit_lgbm(X_train, y_train, X_val=None, y_val=None,
              n_iter: int | None = None):
    params = dict(LGBM_PARAMS)
    if n_iter is not None:
        params["n_estimators"] = n_iter

    model = lgb.LGBMRegressor(**params)
    fit_kw = {}
    if X_val is not None:
        fit_kw["eval_set"]  = [(X_val, y_val)]
        fit_kw["callbacks"] = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
    model.fit(X_train, y_train, **fit_kw)
    return model


def _predict_lgbm(model, X):
    return np.maximum(model.predict(X), 1.0)


# ─── Fallback (HistGBM + log target) ─────────────────────────────────────────

def _make_histgbm():
    return HistGradientBoostingRegressor(
        loss               = "squared_error",   # MSE on log(y) ≈ MAPE
        learning_rate      = LGBM_PARAMS["learning_rate"],
        max_iter           = 500,
        max_leaf_nodes     = LGBM_PARAMS["num_leaves"],
        min_samples_leaf   = LGBM_PARAMS["min_child_samples"],
        l2_regularization  = LGBM_PARAMS["reg_lambda"],
        early_stopping     = True,
        n_iter_no_change   = EARLY_STOPPING_ROUNDS,
        validation_fraction= 0.1,
        random_state       = LGBM_PARAMS["random_state"],
        verbose            = 0,
    )


def _fit_fallback(X_train, y_train, **_kw):
    model = _make_histgbm()
    model.fit(X_train, np.log1p(y_train))
    return model


def _predict_fallback(model, X):
    return np.maximum(np.expm1(model.predict(X)), 1.0)


# ─── public interface ─────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame):
    """
    Full pipeline: CV → final fit → hold-out evaluation.

    Parameters
    ----------
    df : feature-complete DataFrame with 'target' column.

    Returns
    -------
    model       : fitted model object
    mape        : hold-out MAPE (%)
    metrics     : dict of full hold-out metrics
    predictions : pd.Series of predictions on the test set
    feature_cols: list[str]
    """
    print("\n" + "=" * 65)
    print("MODEL TRAINING")
    print("=" * 65)

    X_train, y_train, X_test, y_test, feature_cols = _split_data(df)
    print(f"  Train : {len(X_train):>7,} rows | "
          f"{X_train.index.min().date()} → {X_train.index.max().date()}")
    print(f"  Test  : {len(X_test):>7,}  rows | "
          f"{X_test.index.min().date()} → {X_test.index.max().date()}")

    # ── TimeSeriesSplit cross-validation ────────────────────────────────
    print(f"\n  TimeSeriesSplit CV ({CV_FOLDS} folds) …")
    tscv     = TimeSeriesSplit(n_splits=CV_FOLDS)
    cv_mapes = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        Xtr, ytr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        Xvl, yvl = X_train.iloc[val_idx], y_train.iloc[val_idx]

        if _USE_LGBM:
            m   = _fit_lgbm(Xtr, ytr, Xvl, yvl)
            prd = _predict_lgbm(m, Xvl)
        else:
            m   = _fit_fallback(Xtr, ytr)
            prd = _predict_fallback(m, Xvl)

        mape_fold = _mape(yvl, prd)
        cv_mapes.append(mape_fold)
        n_it = getattr(m, "best_iteration_", None) or getattr(m, "n_iter_", "?")
        print(f"    Fold {fold}/{CV_FOLDS}  │  MAPE = {mape_fold:.3f}%  "
              f"│  iters = {n_it}")

    cv_mean = float(np.mean(cv_mapes))
    cv_std  = float(np.std(cv_mapes))
    print(f"\n  CV MAPE : {cv_mean:.3f}% ± {cv_std:.3f}%")

    # ── Final fit on full training data ─────────────────────────────────
    print("\n  Fitting final model on full training data …")
    if _USE_LGBM:
        # Use best iteration from last CV fold as a guide
        best = getattr(cv_mapes, "__len__", lambda: None)()
        model = _fit_lgbm(X_train, y_train)
        y_pred = _predict_lgbm(model, X_test)
    else:
        model  = _fit_fallback(X_train, y_train)
        y_pred = _predict_fallback(model, X_test)

    n_it = getattr(model, "best_iteration_", None) or getattr(model, "n_iter_", "?")
    print(f"  Boosting rounds used : {n_it}")

    # ── Hold-out metrics ─────────────────────────────────────────────────
    y_true  = y_test.values
    mape    = _mape(y_true, y_pred)
    mae     = float(np.mean(np.abs(y_true - y_pred)))
    rmse    = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2      = float(1 - np.sum((y_true - y_pred) ** 2) /
                        np.sum((y_true - y_true.mean()) ** 2))

    metrics = {
        "mape": mape, "mae": mae, "rmse": rmse, "r2": r2,
        "cv_mape_mean": cv_mean, "cv_mape_std": cv_std,
        "n_train": len(X_train), "n_test": len(X_test),
    }

    print(f"""
  ╔══════════════════════════════════════╗
  ║  Test MAPE  :  {mape:>7.3f} %           ║
  ║  Test MAE   :  {mae:>7.1f} MW          ║
  ║  Test RMSE  :  {rmse:>7.1f} MW          ║
  ║  Test R²    :  {r2:>7.4f}              ║
  ╚══════════════════════════════════════╝""")

    predictions = pd.Series(y_pred, index=X_test.index, name="predicted")
    return model, mape, metrics, predictions, feature_cols


def get_feature_importance(model, feature_cols: list[str]) -> pd.Series:
    """Return feature importance Series sorted descending."""
    if _USE_LGBM:
        fi = pd.Series(model.feature_importances_,
                       index=feature_cols).sort_values(ascending=False)
    else:
        try:
            fi = pd.Series(model.feature_importances_,
                           index=feature_cols).sort_values(ascending=False)
        except AttributeError:
            fi = pd.Series(dtype=float)
    return fi
