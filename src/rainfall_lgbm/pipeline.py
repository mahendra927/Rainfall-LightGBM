"""One experimental run: preprocessing -> treatment -> SMOTENC -> LightGBM -> metrics.

Order of operations (identical for every scenario):
 1. lag features are already present on the full dataset (generated before partitioning)
 2. training-based median/mode imputation and categorical encoding
 3. optional data-quality treatment, estimated on training data only
      - iqr / zscore: training-derived bounds, applied to train/val/test (``apply_to: all``)
      - isolation_forest: anomalous TRAINING rows removed; val/test untouched
 4. optional SMOTENC on the (filtered) training subset
 5. LightGBM with early stopping on validation log loss; fixed threshold 0.50
"""
from __future__ import annotations

import time

import pandas as pd

from .features import predictor_columns
from .metrics import compute_metrics
from .model import train_lgbm
from .preprocessing import Preprocessor
from .treatments import apply_smotenc, clip, iqr_bounds, isolation_forest_mask, zscore_bounds


def run_scenario(df: pd.DataFrame, subset: pd.Series, scenario: dict, cfg: dict, seed: int,
                 lags: list[int] | None = None, keep_predictions: bool = True) -> dict:
    t0 = time.time()
    use_lags = (lags if lags is not None else cfg["lag_features"]["lags"]) if scenario["temporal_lag"] else None
    cols, num, cat = predictor_columns(cfg, use_lags)

    tr, va, te = (subset == "train"), (subset == "validation"), (subset == "test")
    pre = Preprocessor(num, cat).fit(df.loc[tr, cols])
    X_tr, X_va, X_te = (pre.transform(df.loc[m, cols]) for m in (tr, va, te))
    y_tr, y_va, y_te = (df.loc[m, "y"] for m in (tr, va, te))
    info = {"n_features": len(cols), "n_train_before": int(tr.sum()),
            "train_yes_before": int(y_tr.sum()), "train_no_before": int((1 - y_tr).sum())}

    treatment = scenario["treatment"]
    if treatment in ("iqr", "zscore"):
        if treatment == "iqr":
            lo, hi = iqr_bounds(X_tr, num, cfg["iqr"]["k"])
            apply_to = cfg["iqr"]["apply_to"]
        else:
            lo, hi = zscore_bounds(X_tr, num, cfg["zscore"]["threshold"])
            apply_to = cfg["zscore"]["apply_to"]
        X_tr, info["capped_values_train"] = clip(X_tr, num, lo, hi)
        if apply_to == "all":
            X_va, _ = clip(X_va, num, lo, hi)
            X_te, _ = clip(X_te, num, lo, hi)
    elif treatment == "isolation_forest":
        keep = isolation_forest_mask(X_tr, num, cfg["isolation_forest"]["n_estimators"],
                                     cfg["isolation_forest"]["contamination"], seed)
        removed_y = y_tr[~keep]
        info.update({"if_removed_total": int((~keep).sum()), "if_removed_yes": int(removed_y.sum()),
                     "if_removed_no": int((1 - removed_y).sum())})
        X_tr, y_tr = X_tr[keep], y_tr[keep]

    if scenario["smotenc"]:
        X_tr, y_tr = apply_smotenc(X_tr, y_tr, num, cat, cfg["smotenc"]["sampling_strategy"],
                                   cfg["smotenc"]["k_neighbors"], seed)
    info.update({"train_yes_final": int(y_tr.sum()), "train_no_final": int((1 - y_tr).sum())})

    model = train_lgbm(X_tr, y_tr, X_va, y_va, cfg["lightgbm"], cfg["early_stopping_rounds"], seed)
    info["best_iteration"] = int(model.best_iteration_ or cfg["lightgbm"]["n_estimators"])

    thr = cfg["decision_threshold"]
    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    out = {"scenario": scenario["code"], "seed": seed, "info": info,
           "validation": compute_metrics(y_va, p_va, thr), "test": compute_metrics(y_te, p_te, thr),
           "runtime_s": round(time.time() - t0, 1)}
    if keep_predictions:
        pred = df.loc[te, ["Location", "Date", "Rainfall"]].copy()
        pred["y_true"] = y_te.to_numpy()
        pred["prob_yes"] = p_te
        pred["y_pred"] = (p_te >= thr).astype(int)
        out["test_predictions"] = pred.reset_index(drop=True)
    return out


def flatten(result: dict) -> dict:
    row = {"scenario": result["scenario"], "seed": result["seed"], "runtime_s": result["runtime_s"]}
    row.update(result["info"])
    for part in ("validation", "test"):
        row.update({f"{part}_{k}": v for k, v in result[part].items()})
    return row
