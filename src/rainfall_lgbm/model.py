"""LightGBM training with early stopping on validation binary log loss."""
from __future__ import annotations

import lightgbm as lgb


def train_lgbm(X_tr, y_tr, X_va, y_va, params: dict, early_stopping_rounds: int, seed: int):
    model = lgb.LGBMClassifier(**params, random_state=seed)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model
