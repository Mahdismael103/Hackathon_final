"""Scoring helpers. All scores are returned as *higher is better*."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

CLS_METRICS = {"f1_macro", "f1", "accuracy", "balanced_accuracy", "roc_auc", "log_loss", "f1_weighted"}
REG_METRICS = {"rmse", "mse", "mae", "r2", "rmsle"}

_ALIASES = {
    "f1macro": "f1_macro", "macro_f1": "f1_macro", "f1-macro": "f1_macro", "f1": "f1_macro",
    "acc": "accuracy", "balanced_acc": "balanced_accuracy", "bacc": "balanced_accuracy",
    "auc": "roc_auc", "roc": "roc_auc", "logloss": "log_loss", "cross_entropy": "log_loss",
    "weighted_f1": "f1_weighted",
    "root_mean_squared_error": "rmse", "mean_squared_error": "mse", "mean_absolute_error": "mae",
    "r2_score": "r2", "r^2": "r2", "rmsle": "rmsle", "root_mean_squared_log_error": "rmsle",
}


def normalise_metric(metric: str | None, task: str, default: str) -> str:
    if not metric:
        return default
    m = str(metric).strip().lower().replace(" ", "_")
    m = _ALIASES.get(m, m)
    valid = CLS_METRICS if task == "classification" else REG_METRICS
    return m if m in valid else default


def classification_score(metric: str, y: np.ndarray, proba: np.ndarray, pred: np.ndarray | None = None) -> float:
    if pred is None:
        pred = proba.argmax(1)
    if metric == "accuracy":
        return skm.accuracy_score(y, pred)
    if metric == "balanced_accuracy":
        return skm.balanced_accuracy_score(y, pred)
    if metric == "f1_weighted":
        return skm.f1_score(y, pred, average="weighted")
    if metric == "roc_auc":
        try:
            if proba.shape[1] == 2:
                return skm.roc_auc_score(y, proba[:, 1])
            return skm.roc_auc_score(y, proba, multi_class="ovr", labels=np.arange(proba.shape[1]))
        except ValueError:
            return skm.accuracy_score(y, pred)
    if metric == "log_loss":
        p = np.clip(proba, 1e-7, 1)
        p = p / p.sum(1, keepdims=True)
        return -skm.log_loss(y, p, labels=np.arange(proba.shape[1]))
    return skm.f1_score(y, pred, average="macro")


def regression_score(metric: str, y: np.ndarray, pred: np.ndarray) -> float:
    if metric == "mae":
        return -skm.mean_absolute_error(y, pred)
    if metric == "r2":
        return skm.r2_score(y, pred)
    if metric == "mse":
        return -skm.mean_squared_error(y, pred)
    if metric == "rmsle":
        return -float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2)))
    return -float(np.sqrt(skm.mean_squared_error(y, pred)))
