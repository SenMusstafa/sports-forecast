"""Probabilistic scoring for 3-way (H/D/A) forecasts. Lower is better except accuracy."""
from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def one_hot(results: pd.Series) -> np.ndarray:
    y = np.zeros((len(results), 3))
    y[np.arange(len(results)), results.map(OUTCOME_INDEX).to_numpy()] = 1.0
    return y


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip((p * y).sum(axis=1), 1e-12, None))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(((p - y) ** 2).sum(axis=1)))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score: respects H > D > A ordering, standard in football forecasting."""
    cp, cy = np.cumsum(p, axis=1)[:, :2], np.cumsum(y, axis=1)[:, :2]
    return float(np.mean(((cp - cy) ** 2).sum(axis=1) / 2))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(p.argmax(axis=1) == y.argmax(axis=1)))


def evaluate(preds: pd.DataFrame, models: list[str], common: bool = False) -> pd.DataFrame:
    """common=True scores every model on the same rows (where all of them have a forecast)."""
    models = [m for m in models if f"{m}_H" in preds]
    if common:
        preds = preds.dropna(subset=[f"{m}_{o}" for m in models for o in "HDA"])
    rows = []
    for m in models:
        cols = [f"{m}_H", f"{m}_D", f"{m}_A"]
        sub = preds.dropna(subset=cols)
        if sub.empty:
            continue
        p, y = sub[cols].to_numpy(), one_hot(sub["result"])
        rows.append({
            "model": m, "n": len(sub), "log_loss": log_loss(p, y), "brier": brier(p, y),
            "rps": rps(p, y), "accuracy": accuracy(p, y),
        })
    return pd.DataFrame(rows, columns=["model", "n", "log_loss", "brier", "rps", "accuracy"]).set_index("model")


def calibration(preds: pd.DataFrame, model: str, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency, pooled over the three outcomes."""
    p = preds[[f"{model}_H", f"{model}_D", f"{model}_A"]].to_numpy().ravel()
    y = one_hot(preds["result"]).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y}).groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return df
