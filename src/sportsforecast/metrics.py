"""Probabilistic scoring for categorical match markets (1X2, over/under). Lower is better except accuracy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


@dataclass(frozen=True)
class Market:
    """Column conventions for one betting market.

    Model forecasts live in f"{model}_{outcome}"; de-vigged market forecasts use the same
    convention with the `open_prob` / `close_prob` model names.
    """
    name: str
    outcomes: str
    result_col: str
    open_odds: tuple[str, ...]
    close_odds: tuple[str, ...]
    open_prob: str
    close_prob: str

    def cols(self, model: str) -> list[str]:
        return [f"{model}_{o}" for o in self.outcomes]

    def index(self, results: pd.Series) -> np.ndarray:
        return results.map({o: i for i, o in enumerate(self.outcomes)}).to_numpy()


ONE_X_TWO = Market("1X2", "HDA", "result", ("open_h", "open_d", "open_a"), ("odds_h", "odds_d", "odds_a"),
                   open_prob="open", close_prob="mkt")
TOTALS = Market("Over/Under 2.5", "OU", "total25", ("ou_open_over", "ou_open_under"), ("ou_over", "ou_under"),
                open_prob="open25", close_prob="mkt25")


def one_hot(results: pd.Series, outcomes: str = "HDA") -> np.ndarray:
    y = np.zeros((len(results), len(outcomes)))
    y[np.arange(len(results)), results.map({o: i for i, o in enumerate(outcomes)}).to_numpy()] = 1.0
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


def evaluate(preds: pd.DataFrame, models: list[str], common: bool = False,
             market: Market = ONE_X_TWO) -> pd.DataFrame:
    """common=True scores every model on the same rows (where all of them have a forecast)."""
    models = [m for m in models if set(market.cols(m)) <= set(preds.columns)]
    if common:
        preds = preds.dropna(subset=[c for m in models for c in market.cols(m)])
    rows = []
    for m in models:
        cols = market.cols(m)
        sub = preds.dropna(subset=cols + [market.result_col])
        if sub.empty:
            continue
        p, y = sub[cols].to_numpy(), one_hot(sub[market.result_col], market.outcomes)
        rows.append({
            "model": m, "n": len(sub), "log_loss": log_loss(p, y), "brier": brier(p, y),
            "rps": rps(p, y), "accuracy": accuracy(p, y),
        })
    return pd.DataFrame(rows, columns=["model", "n", "log_loss", "brier", "rps", "accuracy"]).set_index("model")


def calibration(preds: pd.DataFrame, model: str, bins: int = 10, market: Market = ONE_X_TWO,
                outcome: str | None = None) -> pd.DataFrame:
    """Predicted vs observed frequency, pooled over the market's outcomes or for one outcome only."""
    preds = preds.dropna(subset=market.cols(model) + [market.result_col])
    y = one_hot(preds[market.result_col], market.outcomes)
    if outcome is None:
        p, y = preds[market.cols(model)].to_numpy().ravel(), y.ravel()
    else:
        p, y = preds[f"{model}_{outcome}"].to_numpy(), y[:, market.outcomes.index(outcome)]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y}).groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return df
