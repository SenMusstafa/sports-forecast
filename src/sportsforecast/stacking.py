"""Linear opinion pool: p = Σ_k w_k · p_k with w on the simplex, fitted walk-forward by log loss.

The learned weights are the research output: a model weight near zero next to the market
means the model adds no information the market does not already price.
A logarithmic pool with outcome biases was tried first and scored worse out of sample on EPL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from .metrics import ONE_X_TWO, Market


class LinearPool:
    def __init__(self, components: list[str]) -> None:
        self.components = components
        self.weights = np.full(len(components), 1.0 / len(components))

    def fit(self, probs: np.ndarray, y: np.ndarray) -> LinearPool:
        """probs: (n, K, 3) component probabilities, y: (n,) outcome indices."""
        picked = probs[np.arange(len(y)), :, y]  # (n, K) probability each component gave the outcome

        def nll(logits: np.ndarray) -> float:
            return -np.log(np.clip(picked @ softmax(logits), 1e-12, None)).sum()

        res = minimize(nll, np.log(self.weights + 1e-9), method="L-BFGS-B")
        self.weights = softmax(res.x)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return np.einsum("nko,k->no", probs, self.weights)


def walk_forward(preds: pd.DataFrame, components: list[str], name: str, refit_days: int = 7,
                 min_train: int = 380, market: Market = ONE_X_TWO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (probabilities aligned to preds.index with columns market.cols(name), weight history)."""
    out = pd.DataFrame(np.nan, index=preds.index, columns=market.cols(name))
    usable = preds.dropna(subset=[c for comp in components for c in market.cols(comp)] + [market.result_col])
    if len(usable) <= min_train:
        return out, pd.DataFrame()

    probs = np.stack([usable[market.cols(c)].to_numpy() for c in components], axis=1)
    y = market.index(usable[market.result_col])
    dates = usable["date"].to_numpy()
    pool = LinearPool(components)
    history = []
    next_refit = None

    for d in np.unique(dates):
        train = dates < d
        if train.sum() < min_train:
            continue
        if next_refit is None or d >= next_refit:
            pool.fit(probs[train], y[train])
            next_refit = d + np.timedelta64(refit_days, "D")
            history.append({"date": pd.Timestamp(d), **dict(zip(components, pool.weights))})
        today = dates == d
        out.loc[usable.index[today]] = pool.predict(probs[today])

    return out, pd.DataFrame(history)
