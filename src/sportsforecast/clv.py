"""Betting-side evaluation against the closing line. Research only, not betting advice.

A bet placed at opening odds o_open on outcome X has closing-line value
CLV = o_open · q_close − 1, where q_close is the de-vigged closing probability of X.
Consistently positive CLV is the most reliable evidence of an edge; realised ROI is far noisier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import ONE_X_TWO, Market


def bets(preds: pd.DataFrame, model: str, min_edge: float = 0.03, market: Market = ONE_X_TWO) -> pd.DataFrame:
    """Flat 1-unit bets at opening odds where the model sees edge >= min_edge."""
    required = {market.open_odds[0], *market.cols(market.close_prob), *market.cols(model)}
    if not required <= set(preds.columns):
        return pd.DataFrame()
    rows = []
    for o, open_col, close_col in zip(market.outcomes, market.open_odds, market.close_odds):
        price = preds[open_col]
        close_p = preds[f"{market.close_prob}_{o}"]
        edge = preds[f"{model}_{o}"] * price - 1
        sel = preds[(edge >= min_edge) & price.notna() & close_p.notna()]
        rows.append(pd.DataFrame({
            "date": sel["date"], "home": sel["home"], "away": sel["away"], "pick": o,
            "open_odds": sel[open_col], "close_odds": sel[close_col],
            "edge": edge.loc[sel.index], "clv": sel[open_col] * close_p.loc[sel.index] - 1,
            "profit": np.where(sel[market.result_col] == o, sel[open_col] - 1, -1.0),
        }))
    return pd.concat(rows).sort_values("date").reset_index(drop=True)


def summary(preds: pd.DataFrame, models: list[str], min_edge: float = 0.03,
            market: Market = ONE_X_TWO) -> pd.DataFrame:
    rows = []
    for m in models:
        b = bets(preds, m, min_edge, market)
        if b.empty:
            continue
        rows.append({
            "model": m, "bets": len(b), "mean_edge": b["edge"].mean(),
            "mean_clv": b["clv"].mean(), "clv_positive": (b["clv"] > 0).mean(),
            "roi": b["profit"].mean(),
            # Standard error of ROI under flat stakes: shows how little a season of results proves
            "roi_se": b["profit"].std(ddof=1) / np.sqrt(len(b)),
        })
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
