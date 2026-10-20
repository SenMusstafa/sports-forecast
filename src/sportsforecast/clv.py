"""Betting-side evaluation against the closing line. Research only, not betting advice.

A bet placed at opening odds o_open on outcome X has closing-line value
CLV = o_open · q_close − 1, where q_close is the de-vigged closing probability of X.
Consistently positive CLV is the most reliable evidence of an edge; realised ROI is far noisier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bets(preds: pd.DataFrame, model: str, min_edge: float = 0.03) -> pd.DataFrame:
    """Flat 1-unit bets at opening odds where the model sees edge >= min_edge."""
    required = {"open_h", "mkt_H", f"{model}_H"}
    if not required <= set(preds.columns):
        return pd.DataFrame()
    rows = []
    for o, side in zip("HDA", "hda"):
        price = preds[f"open_{side}"]
        edge = preds[f"{model}_{o}"] * price - 1
        sel = preds[(edge >= min_edge) & price.notna() & preds[f"mkt_{o}"].notna()]
        rows.append(pd.DataFrame({
            "date": sel["date"], "home": sel["home"], "away": sel["away"], "pick": o,
            "open_odds": sel[f"open_{side}"], "close_odds": sel[f"odds_{side}"],
            "edge": edge.loc[sel.index], "clv": sel[f"open_{side}"] * sel[f"mkt_{o}"] - 1,
            "profit": np.where(sel["result"] == o, sel[f"open_{side}"] - 1, -1.0),
        }))
    return pd.concat(rows).sort_values("date").reset_index(drop=True)


def summary(preds: pd.DataFrame, models: list[str], min_edge: float = 0.03) -> pd.DataFrame:
    rows = []
    for m in models:
        b = bets(preds, m, min_edge)
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
