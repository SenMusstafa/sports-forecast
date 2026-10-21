"""Evaluate pre-registered betting rules on walk-forward backtest predictions.

Seasons are split into a discovery period and a later validation period. A rule only counts as
promising when its closing-line value is positive in validation, across leagues, at a strength
that survives the number of rules tested; realised ROI is reported but is far too noisy to decide on.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import ONE_X_TWO, TOTALS, Market


@dataclass(frozen=True)
class Rule:
    name: str
    hypothesis: str
    when: str
    bet: str

    @property
    def market(self) -> Market:
        return TOTALS if self.bet in TOTALS.outcomes else ONE_X_TWO


def load_rules(path: Path) -> list[Rule]:
    return [Rule(**r) for r in tomllib.loads(path.read_text())["rule"]]


def features(preds: pd.DataFrame) -> pd.DataFrame:
    """Pre-match columns the rules may reference (all known before kick-off)."""
    goal = "dcx" if "dcx_lam" in preds else "dc"
    f = preds.copy()
    f["exp_home"], f["exp_away"] = f[f"{goal}_lam"], f[f"{goal}_mu"]
    f["exp_total"] = f["exp_home"] + f["exp_away"]
    f["exp_gap"] = (f["exp_home"] - f["exp_away"]).abs()
    model = "blend" if "blend_H" in f else "ens"
    for o in "HDA":
        f[f"model_{o}"] = f[f"{model}_{o}"]
    f["model25_O"], f["model25_U"] = f[f"{goal}25_O"], f[f"{goal}25_U"]
    return f


def bets(f: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    market = rule.market
    i = market.outcomes.index(rule.bet)
    open_col, close_col = market.open_odds[i], market.close_odds[i]
    close_p = f"{market.close_prob}_{rule.bet}"
    needed = [open_col, close_p, market.result_col]
    if not set(needed) <= set(f.columns):
        return pd.DataFrame()
    sel = f.dropna(subset=needed).query(rule.when)
    return pd.DataFrame({
        "date": sel["date"], "season": sel["season"], "home": sel["home"], "away": sel["away"],
        "odds": sel[open_col], "close_odds": sel[close_col],
        "implied": sel[f"{market.open_prob}_{rule.bet}"],
        "won": (sel[market.result_col] == rule.bet).astype(float),
        "clv": sel[open_col] * sel[close_p] - 1,
    }).assign(profit=lambda b: np.where(b["won"] == 1, b["odds"] - 1, -1.0))


def summarise(b: pd.DataFrame) -> dict:
    n = len(b)
    if n < 2:
        return {"bets": n}
    clv_t = stats.ttest_1samp(b["clv"], 0.0)
    return {
        "bets": n,
        "hit": b["won"].mean(),
        "implied": b["implied"].mean(),  # de-vigged opening probability: what the market expected
        "roi": b["profit"].mean(),
        "roi_se": b["profit"].std(ddof=1) / np.sqrt(n),
        "clv": b["clv"].mean(),
        "clv_t": clv_t.statistic,
        "clv_p": clv_t.pvalue,
    }


def evaluate(per_league: dict[str, pd.DataFrame], rules: list[Rule],
             validation_seasons: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pooled table by rule x period, per-league validation table)."""
    pooled, by_league = [], []
    for rule in rules:
        periods: dict[str, list[pd.DataFrame]] = {"discovery": [], "validation": []}
        for league, preds in per_league.items():
            b = bets(features(preds), rule)
            if b.empty:
                continue
            seasons = sorted(preds["season"].unique())
            valid = set(seasons[-validation_seasons:])
            periods["validation"].append(b[b["season"].isin(valid)])
            periods["discovery"].append(b[~b["season"].isin(valid)])
            by_league.append({"rule": rule.name, "league": league,
                              **summarise(b[b["season"].isin(valid)])})
        for period, frames in periods.items():
            if frames:
                pooled.append({"rule": rule.name, "bet": rule.bet, "period": period,
                               **summarise(pd.concat(frames))})
    return (pd.DataFrame(pooled).set_index(["rule", "period"]),
            pd.DataFrame(by_league).set_index(["rule", "league"]))
