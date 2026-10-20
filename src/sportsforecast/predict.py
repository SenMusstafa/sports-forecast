"""Forecasts for upcoming fixtures.

The backtest showed the de-vigged market is the most accurate 1X2 estimate, so the report treats it
as the reference and uses the models for what the market file does not give directly: expected
goals, score distribution, totals/BTTS, and where (and how far) the models disagree with the market.
"""
from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest
from .backtest import BacktestConfig
from .data import DataSourceError, read_csv
from .dixon_coles import DixonColes, markets, score_matrix
from .elo import Elo, OrderedLogit
from .metrics import OUTCOME_INDEX, evaluate
from .odds import devig

log = logging.getLogger(__name__)

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURE_ODDS = [("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")]


def current_season(today: pd.Timestamp) -> str:
    """Football season code for a date: Aug 2026 -> '2627', Mar 2027 -> '2627'."""
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def fetch_fixtures(code: str, cache_dir: Path, max_age_hours: float = 3.0) -> pd.DataFrame:
    path = cache_dir / "fixtures.csv"
    stale = not path.exists() or time.time() - path.stat().st_mtime > max_age_hours * 3600
    if stale:
        req = urllib.request.Request(FIXTURES_URL, headers={"User-Agent": "sportsforecast/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if not path.exists():
                raise DataSourceError(f"fixtures: {exc}") from exc
            log.warning("fixtures refresh failed (%s), using cached copy", exc)

    raw = read_csv(path)
    if "Div" not in raw.columns:
        path.unlink()  # likely an error page instead of the CSV; force a fresh download next time
        raise DataSourceError(f"fixtures: unexpected format, first columns {list(raw.columns[:5])}")
    raw = raw[raw["Div"] == code]
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "time": raw.get("Time", pd.Series("", index=raw.index)).fillna(""),
        "home": raw["HomeTeam"].str.strip(),
        "away": raw["AwayTeam"].str.strip(),
    })
    for cols in FIXTURE_ODDS:
        if set(cols) <= set(raw.columns):
            df[["odds_h", "odds_d", "odds_a"]] = raw[list(cols)].to_numpy(dtype=float)
            df["odds_source"] = cols[0][:-1]
            break
    if {"MaxH", "MaxD", "MaxA"} <= set(raw.columns):
        df[["max_h", "max_d", "max_a"]] = raw[["MaxH", "MaxD", "MaxA"]].to_numpy(dtype=float)
    if {"Avg>2.5", "Avg<2.5"} <= set(raw.columns):
        df[["odds_over", "odds_under"]] = raw[["Avg>2.5", "Avg<2.5"]].to_numpy(dtype=float)
    return df.sort_values(["date", "time"]).reset_index(drop=True)


@dataclass
class ForecastResult:
    table: pd.DataFrame
    blend_weights: dict[str, float]
    lineup_beta: float | None
    validation: pd.DataFrame  # backtest scores of blend vs market, shown in the report for context
    as_of: pd.Timestamp


def _blend_weights(result: backtest.BacktestResult) -> dict[str, float]:
    history = result.pool_weights.get("blend")
    if history is None or history.empty:
        return {"elo": 0.5, "dc": 0.5}
    w = history.iloc[-1].drop("date").to_dict()
    # Without known lineups the lineup model equals its base (dcx), so its weight moves there
    if "dcl" in w:
        w["dcx" if "dcx" in w else "dc"] = w.get("dcx", w.get("dc", 0.0)) + w.pop("dcl")
    return w


def forecast(history: pd.DataFrame, fixtures: pd.DataFrame, cfg: BacktestConfig | None = None) -> ForecastResult:
    cfg = cfg or BacktestConfig()
    as_of = fixtures["date"].min()
    history = history[history["date"] < as_of].reset_index(drop=True)

    # 1) Walk-forward backtest on history: blend weights + honest accuracy context for the report
    result = backtest.run(history, cfg)
    weights = _blend_weights(result)
    beta = result.pool_weights.get("lineup_beta")
    validation = evaluate(result.preds, ["blend", "mkt"], common=True)

    # 2) Final models fitted on everything before the first fixture
    elo, diffs = Elo(cfg.elo), []
    for m in history.itertuples(index=False):
        elo.start_season(m.season)
        diffs.append(elo.diff(m.home, m.away))
        elo.update(m.home, m.away, m.hg, m.ag)
    logit = OrderedLogit().fit(np.array(diffs), history["result"].map(OUTCOME_INDEX).to_numpy())
    dc = DixonColes(cfg.dc).fit(history, as_of)
    has_xg = "hxg" in history and history["hxg"].notna().any()
    goal_model = DixonColes(replace(cfg.dc, xg_weight=cfg.xg_weight)).fit(history, as_of) if has_xg else dc
    elo.start_season(current_season(as_of))

    rows = []
    for f in fixtures.itertuples(index=False):
        probs = {"elo": logit.predict(np.array([elo.diff(f.home, f.away)]))[0]}
        p_dc = dc.predict(f.home, f.away)
        probs["dc"] = np.array([p_dc["H"], p_dc["D"], p_dc["A"]])
        lam, mu = goal_model.rates(f.home, f.away)
        mat = score_matrix(lam, mu, goal_model.rho, cfg.dc.max_goals)
        mk = markets(mat)
        if has_xg:
            probs["dcx"] = np.array([mk["H"], mk["D"], mk["A"]])
        total = sum(weights.get(k, 0.0) for k in probs)
        model = sum(weights.get(k, 0.0) * p for k, p in probs.items()) / total

        top = np.dstack(np.unravel_index(np.argsort(mat, axis=None)[::-1][:3], mat.shape))[0]
        rows.append({
            "date": f.date, "time": f.time, "home": f.home, "away": f.away,
            "model_H": model[0], "model_D": model[1], "model_A": model[2],
            "xg_home": lam, "xg_away": mu, "over25": mk["over25"], "btts": mk["btts"],
            "top_scores": ", ".join(f"{h}-{a} ({mat[h, a]:.0%})" for h, a in top),
            "new_team": ", ".join(t for t in (f.home, f.away) if t not in dc.teams),
        })
    table = pd.DataFrame(rows)
    _attach_fixture_market(table, fixtures, cfg.devig_method)
    return ForecastResult(table, weights, float(beta["beta"].iloc[-1]) if beta is not None and not beta.empty else None,
                          validation, as_of)


def _attach_fixture_market(table: pd.DataFrame, fixtures: pd.DataFrame, method: str) -> None:
    if "odds_h" in fixtures:
        odds = fixtures[["odds_h", "odds_d", "odds_a"]].to_numpy()
        table[["mkt_H", "mkt_D", "mkt_A"]] = devig(odds, method)
        table["odds_source"] = fixtures["odds_source"].to_numpy()
        for o in "HDA":
            table[f"diff_{o}"] = table[f"model_{o}"] - table[f"mkt_{o}"]
    if "max_h" in fixtures:
        table[["max_h", "max_d", "max_a"]] = fixtures[["max_h", "max_d", "max_a"]].to_numpy()
    if "odds_over" in fixtures:
        ou = np.column_stack([fixtures["odds_over"], fixtures["odds_under"]])
        implied = 1.0 / ou
        table["mkt_over25"] = implied[:, 0] / implied.sum(axis=1)
