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
from . import xg
from .data import DataSourceError, current_season, read_csv
from .dixon_coles import DixonColes, markets, score_matrix
from .elo import Elo, OrderedLogit
from .metrics import OUTCOME_INDEX, TOTALS, evaluate
from .odds import devig

log = logging.getLogger(__name__)

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURE_ODDS = [("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")]


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


def understat_fixtures(history: pd.DataFrame, code: str, cache_dir: Path, season: str) -> pd.DataFrame:
    """Fallback schedule without odds, team names mapped onto football-data names via played matches."""
    played = pd.concat([xg.fetch_season(code, s, cache_dir) for s in history["season"].unique()],
                       ignore_index=True)
    mapping = xg.build_team_mapping(history, played)
    fixtures = xg.upcoming(code, season, cache_dir)
    # Teams with no played match in history yet (e.g. promoted, before their first game) map by name only
    known = set(history["home"]) | set(history["away"])
    mapping.update({t: t for t in set(fixtures["home"]) | set(fixtures["away"]) if t in known and t not in mapping})
    fixtures["home"] = fixtures["home"].map(mapping)
    fixtures["away"] = fixtures["away"].map(mapping)
    unmapped = fixtures["home"].isna() | fixtures["away"].isna()
    if unmapped.any():
        log.warning("%d understat fixtures dropped: team names could not be mapped", unmapped.sum())
    return fixtures[~unmapped].sort_values(["date", "time"]).reset_index(drop=True)


def fixture_leagues(cache_dir: Path) -> list[str]:
    path = cache_dir / "fixtures.csv"
    return sorted(read_csv(path)["Div"].dropna().unique()) if path.exists() else []


@dataclass
class ForecastResult:
    table: pd.DataFrame
    blend_weights: dict[str, float]
    lineup_beta: float | None
    validation: pd.DataFrame  # backtest scores of blend vs market, shown in the report for context
    as_of: pd.Timestamp
    scores: pd.DataFrame | None = None  # past-window mode: forecasts scored against actual results


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


def past_fixtures(history: pd.DataFrame, as_of: pd.Timestamp, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Already played matches in [as_of, as_of + days) presented as fixtures, priced at the opening line.

    Returns (fixtures, played rows). forecast() only sees history before the first fixture, so the
    results and closing prices of the window cannot leak into the forecasts.
    """
    played = history[(history["date"] >= as_of) & (history["date"] < as_of + pd.Timedelta(days=days))]
    played = played.reset_index(drop=True)
    fixtures = played[["date", "home", "away"]].assign(time="")
    opening = "open_h" in played and played["open_h"].notna().all()
    src = ("open_h", "open_d", "open_a") if opening else ("odds_h", "odds_d", "odds_a")
    if set(src) <= set(played.columns):
        fixtures[["odds_h", "odds_d", "odds_a"]] = played[list(src)].to_numpy()
        fixtures["odds_source"] = "opening" if opening else "closing (no opening prices)"
    ou_open = "ou_open_over" in played and played["ou_open_over"].notna().all()
    ou_src = ("ou_open_over", "ou_open_under") if ou_open else ("ou_over", "ou_under")
    if set(ou_src) <= set(played.columns):
        fixtures[["odds_over", "odds_under"]] = played[list(ou_src)].to_numpy()
    return fixtures, played


def score_past(result: ForecastResult, played: pd.DataFrame, method: str = "shin") -> ForecastResult:
    """Attach actual results and score model, opening market and closing market on the window."""
    t = result.table.copy()
    t[["hg", "ag", "result"]] = played[["hg", "ag", "result"]].to_numpy()
    t["total25"] = np.where(t["hg"] + t["ag"] > 2.5, "O", "U")
    t["model25_O"], t["model25_U"] = t["over25"], 1 - t["over25"]
    if "mkt_over25" in t:
        t["open25_O"], t["open25_U"] = t["mkt_over25"], 1 - t["mkt_over25"]
    if {"odds_h", "open_h"} <= set(played.columns):
        t[["close_H", "close_D", "close_A"]] = devig(played[["odds_h", "odds_d", "odds_a"]].to_numpy(), method)
    if {"ou_over", "ou_open_over"} <= set(played.columns):
        t[["close25_O", "close25_U"]] = devig(played[["ou_over", "ou_under"]].to_numpy(), method)

    one_x_two = evaluate(t, ["model", "mkt", "close"], common=True).rename(
        index={"model": "model", "mkt": "market (opening)", "close": "market (closing)"})
    totals = evaluate(t, ["model25", "open25", "close25"], common=True, market=TOTALS).rename(
        index={"model25": "model", "open25": "market (opening)", "close25": "market (closing)"})
    scores = pd.concat({"1X2": one_x_two.drop(columns="accuracy"),
                        "Over/Under 2.5": totals.drop(columns=["rps", "accuracy"])})
    result.table, result.scores = t, scores
    return result
