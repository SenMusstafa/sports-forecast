"""Lineup strength from Understat rosters.

Each outfield player gets a time-decayed (xG + xA) per 90 rating, shrunk toward the league average
when he has few minutes. A side's lineup strength is the sum over its outfield starters; the
lineup delta is that sum minus the team's recent typical lineup strength, so it measures who is
missing (or rotated in) rather than how good the team is - team quality stays in Dixon-Coles.
Only attacking contribution is modelled; defensive impact of individual players is not.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .understat import cached_match_rosters

log = logging.getLogger(__name__)


@dataclass
class LineupConfig:
    player_half_life_days: float = 365.0
    team_half_life_days: float = 120.0
    prior_90s: float = 5.0  # pseudo-matches at league average added to every player's record


def fetch_rosters(matches: pd.DataFrame, cache_dir: Path, delay: float = 1.0, limit: int | None = None) -> int:
    """Download missing rosters for matches with an Understat id. Resumable; returns new downloads."""
    ids = matches["us_id"].dropna().astype(str).unique()
    cached = {p.stem for p in (cache_dir / "understat_match").glob("*.json")}
    missing = [i for i in ids if i not in cached][:limit]
    log.info("rosters: %d cached, %d to download (~%.0f min)", len(cached), len(missing), len(missing) * delay / 60)
    done = failed = 0
    for n, match_id in enumerate(missing, 1):
        if cached_match_rosters(match_id, cache_dir, delay) is None:
            failed += 1
        else:
            done += 1
        if n % 50 == 0:
            log.info("rosters: %d/%d (failed %d)", n, len(missing), failed)
    if failed:
        log.warning("rosters: %d downloads failed; rerun to retry", failed)
    return done


def player_matches(matches: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """One row per player appearance, from cached rosters only (no network)."""
    rows = []
    folder = cache_dir / "understat_match"
    for m in matches.dropna(subset=["us_id"]).itertuples():
        path = folder / f"{m.us_id}.json"
        if not path.exists():
            continue
        rosters = json.loads(path.read_text())
        for side, team in (("h", m.home), ("a", m.away)):
            for p in rosters.get(side, {}).values():
                rows.append({
                    "match_index": m.Index, "date": m.date, "side": side, "team": team,
                    "player_id": str(p["player_id"]), "player": p["player"], "position": p["position"],
                    "starter": p["position"] != "Sub", "minutes": float(p["time"]),
                    "contrib": float(p["xG"]) + float(p["xA"]),
                })
    return pd.DataFrame(rows)


class _Decayed:
    """Exponentially decayed running sums keyed by an id."""

    def __init__(self, half_life_days: float) -> None:
        self.rate = np.log(2) / half_life_days
        self.sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2))
        self.when: dict[str, pd.Timestamp] = {}

    def get(self, key: str, date: pd.Timestamp) -> np.ndarray:
        if key in self.when:
            self.sums[key] *= np.exp(-self.rate * (date - self.when[key]).days)
        self.when[key] = date
        return self.sums[key]

    def add(self, key: str, date: pd.Timestamp, values: tuple[float, float]) -> None:
        self.get(key, date)
        self.sums[key] += values


def lineup_strength(matches: pd.DataFrame, appearances: pd.DataFrame,
                    cfg: LineupConfig | None = None) -> pd.DataFrame:
    """Walk-forward: each match uses only appearances from earlier dates.

    Returns columns home_lineup, away_lineup, home_lineup_delta, away_lineup_delta aligned to matches.
    """
    cfg = cfg or LineupConfig()
    out = pd.DataFrame(np.nan, index=matches.index,
                       columns=["home_lineup", "away_lineup", "home_lineup_delta", "away_lineup_delta"])
    if appearances.empty:
        return out

    players = _Decayed(cfg.player_half_life_days)   # sums: contribution, 90s played
    league = _Decayed(cfg.player_half_life_days)
    teams = _Decayed(cfg.team_half_life_days)        # sums: lineup strength, weight

    outfield = appearances[appearances["position"] != "GK"]
    for date, day in outfield.groupby("date", sort=True):
        contrib, nineties = league.get("all", date)
        league_avg = contrib / nineties if nineties > 0 else 0.25

        for (match_index, side), lineup in day[day["starter"]].groupby(["match_index", "side"]):
            ratings = []
            for pid in lineup["player_id"]:
                c, n = players.get(pid, date)
                ratings.append((c + cfg.prior_90s * league_avg) / (n + cfg.prior_90s))
            strength = float(np.sum(ratings))
            team = lineup["team"].iloc[0]
            total, weight = teams.get(team, date)
            typical = total / weight if weight > 0 else strength
            prefix = "home" if side == "h" else "away"
            out.loc[match_index, [f"{prefix}_lineup", f"{prefix}_lineup_delta"]] = [strength, strength - typical]
            teams.add(team, date, (strength, 1.0))

        for p in day.itertuples():
            if p.minutes > 0:
                players.add(p.player_id, date, (p.contrib, p.minutes / 90))
                league.add("all", date, (p.contrib, p.minutes / 90))
    return out
