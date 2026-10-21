"""Expected goals (xG) from Understat, merged onto football-data matches."""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .data import is_fresh
from .understat import LEAGUES, UnderstatError, league_dates

log = logging.getLogger(__name__)


def season_dates(code: str, season: str, cache_dir: Path) -> list[dict]:
    """Raw Understat fixtures (played and upcoming) for season '2324' -> understat year 2023."""
    league = LEAGUES.get(code)
    if league is None:
        raise UnderstatError(f"no understat league for {code}")
    year = 2000 + int(season[:2])
    cache = cache_dir / f"understat_{code}_{season}.json"
    if is_fresh(cache, season):
        return json.loads(cache.read_text())
    try:
        dates = league_dates(league, year)
    except UnderstatError:
        if cache.exists():
            log.warning("understat %s %d refresh failed, using cached copy", league, year)
            return json.loads(cache.read_text())
        raise
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dates))
    log.info("downloaded understat %s %d (%d matches)", league, year, len(dates))
    return dates


def fetch_season(code: str, season: str, cache_dir: Path) -> pd.DataFrame:
    """Finished matches with xG."""
    rows = [{
        "us_id": str(d["id"]),
        "date": pd.Timestamp(d["datetime"]).normalize(),
        "home": d["h"]["title"], "away": d["a"]["title"],
        "hg": int(d["goals"]["h"]), "ag": int(d["goals"]["a"]),
        "hxg": float(d["xG"]["h"]), "axg": float(d["xG"]["a"]),
    } for d in season_dates(code, season, cache_dir)
        if d.get("isResult") and d.get("xG", {}).get("h") is not None]
    return pd.DataFrame(rows)


def upcoming(code: str, season: str, cache_dir: Path) -> pd.DataFrame:
    """Scheduled, not yet played matches (Understat team names)."""
    rows = [{
        "date": pd.Timestamp(d["datetime"]).normalize(),
        "time": pd.Timestamp(d["datetime"]).strftime("%H:%M"),
        "home": d["h"]["title"], "away": d["a"]["title"],
    } for d in season_dates(code, season, cache_dir) if not d.get("isResult")]
    return pd.DataFrame(rows, columns=["date", "time", "home", "away"])


def build_team_mapping(left: pd.DataFrame, right: pd.DataFrame, tolerance_days: int = 1) -> dict[str, str]:
    """Map right-side team names onto left-side names by co-occurrence on the same fixtures.

    Name-agnostic: works for any pair of sources as long as both list the same matches.
    """
    # Candidate fixtures share date (± tolerance) and final score; each vote is split among candidates
    by_key: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for m in left.itertuples(index=False):
        for offset in range(-tolerance_days, tolerance_days + 1):
            by_key[(m.date + pd.Timedelta(days=offset), m.hg, m.ag)].append((m.home, m.away))

    votes: dict[str, Counter] = defaultdict(Counter)
    for r in right.itertuples(index=False):
        candidates = by_key.get((r.date, r.hg, r.ag), [])
        for lh, la in candidates:
            votes[r.home][lh] += 1 / len(candidates)
            votes[r.away][la] += 1 / len(candidates)

    left_names = set(left["home"]) | set(left["away"])
    mapping = {}
    for right_name, counter in votes.items():
        # Identical names need no evidence; matters for promoted teams with only a few matches so far
        if right_name in left_names:
            mapping[right_name] = right_name
            continue
        (best, n), total = counter.most_common(1)[0], sum(counter.values())
        if n >= 3 and n / total >= 0.5:
            mapping[right_name] = best
        else:
            log.warning("no confident mapping for %r (best %r %.1f/%.1f)", right_name, best, n, total)
    return mapping


def attach_xg(matches: pd.DataFrame, code: str, cache_dir: Path) -> pd.DataFrame:
    """Add hxg/axg columns; seasons that cannot be fetched keep NaN so the pipeline degrades gracefully."""
    if code not in LEAGUES:  # Understat covers only the big five leagues
        return matches.assign(hxg=float("nan"), axg=float("nan"),
                              us_id=pd.Series(pd.NA, index=matches.index, dtype="string"))
    frames = []
    for season in matches["season"].unique():
        try:
            frames.append(fetch_season(code, season, cache_dir))
        except UnderstatError as exc:
            log.warning("%s", exc)
    out = matches.copy()
    out[["hxg", "axg"]] = float("nan")
    out["us_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    if not frames:
        return out

    xg = pd.concat(frames, ignore_index=True)
    mapping = build_team_mapping(matches, xg)
    xg["home"] = xg["home"].map(mapping)
    xg["away"] = xg["away"].map(mapping)
    xg = xg.dropna(subset=["home", "away"])

    merged = out.reset_index().merge(xg, on=["home", "away"], how="inner", suffixes=("", "_xg"))
    merged = merged[(merged["date"] - merged["date_xg"]).abs() <= pd.Timedelta(days=3)]
    out.loc[merged["index"], "hxg"] = merged["hxg_xg"].to_numpy()
    out.loc[merged["index"], "axg"] = merged["axg_xg"].to_numpy()
    out.loc[merged["index"], "us_id"] = merged["us_id_xg"].to_numpy()
    log.info("xG attached to %d/%d matches", out["hxg"].notna().sum(), len(out))
    return out
