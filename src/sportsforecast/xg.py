"""Expected goals (xG) from Understat, merged onto football-data matches."""
from __future__ import annotations

import codecs
import json
import logging
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

UNDERSTAT_LEAGUES = {"E0": "EPL", "SP1": "La_liga", "I1": "Serie_A", "D1": "Bundesliga", "F1": "Ligue_1"}
PAGE_URL = "https://understat.com/league/{league}/{year}"
API_URL = "https://understat.com/getLeagueData/{league}/{year}"


class XgSourceError(RuntimeError):
    pass


def _get(url: str, ajax: bool = False, timeout: float = 30.0) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (sportsforecast research)"}
    if ajax:
        headers["X-Requested-With"] = "XMLHttpRequest"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _parse_dates(league: str, year: int) -> list[dict]:
    """Understat used to embed `datesData` in the page; newer versions serve it from a JSON endpoint."""
    errors = []
    try:
        html = _get(PAGE_URL.format(league=league, year=year))
        m = re.search(r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", html)
        if m:
            return json.loads(codecs.decode(m.group(1), "unicode_escape"))
        errors.append("page: datesData not embedded")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        errors.append(f"page: {exc}")
    try:
        payload = json.loads(_get(API_URL.format(league=league, year=year), ajax=True))
        return payload["dates"]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
        errors.append(f"api: {exc}")
    raise XgSourceError(f"understat {league} {year}: " + "; ".join(errors))


def fetch_season(code: str, season: str, cache_dir: Path) -> pd.DataFrame:
    """season '2324' -> understat year 2023. Cached as JSON; only finished matches are returned."""
    league = UNDERSTAT_LEAGUES.get(code)
    if league is None:
        raise XgSourceError(f"no understat league for {code}")
    year = 2000 + int(season[:2])
    cache = cache_dir / f"understat_{code}_{season}.json"
    if cache.exists():
        dates = json.loads(cache.read_text())
    else:
        dates = _parse_dates(league, year)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(dates))
        log.info("downloaded understat %s %d (%d matches)", league, year, len(dates))

    rows = [{
        "date": pd.Timestamp(d["datetime"]).normalize(),
        "home": d["h"]["title"], "away": d["a"]["title"],
        "hg": int(d["goals"]["h"]), "ag": int(d["goals"]["a"]),
        "hxg": float(d["xG"]["h"]), "axg": float(d["xG"]["a"]),
    } for d in dates if d.get("isResult") and d.get("xG", {}).get("h") is not None]
    return pd.DataFrame(rows)


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

    mapping = {}
    for right_name, counter in votes.items():
        (best, n), total = counter.most_common(1)[0], sum(counter.values())
        if n >= 3 and n / total >= 0.5:
            mapping[right_name] = best
        else:
            log.warning("no confident mapping for %r (best %r %.1f/%.1f)", right_name, best, n, total)
    return mapping


def attach_xg(matches: pd.DataFrame, code: str, cache_dir: Path) -> pd.DataFrame:
    """Add hxg/axg columns; seasons that cannot be fetched keep NaN so the pipeline degrades gracefully."""
    frames = []
    for season in matches["season"].unique():
        try:
            frames.append(fetch_season(code, season, cache_dir))
        except XgSourceError as exc:
            log.warning("%s", exc)
    out = matches.copy()
    out[["hxg", "axg"]] = float("nan")
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
    log.info("xG attached to %d/%d matches", out["hxg"].notna().sum(), len(out))
    return out
