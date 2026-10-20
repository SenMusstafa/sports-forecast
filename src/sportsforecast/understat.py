"""Minimal Understat client.

Understat serves its data through AJAX endpoints (getLeagueData, getMatchData); older versions
embedded the same JSON in the HTML page, which is kept as a fallback for league data.
"""
from __future__ import annotations

import codecs
import gzip
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://understat.com/"
LEAGUES = {"E0": "EPL", "SP1": "La_Liga", "I1": "Serie_A", "D1": "Bundesliga", "F1": "Ligue_1"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Encoding": "gzip",
    "X-Requested-With": "XMLHttpRequest",
}


class UnderstatError(RuntimeError):
    pass


def _get(path: str, referer: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(BASE_URL + path, headers={**HEADERS, "Referer": BASE_URL + referer})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            return body.decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise UnderstatError(f"{path}: HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UnderstatError(f"{path}: {exc}") from exc


def league_dates(league: str, year: int) -> list[dict]:
    """All fixtures of a league season (year = season start, e.g. 2024 for 2024/25)."""
    page = f"league/{league}/{year}"
    errors = []
    try:
        return json.loads(_get(f"getLeagueData/{league}/{year}", referer=page))["dates"]
    except (UnderstatError, ValueError, KeyError) as exc:
        errors.append(f"api: {exc}")
    try:
        html = _get(page, referer="")
        m = re.search(r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", html)
        if m:
            return json.loads(codecs.decode(m.group(1), "unicode_escape"))
        errors.append("page: datesData not embedded")
    except (UnderstatError, ValueError) as exc:
        errors.append(f"page: {exc}")
    raise UnderstatError(f"{league} {year}: " + "; ".join(errors))


def match_data(match_id: str) -> dict:
    """{'rosters': {'h': {...}, 'a': {...}}, 'shots': {...}, ...} for one finished match."""
    try:
        data = json.loads(_get(f"getMatchData/{match_id}", referer=f"match/{match_id}"))
    except ValueError as exc:
        raise UnderstatError(f"match {match_id}: invalid JSON") from exc
    if "rosters" not in data:
        raise UnderstatError(f"match {match_id}: no rosters in response (keys: {list(data)[:5]})")
    return data


def cached_match_rosters(match_id: str, cache_dir: Path, delay: float = 1.0) -> dict | None:
    """Rosters from cache, downloading (politely) when missing. None if the download fails."""
    path = cache_dir / "understat_match" / f"{match_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        rosters = match_data(match_id)["rosters"]
    except UnderstatError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rosters))
    time.sleep(delay)
    return rosters
