"""Match data loading in football-data.co.uk format (official site or GitHub mirror)."""
from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# football-data.co.uk league code -> datasets/football-datasets mirror slug
LEAGUES = {
    "E0": "premier-league",
    "SP1": "la-liga",
    "I1": "serie-a",
    "D1": "bundesliga",
    "F1": "ligue-1",
}

OFFICIAL_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
MIRROR_URL = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/{slug}/season-{season}.csv"

# Closing odds preference: Pinnacle > market average > Bet365. Seasons before 2019/20 have no closing columns.
CLOSING_ODDS = [
    ("PSCH", "PSCD", "PSCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("B365CH", "B365CD", "B365CA"),
]
# "Opening" in football-data terms = odds collected a few days before kick-off
OPENING_ODDS = [
    ("PSH", "PSD", "PSA"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
]
# Individual bookmakers' closing prices used for the multi-book consensus
BOOKMAKERS = ["B365", "BW", "IW", "PS", "WH", "VC", "1XB", "BF"]

CURRENT_SEASON_MAX_AGE_HOURS = 6.0  # the running season's files change after every matchday

RESULT_COLUMNS = ["date", "season", "home", "away", "hg", "ag", "result"]


class DataSourceError(RuntimeError):
    pass


def current_season(today: pd.Timestamp | None = None) -> str:
    """Football season code for a date: Aug 2026 -> '2627', Mar 2027 -> '2627'."""
    today = today or pd.Timestamp.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def is_fresh(path: Path, season: str) -> bool:
    """Past seasons never change; the current season's cache expires."""
    if not path.exists():
        return False
    if season != current_season():
        return True
    return time.time() - path.stat().st_mtime < CURRENT_SEASON_MAX_AGE_HOURS * 3600


def _download(url: str, target: Path, timeout: float = 30.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "sportsforecast/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        target.write_bytes(resp.read())


def fetch_season(code: str, season: str, cache_dir: Path) -> Path:
    """Download one season CSV (e.g. code='E0', season='2324'), official source first, then mirror."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{code}_{season}.csv"
    if is_fresh(target, season):
        return target

    urls = [OFFICIAL_URL.format(season=season, code=code)]
    if code in LEAGUES:
        urls.append(MIRROR_URL.format(slug=LEAGUES[code], season=season))

    errors = []
    for url in urls:
        try:
            _download(url, target)
            log.info("downloaded %s", url)
            return target
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
    if target.exists():
        log.warning("%s %s refresh failed, using cached copy", code, season)
        return target
    raise DataSourceError(f"{code} {season} could not be fetched:\n  " + "\n  ".join(errors))


def read_csv(path: Path) -> pd.DataFrame:
    """football-data CSVs: some start with a UTF-8 BOM, older ones are latin-1."""
    raw = path.read_bytes()
    text_encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "latin-1"
    df = pd.read_csv(path, encoding=text_encoding)
    df.columns = df.columns.str.strip()
    return df


def _first_complete(raw: pd.DataFrame, candidates: list[tuple[str, str, str]],
                    min_coverage: float = 0.9) -> tuple[str, str, str] | None:
    for cols in candidates:
        if all(c in raw.columns for c in cols) and raw[list(cols)].notna().all(axis=1).mean() >= min_coverage:
            return cols
    return None


def _normalize(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df = pd.DataFrame({
        # Official files use dd/mm/yy(yy), the mirror uses ISO dates
        "date": pd.to_datetime(raw["Date"], dayfirst=not str(raw["Date"].iloc[0])[:4].isdigit(), format="mixed"),
        "season": season,
        "home": raw["HomeTeam"].str.strip(),
        "away": raw["AwayTeam"].str.strip(),
        "hg": raw["FTHG"].astype(int),
        "ag": raw["FTAG"].astype(int),
        "result": raw["FTR"].str.strip(),
    })
    opening = _first_complete(raw, OPENING_ODDS)
    closing = _first_complete(raw, CLOSING_ODDS)
    if closing:
        df[["odds_h", "odds_d", "odds_a"]] = raw[list(closing)].to_numpy(dtype=float)
        df["odds_source"] = closing[0]
        if opening:
            df[["open_h", "open_d", "open_a"]] = raw[list(opening)].to_numpy(dtype=float)
    elif opening:
        # No closing prices (older seasons): opening odds become the market benchmark, no CLV possible
        df[["odds_h", "odds_d", "odds_a"]] = raw[list(opening)].to_numpy(dtype=float)
        df["odds_source"] = opening[0]
    for book in BOOKMAKERS:
        cols = (f"{book}CH", f"{book}CD", f"{book}CA")
        if _first_complete(raw, [cols], min_coverage=0.5):
            df[[f"book_{book}_h", f"book_{book}_d", f"book_{book}_a"]] = raw[list(cols)].to_numpy(dtype=float)
    return df


def load_league(code: str, seasons: list[str], cache_dir: Path) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = fetch_season(code, season, cache_dir)
        frames.append(_normalize(read_csv(path), season))
    df = pd.concat(frames, ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)
    return df
