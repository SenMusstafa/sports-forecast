"""Match data loading in football-data.co.uk format (official site or GitHub mirror)."""
from __future__ import annotations

import logging
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

# Preference order: Pinnacle closing > market average closing > Pinnacle opening > Bet365
ODDS_COLUMNS = [
    ("PSCH", "PSCD", "PSCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
]

RESULT_COLUMNS = ["date", "season", "home", "away", "hg", "ag", "result"]


class DataSourceError(RuntimeError):
    pass


def _download(url: str, target: Path, timeout: float = 30.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "sportsforecast/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        target.write_bytes(resp.read())


def fetch_season(code: str, season: str, cache_dir: Path) -> Path:
    """Download one season CSV (e.g. code='E0', season='2324'), official source first, then mirror."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{code}_{season}.csv"
    if target.exists():
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
    raise DataSourceError(f"{code} {season} could not be fetched:\n  " + "\n  ".join(errors))


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
    for cols in ODDS_COLUMNS:
        if all(c in raw.columns for c in cols) and raw[list(cols)].notna().all(axis=1).mean() > 0.9:
            df[["odds_h", "odds_d", "odds_a"]] = raw[list(cols)].to_numpy(dtype=float)
            df["odds_source"] = cols[0]
            break
    return df


def load_league(code: str, seasons: list[str], cache_dir: Path) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = fetch_season(code, season, cache_dir)
        frames.append(_normalize(pd.read_csv(path, encoding="latin-1"), season))
    df = pd.concat(frames, ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)
    return df
