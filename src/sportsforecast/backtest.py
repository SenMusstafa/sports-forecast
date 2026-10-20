"""Walk-forward backtest: every prediction uses only information available before kick-off."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dixon_coles import DixonColes, DixonColesConfig
from .elo import Elo, EloConfig, OrderedLogit
from .metrics import OUTCOME_INDEX
from .odds import devig

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    burn_in_seasons: int = 2       # seasons used only for warm-up, not scored
    refit_days: int = 7            # Dixon-Coles / ordered-logit refit cadence
    ensemble_weights: dict[str, float] = field(default_factory=lambda: {"elo": 0.5, "dc": 0.5})
    devig_method: str = "shin"
    elo: EloConfig = field(default_factory=EloConfig)
    dc: DixonColesConfig = field(default_factory=DixonColesConfig)


def run(matches: pd.DataFrame, cfg: BacktestConfig | None = None) -> pd.DataFrame:
    cfg = cfg or BacktestConfig()
    seasons = list(dict.fromkeys(matches["season"]))
    test_seasons = set(seasons[cfg.burn_in_seasons:])
    if not test_seasons:
        raise ValueError(f"need more than {cfg.burn_in_seasons} seasons, got {len(seasons)}")

    elo = Elo(cfg.elo)
    logit = OrderedLogit()
    dc = DixonColes(cfg.dc)
    elo_diffs = np.full(len(matches), np.nan)
    next_refit = None
    rows = []

    for i, m in enumerate(matches.itertuples(index=False)):
        elo.start_season(m.season)
        diff = elo.diff(m.home, m.away)
        elo_diffs[i] = diff

        if m.season in test_seasons:
            if next_refit is None or m.date >= next_refit:
                past = matches.iloc[:i]
                past = past[past["date"] < m.date]
                outcomes = past["result"].map(OUTCOME_INDEX).to_numpy()
                logit.fit(elo_diffs[: len(past)], outcomes)
                dc.fit(past, m.date)
                next_refit = m.date + pd.Timedelta(days=cfg.refit_days)
                log.debug("refit at %s on %d matches", m.date.date(), len(past))

            p_elo = logit.predict(np.array([diff]))[0]
            p_dc = dc.predict(m.home, m.away)
            rows.append({
                "date": m.date, "season": m.season, "home": m.home, "away": m.away,
                "hg": m.hg, "ag": m.ag, "result": m.result, "elo_diff": diff,
                "elo_H": p_elo[0], "elo_D": p_elo[1], "elo_A": p_elo[2],
                "dc_H": p_dc["H"], "dc_D": p_dc["D"], "dc_A": p_dc["A"],
                "dc_over25": p_dc["over25"], "dc_btts": p_dc["btts"],
            })

        elo.update(m.home, m.away, m.hg, m.ag)

    preds = pd.DataFrame(rows)
    w = cfg.ensemble_weights
    for o in "HDA":
        preds[f"ens_{o}"] = w["elo"] * preds[f"elo_{o}"] + w["dc"] * preds[f"dc_{o}"]

    if {"odds_h", "odds_d", "odds_a"} <= set(matches.columns):
        scored = matches[matches["season"].isin(test_seasons)]
        mkt = devig(scored[["odds_h", "odds_d", "odds_a"]].to_numpy(), cfg.devig_method)
        preds[["mkt_H", "mkt_D", "mkt_A"]] = mkt
        preds[["odds_h", "odds_d", "odds_a"]] = scored[["odds_h", "odds_d", "odds_a"]].to_numpy()

    return preds


def value_bets(preds: pd.DataFrame, model: str = "ens", min_edge: float = 0.03) -> pd.DataFrame:
    """Flat 1-unit stakes where model probability * odds - 1 >= min_edge. Research only, not advice."""
    if "odds_h" not in preds:
        return pd.DataFrame()
    bets = []
    for o, col in zip("HDA", ["odds_h", "odds_d", "odds_a"]):
        edge = preds[f"{model}_{o}"] * preds[col] - 1
        sel = preds[edge >= min_edge]
        bets.append(pd.DataFrame({
            "date": sel["date"], "home": sel["home"], "away": sel["away"], "pick": o,
            "odds": sel[col], "edge": edge[edge >= min_edge],
            "profit": np.where(sel["result"] == o, sel[col] - 1, -1.0),
        }))
    return pd.concat(bets).sort_values("date").reset_index(drop=True)
