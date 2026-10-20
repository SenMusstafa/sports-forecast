"""Walk-forward backtest: every prediction uses only information available before kick-off."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import stacking
from .data import BOOKMAKERS
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
    xg_weight: float = 0.7         # used by the xG Dixon-Coles variant ("dcx") when xG is available
    stack_min_train: int = 380     # matches of out-of-sample predictions before the pool is first fitted
    models: tuple[str, ...] = ("elo", "dc", "dcx")  # tuning switches off what it does not need
    elo: EloConfig = field(default_factory=EloConfig)
    dc: DixonColesConfig = field(default_factory=DixonColesConfig)


@dataclass
class BacktestResult:
    preds: pd.DataFrame
    pool_weights: dict[str, pd.DataFrame]  # stack name -> weight history per refit


def run(matches: pd.DataFrame, cfg: BacktestConfig | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    seasons = list(dict.fromkeys(matches["season"]))
    test_seasons = set(seasons[cfg.burn_in_seasons:])
    if not test_seasons:
        raise ValueError(f"need more than {cfg.burn_in_seasons} seasons, got {len(seasons)}")

    elo = Elo(cfg.elo)
    logit = OrderedLogit()
    dc = DixonColes(cfg.dc) if "dc" in cfg.models else None
    use_xg = "dcx" in cfg.models and "hxg" in matches and matches["hxg"].notna().any()
    dcx = DixonColes(replace(cfg.dc, xg_weight=cfg.xg_weight)) if use_xg else None
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
                if dc:
                    dc.fit(past, m.date)
                if dcx:
                    dcx.fit(past, m.date)
                next_refit = m.date + pd.Timedelta(days=cfg.refit_days)
                log.debug("refit at %s on %d matches", m.date.date(), len(past))

            p_elo = logit.predict(np.array([diff]))[0]
            row = {
                "date": m.date, "season": m.season, "home": m.home, "away": m.away,
                "hg": m.hg, "ag": m.ag, "result": m.result, "elo_diff": diff,
                "elo_H": p_elo[0], "elo_D": p_elo[1], "elo_A": p_elo[2],
            }
            if dc:
                p_dc = dc.predict(m.home, m.away)
                row.update({"dc_H": p_dc["H"], "dc_D": p_dc["D"], "dc_A": p_dc["A"],
                            "dc_over25": p_dc["over25"], "dc_btts": p_dc["btts"]})
            if dcx:
                p_dcx = dcx.predict(m.home, m.away)
                row.update({"dcx_H": p_dcx["H"], "dcx_D": p_dcx["D"], "dcx_A": p_dcx["A"]})
            rows.append(row)

        elo.update(m.home, m.away, m.hg, m.ag)

    preds = pd.DataFrame(rows)
    if dc:
        w = cfg.ensemble_weights
        for o in "HDA":
            preds[f"ens_{o}"] = w["elo"] * preds[f"elo_{o}"] + w["dc"] * preds[f"dc_{o}"]

    scored = matches[matches["season"].isin(test_seasons)].reset_index(drop=True)
    _attach_market(preds, scored, cfg.devig_method)

    models = ["elo"] + (["dc"] if dc else []) + (["dcx"] if dcx else [])
    pools = {"blend": models} if len(models) > 1 else {}
    if "mkt_H" in preds:
        pools["stack"] = models + ["mkt"]
    if "open_H" in preds:
        pools["stack_open"] = models + ["open"]
    weights = {}
    for name, components in pools.items():
        probs, weights[name] = stacking.walk_forward(preds, components, name, cfg.refit_days, cfg.stack_min_train)
        preds[probs.columns] = probs

    return BacktestResult(preds, weights)


def _attach_market(preds: pd.DataFrame, scored: pd.DataFrame, method: str) -> None:
    """mkt = de-vigged closing line, open = de-vigged opening line, cons = mean of individual books."""
    triples = {"mkt": ("odds_h", "odds_d", "odds_a"), "open": ("open_h", "open_d", "open_a")}
    for name, cols in triples.items():
        if set(cols) <= set(scored.columns):
            preds[[f"{name}_{o}" for o in "HDA"]] = devig(scored[list(cols)].to_numpy(), method)
            preds[list(cols)] = scored[list(cols)].to_numpy()

    books = [b for b in BOOKMAKERS if f"book_{b}_h" in scored]
    if len(books) >= 2:
        per_book = np.stack([
            devig(scored[[f"book_{b}_{s}" for s in "hda"]].to_numpy(), method) for b in books])
        n_books = np.isfinite(per_book[:, :, 0]).sum(axis=0)
        cons = np.nansum(per_book, axis=0) / np.where(n_books > 0, n_books, np.nan)[:, None]
        preds[["cons_H", "cons_D", "cons_A"]] = cons
        preds["cons_books"] = n_books
