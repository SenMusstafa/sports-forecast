"""Coordinate-descent hyper-parameter search on walk-forward RPS.

The last seasons are held out: parameters are chosen on earlier seasons only, then default and
tuned settings are compared on the holdout, so the reported gain is not an in-sample artefact.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import backtest, metrics
from .backtest import BacktestConfig

log = logging.getLogger(__name__)

DEFAULT_GRID: dict[str, list[float]] = {
    "elo.k": [10, 15, 20, 25, 30, 40],
    "elo.home_advantage": [30, 45, 65, 80, 100],
    "elo.season_regression": [0.0, 0.1, 0.2, 0.33, 0.5],
    "elo.promoted_offset": [-150, -100, -80, -40, 0],
    "dc.xi": [0.0005, 0.001, 0.0019, 0.003, 0.005],
    "dc.ridge": [0.0, 0.01, 0.1, 0.5, 2.0],
    "xg_weight": [0.3, 0.5, 0.7, 0.9, 1.0],
}

# Parameter prefix -> the single model whose RPS it is tuned on
TARGET_MODEL = {"elo": "elo", "dc": "dc", "xg_weight": "dcx"}


def _target(name: str) -> str:
    return TARGET_MODEL[name.split(".")[0]]


def get_param(cfg: BacktestConfig, name: str) -> float:
    obj = cfg
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def set_param(cfg: BacktestConfig, name: str, value: float) -> BacktestConfig:
    if "." not in name:
        return replace(cfg, **{name: value})
    section, field_name = name.split(".", 1)
    return replace(cfg, **{section: replace(getattr(cfg, section), **{field_name: value})})


def apply_params(cfg: BacktestConfig, params: dict[str, float]) -> BacktestConfig:
    for name, value in params.items():
        cfg = set_param(cfg, name, value)
    return cfg


def load_params(path: Path) -> dict[str, float]:
    return json.loads(path.read_text())["params"]


def _score(args: tuple[pd.DataFrame, BacktestConfig, str, list[str]]) -> float:
    matches, cfg, model, seasons = args
    preds = backtest.run(matches, cfg).preds
    return float(metrics.evaluate(preds[preds["season"].isin(seasons)], [model])["rps"].iloc[0])


def tune(matches: pd.DataFrame, base: BacktestConfig, tune_seasons: list[str],
         grid: dict[str, list[float]] | None = None, rounds: int = 2,
         workers: int = 4) -> tuple[BacktestConfig, pd.DataFrame]:
    grid = grid or DEFAULT_GRID
    has_xg = "hxg" in matches and matches["hxg"].notna().any()
    best, history = base, []

    # Fresh single-threaded workers: with fork, each process would inherit numpy's full BLAS thread pool
    # and N workers x N threads oversubscribe the CPU (observed ~6x slowdown on 4 cores)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for rnd in range(1, rounds + 1):
            for name, values in grid.items():
                model = _target(name)
                if model == "dcx" and not has_xg:
                    continue
                candidates = [replace(set_param(best, name, v), models=(model,)) for v in values]
                scores = list(pool.map(_score, [(matches, c, model, tune_seasons) for c in candidates]))
                i = min(range(len(values)), key=scores.__getitem__)
                best = set_param(best, name, values[i])
                history += [{"round": rnd, "param": name, "value": v, "model": model, "rps": s}
                            for v, s in zip(values, scores)]
                log.info("round %d %-24s -> %-8g rps %.5f", rnd, name, values[i], scores[i])

    return best, pd.DataFrame(history)


def compare_on_holdout(matches: pd.DataFrame, default: BacktestConfig, tuned: BacktestConfig,
                       holdout: list[str], models: list[str]) -> pd.DataFrame:
    rows = {}
    for label, cfg in (("default", default), ("tuned", tuned)):
        preds = backtest.run(matches, cfg).preds
        table = metrics.evaluate(preds[preds["season"].isin(holdout)], models)
        rows[label] = table["rps"]
    out = pd.DataFrame(rows)
    out["gain"] = out["default"] - out["tuned"]
    return out
