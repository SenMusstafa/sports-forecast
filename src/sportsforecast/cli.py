"""CLI: sportsforecast backtest --league E0 --seasons 1819-2526"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from . import backtest, clv, metrics, xg
from .data import DataSourceError, load_league


def expand_seasons(spec: str) -> list[str]:
    """'1819-2324' -> ['1819', '1920', ..., '2324']; comma lists are passed through."""
    if "-" not in spec:
        return spec.split(",")
    start, end = spec.split("-")
    first, last = int(start[:2]), int(end[:2])
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first, last + 1)]


MODEL_ORDER = ["elo", "dc", "dcx", "ens", "blend", "open", "cons", "mkt", "stack_open", "stack"]


def cmd_backtest(args: argparse.Namespace) -> int:
    seasons = expand_seasons(args.seasons)
    try:
        matches = load_league(args.league, seasons, Path(args.data_dir))
    except DataSourceError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.xg:
        matches = xg.attach_xg(matches, args.league, Path(args.data_dir))

    odds = matches["odds_source"].dropna().unique().tolist() if "odds_source" in matches else "none"
    xg_cov = matches["hxg"].notna().mean() if "hxg" in matches else 0.0
    print(f"{args.league}: {len(matches)} matches, seasons {seasons[0]}..{seasons[-1]}, "
          f"odds: {odds}, xG coverage: {xg_cov:.0%}")

    cfg = backtest.BacktestConfig(burn_in_seasons=args.burn_in, xg_weight=args.xg_weight)
    t0 = time.perf_counter()
    result = backtest.run(matches, cfg)
    preds = result.preds
    print(f"walk-forward done in {time.perf_counter() - t0:.1f}s, {len(preds)} scored matches\n")

    table = metrics.evaluate(preds, MODEL_ORDER, common=True)
    print("Scores on common matches (lower is better, except accuracy):")
    print(table.round(4).to_string(), "\n")

    for name, history in result.pool_weights.items():
        if not history.empty:
            last = history.iloc[-1].drop("date")
            print(f"{name} weights (last refit): " + ", ".join(f"{k}={v:.2f}" for k, v in last.items()))
    print()

    seasonal = [m for m in ("blend", "mkt", "stack") if f"{m}_H" in preds]
    print("RPS per season:")
    per_season = {season: metrics.evaluate(group, seasonal)["rps"] for season, group in preds.groupby("season")}
    print(pd.DataFrame(per_season).T.round(4).to_string(), "\n")

    print("Blend calibration:")
    print(metrics.calibration(preds.dropna(subset=["blend_H"]), "blend").round(3).to_string(), "\n")

    bet_table = clv.summary(preds, ["ens", "blend", "stack_open"], args.min_edge)
    if not bet_table.empty:
        print(f"Bets at opening odds, edge >= {args.min_edge} (CLV vs de-vigged closing line):")
        print(bet_table.round(4).to_string(), "\n")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.league}_{seasons[0]}_{seasons[-1]}"
    preds.to_csv(out / f"{stem}_predictions.csv", index=False)
    table.to_csv(out / f"{stem}_metrics.csv")
    for name, history in result.pool_weights.items():
        history.to_csv(out / f"{stem}_{name}_weights.csv", index=False)
    print(f"saved {out / stem}_*.csv")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sportsforecast")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="walk-forward backtest of Elo, Dixon-Coles, ensemble and market")
    bt.add_argument("--league", default="E0", help="football-data.co.uk code: E0, SP1, I1, D1, F1, ...")
    bt.add_argument("--seasons", default="1819-2526")
    bt.add_argument("--burn-in", type=int, default=2)
    bt.add_argument("--min-edge", type=float, default=0.03)
    bt.add_argument("--no-xg", dest="xg", action="store_false", help="skip Understat xG enrichment")
    bt.add_argument("--xg-weight", type=float, default=0.7)
    bt.add_argument("--data-dir", default="data/raw")
    bt.add_argument("--out-dir", default="reports")
    bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
