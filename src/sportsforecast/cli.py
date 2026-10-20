"""CLI: sportsforecast backtest --league E0 --seasons 1819-2526"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import backtest, metrics
from .data import DataSourceError, load_league


def expand_seasons(spec: str) -> list[str]:
    """'1819-2324' -> ['1819', '1920', ..., '2324']; comma lists are passed through."""
    if "-" not in spec:
        return spec.split(",")
    start, end = spec.split("-")
    first, last = int(start[:2]), int(end[:2])
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first, last + 1)]


def cmd_backtest(args: argparse.Namespace) -> int:
    seasons = expand_seasons(args.seasons)
    try:
        matches = load_league(args.league, seasons, Path(args.data_dir))
    except DataSourceError as exc:
        print(exc, file=sys.stderr)
        return 2

    has_odds = "odds_h" in matches.columns
    print(f"{args.league}: {len(matches)} matches, seasons {seasons[0]}..{seasons[-1]}, "
          f"odds: {matches['odds_source'].dropna().unique().tolist() if has_odds else 'none'}")

    cfg = backtest.BacktestConfig(burn_in_seasons=args.burn_in)
    t0 = time.perf_counter()
    preds = backtest.run(matches, cfg)
    print(f"walk-forward done in {time.perf_counter() - t0:.1f}s, {len(preds)} scored matches\n")

    models = ["elo", "dc", "ens"] + (["mkt"] if has_odds else [])
    table = metrics.evaluate(preds, models)
    print(table.round(4).to_string(), "\n")

    print("Ensemble per season (RPS):")
    print(preds.groupby("season").apply(
        lambda s: metrics.evaluate(s, ["ens"])["rps"].iloc[0], include_groups=False).round(4).to_string(), "\n")

    print("Ensemble calibration:")
    print(metrics.calibration(preds, "ens").round(3).to_string(), "\n")

    bets = backtest.value_bets(preds, "ens", args.min_edge)
    if not bets.empty:
        print(f"Value bets (edge >= {args.min_edge}): {len(bets)} bets, "
              f"ROI {bets['profit'].sum() / len(bets):+.2%}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.league}_{seasons[0]}_{seasons[-1]}"
    preds.to_csv(out / f"{stem}_predictions.csv", index=False)
    table.to_csv(out / f"{stem}_metrics.csv")
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
    bt.add_argument("--data-dir", default="data/raw")
    bt.add_argument("--out-dir", default="reports")
    bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
