"""CLI: sportsforecast backtest --league E0 --seasons 1819-2526"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from . import backtest, clv, metrics, players, predict, report, tuning, understat, xg
from .data import OFFICIAL_URL, current_season
from .understat import UnderstatError
from .data import DataSourceError, load_league


def expand_seasons(spec: str) -> list[str]:
    """'1819-2324' -> ['1819', '1920', ..., '2324']; comma lists are passed through."""
    if "-" not in spec:
        return spec.split(",")
    start, end = spec.split("-")
    first, last = int(start[:2]), int(end[:2])
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first, last + 1)]


MODEL_ORDER = ["elo", "dc", "dcx", "dcl", "ens", "blend", "open", "cons", "mkt", "stack_open", "stack"]


def _load(args: argparse.Namespace) -> pd.DataFrame | None:
    try:
        matches = load_league(args.league, expand_seasons(args.seasons), Path(args.data_dir))
    except DataSourceError as exc:
        print(exc, file=sys.stderr)
        return None
    if args.xg or getattr(args, "lineups", False):
        matches = xg.attach_xg(matches, args.league, Path(args.data_dir))
    if getattr(args, "lineups", False):
        appearances = players.player_matches(matches, Path(args.data_dir))
        strength = players.lineup_strength(matches, appearances)
        matches = pd.concat([matches, strength], axis=1)
        print(f"lineups: {matches['home_lineup_delta'].notna().mean():.0%} of matches have a rated lineup "
              f"(run `fetch-lineups` to download missing rosters)")
    return matches


def cmd_backtest(args: argparse.Namespace) -> int:
    seasons = expand_seasons(args.seasons)
    matches = _load(args)
    if matches is None:
        return 2

    odds = matches["odds_source"].dropna().unique().tolist() if "odds_source" in matches else "none"
    xg_cov = matches["hxg"].notna().mean() if "hxg" in matches else 0.0
    print(f"{args.league}: {len(matches)} matches, seasons {seasons[0]}..{seasons[-1]}, "
          f"odds: {odds}, xG coverage: {xg_cov:.0%}")

    cfg = backtest.BacktestConfig(burn_in_seasons=args.burn_in, xg_weight=args.xg_weight)
    if args.params:
        cfg = tuning.apply_params(cfg, tuning.load_params(Path(args.params)))
        print(f"params from {args.params}")
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

    _print_totals(result, args.min_edge)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.league}_{seasons[0]}_{seasons[-1]}"
    preds.to_csv(out / f"{stem}_predictions.csv", index=False)
    table.to_csv(out / f"{stem}_metrics.csv")
    for name, history in result.pool_weights.items():
        history.to_csv(out / f"{stem}_{name}_weights.csv", index=False)
    print(f"saved {out / stem}_*.csv")
    return 0


TOTALS_ORDER = ["dc25", "dcx25", "blend25", "open25", "mkt25", "stack25_open", "stack25"]


def _print_totals(result: backtest.BacktestResult, min_edge: float) -> None:
    preds = result.preds
    table = metrics.evaluate(preds, TOTALS_ORDER, common=True, market=metrics.TOTALS)
    if table.empty:
        return
    print("=== Over/Under 2.5 ===")
    print(table.drop(columns="rps").round(4).to_string(), "\n")
    for name in ("blend25", "stack25", "stack25_open"):
        history = result.pool_weights.get(name)
        if history is not None and not history.empty:
            last = history.iloc[-1].drop("date")
            print(f"{name} weights (last refit): " + ", ".join(f"{k}={v:.2f}" for k, v in last.items()))
    best = "dcx25" if "dcx25_O" in preds else "dc25"
    print(f"\n{best} calibration (over 2.5):")
    print(metrics.calibration(preds, best, market=metrics.TOTALS, outcome="O").round(3).to_string(), "\n")
    bet_table = clv.summary(preds, [best, "blend25", "stack25_open"], min_edge, market=metrics.TOTALS)
    if not bet_table.empty:
        print(f"Over/Under bets at opening odds, edge >= {min_edge} (CLV vs de-vigged closing line):")
        print(bet_table.round(4).to_string(), "\n")


def _summary_row(league: str, result: backtest.BacktestResult, min_edge: float) -> dict:
    """One line per league: how far the models are from the market and whether they add anything to it."""
    preds, row = result.preds, {"league": league, "matches": len(result.preds)}
    one = metrics.evaluate(preds, ["blend", "mkt"], common=True)
    if {"blend", "mkt"} <= set(one.index):
        row.update(blend_rps=one.loc["blend", "rps"], mkt_rps=one.loc["mkt", "rps"],
                   gap_rps=one.loc["blend", "rps"] - one.loc["mkt", "rps"])
    tot = metrics.evaluate(preds, ["blend25", "dcx25", "dc25", "mkt25"], common=True, market=metrics.TOTALS)
    model25 = next((m for m in ("blend25", "dcx25", "dc25") if m in tot.index), None)
    if model25 and "mkt25" in tot.index:
        row.update(model25_brier=tot.loc[model25, "brier"], mkt25_brier=tot.loc["mkt25", "brier"],
                   gap25_brier=tot.loc[model25, "brier"] - tot.loc["mkt25", "brier"])
    for name, market_name, col in (("stack", "mkt", "model_w_1x2"), ("stack25", "mkt25", "model_w_ou")):
        h = result.pool_weights.get(name)
        if h is not None and not h.empty:
            row[col] = 1 - h.iloc[-1][market_name]
    for model, market, col in (("stack_open", metrics.ONE_X_TWO, "clv_1x2"),
                               ("stack25_open", metrics.TOTALS, "clv_ou")):
        b = clv.bets(preds, model, min_edge, market)
        if not b.empty:
            row[col], row[f"{col}_bets"] = b["clv"].mean(), len(b)
    h = result.pool_weights.get("lineup_beta")
    if h is not None and not h.empty:
        row["lineup_beta"] = h.iloc[-1]["beta"]
    return row


def cmd_compare(args: argparse.Namespace) -> int:
    """Same frozen configuration on every league: consistency across leagues is the robustness test."""
    rows = []
    for league in args.leagues.split(","):
        args.league = league
        matches = _load(args)
        if matches is None:
            print(f"{league}: skipped (data unavailable)", file=sys.stderr)
            continue
        cfg = backtest.BacktestConfig(burn_in_seasons=args.burn_in)
        rows.append(_summary_row(league, backtest.run(matches, cfg), args.min_edge))
        print(f"{league}: done")
    table = pd.DataFrame(rows).set_index("league")
    print("\ngap_* > 0: market better; model_w_*: weight the stack gives the models next to the closing "
          "market; clv_*: mean closing-line value of stack bets at opening odds\n")
    print(table.round(4).to_string())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seasons = expand_seasons(args.seasons)
    path = out / f"compare_{seasons[0]}_{seasons[-1]}.csv"
    table.to_csv(path)
    print(f"\nsaved {path}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    matches = _load(args)
    if matches is None:
        return 2
    seasons = list(dict.fromkeys(matches["season"]))
    scored = seasons[args.burn_in:]
    if len(scored) <= args.holdout:
        print(f"need more than {args.burn_in + args.holdout} seasons", file=sys.stderr)
        return 2
    tune_seasons, holdout = scored[:-args.holdout], scored[-args.holdout:]
    print(f"tuning on {tune_seasons}, holdout {holdout}")

    default = backtest.BacktestConfig(burn_in_seasons=args.burn_in)
    t0 = time.perf_counter()
    tuned, history = tuning.tune(matches, default, tune_seasons, rounds=args.rounds, workers=args.workers)
    print(f"search done in {time.perf_counter() - t0:.0f}s\n")

    names = [n for n in tuning.DEFAULT_GRID if n in set(history["param"])]
    params = {n: tuning.get_param(tuned, n) for n in names}
    print(pd.DataFrame({"default": {n: tuning.get_param(default, n) for n in names}, "tuned": params}).to_string(), "\n")

    models = [m for m in ("elo", "dc", "dcx", "blend", "mkt") if m != "dcx" or "xg_weight" in params]
    comparison = tuning.compare_on_holdout(matches, default, tuned, holdout, models)
    print(f"Holdout RPS {holdout} (gain > 0 = tuned is better):")
    print(comparison.round(5).to_string(), "\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "league": args.league, "tuned_on": tune_seasons, "holdout": holdout, "params": params,
        "holdout_rps": comparison.drop(columns="gain").round(5).to_dict(),
    }, indent=2))
    history.to_csv(out.with_suffix(".history.csv"), index=False)
    print(f"saved {out}")
    return 0


def cmd_fetch_lineups(args: argparse.Namespace) -> int:
    args.lineups = False
    args.xg = True
    matches = _load(args)
    if matches is None:
        return 2
    if matches["us_id"].isna().all():
        print("no Understat match ids - xG/fixture download failed, see `check-sources`", file=sys.stderr)
        return 2
    n = players.fetch_rosters(matches, Path(args.data_dir), delay=args.delay, limit=args.limit)
    print(f"downloaded {n} rosters")
    return 0


def cmd_check_sources(args: argparse.Namespace) -> int:
    """One request per source, so connectivity and format problems show up before a long run."""
    league = understat.LEAGUES.get(args.league, "EPL")
    year = 2000 + int(args.season[:2])
    checks = {
        "football-data": lambda: _probe_csv(OFFICIAL_URL.format(season=args.season, code=args.league)),
        "understat league": lambda: f"{len(understat.league_dates(league, year))} fixtures",
        "understat match": lambda: _probe_match(league, year),
    }
    ok = True
    for name, check in checks.items():
        try:
            print(f"OK    {name}: {check()}")
        except Exception as exc:  # diagnostics command: report every failure kind instead of stopping
            ok = False
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    return 0 if ok else 1


def _probe_csv(url: str) -> str:
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sportsforecast/0.1"}), timeout=30) as r:
        header = r.readline().decode("latin-1")
    cols = header.strip().split(",")
    return f"{len(cols)} columns, closing odds: {'PSCH' in cols}"


def _probe_match(league: str, year: int) -> str:
    played = [d for d in understat.league_dates(league, year) if d.get("isResult")]
    if not played:
        return "no finished matches yet"
    rosters = understat.match_data(str(played[0]["id"]))["rosters"]
    sample = next(iter(rosters["h"].values()))
    return f"match {played[0]['id']}: {sum(len(v) for v in rosters.values())} players, fields {sorted(sample)[:8]}..."


def cmd_predict(args: argparse.Namespace) -> int:
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()
    if args.seasons is None:
        last = int(current_season(as_of)[:2])
        args.seasons = f"{last - args.history:02d}{last - args.history + 1:02d}-{last:02d}{last + 1:02d}"
    history = _load(args)
    if history is None:
        return 2
    if as_of < pd.Timestamp.today().normalize():
        return _predict_past(args, history, as_of)
    try:
        fixtures = predict.fetch_fixtures(args.league, Path(args.data_dir))
    except DataSourceError as exc:
        print(exc, file=sys.stderr)
        return 2
    def in_window(f: pd.DataFrame) -> pd.DataFrame:
        return f[(f["date"] >= as_of) & (f["date"] < as_of + pd.Timedelta(days=args.days))]

    window = in_window(fixtures)
    if window.empty and args.xg:
        try:
            window = in_window(predict.understat_fixtures(history, args.league, Path(args.data_dir),
                                                          current_season(as_of)))
            if not window.empty:
                print(f"fixtures.csv has no {args.league} matches in the window; using the Understat "
                      f"schedule (no odds, so no market comparison)")
        except UnderstatError as exc:
            print(f"understat fallback failed: {exc}", file=sys.stderr)
    if window.empty:
        span = (f"{len(fixtures)} {args.league} fixtures dated {fixtures['date'].min():%Y-%m-%d}.."
                f"{fixtures['date'].max():%Y-%m-%d}" if len(fixtures) else f"no {args.league} fixtures at all")
        print(f"no {args.league} fixtures in the next {args.days} days from {as_of:%Y-%m-%d}; "
              f"fixtures.csv has {span} (leagues: {', '.join(predict.fixture_leagues(Path(args.data_dir)))}). "
              f"The file is refreshed when older than 3h; delete data/raw/fixtures.csv to force it.",
              file=sys.stderr)
        return 1
    fixtures = window
    fixtures = fixtures.reset_index(drop=True)
    print(f"{args.league}: history {args.seasons} ({len(history)} matches), {len(fixtures)} fixtures")

    result = predict.forecast(history, fixtures, _predict_config(args))
    _save_forecast(args, result, as_of)
    return 0


def _predict_config(args: argparse.Namespace) -> backtest.BacktestConfig:
    cfg = backtest.BacktestConfig(burn_in_seasons=args.burn_in)
    if args.params:
        cfg = tuning.apply_params(cfg, tuning.load_params(Path(args.params)))
    return cfg


def _predict_past(args: argparse.Namespace, history: pd.DataFrame, as_of: pd.Timestamp) -> int:
    """Forecast an already played window as if it were upcoming, then score it."""
    fixtures, played = predict.past_fixtures(history, as_of, args.days)
    if fixtures.empty:
        print(f"no played {args.league} matches between {as_of:%Y-%m-%d} and "
              f"{as_of + pd.Timedelta(days=args.days):%Y-%m-%d}", file=sys.stderr)
        return 1
    print(f"{args.league}: past window {as_of:%Y-%m-%d} +{args.days}d, {len(fixtures)} played matches; "
          f"models see only data before {fixtures['date'].min():%Y-%m-%d}, market = "
          f"{fixtures['odds_source'].iloc[0] if 'odds_source' in fixtures else 'none'} odds")
    result = predict.score_past(predict.forecast(history, fixtures, _predict_config(args)), played)
    _save_forecast(args, result, as_of)
    print(f"\nScores on these {len(fixtures)} matches (lower is better; one round is far too few "
          f"to judge a model, the backtest is the real test):")
    print(result.scores.round(4).to_string())
    return 0


def _save_forecast(args: argparse.Namespace, result: predict.ForecastResult, as_of: pd.Timestamp) -> None:
    t = result.table
    cols = ["date", "home", "away", "model_H", "model_D", "model_A"] + \
        [c for c in ("mkt_H", "mkt_D", "mkt_A") if c in t] + ["xg_home", "xg_away", "over25"] + \
        [c for c in ("mkt_over25", "hg", "ag") if c in t]
    print(t[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{args.league}_forecast_{as_of:%Y%m%d}"
    t.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".html").write_text(report.render(result, args.league), encoding="utf-8")
    print(f"saved {stem}.html and .csv")


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--league", default="E0", help="football-data.co.uk code: E0, SP1, I1, D1, F1, ...")
    p.add_argument("--seasons", default="1819-2526")
    p.add_argument("--burn-in", type=int, default=2)
    p.add_argument("--no-xg", dest="xg", action="store_false", help="skip Understat xG enrichment")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--lineups", action="store_true", help="add lineup-adjusted model from cached rosters")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sportsforecast")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="walk-forward backtest of Elo, Dixon-Coles, ensemble and market")
    _add_data_args(bt)
    bt.add_argument("--min-edge", type=float, default=0.03)
    bt.add_argument("--xg-weight", type=float, default=0.7)
    bt.add_argument("--params", help="JSON written by `tune`; overrides defaults")
    bt.add_argument("--out-dir", default="reports")
    bt.set_defaults(func=cmd_backtest)

    tn = sub.add_parser("tune", help="coordinate-descent search on walk-forward RPS with a held-out period")
    _add_data_args(tn)
    tn.add_argument("--holdout", type=int, default=2, help="last N seasons excluded from the search")
    tn.add_argument("--rounds", type=int, default=2)
    tn.add_argument("--workers", type=int, default=4)
    tn.add_argument("--out", help="default: params/<league>.json")
    tn.set_defaults(func=cmd_tune)

    fl = sub.add_parser("fetch-lineups", help="download Understat rosters (resumable, ~1 request/s)")
    _add_data_args(fl)
    fl.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    fl.add_argument("--limit", type=int, help="download at most N rosters this run")
    fl.set_defaults(func=cmd_fetch_lineups)

    cp = sub.add_parser("compare", help="backtest several leagues with one frozen configuration")
    _add_data_args(cp)
    cp.add_argument("--leagues", default="E0,SP1,I1,D1,F1")
    cp.add_argument("--min-edge", type=float, default=0.03)
    cp.add_argument("--out-dir", default="reports")
    cp.set_defaults(func=cmd_compare)

    pr = sub.add_parser("predict", help="forecast upcoming fixtures and write an HTML report")
    _add_data_args(pr)
    pr.set_defaults(seasons=None)
    pr.add_argument("--history", type=int, default=4, help="past seasons used besides the current one")
    pr.add_argument("--days", type=int, default=10, help="forecast fixtures within this many days")
    pr.add_argument("--as-of", help="YYYY-MM-DD, default today (earlier dates for checking past rounds)")
    pr.add_argument("--params", help="JSON written by `tune`")
    pr.add_argument("--out-dir", default="reports")
    pr.set_defaults(func=cmd_predict)

    cs = sub.add_parser("check-sources", help="probe each data source once")
    cs.add_argument("--league", default="E0")
    cs.add_argument("--season", default="2526")
    cs.set_defaults(func=cmd_check_sources)

    args = parser.parse_args(argv)
    if getattr(args, "out", "") is None and args.command == "tune":
        args.out = f"params/{args.league}.json"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
