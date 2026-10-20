"""Draw ranking: which matches of a week are most likely to end level.

The market is used as an input, not as an opponent. A pooled logistic model, refitted every week
on earlier weeks of all leagues, combines the opening-market draw probability with model and
history features. Its value is judged by how often the top-N matches of each week end in a draw,
compared with ranking by the market alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

FEATURES = ["lg_market_D", "lg_model_D", "exp_total", "exp_gap", "abs_elo", "lg_under", "league_draw_rate"]
MARKET_ONLY = ["lg_market_D"]


def _lg(p: pd.Series) -> pd.Series:
    return pd.Series(logit(np.clip(p.astype(float), 0.01, 0.99)), index=p.index)


def features(preds: pd.DataFrame, league: str) -> pd.DataFrame:
    """Pre-match draw features from walk-forward backtest predictions of one league.

    Only the opening market is used: closing prices are not known when the list is made.
    """
    f = pd.DataFrame({"date": preds["date"], "league": league, "season": preds["season"],
                      "home": preds["home"], "away": preds["away"]})
    if "result" in preds:
        f["draw"] = (preds["result"] == "D").astype(float)
        f["score"] = preds["hg"].astype(int).astype(str) + "-" + preds["ag"].astype(int).astype(str)
    f["market_D"] = preds["open_D"] if "open_D" in preds else np.nan
    model_d = preds["blend_D"] if "blend_D" in preds else pd.Series(np.nan, index=preds.index)
    f["model_D"] = model_d.fillna(preds.get("ens_D", preds["elo_D"]))
    goal = "dcx" if "dcx_lam" in preds else "dc"
    f["exp_total"] = preds[f"{goal}_lam"] + preds[f"{goal}_mu"]
    f["exp_gap"] = (preds[f"{goal}_lam"] - preds[f"{goal}_mu"]).abs()
    f["abs_elo"] = preds["elo_diff"].abs() / 100
    f["under"] = preds["open25_U"] if "open25_U" in preds else np.nan
    if "open_d" in preds:
        f["draw_odds"] = preds["open_d"]
    return _finish(f)


def _finish(f: pd.DataFrame) -> pd.DataFrame:
    f["lg_market_D"] = _lg(f["market_D"])
    f["lg_model_D"] = _lg(f["model_D"])
    f["lg_under"] = _lg(f["under"].fillna(0.5))
    return f


def add_league_draw_rate(f: pd.DataFrame, prior_n: float = 200, prior_rate: float = 0.26) -> pd.DataFrame:
    """Draw frequency of the league before each match date (walk-forward, shrunk toward a prior)."""
    f = f.sort_values("date").copy()
    rates = []
    for _, g in f.groupby("league", sort=False):
        daily = g.groupby("date")["draw"].agg(["sum", "count"]).cumsum().shift(1).fillna(0)
        rate = (daily["sum"] + prior_n * prior_rate) / (daily["count"] + prior_n * 1.0)
        rates.append(g["date"].map(rate))
    f["league_draw_rate"] = pd.concat(rates).reindex(f.index)
    return f


@dataclass
class DrawModel:
    """L2-regularised logistic regression on standardised features."""
    columns: list[str]
    ridge: float = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> DrawModel:
        A = X[self.columns].to_numpy(dtype=float)
        self.mean_, self.std_ = A.mean(axis=0), A.std(axis=0) + 1e-9
        Z = np.column_stack([np.ones(len(A)), (A - self.mean_) / self.std_])

        def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
            p = expit(Z @ w)
            ll = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum()
            grad = Z.T @ (p - y)
            ll += self.ridge * (w[1:] @ w[1:])
            grad[1:] += 2 * self.ridge * w[1:]
            return ll, grad

        self.coef_ = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        A = (X[self.columns].to_numpy(dtype=float) - self.mean_) / self.std_
        return expit(self.coef_[0] + A @ self.coef_[1:])

    def weights(self) -> dict[str, float]:
        """Standardised coefficients: how strongly each feature moves the draw probability."""
        return dict(zip(self.columns, self.coef_[1:]))


def usable(f: pd.DataFrame) -> pd.DataFrame:
    return f.dropna(subset=["market_D", "model_D", "exp_total", "abs_elo", "league_draw_rate"])


def fixture_features(table: pd.DataFrame, league: str) -> pd.DataFrame:
    """Draw features for upcoming fixtures from a predict.forecast table (current prices ~ opening line)."""
    f = pd.DataFrame({"date": table["date"], "time": table["time"], "league": league,
                      "home": table["home"], "away": table["away"], "draw": np.nan,
                      "market_D": table["mkt_D"] if "mkt_D" in table else np.nan,
                      "model_D": table["model_D"], "exp_total": table["xg_home"] + table["xg_away"],
                      "exp_gap": (table["xg_home"] - table["xg_away"]).abs(), "abs_elo": table["elo_diff"].abs() / 100,
                      "under": 1 - table["mkt_over25"] if "mkt_over25" in table else np.nan})
    if "odds_d" in table:
        f["draw_odds"] = table["odds_d"]
    return _finish(f)


def walk_forward(f: pd.DataFrame, min_train: int = 2000, from_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Weekly refits on all earlier weeks (all leagues pooled). Adds p_combined and p_market.

    from_date limits which weeks are scored; training always uses every earlier week.
    """
    f = usable(f).copy()
    f["week"] = f["date"].dt.to_period("W-SUN").dt.start_time
    f["p_combined"], f["p_market"] = np.nan, np.nan
    weeks = sorted(f["week"].unique())
    if from_date is not None:
        first = pd.Timestamp(from_date).to_period("W-SUN").start_time
        weeks = [w for w in weeks if w >= first]
    for week in weeks:
        train = f[f["date"] < week]
        if len(train) < min_train:
            continue
        idx = f.index[f["week"] == week]
        y = train["draw"].to_numpy()
        f.loc[idx, "p_combined"] = DrawModel(FEATURES).fit(train, y).predict(f.loc[idx])
        f.loc[idx, "p_market"] = DrawModel(MARKET_ONLY).fit(train, y).predict(f.loc[idx])
    return f.dropna(subset=["p_combined"])


def top_n(scored: pd.DataFrame, score: str, n: int) -> pd.DataFrame:
    """The n highest-scored matches of each week."""
    return (scored.sort_values(score, ascending=False).groupby("week", group_keys=False).head(n)
            .sort_values(["week", score], ascending=[True, False]))


def evaluate(scored: pd.DataFrame, ns: tuple[int, ...] = (3, 5, 7)) -> pd.DataFrame:
    """Per ranking and list size: draw hit rate, mean predicted probability, draws per week, ROI at opening odds."""
    rows = []
    for n in ns:
        for score, label in (("p_combined", "combined"), ("p_market", "market only")):
            picks = top_n(scored, score, n)
            weekly = picks.groupby("week")["draw"].sum()
            row = {"list": n, "ranking": label, "weeks": len(weekly), "picks": len(picks),
                   "hit_rate": picks["draw"].mean(), "predicted": picks[score].mean(),
                   "draws_per_week": weekly.mean(), "weeks_with_0_draws": (weekly == 0).mean()}
            if "draw_odds" in picks:
                profit = np.where(picks["draw"] == 1, picks["draw_odds"] - 1, -1.0)
                row["roi_at_opening"] = np.nanmean(profit)
            rows.append(row)
    base = {"list": "all", "ranking": "base rate", "picks": len(scored), "hit_rate": scored["draw"].mean(),
            "predicted": scored["p_market"].mean()}
    return pd.DataFrame(rows + [base]).set_index(["list", "ranking"])


def binary_scores(scored: pd.DataFrame) -> pd.DataFrame:
    y = scored["draw"].to_numpy()
    out = {}
    for col, label in (("market_D", "market raw"), ("p_market", "market recalibrated"), ("p_combined", "combined")):
        p = np.clip(scored[col].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        out[label] = {"log_loss": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
                      "brier": np.mean((p - y) ** 2)}
    return pd.DataFrame(out).T
