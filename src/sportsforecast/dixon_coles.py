"""Time-weighted Dixon-Coles (1997) bivariate Poisson model with low-score correction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass
class DixonColesConfig:
    xi: float = 0.0019          # time decay per day (half-life ~ 1 year)
    window_days: int = 3 * 365  # matches older than this are ignored
    ridge: float = 0.01         # L2 on attack/defence, stabilises teams with few matches
    max_goals: int = 10


def _tau(hg: np.ndarray, ag: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lam)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    return tau


class DixonColes:
    def __init__(self, config: DixonColesConfig | None = None) -> None:
        self.config = config or DixonColesConfig()
        self.teams: dict[str, int] = {}
        self.attack = np.empty(0)
        self.defence = np.empty(0)
        self.home = 0.25
        self.rho = -0.05
        self._x0: dict[str, tuple[float, float]] = {}

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp) -> DixonColes:
        cfg = self.config
        age = (as_of - matches["date"]).dt.days.to_numpy()
        m = matches[(age > 0) & (age <= cfg.window_days)]
        age = (as_of - m["date"]).dt.days.to_numpy()
        w = np.exp(-cfg.xi * age)

        teams = sorted(set(m["home"]) | set(m["away"]))
        self.teams = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = m["home"].map(self.teams).to_numpy()
        ai = m["away"].map(self.teams).to_numpy()
        hg = m["hg"].to_numpy()
        ag = m["ag"].to_numpy()

        def unpack(p: np.ndarray):
            att = p[:n] - p[:n].mean()
            return att, p[n:2 * n], p[2 * n], p[2 * n + 1]

        def nll(p: np.ndarray) -> float:
            att, dfc, home, rho = unpack(p)
            lam = np.exp(home + att[hi] + dfc[ai])
            mu = np.exp(att[ai] + dfc[hi])
            tau = np.clip(_tau(hg, ag, lam, mu, rho), 1e-10, None)
            ll = np.log(tau) + hg * np.log(lam) - lam + ag * np.log(mu) - mu
            return -(w * ll).sum() + cfg.ridge * (att @ att + dfc @ dfc)

        # Warm start from previous fit for teams already known
        x0 = np.zeros(2 * n + 2)
        for t, i in self.teams.items():
            x0[i], x0[n + i] = self._x0.get(t, (0.0, 0.0))
        x0[2 * n], x0[2 * n + 1] = self.home, self.rho
        bounds = [(None, None)] * (2 * n + 1) + [(-0.3, 0.3)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)

        self.attack, self.defence, self.home, self.rho = unpack(res.x)
        self._x0 = {t: (self.attack[i], self.defence[i]) for t, i in self.teams.items()}
        return self

    def _params(self, team: str) -> tuple[float, float]:
        if team in self.teams:
            i = self.teams[team]
            return self.attack[i], self.defence[i]
        # Unseen (typically promoted) team: average of the three weakest attacks/defences
        weakest = np.argsort(self.attack - self.defence)[:3]
        return self.attack[weakest].mean(), self.defence[weakest].mean()

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        ah, dh = self._params(home)
        aa, da = self._params(away)
        lam = np.exp(self.home + ah + da)
        mu = np.exp(aa + dh)
        goals = np.arange(self.config.max_goals + 1)
        mat = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        mat[0, 0] *= 1 - lam * mu * self.rho
        mat[0, 1] *= 1 + lam * self.rho
        mat[1, 0] *= 1 + mu * self.rho
        mat[1, 1] *= 1 - self.rho
        return mat / mat.sum()

    def predict(self, home: str, away: str) -> dict[str, float]:
        mat = self.score_matrix(home, away)
        total = np.add.outer(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
        return {
            "H": np.tril(mat, -1).sum(),
            "D": np.trace(mat),
            "A": np.triu(mat, 1).sum(),
            "over25": mat[total > 2.5].sum(),
            "btts": mat[1:, 1:].sum(),
        }
