"""Elo ratings with home advantage, goal-difference multiplier and season regression,
plus an ordered-logit layer that maps rating difference to H/D/A probabilities."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class EloConfig:
    k: float = 20.0
    home_advantage: float = 65.0
    initial: float = 1500.0
    season_regression: float = 0.2  # share pulled back to league mean at season start
    promoted_offset: float = -80.0  # unseen teams start below average


@dataclass
class Elo:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)
    _season: str | None = None

    def _rating(self, team: str) -> float:
        if team not in self.ratings:
            base = np.mean(list(self.ratings.values())) if self.ratings else self.config.initial
            self.ratings[team] = base + (self.config.promoted_offset if self.ratings else 0.0)
        return self.ratings[team]

    def start_season(self, season: str) -> None:
        if self._season is not None and season != self._season and self.ratings:
            mean = np.mean(list(self.ratings.values()))
            r = self.config.season_regression
            self.ratings = {t: (1 - r) * v + r * mean for t, v in self.ratings.items()}
        self._season = season

    def diff(self, home: str, away: str) -> float:
        """Pre-match rating difference including home advantage."""
        return self._rating(home) + self.config.home_advantage - self._rating(away)

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        d = self.diff(home, away)
        expected = 1.0 / (1.0 + 10 ** (-d / 400))
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        # Goal-difference multiplier (World Football Elo style)
        mult = np.log(gd + 1) + 1.0 if gd > 0 else 1.0
        delta = self.config.k * mult * (actual - expected)
        self.ratings[home] += delta
        self.ratings[away] -= delta


class OrderedLogit:
    """P(A)=σ(c1-βx), P(A∪D)=σ(c2-βx), x = Elo diff / 100."""

    def __init__(self) -> None:
        self.params = np.array([0.3, -0.6, 0.6])  # beta, c1, c2 (c2 = c1 + exp(gap))

    def _probs(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        beta, c1, log_gap = params
        c2 = c1 + np.exp(log_gap)
        p_a = expit(c1 - beta * x)
        p_ad = expit(c2 - beta * x)
        return np.column_stack([1 - p_ad, p_ad - p_a, p_a])

    def fit(self, diffs: np.ndarray, outcomes: np.ndarray) -> OrderedLogit:
        """outcomes: 0=H, 1=D, 2=A."""
        x = diffs / 100.0
        idx = np.arange(len(x))

        def nll(p: np.ndarray) -> float:
            probs = self._probs(p, x)[idx, outcomes]
            return -np.log(np.clip(probs, 1e-12, None)).sum()

        init = np.array([self.params[0], self.params[1], np.log(max(self.params[2] - self.params[1], 0.1))])
        res = minimize(nll, init, method="L-BFGS-B")
        self.params = np.array([res.x[0], res.x[1], res.x[1] + np.exp(res.x[2])])
        return self

    def predict(self, diffs: np.ndarray) -> np.ndarray:
        beta, c1, c2 = self.params
        return self._probs(np.array([beta, c1, np.log(c2 - c1)]), np.asarray(diffs) / 100.0)
