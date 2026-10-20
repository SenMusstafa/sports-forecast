"""Convert bookmaker decimal odds to margin-free (de-vigged) probabilities."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def normalize(odds: np.ndarray) -> np.ndarray:
    implied = 1.0 / odds
    return implied / implied.sum()


def power(odds: np.ndarray) -> np.ndarray:
    """p_i = (1/o_i)^k with k chosen so probabilities sum to 1; corrects favourite-longshot bias better than normalize."""
    implied = 1.0 / odds
    k = brentq(lambda k: (implied ** k).sum() - 1.0, 1e-6, 10.0)
    return implied ** k


def shin(odds: np.ndarray) -> np.ndarray:
    """Shin (1993) model: margin attributed to insider trading share z."""
    implied = 1.0 / odds
    booksum = implied.sum()

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z ** 2 + 4 * (1 - z) * implied ** 2 / booksum) - z) / (2 * (1 - z))

    z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.4)
    return probs(z)


METHODS = {"normalize": normalize, "power": power, "shin": shin}


def devig(odds_matrix: np.ndarray, method: str = "shin") -> np.ndarray:
    """odds_matrix: (n, 3) decimal odds for H/D/A. Rows with missing odds yield NaN."""
    fn = METHODS[method]
    out = np.full(odds_matrix.shape, np.nan)
    for i, row in enumerate(odds_matrix):
        if np.all(np.isfinite(row)) and np.all(row > 1.0) and (1.0 / row).sum() > 1.0:
            out[i] = fn(row)
    return out
