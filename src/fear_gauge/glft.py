from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

# Aproximação de Guéant, Lehalle e Fernandez-Tapia (2013) para horizonte longo, com lote Δ:
#   δ_bid(q) = c1 + (2q + Δ)/2 · c2      δ_ask(q) = c1 − (2q − Δ)/2 · c2
# Centro das cotações = mid − q·c2      Spread = 2·c1 + Δ·c2


@dataclass(frozen=True)
class MarketParams:
    sigma: float
    A: float
    k: float
    delta: float


def c1(gamma: float, p: MarketParams) -> float:
    x = gamma * p.delta / p.k
    return np.log1p(x) / (gamma * p.delta)


def c2(gamma: float, p: MarketParams) -> float:
    x = gamma * p.delta / p.k
    return p.sigma * np.sqrt(gamma / (2 * p.A * p.delta * p.k) * (1 + x) ** (1 / x + 1))


def quotes(q: np.ndarray | float, gamma: float, p: MarketParams) -> tuple[np.ndarray, np.ndarray]:
    a, b = c1(gamma, p), c2(gamma, p)
    q = np.asarray(q, dtype=float)
    return a + (2 * q + p.delta) / 2 * b, a - (2 * q - p.delta) / 2 * b


def spread(gamma: float, p: MarketParams) -> float:
    return 2 * c1(gamma, p) + p.delta * c2(gamma, p)


def _solve(f, target: float, lo: float = 1e-8, hi: float = 1e6) -> float:
    g = lambda lg: f(np.exp(lg)) - target
    a, b = np.log(lo), np.log(hi)
    ga, gb = g(a), g(b)
    if not (np.isfinite(ga) and np.isfinite(gb)) or ga * gb > 0:
        return np.nan
    return float(np.exp(brentq(g, a, b, xtol=1e-10)))


def gamma_from_skew(skew_per_unit: float, p: MarketParams) -> float:
    if not (p.k > 0 and p.A > 0 and skew_per_unit > 0):
        return np.nan
    return _solve(lambda g: c2(g, p), skew_per_unit)


def gamma_from_spread(observed_spread: float, p: MarketParams) -> float:
    if not (p.k > 0 and p.A > 0):
        return np.nan
    return _solve(lambda g: spread(g, p), observed_spread)
