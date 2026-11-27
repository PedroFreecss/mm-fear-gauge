from __future__ import annotations

import numpy as np
import pandas as pd

from .glft import MarketParams, quotes


def simulate_mm(gamma: float | np.ndarray, params: MarketParams, seconds: int = 6 * 3600, s0: float = 100.0,
                inventory_cap: float | None = None, adverse_jump: float = 0.0, tick: float | None = None,
                quote_noise: float = 0.001, sigma_path: np.ndarray | None = None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = params
    mid = np.empty(seconds)
    bid = np.full(seconds, np.nan)
    ask = np.full(seconds, np.nan)
    inv = np.empty(seconds)
    fb = np.zeros(seconds, dtype=int)
    fa = np.zeros(seconds, dtype=int)
    s, q = s0, 0.0
    sigmas = np.full(seconds, p.sigma) if sigma_path is None else np.asarray(sigma_path, dtype=float)
    shocks = rng.normal(0, 1, seconds) * sigmas
    noise = rng.normal(0, quote_noise, seconds) if quote_noise else np.zeros(seconds)
    gammas = np.broadcast_to(np.asarray(gamma, dtype=float), (seconds,))
    for t in range(seconds):
        mid[t], inv[t] = s, q
        pt = p if sigma_path is None else MarketParams(sigmas[t], p.A, p.k, p.delta)
        db, da = quotes(q, gammas[t], pt)
        b, a = s - max(db, 0.0) + noise[t], s + max(da, 0.0) + noise[t]
        if tick:
            b, a = np.floor(b / tick) * tick, np.ceil(a / tick) * tick
            a = max(a, b + tick)
        quote_bid = inventory_cap is None or q < inventory_cap
        quote_ask = inventory_cap is None or q > -inventory_cap
        if quote_bid:
            bid[t] = b
            if rng.random() < 1 - np.exp(-p.A * np.exp(-p.k * (s - b))):
                fb[t], q = 1, q + p.delta
        if quote_ask:
            ask[t] = a
            if rng.random() < 1 - np.exp(-p.A * np.exp(-p.k * (a - s))):
                fa[t], q = 1, q - p.delta
        s += shocks[t] - adverse_jump * (fb[t] - fa[t])
    idx = pd.date_range("2026-01-01", periods=seconds, freq="s")
    return pd.DataFrame({"mid": mid, "bid": bid, "ask": ask, "q": inv, "fill_bid": fb, "fill_ask": fa,
                         "size": p.delta}, index=idx)
