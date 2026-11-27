from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

from .glft import MarketParams, c2, gamma_from_skew, gamma_from_spread, spread


def usable(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["cotacao_atual"]] if "cotacao_atual" in panel else panel


def sigma_per_second(panel: pd.DataFrame) -> float:
    return float(panel["mid"].diff().dropna().std())


def _distances(panel: pd.DataFrame, side: str):
    panel = usable(panel)
    parts = []
    if side in ("both", "bid"):
        m = panel["bid"].notna()
        parts.append(((panel["mid"] - panel["bid"])[m].to_numpy(), (panel["fill_bid"][m] > 0).to_numpy()))
    if side in ("both", "ask"):
        m = panel["ask"].notna()
        parts.append(((panel["ask"] - panel["mid"])[m].to_numpy(), (panel["fill_ask"][m] > 0).to_numpy()))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def fit_intensity(panel: pd.DataFrame, side: str = "both", form: str = "exp") -> dict:
    # Com no máximo uma execução contada por segundo, P(fill) = 1 − exp(−λ·dt). Poisson puro subestimaria A.
    d, y = _distances(panel, side)
    keep = d > 0 if form == "power" else np.isfinite(d)
    d, y = d[keep], y[keep]
    x = np.log(d) if form == "power" else d

    def nll(th):
        lam = np.exp(th[0] - th[1] * x)
        pr = np.clip(-np.expm1(-lam), 1e-300, 1.0)
        return -(np.log(pr[y]).sum() - lam[~y].sum())

    k0 = 1.0 if form == "power" else 1.0 / max(np.median(d), 1e-12)
    res = minimize(nll, [np.log(max(y.mean(), 1e-6)), k0], method="BFGS")
    se = np.sqrt(np.diag(res.hess_inv)) if res.success else np.full(2, np.nan)
    return {"A": float(np.exp(res.x[0])), "k": float(res.x[1]), "se_logA": float(se[0]), "se_k": float(se[1]),
            "loglik": float(-res.fun), "aic": float(2 * res.fun + 4), "n": int(len(d)), "form": form}


def fit_skew(panel: pd.DataFrame, maxlags: int = 60):
    panel = usable(panel)
    m = panel["bid"].notna() & panel["ask"].notna()
    center = (panel["bid"] + panel["ask"])[m] / 2 - panel["mid"][m]
    X = sm.add_constant(panel["q"][m].rename("q"))
    return sm.OLS(center.rename("centro"), X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


@dataclass
class GammaEstimate:
    sigma: float
    A: float
    k: float
    delta: float
    skew: float
    skew_se: float
    spread: float
    gamma_skew: float
    gamma_skew_lo: float
    gamma_skew_hi: float
    gamma_spread: float
    n_two_sided: int

    def as_series(self) -> pd.Series:
        return pd.Series(asdict(self))


def estimate_gamma(panel: pd.DataFrame, intensity: dict | None = None, maxlags: int = 60) -> GammaEstimate:
    intensity = intensity or fit_intensity(panel)
    p = MarketParams(sigma_per_second(panel), intensity["A"], intensity["k"], float(panel["size"].median()))
    fit = fit_skew(panel, maxlags)
    skew, se = -float(fit.params["q"]), float(fit.bse["q"])
    up = usable(panel)
    two = up["bid"].notna() & up["ask"].notna()
    spr = float((up["ask"] - up["bid"])[two].mean())
    return GammaEstimate(
        sigma=p.sigma, A=p.A, k=p.k, delta=p.delta, skew=skew, skew_se=se, spread=spr,
        gamma_skew=gamma_from_skew(skew, p),
        gamma_skew_lo=gamma_from_skew(max(skew - 1.96 * se, 1e-15), p),
        gamma_skew_hi=gamma_from_skew(skew + 1.96 * se, p),
        gamma_spread=gamma_from_spread(spr, p),
        n_two_sided=int(two.sum()),
    )


def rolling_gauge(panel: pd.DataFrame, window: str = "1h", step: str = "15min",
                  intensity: dict | None = None, min_rows: int = 600) -> pd.DataFrame:
    intensity = intensity or fit_intensity(panel)
    delta = float(panel["size"].median())
    w, st = pd.Timedelta(window), pd.Timedelta(step)
    rows = []
    t = panel.index[0]
    while t + w <= panel.index[-1]:
        full = panel.loc[t:t + w]
        chunk = usable(full)
        two = chunk["bid"].notna() & chunk["ask"].notna()
        if two.sum() >= min_rows and chunk["q"].std() > 0:
            fit = fit_skew(chunk)
            p = MarketParams(sigma_per_second(full), intensity["A"], intensity["k"], delta)
            skew = -float(fit.params["q"])
            rows.append({"fim": t + w, "sigma": p.sigma, "n_mid": int(full["mid"].diff().notna().sum()), "skew": skew, "skew_pvalor": float(fit.pvalues["q"]),
                         "gamma": gamma_from_skew(skew, p) if skew > 0 else np.nan,
                         "spread": float((chunk["ask"] - chunk["bid"])[two].mean()),
                         "inventario_abs": float(chunk["q"].abs().mean()),
                         "fills": int(full["fill_bid"].sum() + full["fill_ask"].sum())})
        t += st
    return pd.DataFrame(rows).set_index("fim") if rows else pd.DataFrame()


def model_quotes_check(est: GammaEstimate) -> pd.Series:
    p = MarketParams(est.sigma, est.A, est.k, est.delta)
    g = est.gamma_skew
    return pd.Series({"skew_previsto": c2(g, p), "skew_observado": est.skew,
                      "spread_previsto": spread(g, p),
                      "spread_observado": est.spread})
