from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .estimate import fit_intensity, usable


def _hac_mean(y: pd.Series, maxlags: int = 10):
    fit = sm.OLS(y.to_numpy(), np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(fit.params[0]), float(fit.bse[0]), float(fit.pvalues[0])


def skew_shape(panel: pd.DataFrame, bins: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    panel = usable(panel)
    two = panel["bid"].notna() & panel["ask"].notna()
    df = pd.DataFrame({"q": panel["q"], "centro": (panel["bid"] + panel["ask"]) / 2 - panel["mid"],
                       "so_bid": panel["bid"].notna() & panel["ask"].isna(),
                       "so_ask": panel["ask"].notna() & panel["bid"].isna(),
                       "dois_lados": two})
    df["faixa_q"] = pd.qcut(df["q"], bins, duplicates="drop")
    tabela = df.groupby("faixa_q", observed=True).agg(
        q_medio=("q", "mean"), centro_medio=("centro", "mean"), so_bid=("so_bid", "mean"),
        so_ask=("so_ask", "mean"), dois_lados=("dois_lados", "mean"), segundos=("q", "size"))
    d = df[two]
    X = sm.add_constant(pd.DataFrame({"q": d["q"], "q3": d["q"] ** 3 / max(d["q"].std(), 1e-12) ** 2}))
    fit = sm.OLS(d["centro"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 60})
    teste = pd.Series({"coef_q": fit.params["q"], "coef_q3": fit.params["q3"], "pvalor_q3": fit.pvalues["q3"]})
    return tabela, teste


def markouts(panel: pd.DataFrame, horizons=(1, 5, 30, 60, 300)) -> pd.DataFrame:
    mid = panel["mid"].to_numpy()
    rate = (panel["fill_bid"].sum() + panel["fill_ask"].sum()) / max(len(panel), 1)
    rows = []
    for col, sign in (("fill_bid", 1), ("fill_ask", -1)):
        idx = np.flatnonzero(panel[col].to_numpy() > 0)
        for h in horizons:
            ok = idx[idx + h < len(mid)]
            mv = pd.Series(sign * (mid[ok + h] - mid[ok]))
            if len(mv) > 5:
                # Janelas sobrepostas: o lag do HAC precisa cobrir quantos fills cabem no horizonte.
                m, se, pv = _hac_mean(mv, maxlags=max(10, int(2 * h * rate)))
                rows.append({"lado": "compra (bid)" if sign == 1 else "venda (ask)", "horizonte_s": h,
                             "movimento_medio": m, "erro_padrao": se, "pvalor": pv, "n": len(mv)})
    return pd.DataFrame(rows)


def intensity_shape(panel: pd.DataFrame) -> pd.DataFrame:
    exp_fit = fit_intensity(panel, form="exp")
    pow_fit = fit_intensity(panel, form="power")
    bid, ask = fit_intensity(panel, side="bid"), fit_intensity(panel, side="ask")
    return pd.DataFrame({"exponencial": exp_fit, "potencia": pow_fit, "so_bid": bid, "so_ask": ask}).T


def tick_size(panel: pd.DataFrame) -> float:
    px = pd.concat([panel["bid"], panel["ask"]]).dropna().unique()
    diffs = np.diff(np.sort(px))
    diffs = diffs[diffs > 1e-12]
    return float(np.round(diffs.min(), 10)) if len(diffs) else np.nan


def tick_pinning(panel: pd.DataFrame, tick: float | None = None) -> pd.Series:
    panel = usable(panel)
    tick = tick or tick_size(panel)
    two = panel["bid"].notna() & panel["ask"].notna()
    spr = (panel["ask"] - panel["bid"])[two]
    return pd.Series({"tick": tick, "spread_medio_em_ticks": float((spr / tick).mean()),
                      "fracao_spread_1_tick": float((spr <= 1.5 * tick).mean()),
                      "fracao_dois_lados": float(two.mean())})


def gamma_vs_vol(gauge: pd.DataFrame, overlap: int = 5) -> pd.Series:
    # Com γ constante, o modelo prevê skew ∝ σ (inclinação 1 em log-log). Erro na estimativa de σ puxa a
    # inclinação para zero (e a elasticidade de γ para −2), então corrijo pela confiabilidade de log σ.
    g = gauge[(gauge["skew"] > 0) & (gauge["sigma"] > 0)].dropna(subset=["skew", "sigma"])
    if len(g) < 8:
        return pd.Series({"inclinacao_skew_sigma": np.nan, "elasticidade_gamma": np.nan, "pvalor": np.nan,
                          "confiabilidade_sigma": np.nan, "conclusivo": False, "n": len(g)})
    x = np.log(g["sigma"])
    fit = sm.OLS(np.log(g["skew"]), sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": overlap})
    ruido = float(np.mean(1 / (2 * g["n_mid"]))) if "n_mid" in g else 0.0
    conf = max(0.0, 1 - ruido / float(x.var())) if x.var() > 0 else 0.0
    b = float(fit.params.iloc[1]) / conf if conf > 0 else np.nan
    se = float(fit.bse.iloc[1]) / conf if conf > 0 else np.nan
    pv = float(2 * stats.norm.sf(abs((b - 1) / se))) if conf > 0 else np.nan
    ok = conf >= 0.5
    return pd.Series({"inclinacao_skew_sigma": b if ok else np.nan, "elasticidade_gamma": 2 * (b - 1) if ok else np.nan,
                      "pvalor": pv if ok else np.nan, "confiabilidade_sigma": conf, "conclusivo": bool(ok),
                      "n": len(g)})


def inventory_half_life(panel: pd.DataFrame, freq: str = "1min") -> pd.Series:
    q = panel["q"].resample(freq).last().dropna()
    fit = sm.OLS(q.iloc[1:].to_numpy(), sm.add_constant(q.iloc[:-1].to_numpy())).fit()
    phi = float(fit.params[1])
    hl = np.log(0.5) / np.log(phi) if 0 < phi < 1 else (0.0 if phi <= 0 else np.inf)
    return pd.Series({"phi": phi, "meia_vida_min": hl * pd.Timedelta(freq).total_seconds() / 60,
                      "q_desvio": float(q.std()), "q_min": float(q.min()), "q_max": float(q.max())})
