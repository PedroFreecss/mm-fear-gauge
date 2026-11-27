from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .estimate import usable
from .glft import MarketParams, c2

BLUE, ORANGE, AQUA, YELLOW = "#257de0", "#f47230", "#00b894", "#f2a600"
NEUTRAL, TEXT, GRID = "#6b6b6b", "#1f1f1f", "#e6e6e6"


def _style(ax, title: str, ylabel: str = "", xlabel: str = ""):
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=12)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bdbdbd")
    ax.tick_params(which="both", colors="#555555", labelsize=8, length=0)


def skew_vs_inventory(panel: pd.DataFrame, est, bins: int = 15, ax=None):
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    up = usable(panel)
    two = up["bid"].notna() & up["ask"].notna()
    d = pd.DataFrame({"q": up["q"][two], "c": ((up["bid"] + up["ask"]) / 2 - up["mid"])[two]})
    g = d.groupby(pd.qcut(d["q"], bins, duplicates="drop"), observed=True).mean()
    ax.scatter(g["q"], g["c"], s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3,
               label="Observado (média por faixa)")
    qq = np.linspace(d["q"].min(), d["q"].max(), 50)
    ax.plot(qq, -est.skew * qq, color=ORANGE, lw=2, label=f"Modelo linear (γ = {est.gamma_skew:.3g})")
    ax.axhline(0, color="#bdbdbd", lw=0.8)
    _style(ax, "Quanto o centro das cotações se desloca com o inventário", "Centro − mid (US$)", "Inventário")
    ax.legend(frameon=False, fontsize=8)
    return ax


def intensity_curve(panel: pd.DataFrame, fit: dict, bins: int = 20, ax=None):
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    up = usable(panel)
    d = pd.concat([pd.DataFrame({"d": up["mid"] - up["bid"], "y": up["fill_bid"] > 0}),
                   pd.DataFrame({"d": up["ask"] - up["mid"], "y": up["fill_ask"] > 0})]).dropna()
    g = d.groupby(pd.qcut(d["d"], bins, duplicates="drop"), observed=True).agg(d=("d", "mean"), p=("y", "mean"))
    ax.scatter(g["d"], g["p"], s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3,
               label="Observado")
    dd = np.linspace(g["d"].min(), g["d"].max(), 60)
    ax.plot(dd, -np.expm1(-fit["A"] * np.exp(-fit["k"] * dd)), color=ORANGE, lw=2,
            label=f"Ajuste exponencial (A = {fit['A']:.3g}/s, k = {fit['k']:.3g})")
    _style(ax, "Chance de ser executado em 1 segundo vs distância do mid", "Probabilidade de fill",
           "Distância da cotação ao mid (US$)")
    ax.legend(frameon=False, fontsize=8)
    return ax


def gauge(gauge_df: pd.DataFrame, true_gamma: pd.Series | None = None):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.35})
    a1.plot(gauge_df.index, gauge_df["gamma"], color=BLUE, lw=2, marker="o", ms=3, label="γ estimado")
    if true_gamma is not None:
        a1.plot(true_gamma.index, true_gamma, color=NEUTRAL, ls="--", lw=1.5, label="γ verdadeiro")
        a1.legend(frameon=False, fontsize=8)
    _style(a1, "Fear gauge: aversão a risco implícita, janela móvel", "γ (1/US$)")
    a2.plot(gauge_df.index, gauge_df["sigma"], color=AQUA, lw=1.6)
    _style(a2, "Volatilidade do mid na mesma janela", "σ (US$ por √s)")
    return fig


def markout_chart(m: pd.DataFrame, ax=None):
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    for i, (lado, cor) in enumerate((("compra (bid)", BLUE), ("venda (ask)", ORANGE))):
        d = m[m["lado"] == lado]
        x = np.arange(len(d)) + (i - 0.5) * 0.18
        ax.errorbar(x, d["movimento_medio"], yerr=1.96 * d["erro_padrao"], fmt="o", color=cor, ms=6,
                    capsize=3, lw=1.5, label=f"Fills de {lado}")
        ax.set_xticks(np.arange(len(d)), [f"{h}s" for h in d["horizonte_s"]])
    ax.axhline(0, color=NEUTRAL, lw=1, ls="--")
    _style(ax, "Para onde o preço vai depois de um fill (negativo = contra o market maker)",
           "Movimento médio do mid (US$)", "Tempo depois do fill")
    ax.legend(frameon=False, fontsize=8)
    return ax


def inventory_and_mid(panel: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.35})
    m = panel["mid"].resample("1min").last()
    q = panel["q"].resample("1min").last()
    a1.plot(m.index, m, color=BLUE, lw=1.4)
    _style(a1, "Preço médio (mid)", "US$")
    a2.plot(q.index, q, color=ORANGE, lw=1.4)
    a2.axhline(0, color="#bdbdbd", lw=0.8)
    _style(a2, "Inventário do market maker", "Unidades do ativo")
    return fig
