import gzip
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from fear_gauge import breaks
from fear_gauge.dataset import build_panel
from fear_gauge.estimate import estimate_gamma, fit_intensity, rolling_gauge
from fear_gauge.glft import MarketParams, c2, gamma_from_skew, gamma_from_spread, spread
from fear_gauge.simulate import simulate_mm

P = MarketParams(sigma=0.02, A=1.0, k=100.0, delta=1.0)


@pytest.fixture(scope="module")
def sim():
    return simulate_mm(0.5, P, seconds=6 * 3600, seed=1)


def test_formulas_invertem():
    for g in (0.05, 0.5, 5.0):
        assert gamma_from_skew(c2(g, P), P) == pytest.approx(g, rel=1e-6)
        assert gamma_from_spread(spread(g, P), P) == pytest.approx(g, rel=1e-4)


def test_intensidade_recupera_A_e_k(sim):
    fit = fit_intensity(sim)
    assert fit["A"] == pytest.approx(1.0, rel=0.1)
    assert fit["k"] == pytest.approx(100.0, rel=0.1)


def test_gamma_pelo_skew_recupera_o_verdadeiro(sim):
    assert estimate_gamma(sim).gamma_skew == pytest.approx(0.5, rel=0.15)


def test_gauge_detecta_mudanca_de_gamma():
    g = np.r_[np.full(3 * 3600, 0.3), np.full(3 * 3600, 2.0)]
    gauge = rolling_gauge(simulate_mm(g, P, seed=4))
    assert gauge["gamma"].iloc[0] < 0.5 < 1.5 < gauge["gamma"].iloc[-1]


def test_markout_detecta_selecao_adversa():
    m = breaks.markouts(simulate_mm(0.5, P, seed=5, adverse_jump=0.01), horizons=(1,))
    assert (m["movimento_medio"] < 0).all() and (m["pvalor"] < 0.01).all()


def test_sem_selecao_adversa_markout_nao_significativo(sim):
    m = breaks.markouts(sim, horizons=(5,))
    assert (m["pvalor"] > 0.01).all()


def test_limite_de_inventario_gera_cotacao_de_um_lado():
    tabela, _ = breaks.skew_shape(simulate_mm(0.5, P, seed=6, inventory_cap=3))
    assert tabela["so_ask"].iloc[-1] > 0.9


def _write_fake_recording(folder, panel, user="0xabc", coin="TST", with_events=False):
    folder.mkdir(parents=True, exist_ok=True)
    t_ms = panel.index.as_unit("ms").asi8
    lag = 150
    (folder / "alvo.json").write_text(json.dumps({"user": user, "coin": coin, "inicio_ms": int(t_ms[0])}))
    with open(folder / "bbo.jsonl", "w") as f:
        for t, m in zip(t_ms, panel["mid"]):
            bbo = [{"px": str(m - 0.005), "sz": "1", "n": 1}, {"px": str(m + 0.005), "sz": "1", "n": 1}]
            f.write(json.dumps({"t": int(t + lag), "data": {"coin": coin, "time": int(t), "bbo": bbo}}) + "\n")
    with open(folder / "rest_open_orders.jsonl", "w") as f:
        for i in range(0, len(panel), 3):
            r = panel.iloc[i]
            orders = []
            if not np.isnan(r["bid"]):
                orders.append({"coin": coin, "side": "B", "limitPx": str(r["bid"]), "sz": "1.0", "oid": 1})
            if not np.isnan(r["ask"]):
                orders.append({"coin": coin, "side": "A", "limitPx": str(r["ask"]), "sz": "1.0", "oid": 2})
            f.write(json.dumps({"t": int(t_ms[i] + lag), "data": orders}) + "\n")
    with open(folder / "rest_clearinghouse.jsonl", "w") as f:
        state = {"assetPositions": [{"position": {"coin": coin, "szi": str(panel["q"].iloc[0])}}]}
        f.write(json.dumps({"t": int(t_ms[0]), "data": state}) + "\n")
    fills, tid = [], 0
    for i in np.flatnonzero((panel["fill_bid"] + panel["fill_ask"]).to_numpy() > 0):
        q = panel["q"].iloc[i]
        for side, col, px in (("B", "fill_bid", panel["bid"].iloc[i]), ("A", "fill_ask", panel["ask"].iloc[i])):
            if panel[col].iloc[i]:
                tid += 1
                fills.append({"coin": coin, "side": side, "px": str(px), "sz": "1.0", "startPosition": str(q),
                              "crossed": False, "time": int(t_ms[i] + 500), "tid": tid, "oid": tid})
                q += 1 if side == "B" else -1
    with open(folder / "userFills.jsonl", "w") as f:
        f.write(json.dumps({"t": int(t_ms[0]), "data": {"user": user, "fills": fills}}) + "\n")
    if with_events:
        with open(folder / "orderUpdates.jsonl", "w") as f, open(folder / "rest_open_orders.jsonl", "w") as snaps:
            live, oid = {}, 10
            for i in range(len(panel)):
                r, t, ups = panel.iloc[i], int(t_ms[i]), []
                for side, col in (("B", "bid"), ("A", "ask")):
                    if side in live:
                        ups.append({"order": live.pop(side), "status": "canceled", "statusTimestamp": t})
                    if not np.isnan(r[col]):
                        oid += 1
                        live[side] = {"coin": coin, "side": side, "limitPx": str(r[col]), "sz": "1.0", "oid": oid}
                        ups.append({"order": live[side], "status": "open", "statusTimestamp": t})
                f.write(json.dumps({"t": t + lag, "data": ups}) + "\n")
                if i % 30 == 0:
                    snaps.write(json.dumps({"t": t + lag + 400, "data": list(live.values())}) + "\n")
    with open(folder / "bbo.jsonl", "rb") as src, gzip.open(folder / "bbo.jsonl.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    (folder / "bbo.jsonl").unlink()


def test_dataset_reconstroi_painel_de_uma_gravacao(tmp_path):
    sim = simulate_mm(0.5, P, seconds=2 * 3600, seed=9)
    _write_fake_recording(tmp_path / "rec", sim)
    panel, info = build_panel(tmp_path / "rec")
    assert info["fonte_cotacoes"] == "REST a cada 3s"
    assert panel["fill_bid"].sum() == sim["fill_bid"].reindex(panel.index).sum()
    joined = panel.join(sim[["q"]], rsuffix="_sim").dropna()
    assert (joined["q"] == joined["q_sim"]).mean() > 0.99
    assert estimate_gamma(panel).gamma_skew == pytest.approx(0.5, rel=0.35)


def test_dataset_com_eventos_de_ordens(tmp_path):
    sim = simulate_mm(0.5, P, seconds=2 * 3600, seed=9)
    _write_fake_recording(tmp_path / "rec", sim, with_events=True)
    panel, info = build_panel(tmp_path / "rec")
    assert info["fonte_cotacoes"] == "orderUpdates"
    assert info["fracao_dois_lados"] > 0.99
    assert estimate_gamma(panel).gamma_skew == pytest.approx(0.5, rel=0.2)


def test_gamma_vs_vol_sem_variacao_de_sigma_e_inconclusivo():
    gauge = rolling_gauge(simulate_mm(0.5, P, seconds=70 * 60, seed=21), window="10min", step="2min", min_rows=300)
    assert not breaks.gamma_vs_vol(gauge)["conclusivo"]


@pytest.mark.parametrize("gamma_sobe_com_sigma, esperado", [(False, 0.0), (True, 2.0)])
def test_gamma_vs_vol_recupera_elasticidade(gamma_sobe_com_sigma, esperado):
    n = 3 * 3600
    sig = 0.02 * np.exp(np.linspace(np.log(0.5), np.log(2.0), n))
    gam = 0.5 * (sig / 0.02) ** 2 if gamma_sobe_com_sigma else 0.5
    gauge = rolling_gauge(simulate_mm(gam, P, seconds=n, sigma_path=sig, seed=3), window="10min", step="5min",
                          min_rows=300)
    r = breaks.gamma_vs_vol(gauge, overlap=2)
    assert r["conclusivo"]
    assert r["elasticidade_gamma"] == pytest.approx(esperado, abs=0.6)
