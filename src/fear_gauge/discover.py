from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict

import pandas as pd

from .hl_api import HLP_VAULT, Info, position_of
from .recorder import DATA_DIR, Sink, log, stream


def collect_trades(coins: list[str], minutes: float) -> pd.DataFrame:
    rows = []

    def on_message(ch, data):
        if ch == "trades":
            for t in data:
                users = t.get("users") or [None, None]
                rows.append({"coin": t["coin"], "time": t["time"], "px": float(t["px"]), "sz": float(t["sz"]),
                             "side": t["side"], "buyer": users[0], "seller": users[1]})

    sink = Sink(DATA_DIR / "descoberta")
    subs = [{"type": "trades", "coin": c} for c in coins]
    log(f"ouvindo trades de {', '.join(coins)} por {minutes:.0f} minutos", sink)
    asyncio.run(stream(subs, sink, time.time() + minutes * 60, on_message))
    sink.close()
    return pd.DataFrame(rows)


def rank_participants(trades: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    stats = defaultdict(lambda: {"compras": 0, "vendas": 0, "volume_usd": 0.0})
    for r in trades.itertuples():
        notional = r.px * r.sz
        for addr, key in ((r.buyer, "compras"), (r.seller, "vendas")):
            if addr:
                s = stats[(r.coin, addr.lower())]
                s[key] += 1
                s["volume_usd"] += notional
    df = pd.DataFrame([{"coin": c, "user": u, **v} for (c, u), v in stats.items()])
    if df.empty:
        return df
    df["trades"] = df["compras"] + df["vendas"]
    df["equilibrio"] = 1 - (df["compras"] - df["vendas"]).abs() / df["trades"]
    return df.sort_values("trades", ascending=False).groupby("coin").head(top)


def profile(info: Info, user: str, coin: str) -> dict:
    fills = [f for f in info.user_fills(user) if f["coin"] == coin]
    orders = [o for o in info.open_orders(user) if o["coin"] == coin and not o.get("isTrigger")]
    state = info.clearinghouse(user)
    out = {"coin": coin, "user": user, "fills_recentes": len(fills), "hlp": user == HLP_VAULT,
           "ordens_abertas_bid": sum(o["side"] == "B" for o in orders),
           "ordens_abertas_ask": sum(o["side"] == "A" for o in orders),
           "posicao": position_of(state, coin)}
    if fills:
        f = pd.DataFrame(fills)
        horas = max((f["time"].max() - f["time"].min()) / 3.6e6, 1 / 60)
        out.update({"fracao_maker": float((~f["crossed"]).mean()),
                    "fracao_compras": float((f["side"] == "B").mean()),
                    "fills_por_hora": len(f) / horas,
                    "tamanho_mediano": float(f["sz"].astype(float).median())})
    return out


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("fracao_maker", "fracao_compras", "fills_por_hora"):
        if c not in df:
            df[c] = float("nan")
    df["dois_lados"] = (df["ordens_abertas_bid"] > 0) & (df["ordens_abertas_ask"] > 0)
    df["parece_mm"] = (df["fracao_maker"] >= 0.85) & df["dois_lados"] & df["fracao_compras"].between(0.3, 0.7)
    return df.sort_values(["parece_mm", "fills_por_hora"], ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Encontra carteiras com comportamento de market maker")
    ap.add_argument("--coins", default="ETH,SOL,HYPE")
    ap.add_argument("--minutos", type=float, default=30)
    args = ap.parse_args()
    coins = [c.strip() for c in args.coins.split(",")]

    trades = collect_trades(coins, args.minutos)
    trades.to_csv(DATA_DIR / "descoberta" / "trades.csv", index=False)
    log(f"{len(trades)} trades coletados")
    ranked = rank_participants(trades)
    info = Info()
    perfis = []
    for r in ranked.itertuples():
        try:
            perfis.append({**profile(info, r.user, r.coin), "trades_observados": r.trades,
                           "volume_usd_observado": r.volume_usd})
            log(f"perfil {r.coin} {r.user[:10]}: ok")
        except Exception as e:
            log(f"perfil {r.coin} {r.user[:10]} falhou: {e}")
    cand = score(pd.DataFrame(perfis))
    cand.to_csv(DATA_DIR / "candidatos.csv", index=False)
    escolhido = cand[cand["parece_mm"]].head(1)
    if escolhido.empty:
        log("nenhuma carteira passou nos critérios de market maker; veja data/candidatos.csv")
        return
    alvo = {"user": escolhido["user"].iloc[0], "coin": escolhido["coin"].iloc[0]}
    (DATA_DIR / "alvo.json").write_text(json.dumps(alvo, indent=2))
    log(f"alvo escolhido: {alvo}")
    print(cand.head(15).to_string())


if __name__ == "__main__":
    main()
