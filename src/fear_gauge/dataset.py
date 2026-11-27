from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hl_api import position_of

_TERMINAL = {"filled", "canceled", "rejected", "marginCanceled", "triggered"}


def read_jsonl(path: Path):
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        opener = lambda: gzip.open(gz, "rt", encoding="utf-8")
    elif path.exists():
        opener = lambda: open(path, encoding="utf-8")
    else:
        return
    with opener() as f:
        try:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        except (EOFError, gzip.BadGzipFile):
            return


def load_mid(folder: Path, coin: str) -> pd.Series:
    rows = []
    for rec in read_jsonl(folder / "bbo.jsonl"):
        d = rec["data"]
        b, a = d.get("bbo", [None, None])
        if d.get("coin") == coin and b and a:
            rows.append((d["time"], (float(b["px"]) + float(a["px"])) / 2))
    s = pd.Series(dict(rows), name="mid").sort_index()
    s.index = pd.to_datetime(s.index, unit="ms")
    return s[~s.index.duplicated(keep="last")]


def load_fills(folder: Path, coin: str, user: str) -> pd.DataFrame:
    fills = []
    for rec in read_jsonl(folder / "userFills.jsonl"):
        d = rec["data"]
        if str(d.get("user", user)).lower() == user:
            fills.extend(d.get("fills", []))
    rest = folder / "rest_fills.json"
    if rest.exists():
        fills.extend(json.loads(rest.read_text()))
    df = pd.DataFrame(fills)
    if df.empty:
        return df
    df = df[df["coin"] == coin].drop_duplicates(subset=["tid", "oid"])
    for c in ("px", "sz", "startPosition"):
        df[c] = df[c].astype(float)
    df["sinal"] = np.where(df["side"] == "B", 1.0, -1.0)
    df["q_depois"] = df["startPosition"] + df["sinal"] * df["sz"]
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def _best(orders: list[dict], coin: str) -> tuple[float, float, float, float]:
    bids = [(float(o["limitPx"]), float(o["sz"])) for o in orders
            if o.get("coin") == coin and o.get("side") == "B" and not o.get("isTrigger")]
    asks = [(float(o["limitPx"]), float(o["sz"])) for o in orders
            if o.get("coin") == coin and o.get("side") == "A" and not o.get("isTrigger")]
    b = max(bids) if bids else (np.nan, np.nan)
    a = min(asks) if asks else (np.nan, np.nan)
    return b[0], a[0], b[1], a[1]


def clock_offset_ms(folder: Path) -> float:
    gaps = [rec["t"] - rec["data"]["time"] for i, rec in zip(range(5000), read_jsonl(folder / "bbo.jsonl"))
            if isinstance(rec.get("data"), dict) and "time" in rec["data"]]
    return float(np.median(gaps)) if gaps else 0.0


def quotes_from_rest(folder: Path, coin: str) -> pd.DataFrame:
    # O "t" das consultas REST vem do relógio do PC; o desvio é estimado comparando com o horário da corretora no bbo.
    off = clock_offset_ms(folder)
    rows = [(rec["t"] - off, *_best(rec["data"], coin)) for rec in read_jsonl(folder / "rest_open_orders.jsonl")
            if isinstance(rec["data"], list)]
    df = pd.DataFrame(rows, columns=["t", "bid", "ask", "bid_sz", "ask_sz"])
    df.index = pd.to_datetime(df.pop("t").round().astype("int64"), unit="ms")
    return df


def quotes_from_order_updates(folder: Path, coin: str, prune_margin_ms: int = 5000) -> pd.DataFrame:
    off = clock_offset_ms(folder)
    snaps = [(rec["t"] - off, rec["data"]) for rec in read_jsonl(folder / "rest_open_orders.jsonl")
             if isinstance(rec["data"], list)]
    events = []
    for rec in read_jsonl(folder / "orderUpdates.jsonl"):
        for u in rec["data"] if isinstance(rec["data"], list) else []:
            o = u.get("order", {})
            if o.get("coin") == coin:
                events.append((u.get("statusTimestamp", rec["t"] - off), 1, u.get("status"), o))
    if not snaps or len(events) < 100:
        return pd.DataFrame()
    # Retratos REST chegam atrasados: o primeiro inicia o livro e os seguintes só removem ordens que sumiram
    # (um cancelamento anterior ao início do websocket nunca chega como evento).
    t0, first = snaps[0]
    book = {o["oid"]: (o, t0) for o in first if o.get("coin") == coin}
    timeline = events + [(t, 0, "snapshot", orders) for t, orders in snaps[1:]]
    timeline.sort(key=lambda e: (e[0], e[1]))
    rows = []
    for ts, _, status, payload in timeline:
        if status == "snapshot":
            alive = {o["oid"] for o in payload}
            book = {k: v for k, v in book.items() if k in alive or v[1] > ts - prune_margin_ms}
        elif status == "open":
            book[payload["oid"]] = (payload, ts)
        elif status in _TERMINAL or str(status).endswith(("Canceled", "Rejected")):
            book.pop(payload["oid"], None)
        rows.append((ts, *_best([v[0] for v in book.values()], coin)))
    df = pd.DataFrame(rows, columns=["t", "bid", "ask", "bid_sz", "ask_sz"])
    df.index = pd.to_datetime(df.pop("t").round().astype("int64"), unit="ms")
    return df[~df.index.duplicated(keep="last")]


def initial_position(folder: Path, coin: str) -> float:
    for rec in read_jsonl(folder / "rest_clearinghouse.jsonl"):
        return position_of(rec["data"], coin)
    return 0.0


def build_panel(folder: str | Path, freq: str = "1s") -> tuple[pd.DataFrame, dict]:
    folder = Path(folder)
    alvo = json.loads((folder / "alvo.json").read_text())
    user, coin = alvo["user"].lower(), alvo["coin"]
    mid = load_mid(folder, coin)
    fills = load_fills(folder, coin, user)
    q_ev = quotes_from_order_updates(folder, coin)
    fonte = "orderUpdates"
    if q_ev.empty:
        q_ev, fonte = quotes_from_rest(folder, coin), "REST a cada 3s"

    start = max(mid.index[0], q_ev.index[0]).ceil(freq)
    end = min(mid.index[-1], q_ev.index[-1]).floor(freq)
    grid = pd.date_range(start, end, freq=freq)
    panel = pd.DataFrame(index=grid)
    panel["mid"] = mid.reindex(grid, method="ffill")
    quotes = q_ev.reindex(grid, method="ffill")
    panel[["bid", "ask"]] = quotes[["bid", "ask"]]
    if fonte == "orderUpdates":
        panel["cotacao_atual"] = True
    else:
        last_seen = pd.Series(q_ev.index, index=q_ev.index).reindex(grid, method="ffill")
        panel["cotacao_atual"] = (grid - last_seen.to_numpy()) < pd.Timedelta(freq)
    sizes = pd.concat([quotes["bid_sz"], quotes["ask_sz"]]).dropna()
    panel["size"] = float(sizes.median()) if len(sizes) else np.nan

    q0 = initial_position(folder, coin)
    if len(fills):
        q_path = pd.concat([pd.Series([fills["startPosition"].iloc[0]], index=[grid[0] - pd.Timedelta(freq)]),
                            fills.set_index("time")["q_depois"]])
        q_path = q_path[~q_path.index.duplicated(keep="last")].sort_index()
        panel["q"] = q_path.reindex(grid, method="ffill").fillna(q_path.iloc[0])
        maker = fills[~fills["crossed"]]
        b = maker[maker["side"] == "B"].groupby(maker["time"].dt.floor(freq)).size()
        a = maker[maker["side"] == "A"].groupby(maker["time"].dt.floor(freq)).size()
        panel["fill_bid"] = b.reindex(grid, fill_value=0).astype(int)
        panel["fill_ask"] = a.reindex(grid, fill_value=0).astype(int)
    else:
        panel["q"], panel["fill_bid"], panel["fill_ask"] = q0, 0, 0

    for side, sign in (("bid", 1), ("ask", -1)):
        bad = sign * (panel["mid"] - panel[side]) < 0
        panel.loc[bad, side] = np.nan
    panel["cotacao_atual"] = panel["cotacao_atual"].astype(bool)

    info = {"user": user, "coin": coin, "fonte_cotacoes": fonte, "inicio": str(grid[0]), "fim": str(grid[-1]),
            "horas": len(grid) / 3600, "fills_maker": int(panel["fill_bid"].sum() + panel["fill_ask"].sum()),
            "fills_total": int(len(fills)), "posicao_inicial_api": q0,
            "fracao_cotacao_atual": float(panel["cotacao_atual"].mean()),
            "fracao_dois_lados": float((panel["bid"].notna() & panel["ask"].notna())[panel["cotacao_atual"]].mean())}
    return panel.dropna(subset=["mid"]), info


def load_context(folder: str | Path, freq: str = "5min") -> pd.DataFrame:
    rows = []
    for rec in read_jsonl(Path(folder) / "activeAssetCtx.jsonl"):
        ctx = rec["data"].get("ctx", {})
        rows.append({"t": rec["t"], "funding": float(ctx.get("funding", np.nan)),
                     "open_interest": float(ctx.get("openInterest", np.nan)),
                     "mark": float(ctx.get("markPx", np.nan))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.pop("t").round().astype("int64"), unit="ms")
    return df.resample(freq).last()
