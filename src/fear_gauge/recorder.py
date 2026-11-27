from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import time
from pathlib import Path

from .hl_api import WS_URL, Info

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class Sink:
    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        self.files: dict[str, object] = {}
        self.counts: dict[str, int] = {}
        self.last_flush = time.time()

    def write(self, name: str, payload) -> None:
        if name not in self.files:
            self.files[name] = gzip.open(self.folder / f"{name}.jsonl.gz", "at", encoding="utf-8", compresslevel=6)
        self.files[name].write(json.dumps({"t": int(time.time() * 1000), "data": payload}) + "\n")
        self.counts[name] = self.counts.get(name, 0) + 1
        if time.time() - self.last_flush > 5:
            for f in self.files.values():
                f.flush()
            self.last_flush = time.time()

    def close(self) -> None:
        for f in self.files.values():
            f.close()


def log(msg: str, sink: Sink | None = None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if sink:
        with open(sink.folder / "gravacao_log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")


async def stream(subscriptions: list[dict], sink: Sink, until: float, on_message=None, transform=None) -> None:
    import websockets

    while time.time() < until:
        try:
            async with websockets.connect(WS_URL, max_size=None, ping_interval=None, open_timeout=30) as ws:
                for s in subscriptions:
                    await ws.send(json.dumps({"method": "subscribe", "subscription": s}))
                log(f"websocket conectado, {len(subscriptions)} assinaturas", sink)
                last_ping = time.time()
                while time.time() < until:
                    if time.time() - last_ping > 30:
                        await ws.send(json.dumps({"method": "ping"}))
                        last_ping = time.time()
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    except asyncio.TimeoutError:
                        continue
                    msg = json.loads(raw)
                    ch = msg.get("channel")
                    if ch == "pong":
                        continue
                    if ch == "subscriptionResponse":
                        sink.write("assinaturas", msg.get("data"))
                        continue
                    if ch == "error":
                        log(f"erro do servidor: {msg.get('data')}", sink)
                        sink.write("erros", msg.get("data"))
                        continue
                    data = transform(ch, msg.get("data")) if transform else msg.get("data")
                    if data is None:
                        continue
                    sink.write(ch, data)
                    if on_message:
                        on_message(ch, msg.get("data"))
        except Exception as e:
            log(f"conexão caiu ({type(e).__name__}: {e}), reconectando em 5s", sink)
            await asyncio.sleep(5)


async def poll_rest(user: str, coin: str, sink: Sink, until: float, orders_every: float = 30.0,
                    state_every: float = 60.0) -> None:
    info = Info(budget_per_min=700)
    last_state = 0.0
    while time.time() < until:
        t0 = time.time()
        try:
            orders = await asyncio.to_thread(info.open_orders, user)
            sink.write("rest_open_orders", [o for o in orders if o.get("coin") == coin])
            if t0 - last_state > state_every:
                sink.write("rest_clearinghouse", await asyncio.to_thread(info.clearinghouse, user))
                last_state = t0
        except Exception as e:
            log(f"REST falhou ({type(e).__name__}: {e})", sink)
        await asyncio.sleep(max(0.0, orders_every - (time.time() - t0)))


async def heartbeat(sink: Sink, until: float, every: float = 600) -> None:
    while time.time() < until:
        await asyncio.sleep(min(every, max(1.0, until - time.time())))
        resumo = ", ".join(f"{k}={v}" for k, v in sorted(sink.counts.items()))
        log(f"mensagens gravadas: {resumo}", sink)


def record_target(user: str, coin: str, hours: float, out: Path | None = None) -> Path:
    user = user.lower()
    out = out or DATA_DIR / "gravacoes" / f"{coin}_{user[:10]}_{time.strftime('%Y%m%d_%H%M')}"
    sink = Sink(out)
    (out / "alvo.json").write_text(json.dumps({"user": user, "coin": coin, "inicio_ms": int(time.time() * 1000),
                                               "horas": hours}, indent=2))
    subs = [{"type": "bbo", "coin": coin}, {"type": "trades", "coin": coin},
            {"type": "activeAssetCtx", "coin": coin}, {"type": "l2Book", "coin": coin},
            {"type": "orderUpdates", "user": user}, {"type": "userFills", "user": user}]

    def only_coin(ch, data):
        # O market maker cota centenas de ativos; guardar todas as ordens geraria dezenas de GB por dia.
        if ch == "orderUpdates" and isinstance(data, list):
            data = [u for u in data if u.get("order", {}).get("coin") == coin]
            return data or None
        return data

    until = time.time() + hours * 3600
    log(f"gravando {coin} de {user} por {hours}h em {out}", sink)

    async def main():
        await asyncio.gather(stream(subs, sink, until, transform=only_coin), poll_rest(user, coin, sink, until),
                             heartbeat(sink, until))

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("interrompido pelo usuário", sink)
    finally:
        sink.close()
        finalize(out)
    return out


def finalize(out: Path) -> None:
    alvo = json.loads((out / "alvo.json").read_text())
    try:
        fills = Info().user_fills_by_time(alvo["user"], alvo["inicio_ms"] - 3600_000)
        (out / "rest_fills.json").write_text(json.dumps(fills))
        log(f"{len(fills)} fills baixados pela API no final", None)
    except Exception as e:
        log(f"não consegui baixar os fills finais ({e})", None)
    alvo["fim_ms"] = int(time.time() * 1000)
    (out / "alvo.json").write_text(json.dumps(alvo, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Grava cotações, fills e inventário de um market maker na Hyperliquid")
    ap.add_argument("--horas", type=float, default=24)
    ap.add_argument("--user")
    ap.add_argument("--coin")
    args = ap.parse_args()
    user, coin = args.user, args.coin
    if not user or not coin:
        alvo = json.loads((DATA_DIR / "alvo.json").read_text())
        user, coin = user or alvo["user"], coin or alvo["coin"]
    record_target(user, coin, args.horas)


if __name__ == "__main__":
    main()
