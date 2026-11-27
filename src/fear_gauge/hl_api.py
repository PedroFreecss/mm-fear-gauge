from __future__ import annotations

import time
from collections import deque

import requests

INFO_URL = "https://api.hyperliquid.xyz/info"
WS_URL = "wss://api.hyperliquid.xyz/ws"
HLP_VAULT = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"

_WEIGHTS = {"l2Book": 2, "allMids": 2, "clearinghouseState": 2, "orderStatus": 2}
_ITEM_WEIGHTED = {"userFills", "userFillsByTime", "historicalOrders", "recentTrades"}


class Info:
    def __init__(self, budget_per_min: int = 900):
        self.session = requests.Session()
        self.budget = budget_per_min
        self.spent: deque[tuple[float, int]] = deque()

    def _wait(self, weight: int) -> None:
        while True:
            now = time.time()
            while self.spent and now - self.spent[0][0] > 60:
                self.spent.popleft()
            if sum(w for _, w in self.spent) + weight <= self.budget:
                self.spent.append((now, weight))
                return
            time.sleep(1)

    def post(self, body: dict):
        self._wait(_WEIGHTS.get(body["type"], 20))
        for attempt in range(5):
            r = self.session.post(INFO_URL, json=body, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            if body["type"] in _ITEM_WEIGHTED and isinstance(data, list):
                self._wait(len(data) // 20)
            return data
        raise RuntimeError(f"Limite de requisições atingido em {body['type']}")

    def user_fills(self, user: str) -> list[dict]:
        return self.post({"type": "userFills", "user": user})

    def user_fills_by_time(self, user: str, start_ms: int, end_ms: int | None = None) -> list[dict]:
        out, seen, cur = [], set(), start_ms
        while True:
            body = {"type": "userFillsByTime", "user": user, "startTime": cur}
            if end_ms:
                body["endTime"] = end_ms
            batch = self.post(body)
            new = [f for f in batch if (f["tid"], f["oid"]) not in seen]
            if not new:
                return out
            out.extend(new)
            seen.update((f["tid"], f["oid"]) for f in new)
            cur = max(f["time"] for f in batch)
            if len(batch) < 500:
                return out

    def open_orders(self, user: str) -> list[dict]:
        return self.post({"type": "frontendOpenOrders", "user": user})

    def clearinghouse(self, user: str) -> dict:
        return self.post({"type": "clearinghouseState", "user": user})

    def meta(self) -> dict:
        return self.post({"type": "meta"})

    def l2_book(self, coin: str) -> dict:
        return self.post({"type": "l2Book", "coin": coin})


def position_of(state: dict, coin: str) -> float:
    for ap in state.get("assetPositions", []):
        pos = ap.get("position", {})
        if pos.get("coin") == coin:
            return float(pos.get("szi", 0.0))
    return 0.0
