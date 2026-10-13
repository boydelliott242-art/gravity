"""Real bid/ask (and top-of-book size) for every "$30K-ready" name, several
times during one regular session — to calibrate the cost model for liquid
names (the live job only quotes the board). Waits on the wall clock (survives
Mac sleep), so it can be started the evening before.

usage: quote_sampler.py [YYYY-MM-DD]   (default: the next session)
Output: data/spreads_sample/<session>.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gravity import capacity as CAP, cli, model as M, net  # noqa: E402
from gravity.util import ET, now_et, num  # noqa: E402

OUT = ROOT / "data" / "spreads_sample"
TIMES = ("09:50", "10:45", "11:45", "13:00", "14:15", "15:20")


def quote(sym):
    try:
        d = net.get_json(f"https://api.nasdaq.com/api/quote/{sym}/info", headers=net.NASDAQ_HEADERS,
                         params={"assetclass": "stocks"}, timeout=15.0, retries=1)
        data = d["data"]; p = data["primaryData"] or {}
    except Exception:  # noqa: BLE001
        return None
    if str(data.get("marketStatus") or "") not in ("Market Open", "Open"):
        return None
    bid, ask = num(p.get("bidPrice")), num(p.get("askPrice"))
    if not bid or not ask or ask <= bid:
        return None
    return {"bid": bid, "ask": ask, "bid_size": num(p.get("bidSize")), "ask_size": num(p.get("askSize")),
            "last": num(p.get("lastSalePrice")), "spread": (ask - bid) / ((ask + bid) / 2)}


def names():
    st = cli.load_state()
    lat = st["latest"]
    lat = lat[pd.to_datetime(lat["date"]) == pd.to_datetime(lat["date"]).max()].copy()
    close = pd.to_numeric(lat["close"], errors="coerce").to_numpy(float)
    dv20 = pd.to_numeric(lat["dvol20"], errors="coerce").to_numpy(float)
    dv1 = close * pd.to_numeric(lat["volume"], errors="coerce").to_numpy(float)
    cap = CAP.market_capacity(CAP.exp_dvol(dv20, dv1), pd.to_numeric(lat["vol20"], errors="coerce").to_numpy(float))
    ok = M.publishable(lat) & (close >= 1)
    ready = list(lat["symbol"].to_numpy()[ok & (cap >= 30_000)])
    edge = list(lat["symbol"].to_numpy()[ok & (cap >= 10_000) & (cap < 30_000)])
    meta = {s: {"cap": float(c), "dvol20": float(d), "close": float(p)} for s, c, d, p in zip(lat["symbol"], cap, dv20, close)}
    return ready, edge, meta


def main():
    session = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
    ready, edge, meta = names()
    syms = ready + edge
    print(time.strftime("%H:%M:%S"), f"{len(ready)} ready + {len(edge)} edge names", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for t in TIMES:
        hh, mm = map(int, t.split(":"))
        while True:
            now = now_et()
            if session is None:
                session = now.date() if now.time() < datetime.min.time().replace(hour=16) else None
            if session is not None:
                target = datetime.combine(session, datetime.min.time(), tzinfo=ET).replace(hour=hh, minute=mm)
                if now >= target:
                    break
            time.sleep(30)
        if (now_et() - target).total_seconds() > 1800:
            print("skip", t, "(woke too late)", flush=True)
            continue
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=5) as ex:
            res = dict(zip(syms, ex.map(quote, syms)))
        stamp = now_et().isoformat(timespec="seconds")
        n = 0
        with open(OUT / f"{session.isoformat()}.jsonl", "a") as f:
            for s, q in res.items():
                if q:
                    n += 1
                    f.write(json.dumps({"ts": stamp, "symbol": s, "ready": s in ready, **meta.get(s, {}), **q}) + "\n")
        print(time.strftime("%H:%M:%S"), t, f"{n}/{len(syms)} quotes in {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
