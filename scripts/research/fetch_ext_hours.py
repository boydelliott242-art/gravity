"""Fetch Yahoo hourly bars WITH extended hours (prepost) for every symbol in
the research lab and reduce them to per-session aggregates.

For each (symbol, session date D) it stores what was knowable at ~9:00 ET on D:
  ah_last   last after-hours trade on the previous session (16:00–20:00)
  pm_last   last pre-market trade on D before 9:00 ET (bars 4:00–8:00)
  ext_last  pm_last if any pre-market trade, else ah_last
  ext_high / ext_low   extremes of those extended-hours bars
  n_ext     number of extended-hours hourly bars that had trades (activity proxy:
            Yahoo reports 0 volume for extended hours, but only emits a bar
            when something traded)
and the regular-session hourly path on D (for the entry/exit timing study):
  h_open, c1030, c1130, c1230, c1330, c1430, close_h, hi1030, hi1230, hi_all,
  lo1030, lo_all
Output: data/research/ext_hours.pkl
"""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "research"
PART = OUT / "ext_parts"
PART.mkdir(parents=True, exist_ok=True)
ET = "America/New_York"
_today = pd.Timestamp.now(tz=ET).normalize().tz_localize(None)
START = (_today - pd.Timedelta(days=727)).strftime("%Y-%m-%d")   # Yahoo: hourly only within the last 730 days
END = (_today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def reduce(sym: str, h: pd.DataFrame) -> pd.DataFrame:
    h = h.dropna(subset=["Close"])
    h = h[h["Close"] > 0]
    if h.empty:
        return pd.DataFrame()
    idx = h.index.tz_convert(ET)
    day = pd.DatetimeIndex(idx.date)
    mins = idx.hour * 60 + idx.minute
    df = pd.DataFrame({"day": day, "m": mins, "o": h["Open"].to_numpy(float), "h": h["High"].to_numpy(float),
                       "l": h["Low"].to_numpy(float), "c": h["Close"].to_numpy(float)})
    reg = df[(df.m >= 570) & (df.m < 960)]
    pm = df[(df.m >= 240) & (df.m < 540)]          # bars starting 4:00 … 8:00 → trades before 9:00
    ah = df[(df.m >= 960) & (df.m < 1200)]          # 16:00 … 19:00
    rows = {}
    g_reg = {d: x for d, x in reg.groupby("day")}
    sessions = sorted(g_reg)
    if not sessions:
        return pd.DataFrame()
    g_pm = {d: x for d, x in pm.groupby("day")}
    g_ah = {d: x for d, x in ah.groupby("day")}
    prev = None
    for d in sessions:
        r = {"symbol": sym, "date": d}
        a = g_ah.get(prev) if prev is not None else None
        p = g_pm.get(d)
        r["ah_last"] = float(a["c"].iloc[-1]) if a is not None and len(a) else np.nan
        r["pm_last"] = float(p["c"].iloc[-1]) if p is not None and len(p) else np.nan
        ext = pd.concat([x for x in (a, p) if x is not None and len(x)]) if ((a is not None and len(a)) or (p is not None and len(p))) else None
        r["ext_last"] = r["pm_last"] if np.isfinite(r["pm_last"]) else r["ah_last"]
        r["ext_high"] = float(ext["h"].max()) if ext is not None else np.nan
        r["ext_low"] = float(ext["l"].min()) if ext is not None else np.nan
        r["n_ext"] = float(len(ext)) if ext is not None else 0.0
        r["n_pm"] = float(len(p)) if p is not None else 0.0
        g = g_reg[d].sort_values("m")
        r["h_open"] = float(g["o"].iloc[0]) if g["m"].iloc[0] == 570 else np.nan
        for lab_, end in (("c1030", 630), ("c1130", 690), ("c1230", 750), ("c1330", 810), ("c1430", 870)):
            x = g[g.m < end]
            r[lab_] = float(x["c"].iloc[-1]) if len(x) and x["m"].iloc[-1] == end - 60 else np.nan
        r["close_h"] = float(g["c"].iloc[-1]) if g["m"].iloc[-1] >= 930 else np.nan
        x = g[g.m < 630]; r["hi1030"] = float(x["h"].max()) if len(x) else np.nan; r["lo1030"] = float(x["l"].min()) if len(x) else np.nan
        x = g[g.m < 750]; r["hi1230"] = float(x["h"].max()) if len(x) else np.nan
        r["hi_all"] = float(g["h"].max()); r["lo_all"] = float(g["l"].min())
        r["n_reg"] = float(len(g))
        rows[d] = r
        prev = d
    return pd.DataFrame(rows.values())


def main():
    syms = sorted(pd.read_pickle(OUT / "wide_lab.pkl", )["symbol"].unique())
    print(f"{len(syms)} symbols", flush=True)
    B = 40
    t0 = time.time()
    for bi in range(0, len(syms), B):
        part = PART / f"p{bi:05d}.pkl"
        if part.exists():
            continue
        batch = syms[bi:bi + B]
        ysyms = [s.replace(".", "-").replace("/", "-") for s in batch]
        for attempt in range(4):
            try:
                df = yf.download(ysyms, start=START, end=END, interval="60m", prepost=True, group_by="ticker",
                                 threads=True, progress=False, auto_adjust=False)
                break
            except Exception as e:  # rate limit etc.
                print("retry", bi, e, flush=True); time.sleep(30 * (attempt + 1))
        else:
            continue
        outs = []
        for s, ys in zip(batch, ysyms):
            try:
                h = df[ys] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                continue
            try:
                r = reduce(s, h)
            except Exception as e:
                print("reduce fail", s, e, flush=True); continue
            if len(r):
                outs.append(r)
        pd.concat(outs).to_pickle(part) if outs else pd.DataFrame().to_pickle(part)
        print(f"{bi + len(batch)}/{len(syms)}  {sum(len(o) for o in outs)} rows  {time.time()-t0:.0f}s", flush=True)
        time.sleep(1.0)
    got = set()
    for p in sorted(PART.glob("p*.pkl")):
        x = pd.read_pickle(p)
        if len(x):
            got |= set(x["symbol"].unique())
    miss = [s for s in syms if s not in got]
    rp = PART / "retry.pkl"
    if miss and not rp.exists():
        print(f"retrying {len(miss)} symbols with explicit dates", flush=True)
        outs = []
        for bi in range(0, len(miss), B):
            batch = miss[bi:bi + B]
            ysyms = [s.replace(".", "-").replace("/", "-") for s in batch]
            try:
                df = yf.download(ysyms, start=START, end=END, interval="60m", prepost=True, group_by="ticker",
                                 threads=True, progress=False, auto_adjust=False)
            except Exception as e:
                print("retry fail", e, flush=True); continue
            for s, ys in zip(batch, ysyms):
                try:
                    r = reduce(s, df[ys] if isinstance(df.columns, pd.MultiIndex) else df)
                except Exception:
                    continue
                if len(r):
                    outs.append(r)
            time.sleep(1.0)
        (pd.concat(outs) if outs else pd.DataFrame()).to_pickle(rp)
        print(f"retry recovered {len(set(pd.concat(outs).symbol)) if outs else 0}", flush=True)
    allp = [pd.read_pickle(p) for p in sorted(PART.glob("*.pkl"))]
    allp = [p for p in allp if len(p)]
    ext = pd.concat(allp, ignore_index=True)
    ext["date"] = pd.to_datetime(ext["date"])
    ext.to_pickle(OUT / "ext_hours.pkl")
    print("DONE", ext.shape, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
