"""R11c — stage 2 of R11 re-tested without the look-ahead found in review.

The R11 stage-2 test (L4 + overnight filing counts) took filing events from
data/state/context.pkl, which only holds TODAY's <= $2B universe: names that
were <= $2B then but grew later had on_* = NaN — a marker of future growth.
Here the counts come from production's own code (sec.events_for_universe for
every symbol in the point-in-time panel + features.overnight_counts, 8:00 ET
cutoff), exactly as training builds them. Same pre-registered threshold:
adopt only if DEV and CONFIRM ALL100 net each improve by >= 0.05 pt over L4.

Also reported (fidelity, no decision): the morning run rescores only the top 450
$30K-ready names (by the close-only ranker) with the morning ranker; the rest
keep close-only scores. Output: data/research/r11c.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r11_liquid as X  # noqa: E402
from gravity import features as F  # noqa: E402
from gravity.sources import sec  # noqa: E402

OUT = X.OUT
SHORTLIST = 450


def hybrid(sub, s_m0, s_m1):
    """Production's morning scoring: per session, the top SHORTLIST L names by the
    close-only score get the morning score; everyone else keeps the close-only one."""
    L = sub["L"].to_numpy()
    d = pd.DataFrame({"date": sub["date"].to_numpy()[L], "s0": np.asarray(s_m0)[L], "s1": np.asarray(s_m1)[L]})
    rk = d.groupby("date")["s0"].rank(ascending=False, method="first").to_numpy()
    out = np.full(len(sub), np.nan)
    out[np.flatnonzero(L)] = np.where(rk <= SHORTLIST, d["s1"].to_numpy(), d["s0"].to_numpy())
    return out


def main():
    t0 = time.time()
    sub = X.prep()
    syms = sorted(sub["symbol"].unique())
    events = sec.events_for_universe(syms)
    X.log(f"events for {len(events)} of {len(syms)} symbols ({time.time() - t0:.0f}s)")
    on = F.overnight_counts(sub["symbol"].to_numpy(), sub["date"], sub["D"], events, cutoff_hour=8)
    for c in F.ON_FEATURES:
        sub[c] = on[c]
    L = sub["L"].to_numpy()
    X.log(f"on_filings finite in L: {np.isfinite(sub['on_filings'].to_numpy()[L]).mean():.3f} (context-file version: see r11.json)")
    f = OUT / "oos_r11c_L4_on.npy"
    if f.exists():
        p_on = np.load(f)
    else:
        p_on = X.walk(sub, list(F.FEATURES) + ["gap_open"] + list(F.ON_FEATURES), L, "reg", "y_pct")
        np.save(f, p_on)
    p4 = np.load(OUT / "oos_r11_L4.npy")
    p4m0 = np.load(OUT / "oos_r11_L4_m0.npy")
    res = {"variants": {"L4": X.evaluate(sub, p4), "L4_on_allsym": X.evaluate(sub, p_on),
                        "L4_m0": X.evaluate(sub, p4m0),
                        "hybrid_L4": X.evaluate(sub, hybrid(sub, p4m0, p4)),
                        "hybrid_L4_on_allsym": X.evaluate(sub, hybrid(sub, p4m0, p_on))},
           "paired_on_vs_L4": X.paired(sub, p_on, p4)}
    for k, v in res["variants"].items():
        X.line(k, v)
    V = res["variants"]
    adopt = all(V["L4_on_allsym"][per]["1-100"]["net30"] - V["L4"][per]["1-100"]["net30"] >= 0.0005 for per in ("dev", "confirm"))
    res["decision"] = {"overnight_filings": bool(adopt)}
    X.log(f"RULE (R11c): overnight filings adopted {adopt}; paired {json.dumps(res['paired_on_vs_L4'])}")
    (OUT / "r11c.json").write_text(json.dumps(res, indent=1, default=float))
    X.log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
