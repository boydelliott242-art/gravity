"""Round 9b — re-score the saved R9 out-of-sample predictions with the
CORRECTED cost model (capacity.spread_used: the band cap, not its midpoint,
when the spread estimator can't read a spread; 6% tier under $250K/day), and
test one exit rule found in the R9 timing study:

  "10:30 check": short at the open; if the price at 10:30 is ABOVE the open,
  cover at 10:30; otherwise hold to the close.

Rows: the published #1 per day (no SSR, ≥ $300K/day), dev < 2025-11-01 /
confirm. Hourly-path rows only for the exit rule (Yahoo hourly agrees with the
daily bars). Output: data/research/r9b.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_premarket as R  # noqa: E402
from gravity import capacity as CAP, model as M  # noqa: E402

OUT = R.OUT
LEAK_FREE = "--leak-free" in sys.argv
VARIANTS = (["A_m0", "Bp_m1_official", "B_m1_live_lf", "C_m1_honest_lf", "D_ext_shape_lf", "D2_no_counts_lf"] if LEAK_FREE else
            ["A_m0", "Bp_m1_official", "B_m1_live", "C_m1_honest", "D_ext_shape", "E_overnight", "F_overnight_only"])
R.LEAK_FREE = LEAK_FREE


def main():
    sub = R.build_sub()
    sub = R.add_ext(sub)
    sub["pub"] = (~M.ssr_next(sub)) & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR)
    sub["spr"] = CAP.spread_used_vec(sub["spread_ar"].to_numpy(float), sub["close"].to_numpy(float), sub["dvol20"].to_numpy(float))
    preds = {k: np.load(OUT / f"oos_r9_{k}.npy") for k in VARIANTS}
    has_ext = np.isfinite(sub["gap_ext"].to_numpy(float))
    bkey = "B_m1_live_lf" if LEAK_FREE else "B_m1_live"
    preds[bkey] = np.where(has_ext, preds[bkey], preds["A_m0"])
    if not LEAK_FREE:
        for k in ("C_m1_honest", "D_ext_shape", "E_overnight"):
            preds[k + "_mix"] = np.where(has_ext, preds[k], preds["A_m0"])
    res = {"cost_model": "spread = estimate clipped to [tick, cap]; cap when unreadable; cap 6% <$250K/day, 3% <$1M, 2% <$5M, 1% above",
           "variants": {}, "exit_rule": {}, "baskets": {}}
    for k, p in preds.items():
        res["variants"][k] = {}
        for per, pm in (("dev", sub["date"] < R.SPLIT), ("confirm", sub["date"] >= R.SPLIT)):
            t = R.daily_top(sub, np.where(pm.to_numpy(), p, np.nan))
            res["variants"][k][per] = R.summarize(t)
            # exit rule on hourly-path rows
            tp = t[t.path_ok.astype(bool) & t.h_open.notna() & t.c1030.notna() & t.close_h.notna()]
            o, c10, cl = (tp[c].to_numpy(float) for c in ("h_open", "c1030", "close_h"))
            cost = np.asarray(CAP.round_trip_cost(10_000, tp.dvx.to_numpy(float), tp.vol20.to_numpy(float), tp.spr.to_numpy(float)), float)
            hold = 1 - cl / o
            rule = np.where(c10 > o, 1 - c10 / o, 1 - cl / o)
            res["exit_rule"].setdefault(k, {})[per] = {
                "n": int(len(tp)),
                "hold_mean": float(hold.mean()), "hold_net10k": float((hold - cost).mean()), "hold_win": float((hold - cost > 0).mean()),
                "hold_worst": float(hold.min()) if len(hold) else None,
                "rule_mean": float(rule.mean()), "rule_net10k": float((rule - cost).mean()), "rule_win": float((rule - cost > 0).mean()),
                "rule_worst": float(rule.min()) if len(rule) else None,
                "share_covered_1030": float((c10 > o).mean()),
            }
        if k in ("D_ext_shape", "E_overnight", "B_m1_live", "A_m0", "D2_no_counts_lf", "B_m1_live_lf"):
            res["baskets"][k] = R.baskets(sub, p)
        v = res["variants"][k]
        print(f"{k:20s} dev hit {v['dev']['hit']:.3f} mean {v['dev']['mean_oc']:+.4f} net10k {v['dev']['net_10k']:+.4f} net50k {v['dev']['net_50k']:+.4f} | "
              f"conf hit {v['confirm']['hit']:.3f} mean {v['confirm']['mean_oc']:+.4f} net10k {v['confirm']['net_10k']:+.4f} net50k {v['confirm']['net_50k']:+.4f}", flush=True)
        er = res["exit_rule"][k]
        print(f"   exit rule: dev hold {er['dev']['hold_net10k']:+.4f} → rule {er['dev']['rule_net10k']:+.4f} (worst {er['dev']['hold_worst']:+.2f}→{er['dev']['rule_worst']:+.2f}) | "
              f"conf hold {er['confirm']['hold_net10k']:+.4f} → rule {er['confirm']['rule_net10k']:+.4f} (worst {er['confirm']['hold_worst']:+.2f}→{er['confirm']['rule_worst']:+.2f})", flush=True)
    (OUT / ("r9d.json" if LEAK_FREE else "r9b.json")).write_text(json.dumps(res, indent=1, default=float))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
