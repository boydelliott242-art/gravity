"""Round 9e — tail guards for the leak-free morning model (D2). Pre-registered in
data/research/r9_selection_rule.txt (addendum 2). Uses the saved OOS predictions
(oos_r9_D2_no_counts_lf.npy) and the leak-free research frame. Output: r9e.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_premarket as R  # noqa: E402
from gravity import capacity as CAP, model as M  # noqa: E402
R.LEAK_FREE = True

sub = R.build_sub()
sub["spr"] = CAP.spread_used_vec(sub["spread_ar"].to_numpy(float), sub["close"].to_numpy(float), sub["dvol20"].to_numpy(float))
sub = R.add_ext(sub)
sub["pub"] = (~M.ssr_next(sub)) & (sub["dvol20"].to_numpy() >= M.LIQ_FLOOR)
p = np.load(R.OUT / "oos_r9_D2_no_counts_lf.npy")
pa = np.load(R.OUT / "oos_r9_A_m0.npy")
gap, ehi = sub["gap_ext"].to_numpy(float), sub["ext_hi"].to_numpy(float)
guards = {"none": np.zeros(len(sub), bool), "G1_gap20": gap >= 0.20, "G2_gap30": gap >= 0.30,
          "G3_gap50": gap >= 0.50, "G4_exthi40": ehi >= 0.40}
res = {}
for g, block in guards.items():
    pg = np.where(block, np.nan, p)
    res[g] = {}
    for per, pm in (("dev", sub["date"] < R.SPLIT), ("confirm", sub["date"] >= R.SPLIT)):
        t = R.daily_top(sub, np.where(pm.to_numpy(), pg, np.nan))
        s = R.summarize(t)
        s["worst"] = float(t.y_oc.max()); s["p95"] = float(np.percentile(t.y_oc, 95))
        s["blocked_share"] = float(block[pm.to_numpy() & sub["pub"].to_numpy()].mean())
        res[g][per] = s
    print(g, " | ".join(f"{per}: hit {res[g][per]['hit']:.3f} mean {res[g][per]['mean_oc']:+.4f} sqz {res[g][per]['squeeze']:.3f} worst {res[g][per]['worst']:+.2f} net10k {res[g][per]['net_10k']:+.4f}" for per in ("dev", "confirm")), flush=True)
for per, pm in (("dev", sub["date"] < R.SPLIT), ("confirm", sub["date"] >= R.SPLIT)):
    t = R.daily_top(sub, np.where(pm.to_numpy(), pa, np.nan))
    print("A_m0", per, f"hit {t.y_dump.mean():.3f} mean {t.y_oc.mean():+.4f} sqz {(t.y_oh >= .2).mean():.3f} worst {t.y_oc.max():+.2f}", flush=True)
(R.OUT / "r9e.json").write_text(json.dumps(res, indent=1, default=float))
print("DONE", flush=True)
