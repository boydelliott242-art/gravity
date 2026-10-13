"""R11 cost anatomy for the adopted $30K board (review follow-up): how often the
spread used is the liquidity-tier cap and why (estimator unreadable vs read
wider and clipped), and the board's net at 0.5x / 1x / 2x the assumed spread.
Reporting only — no decision depends on it. Output: data/research/r11_costs.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r11_liquid as X  # noqa: E402
from gravity import capacity as CAP  # noqa: E402

OUT = X.OUT
r11 = json.loads((OUT / "r11.json").read_text())
won = r11["decision"]["adopted"]
sub = X.prep()
# the board as production builds it each morning (R11c): close-only ranker for every $30K-ready name, the morning
# ranker (+ overnight filings if R11c adopted them) for the top 450 of those
import r11c_overnight as C  # noqa: E402
r11c = json.loads((OUT / "r11c.json").read_text()) if (OUT / "r11c.json").exists() else {}
on = bool(r11c.get("decision", {}).get("overnight_filings"))
p_m1 = np.load(OUT / ("oos_r11c_L4_on.npy" if on else f"oos_r11_{won}.npy"))
p = C.hybrid(sub, np.load(OUT / f"oos_r11_{won}_m0.npy"), p_m1)
b = X.board(sub, p, sub["L"].to_numpy())
idx = b.index
est = sub.loc[idx, "spread_ar"].to_numpy(float)
dv = sub.loc[idx, "dvol20"].to_numpy(float)
spr = sub.loc[idx, "spr"].to_numpy(float)
cap = np.array([CAP.spread_cap(v) for v in dv])            # the current liquidity-tier caps
unread = ~(np.isfinite(est) & (est >= 0.001))
clipped = ~unread & (est > cap)
dvx, sig = sub.loc[idx, "dvx"].to_numpy(float), sub.loc[idx, "vol20"].to_numpy(float)
y = sub.loc[idx, "y_oc"].to_numpy(float)
res = {"adopted": won + ("+overnight filings" if on else "") + " (production hybrid)", "n": int(len(b)), "share_at_cap": float(np.isclose(spr, cap).mean()),
       "share_unreadable": float(unread.mean()), "share_read_wider_clipped": float(clipped.mean()),
       "median_read_over_cap_when_clipped": float(np.median(est[clipped] / cap[clipped])) if clipped.any() else None,
       "mean_spread_used": float(spr.mean()), "mean_two_way_impact": float(np.mean(2 * CAP.impact(30_000, dvx, sig))),
       "net_by_spread_multiple": {}}
for mlt in (0.5, 1.0, 2.0):
    c = np.asarray(CAP.round_trip_cost(30_000, dvx, sig, mlt * spr), float)
    net = -y - c
    res["net_by_spread_multiple"][str(mlt)] = {per: float(net[(b.date < X.SPLIT).to_numpy() if per == "dev" else (b.date >= X.SPLIT).to_numpy()].mean())
                                               for per in ("dev", "confirm")}
# the old board's #1 (dump odds among publishable names, any capacity): how often could it take $30K at all?
ob = X.board(sub, np.load(OUT / "oos_r10_D3_8.npy"), sub["pub"].to_numpy())
o1 = ob[ob.rk == 1]
res["old_top1_cap30_share"] = {per: float((o1["cap"][(o1.date < X.SPLIT) if per == "dev" else (o1.date >= X.SPLIT)] >= 30_000).mean())
                               for per in ("dev", "confirm")}
(OUT / "r11_costs.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
