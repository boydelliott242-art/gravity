"""R13 — "the ONE": the single main-page pick, shortable at $100K+, chosen for
the highest #1 dump rate. Rule: scripts/research/r9_selection_rule.txt (R13).
Output: data/research/r13.json, oos_r13_<name>.npy"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import r11_liquid as X  # noqa: E402
from gravity import capacity as CAP, features as F  # noqa: E402
from gravity.sources import sec  # noqa: E402

OUT = X.OUT
FLOOR = 100_000
log = X.log


def evaluate_one(sub, score, mask, ks=(1, 3)):
    m = np.asarray(mask, bool) & np.isfinite(score)
    d = sub.loc[m, ["date", "y_dump", "y_oc", "y_oh", "c100", "c500", "cap"]].assign(s=np.asarray(score)[m])
    d = d.sort_values(["date", "s"], ascending=[True, False])
    d["rk"] = d.groupby("date").cumcount() + 1
    out = {}
    for per, pm in (("dev", d.date < X.SPLIT), ("confirm", d.date >= X.SPLIT)):
        x = d[pm.to_numpy()]
        r = {}
        for k in ks:
            t = x[x.rk <= k]
            r[f"top{k}"] = {"n": int(len(t)), "hit": float(t.y_dump.mean()), "big10": float((t.y_oc <= -0.10).mean()),
                            "mean_oc": float(t.y_oc.mean()), "median_oc": float(t.y_oc.median()),
                            "squeeze": float((t.y_oh >= 0.20).mean()), "net100k": float((-t.y_oc - t.c100).mean()),
                            "net500k": float((-t.y_oc - t.c500).mean()), "cap_median": float(t.cap.median()),
                            "win100k": float(((-t.y_oc - t.c100) > 0).mean())}
        out[per] = r
    return out


def main():
    t0 = time.time()
    sub = X.prep()
    px = np.where(np.isfinite(sub["price"].to_numpy(float)), sub["price"].to_numpy(float), sub["close"].to_numpy(float))
    U = sub["pub"].to_numpy() & (px >= 1.0) & (sub["cap"].to_numpy(float) >= FLOOR)
    sub["U"] = U
    for q, c in ((100_000, "c100"), (500_000, "c500")):
        sub[c] = np.asarray(CAP.round_trip_cost(q, sub["dvx"].to_numpy(float), sub["vol20"].to_numpy(float), sub["spr"].to_numpy(float)), float)
    sub["y_d10"] = np.where(np.isfinite(sub["y_oc"]), (sub["y_oc"] <= -0.10).astype(float), np.nan)
    L = sub["L"].to_numpy()
    D3 = list(F.FEATURES) + ["gap_open"]
    log(f"U rows {U.sum():,}  per session {pd.Series(U).groupby(sub['date'].to_numpy()).sum().median():.0f}")
    preds = {"O0": np.load(OUT / "oos_r10_D3_8.npy"), "O1": np.load(OUT / "oos_r11_L1.npy")}

    def run(name, feats, train, kind, target, seed=0):
        f = OUT / f"oos_r13_{name}.npy"
        if f.exists():
            preds[name] = np.load(f); log(f"{name}: cached"); return
        t1 = time.time()
        preds[name] = X.walk(sub, feats, train, kind, target, seed=seed)
        np.save(f, preds[name]); log(f"{name}: {time.time() - t1:.0f}s")

    run("O2", D3, U, "cls", "y_dump")
    for sd in (1, 2, 3, 4):
        run(f"O1s{sd}", D3, L, "cls", "y_dump", seed=sd)
    preds["O3"] = np.nanmean(np.vstack([preds["O1"]] + [preds[f"O1s{sd}"] for sd in (1, 2, 3, 4)]), axis=0)
    preds["O4"] = X.rank_avg(sub, [preds["O0"], preds["O1"], np.load(OUT / "oos_r11_L2.npy")])
    # O5: + overnight filing counts (every symbol's history, 8:00 ET cutoff) incl. earnings releases (8-K item 2.02)
    events = sec.events_for_universe(sorted(sub["symbol"].unique()))
    on = F.overnight_counts(sub["symbol"].to_numpy(), sub["date"], sub["D"], events, cutoff_hour=8)
    for c in F.ON_FEATURES:
        sub[c] = on[c]
    earn = {s: [e for e in evs if str(e.get("form") or "").upper().split("/")[0] in ("8-K", "6-K")
                and "2.02" in [str(i) for i in (e.get("items") or [])]] for s, evs in events.items()}
    sub["on_earn"] = F.overnight_counts(sub["symbol"].to_numpy(), sub["date"], sub["D"], earn, cutoff_hour=8)["on_filings"]
    log(f"overnight earnings releases: {(sub['on_earn'].to_numpy()[U] > 0).mean():.4f} of U rows")
    run("O5", D3 + list(F.ON_FEATURES) + ["on_earn"], L, "cls", "y_dump")
    run("O6", D3, L, "cls", "y_d10")

    cands = ["O0", "O1", "O2", "O3", "O4", "O5", "O6"]
    res = {"universe": {"floor": FLOOR, "rows": int(U.sum()), "base_dump": float(sub["y_dump"][U].mean())},
           "variants": {k: evaluate_one(sub, preds[k], U) for k in cands}}
    for k in cands:
        v = res["variants"][k]
        log(f"{k}: " + " | ".join(f"{per} #1 hit {v[per]['top1']['hit']*100:.1f} big {v[per]['top1']['big10']*100:.1f} mean {v[per]['top1']['mean_oc']*100:+.2f} "
                                  f"net100k {v[per]['top1']['net100k']*100:+.2f} top3 hit {v[per]['top3']['hit']*100:.1f}" for per in ("dev", "confirm")))
    V = res["variants"]
    best = max(V[c]["dev"]["top1"]["hit"] for c in cands)
    tied = [c for c in cands if V[c]["dev"]["top1"]["hit"] >= best - 0.02]
    W = max(tied, key=lambda c: V[c]["dev"]["top3"]["hit"])
    ok = W == "O0" or (V[W]["confirm"]["top1"]["hit"] >= V["O0"]["confirm"]["top1"]["hit"] - 0.02
                       and V[W]["confirm"]["top3"]["hit"] >= V["O0"]["confirm"]["top3"]["hit"])
    chosen = W if ok else "O0"
    res["decision"] = {"dev_winner": W, "tied": tied, "adopted": chosen}
    log(f"RULE: tied {tied}; dev winner {W}; adopted {chosen}")

    # reported: conviction tiers (DEV percentiles of the session's top probability) and other floors
    p = preds[chosen]
    m = U & np.isfinite(p)
    d = pd.DataFrame({"date": sub["date"].to_numpy()[m], "p": p[m], "y": sub["y_dump"].to_numpy()[m],
                      "oc": sub["y_oc"].to_numpy()[m]}).sort_values(["date", "p"], ascending=[True, False])
    top = d.groupby("date").head(1)
    dev_top = top[top.date < X.SPLIT]
    q50, q80 = float(dev_top.p.quantile(0.5)), float(dev_top.p.quantile(0.8))
    tiers = {}
    for per, tm in (("dev", top.date < X.SPLIT), ("confirm", top.date >= X.SPLIT)):
        t = top[tm]
        for name, sel in (("low", t.p < q50), ("medium", (t.p >= q50) & (t.p < q80)), ("high", t.p >= q80)):
            x = t[sel]
            tiers.setdefault(name, {})[per] = {"days": int(len(x)), "hit": float(x.y.mean()) if len(x) else None,
                                               "mean_oc": float(x.oc.mean()) if len(x) else None}
    res["tiers"] = {"thresholds": {"medium": q50, "high": q80}, "by_tier": tiers}
    log(f"tiers (top prob >= {q50:.3f} medium, >= {q80:.3f} high): {json.dumps(tiers)}")
    res["floors"] = {}
    for fl in (250_000, 500_000, 1_000_000):
        Uf = U & (sub["cap"].to_numpy(float) >= fl)
        res["floors"][str(fl)] = evaluate_one(sub, p, Uf)
        v = res["floors"][str(fl)]
        log(f"floor ${fl:,}: " + " | ".join(f"{per} #1 hit {v[per]['top1']['hit']*100:.1f} mean {v[per]['top1']['mean_oc']*100:+.2f}" for per in ("dev", "confirm")))
    (OUT / "r13.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
