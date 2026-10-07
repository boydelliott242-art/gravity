"""Round 2: direction-aware ranking, pre-market gap (M1) and protective stops
for the $500K / $1M tiers. Same walk-forward recipe and dev/confirm split as
round 1 (size_tier.py)."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import size_tier as R1  # noqa: E402
from gravity import cli, model as M  # noqa: E402
from gravity.features import FEATURES, M1_EXTRA  # noqa: E402
from gravity.sources import prices  # noqa: E402

OUT = R1.OUT


def walk_forward_target(lab, feats, target, train_mask=None, regress=False, seed=0):
    s = M._settings(1, None)
    dates = np.unique(lab["date"].to_numpy("datetime64[ns]"))
    di = np.searchsorted(dates, lab["date"].to_numpy("datetime64[ns]"))
    X = lab[feats].to_numpy(np.float32)
    y = lab[target].to_numpy(float)
    tm = np.ones(len(lab), bool) if train_mask is None else train_mask
    oos = np.full(len(lab), np.nan)
    rng = np.random.default_rng(seed)
    for fw in M._fold_windows(dates, s):
        fit_idx = np.flatnonzero((di <= fw["fit_end"]) & tm & np.isfinite(y))
        cal_idx = np.flatnonzero((di >= fw["cal_start"]) & (di <= fw["train_end"]) & tm & np.isfinite(y))
        te_idx = np.flatnonzero((di >= fw["test_start"]) & (di <= fw["test_end"]))
        if not len(te_idx) or len(cal_idx) < 200:
            continue
        if regress:
            reg = M._fit_regressor(X[np.r_[fit_idx, cal_idx]], np.clip(y[np.r_[fit_idx, cal_idx]], *M.OC_CLIP), s, rng)
            oos[te_idx] = reg.predict(X[te_idx])
        else:
            clf = M._fit_classifier(X[fit_idx], y[fit_idx], s, rng)
            iso = M._fit_iso(M._raw(clf, X[cal_idx]), y[cal_idx])
            rt = M._raw(clf, X[te_idx])
            cal = iso.predict(rt) if iso is not None else rt
            oos[te_idx] = (1 - M.TIE_EPS) * cal + M.TIE_EPS * rt
    return oos


def eval_rank(lab, score, mask_dev, label, ascending=False):
    base_ok = (~M.ssr_next(lab)) & np.isfinite(score)
    tiers = {
        "$500K": (base_ok & (lab["close"].to_numpy() >= 1) & (lab["cap"].to_numpy() >= 5e5), 5e5),
        "$1M": (base_ok & (lab["close"].to_numpy() >= 1) & (lab["cap"].to_numpy() >= 1e6), 1e6),
    }
    out = {}
    lines = []
    for period, pm in (("dev", mask_dev), ("confirm", ~mask_dev)):
        for tname, (tmask, Q) in tiers.items():
            m = tmask & pm
            df = pd.DataFrame({"date": lab["date"].to_numpy()[m], "s": score[m], "y": lab["y_dump"].to_numpy()[m],
                               "oc": lab["y_oc"].to_numpy()[m], "oh": lab["y_oh"].to_numpy()[m],
                               "dvx": lab["dvx"].to_numpy()[m], "sig": lab["vol20"].to_numpy()[m],
                               "spr": np.clip(np.nan_to_num(lab["spread_ar"].to_numpy()[m], nan=0.005), 0.001, 0.02)})
            df = df.sort_values(["date", "s"], ascending=[True, ascending])
            top = df.groupby("date").head(1)
            cost = 2 * R1.impact(Q, top.dvx.to_numpy(), top.sig.to_numpy()) + top.spr.to_numpy()
            row = {"days": int(len(top)), "hit": float(top.y.mean()), "oc": float(top.oc.mean()),
                   "med": float(top.oc.median()), "sq": float((top.oh >= 0.2).mean()), "cost": float(cost.mean())}
            for stop in (None, 0.03, 0.05, 0.08, 0.10):
                g = -top.oc.to_numpy()
                if stop is not None:   # conservative: if the high touched the stop, assume it hit first
                    hit = top.oh.to_numpy() >= stop
                    g = np.where(hit, -stop - 0.005, g)   # +50 bp slippage on the stop
                net = g - cost
                key = "nostop" if stop is None else f"stop{int(stop*100)}"
                eq = np.cumprod(1 + 0.10 * net)
                row[key] = {"gross": float(g.mean()), "net": float(net.mean()), "win": float((net > 0).mean()),
                            "comp10": float(eq[-1]), "dd": float(np.max(1 - eq / np.maximum.accumulate(eq)))}
            out[f"{period}|{tname}"] = row
            lines.append(f"{period:7s} {tname:5s} hit {row['hit']:.3f} oc {row['oc']:+.4f} med {row['med']:+.4f} sq {row['sq']:.3f} cost {row['cost']*1e4:>3.0f}bp | "
                         + " | ".join(f"{k} g{v['gross']:+.4f} n{v['net']:+.4f} w{v['win']:.2f} x{v['comp10']:.2f}" for k, v in row.items() if isinstance(v, dict)))
    print(f"\n=== {label}\n" + "\n".join(lines), flush=True)
    return out


def main():
    t0 = time.time()
    panel = pickle.loads(cli.PANEL_PATH.read_bytes())
    lab = panel[panel["y_dump"].notna() & panel["y_oc"].notna()].copy()
    hist = prices.load_history(sorted(lab["symbol"].unique()), refresh=False)
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab.merge(R1.extra_features(hist), on=["date", "symbol"], how="left")
    lab["dvx"] = R1.exp_dvol(lab["dvol20"].to_numpy(float), lab["dv1"].to_numpy(float))
    lab["cap"] = R1.capacity(lab["dvx"].to_numpy(float), lab["vol20"].to_numpy(float))
    lab["log_dvx"] = np.log10(lab["dvx"].clip(lower=1)); lab["log_cap"] = np.log10(lab["cap"].clip(lower=1))
    lab["y_pump"] = (lab["y_oc"] >= 0.05).astype(float)
    lab = lab.sort_values(["date", "symbol"]).reset_index(drop=True)
    mask_dev = (lab["date"] < pd.Timestamp("2025-11-01")).to_numpy()
    liquid = lab["dvx"].to_numpy() >= 2e6
    F0 = FEATURES + R1.EXTRA
    F1 = F0 + list(M1_EXTRA)
    print(f"prepared in {time.time()-t0:.0f}s", flush=True)
    res = {}
    p0 = np.load(OUT / "oos_V1.npy") if (OUT / "oos_V1.npy").exists() else walk_forward_target(lab, F0, "y_dump")
    res["A M0 dump"] = eval_rank(lab, p0, mask_dev, "A  M0 P(dump) [round-1 V1]")
    p1 = walk_forward_target(lab, F1, "y_dump"); np.save(OUT / "oos_M1_dump.npy", p1)
    res["B M1 dump"] = eval_rank(lab, p1, mask_dev, "B  M1 P(dump) — knows the open (live: pre-market proxy)")
    q1 = walk_forward_target(lab, F1, "y_pump"); np.save(OUT / "oos_M1_pump.npy", q1)
    res["C M1 dump-pump"] = eval_rank(lab, p1 - q1, mask_dev, "C  M1 P(dump) − P(pump)")
    q0 = walk_forward_target(lab, F0, "y_pump"); np.save(OUT / "oos_M0_pump.npy", q0)
    res["D M0 dump-pump"] = eval_rank(lab, p0 - q0, mask_dev, "D  M0 P(dump) − P(pump)")
    r1 = walk_forward_target(lab, F1, "y_oc", train_mask=liquid, regress=True); np.save(OUT / "oos_M1_regliq.npy", r1)
    res["E M1 reg liquid"] = eval_rank(lab, r1, mask_dev, "E  M1 expected open→close, trained on liquid names (lowest first)", ascending=True)
    (OUT / "size_tier_r2.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
