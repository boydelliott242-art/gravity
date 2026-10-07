"""Round 3: multi-day shorts (5 / 10 / 20 sessions) in names that absorb
$500K / $1M. Entry = next session's open, exit = close H sessions later.
Embargo = H sessions (labels overlap in time), monthly walk-forward, the
same dev / confirm split. Costs: impact + spread on both legs + a 5%/yr
borrow allowance. Reports worst squeeze during the hold (max high / entry)."""
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
from gravity.features import FEATURES  # noqa: E402
from gravity.sources import prices  # noqa: E402

OUT = R1.OUT
BORROW_YR = 0.05
HORIZONS = (5, 10, 20)


def forward_labels(hist):
    rows = []
    for s, df in hist.items():
        if df is None or len(df) < 30:
            continue
        o, h, c = df["open"].astype(float), df["high"].astype(float), df["close"].astype(float)
        e = pd.DataFrame(index=df.index)
        o1 = o.shift(-1)
        for H in HORIZONS:
            e[f"fwd{H}"] = c.shift(-H) / o1 - 1                                  # entry next open, exit close t+H
            e[f"mae{H}"] = h[::-1].rolling(H, min_periods=H).max()[::-1].shift(-1) / o1 - 1  # worst high t+1..t+H
        e["symbol"] = s
        rows.append(e.reset_index().rename(columns={"index": "date"}))
    x = pd.concat(rows, ignore_index=True)
    x["date"] = pd.to_datetime(x["date"])
    return x


def wf(lab, feats, ycol, H, regress=False, train_mask=None):
    s = M._settings(1, None)
    s["embargo"] = max(int(s["embargo"]), H + 1)
    dates = np.unique(lab["date"].to_numpy("datetime64[ns]"))
    di = np.searchsorted(dates, lab["date"].to_numpy("datetime64[ns]"))
    X = lab[feats].to_numpy(np.float32)
    y = lab[ycol].to_numpy(float)
    tm = (np.ones(len(lab), bool) if train_mask is None else train_mask) & np.isfinite(y)
    oos = np.full(len(lab), np.nan)
    rng = np.random.default_rng(0)
    for fw in M._fold_windows(dates, s):
        fit_idx = np.flatnonzero((di <= fw["fit_end"]) & tm)
        cal_idx = np.flatnonzero((di >= fw["cal_start"]) & (di <= fw["train_end"]) & tm)
        te_idx = np.flatnonzero((di >= fw["test_start"]) & (di <= fw["test_end"]))
        if not len(te_idx) or len(cal_idx) < 200:
            continue
        if regress:
            idx = np.r_[fit_idx, cal_idx]
            reg = M._fit_regressor(X[idx], np.clip(y[idx], -0.6, 0.6), s, rng)
            oos[te_idx] = reg.predict(X[te_idx])
        else:
            clf = M._fit_classifier(X[fit_idx], y[fit_idx], s, rng)
            iso = M._fit_iso(M._raw(clf, X[cal_idx]), y[cal_idx])
            rt = M._raw(clf, X[te_idx])
            oos[te_idx] = (iso.predict(rt) if iso is not None else rt) + M.TIE_EPS * rt
    return oos


def evaluate(lab, score, H, mask_dev, label, ascending=False):
    ok = (~M.ssr_next(lab)) & np.isfinite(score) & np.isfinite(lab[f"fwd{H}"].to_numpy())
    out, lines = {}, []
    for period, pm in (("dev", mask_dev), ("confirm", ~mask_dev)):
        for tname, Q in (("$500K", 5e5), ("$1M", 1e6)):
            m = ok & pm & (lab["close"].to_numpy() >= 1) & (lab["cap"].to_numpy() >= Q)
            df = pd.DataFrame({"date": lab["date"].to_numpy()[m], "s": score[m], "r": lab[f"fwd{H}"].to_numpy()[m],
                               "mae": lab[f"mae{H}"].to_numpy()[m], "dvx": lab["dvx"].to_numpy()[m],
                               "sig": lab["vol20"].to_numpy()[m],
                               "spr": np.clip(np.nan_to_num(lab["spread_ar"].to_numpy()[m], nan=0.005), 0.001, 0.02)})
            df = df.sort_values(["date", "s"], ascending=[True, ascending])
            top = df.groupby("date").head(1).reset_index(drop=True)
            cost = 2 * R1.impact(Q, top.dvx.to_numpy(), top.sig.to_numpy()) + top.spr.to_numpy() + BORROW_YR * H / 252
            gross = -top.r.to_numpy()
            net = gross - cost
            # non-overlapping: one position at a time, a new entry every H sessions
            no = net[::H]
            eq = np.cumprod(1 + 0.25 * no)       # 25% of equity per position
            # staggered book: 1/H of capital enters each day → daily P&L ≈ mean of open positions
            row = {"trades": int(len(top)), "hit15": float(np.mean(top.r <= -0.15)), "mean_r": float(top.r.mean()),
                   "med_r": float(top.r.median()), "gross": float(gross.mean()), "cost": float(cost.mean()),
                   "net": float(net.mean()), "win": float(np.mean(net > 0)),
                   "mae_p50": float(np.nanmedian(top.mae)), "mae_p90": float(np.nanquantile(top.mae, 0.9)),
                   "nonoverlap_n": int(len(no)), "nonoverlap_mult25": float(eq[-1]) if len(eq) else None,
                   "nonoverlap_dd": float(np.max(1 - eq / np.maximum.accumulate(eq))) if len(eq) else None,
                   "tier_mean_r": float(df.r.mean())}
            out[f"{period}|{tname}"] = row
            lines.append(f"{period:7s} {tname:5s} trades {row['trades']:>3} | ≤−15% {row['hit15']:.3f} mean {row['mean_r']:+.4f} med {row['med_r']:+.4f} "
                         f"(tier avg {row['tier_mean_r']:+.4f}) | cost {row['cost']*1e4:>3.0f}bp net {row['net']:+.4f} win {row['win']:.2f} | "
                         f"squeeze p50 {row['mae_p50']:+.3f} p90 {row['mae_p90']:+.3f} | non-overlap {row['nonoverlap_n']} trades ×{row['nonoverlap_mult25']:.2f} dd {row['nonoverlap_dd']:.2f}")
    print(f"\n=== {label}\n" + "\n".join(lines), flush=True)
    return out


def main():
    t0 = time.time()
    panel = pickle.loads(cli.PANEL_PATH.read_bytes())
    lab = panel[panel["y_dump"].notna()].copy()
    hist = prices.load_history(sorted(lab["symbol"].unique()), refresh=False)
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab.merge(R1.extra_features(hist), on=["date", "symbol"], how="left")
    lab = lab.merge(forward_labels(hist), on=["date", "symbol"], how="left")
    lab["dvx"] = R1.exp_dvol(lab["dvol20"].to_numpy(float), lab["dv1"].to_numpy(float))
    lab["cap"] = R1.capacity(lab["dvx"].to_numpy(float), lab["vol20"].to_numpy(float))
    lab["log_dvx"] = np.log10(lab["dvx"].clip(lower=1)); lab["log_cap"] = np.log10(lab["cap"].clip(lower=1))
    lab = lab.sort_values(["date", "symbol"]).reset_index(drop=True)
    mask_dev = (lab["date"] < pd.Timestamp("2025-11-01")).to_numpy()
    liquid = lab["dvx"].to_numpy() >= 2e6
    F = FEATURES + R1.EXTRA
    print(f"prepared in {time.time()-t0:.0f}s", flush=True)
    res = {}
    for H in HORIZONS:
        thr = {5: -0.10, 10: -0.15, 20: -0.20}[H]
        lab[f"y_d{H}"] = np.where(np.isfinite(lab[f"fwd{H}"]), (lab[f"fwd{H}"] <= thr).astype(float), np.nan)
        p = wf(lab, F, f"y_d{H}", H); np.save(OUT / f"oos_h{H}_cls.npy", p)
        res[f"H{H} cls all"] = evaluate(lab, p, H, mask_dev, f"H={H}: P(fall ≥{-thr:.0%}) trained on all names")
        pl = wf(lab, F, f"y_d{H}", H, train_mask=liquid); np.save(OUT / f"oos_h{H}_cls_liq.npy", pl)
        res[f"H{H} cls liquid"] = evaluate(lab, pl, H, mask_dev, f"H={H}: P(fall ≥{-thr:.0%}) trained on liquid names")
        r = wf(lab, F, f"fwd{H}", H, regress=True, train_mask=liquid); np.save(OUT / f"oos_h{H}_reg_liq.npy", r)
        res[f"H{H} reg liquid"] = evaluate(lab, r, H, mask_dev, f"H={H}: expected return, trained on liquid names (lowest first)", ascending=True)
    (OUT / "size_tier_r3.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
