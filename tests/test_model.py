"""Tests for gravity.model (CONTRACTS.md §8, §12).

Synthetic panels with a known signal (and one with none) exercise the whole
train → report → bundle → load → predict path quickly and offline. The
walk-forward is checked for out-of-sample integrity two ways: the fold
windows themselves (embargo, ordering) and an end-to-end test that
scrambling every label after a date leaves the results of earlier folds
bit-for-bit unchanged.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gravity import config  # noqa: E402
from gravity import model as M  # noqa: E402
from gravity.features import FEATURES, M1_EXTRA  # noqa: E402

FAST = {
    "min_train_sessions": 130, "cal_sessions": 40, "embargo": 5, "max_fit_rows": 5000,
    "importance_rows": 1500, "hgb": {"max_iter": 40, "learning_rate": 0.15, "min_samples_leaf": 40},
}
FOLD_MONTHS = 2


def synth_panel(n_sym: int = 30, n_days: int = 300, seed: int = 0, signal: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    d = np.repeat(dates.values, n_sym)
    s = np.tile([f"S{i:02d}" for i in range(n_sym)], n_days)
    n = len(d)
    X = rng.normal(0.0, 1.0, (n, len(FEATURES))).astype(np.float32)
    X[rng.random((n, len(FEATURES))) < 0.05] = np.nan               # NaN-native
    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "symbol", s)
    df.insert(0, "date", d)
    r1 = np.nan_to_num(df["r1"].to_numpy(float))
    rg = np.nan_to_num(df["range14"].to_numpy(float))
    gap = rng.normal(0.0, 0.05, n)
    latent = (1.2 * r1 + 0.8 * rg + 12.0 * gap) if signal else np.zeros(n)
    y_oc = -0.02 * latent + rng.normal(-0.005, 0.04, n)
    y_oh = np.abs(rng.normal(0.03, 0.06, n)) + 0.03 * np.maximum(latent, 0)
    df["gap_open"] = gap
    df["y_gap"] = gap
    df["y_oc"] = y_oc
    df["y_co"] = (1 + gap) * (1 + y_oc) - 1
    df["y_ol"] = np.minimum(y_oc, 0) - 0.01
    df["y_oh"] = y_oh
    df["y_c5"] = y_oc + rng.normal(0, 0.03, n)
    df["y_dump"] = (y_oc <= config.DUMP_THRESHOLD).astype(float)
    df["y_bigdump"] = (y_oc <= config.BIG_DUMP_THRESHOLD).astype(float)
    df["y_squeeze"] = (y_oh >= config.SQUEEZE_THRESHOLD).astype(float)
    last = df["date"] == df["date"].max()                           # live rows: no labels yet
    for c in ["y_oc", "y_co", "y_gap", "y_ol", "y_oh", "y_c5", "y_dump", "y_bigdump", "y_squeeze", "gap_open"]:
        df.loc[last, c] = np.nan
    df["close"] = 2.0
    df["price"] = 2.0
    return df


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    site = out / "site" / "model.json"
    panel = synth_panel()
    rep = M.train(panel, out_dir=out, site_json=site, fold_months=FOLD_MONTHS, params=FAST)
    return {"panel": panel, "report": rep, "out": out, "site": site}


# ── report / files ───────────────────────────────────────────────────────
def test_report_schema_and_files(trained):
    rep, out, site = trained["report"], trained["out"], trained["site"]
    for k in ("trained_at", "trained_through", "n_rows", "n_symbols", "n_days", "targets", "base_rate",
              "oos", "calibration", "importance", "sim", "caveats", "training_notes", "walk_forward", "folds"):
        assert k in rep, k
    assert {"dump", "bigdump", "squeeze"} <= set(rep["base_rate"])
    assert set(rep["targets"]) >= {"dump", "bigdump", "squeeze"}
    for m in ("m0", "m1"):
        met = rep["oos"][m]
        for k in ("auc", "brier", "top1_hit", "top10_hit", "top_decile_hit", "top1_mean_oc", "top10_mean_oc", "days"):
            assert k in met, (m, k)
        sim = rep["sim"][m]
        for k in ("daily", "gross_total", "net_total", "win_rate", "max_drawdown", "cost_assumption"):
            assert k in sim, (m, k)
        assert sim["cost_assumption"] == 0.01
        assert len(sim["daily"]) == met["days"] > 0
        assert all(len(r) == 6 and isinstance(r[0], str) for r in sim["daily"])
        cal = rep["calibration"][m]
        assert [b["bin"] for b in cal] == list(range(1, 11))
        assert sum(b["n"] for b in cal) == rep["walk_forward"]["oos_rows"]
    assert rep["n_rows"] == int(trained["panel"]["y_dump"].notna().sum())
    assert rep["trained_through"] == str(trained["panel"]["date"].max().date())
    assert any("Point-in-time universe" in c for c in rep["caveats"])
    assert any("pre-market" in c for c in rep["caveats"])
    assert rep["importance"] and {"family", "feature", "importance"} <= set(rep["importance"][0])
    assert {f["family"] for f in rep["importance_family"]} <= set(M.FAMILIES)

    # model.json: strict JSON, identical to the returned report
    txt = site.read_text()

    def no_constants(tok):
        raise AssertionError(f"non-standard JSON constant {tok} in model.json")

    assert json.loads(txt, parse_constant=no_constants) == json.loads(json.dumps(rep))

    b = M.load(out)
    assert b is not None
    assert b["trained_at"] == rep["trained_at"]
    assert b["features"] == FEATURES and b["m1_features"] == FEATURES + M1_EXTRA
    for m in ("m0", "m1"):
        assert {"dump", "bigdump", "squeeze", "oc", "swing", "pump"} <= set(b[m])
        for t in ("dump", "bigdump", "squeeze"):
            assert b[m][t]["clf"] is not None and b[m][t]["iso"] is not None
    assert set(b["report"]["oos"]) >= {"m0", "m1"}
    assert {"dump", "bigdump", "squeeze", "swing", "pump"} <= set(b["report"]["base_rate"])


def test_walk_forward_windows_are_out_of_sample(trained):
    rep = trained["report"]
    dates = sorted(pd.to_datetime(trained["panel"].loc[trained["panel"]["y_dump"].notna(), "date"]).unique())
    pos = {pd.Timestamp(d).strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    folds = rep["folds"]
    assert len(folds) >= 3
    prev_end = -1
    for f in folds:
        fit_end, cal_start, train_end = pos[f["fit_end"]], pos[f["cal_start"]], pos[f["train_end"]]
        a, z = pos[f["test_start"]], pos[f["test_end"]]
        assert fit_end < cal_start <= train_end < a <= z
        assert a - train_end - 1 >= 5 and f["embargo_sessions"] >= 5         # embargo before test
        assert cal_start - fit_end - 1 >= 5                                   # and before calibration
        assert a > prev_end                                                    # folds don't overlap
        assert a == prev_end + 1 or prev_end == -1                             # and leave no gaps
        prev_end = z
    assert pos[folds[0]["test_start"]] >= FAST["min_train_sessions"]
    assert rep["walk_forward"]["n_folds"] == len(folds)


def test_signal_is_found_and_calibrated(trained):
    rep = trained["report"]
    o0, o1 = rep["oos"]["m0"], rep["oos"]["m1"]
    assert o0["auc"] > 0.65
    assert o1["auc"] > o0["auc"]                     # the open carries extra information here
    assert o0["top1_hit"] > o0["base_rate"]
    assert o0["top1_mean_oc"] < o0["universe_mean_oc"]
    # calibrated: mean prediction close to the realised rate, deciles increasing
    assert abs(o0["mean_pred"] - o0["base_rate"]) < 0.05
    acts = [b["actual"] for b in rep["calibration"]["m0"]]
    assert acts[-1] > acts[0]


def test_no_signal_means_chance_auc(tmp_path):
    rep = M.train(synth_panel(seed=3, signal=False), out_dir=tmp_path, site_json=None,
                  fold_months=FOLD_MONTHS, params=FAST)
    assert abs(rep["oos"]["m0"]["auc"] - 0.5) < 0.06
    assert abs(rep["oos"]["m1"]["auc"] - 0.5) < 0.06
    assert not (tmp_path / "model.json").exists()


def test_future_labels_cannot_change_past_folds(trained, tmp_path):
    """End-to-end leak test: scramble every label (and the open) after the
    second fold's test window — folds 1–2 must be identical."""
    rep0 = trained["report"]
    cut = pd.Timestamp(rep0["folds"][1]["test_end"])
    p = trained["panel"].copy()
    late = (p["date"] > cut) & p["y_dump"].notna()
    rng = np.random.default_rng(11)
    for c in ("y_oc", "y_oh", "y_gap", "gap_open"):
        p.loc[late, c] = rng.permutation(p.loc[late, c].to_numpy())
    p.loc[late, "y_dump"] = (p.loc[late, "y_oc"] <= config.DUMP_THRESHOLD).astype(float)
    p.loc[late, "y_bigdump"] = (p.loc[late, "y_oc"] <= config.BIG_DUMP_THRESHOLD).astype(float)
    p.loc[late, "y_squeeze"] = (p.loc[late, "y_oh"] >= config.SQUEEZE_THRESHOLD).astype(float)
    rep1 = M.train(p, out_dir=tmp_path, site_json=None, fold_months=FOLD_MONTHS, params=FAST)
    for k in (0, 1):
        a, b = rep0["folds"][k], rep1["folds"][k]
        assert a["test_end"] <= str(cut.date())
        assert a["auc_m0"] == b["auc_m0"] and a["auc_m1"] == b["auc_m1"], k
    assert rep0["folds"][-1]["auc_m0"] != rep1["folds"][-1]["auc_m0"]


# ── predict ──────────────────────────────────────────────────────────────
def test_predict_contract(trained):
    b = M.load(trained["out"])
    p = trained["panel"]
    rows = p[p["date"] == p["date"].max()].set_index("symbol")        # live rows, symbol index like cli
    assert rows["gap_open"].isna().all()
    pr = M.predict(b, rows, use_open=False)
    assert list(pr.index) == list(rows.index)
    for c in ("prob_dump", "prob_bigdump", "prob_squeeze", "exp_oc"):
        assert c in pr.columns and pr[c].notna().all()
    for c in ("prob_dump", "prob_bigdump", "prob_squeeze"):
        assert ((pr[c] >= 0) & (pr[c] <= 1)).all()
    assert (pr["prob_bigdump"] <= pr["prob_dump"] + 1e-12).all()
    assert pr["exp_oc"].between(*M.OC_CLIP).all()
    assert (pr["model"] == "m0").all()
    # row order doesn't matter (no cross-row dependence)
    rev = M.predict(b, rows.iloc[::-1], use_open=False)
    pd.testing.assert_frame_equal(rev.loc[pr.index], pr)

    # use_open: rows with a gap proxy → M1, the others fall back to M0
    sub = rows.copy()
    sub["gap_open"] = np.nan
    sub.iloc[:10, sub.columns.get_loc("gap_open")] = np.linspace(-0.1, 0.4, 10)
    p1 = M.predict(b, sub, use_open=True)
    assert (p1["model"].iloc[:10] == "m1").all() and (p1["model"].iloc[10:] == "m0").all()
    pd.testing.assert_frame_equal(p1.iloc[10:], pr.iloc[10:])
    assert not np.allclose(p1["prob_dump"].iloc[:10], pr["prob_dump"].iloc[:10])
    # no gap_open column at all → everything M0
    p2 = M.predict(b, rows.drop(columns=["gap_open"]), use_open=True)
    pd.testing.assert_frame_equal(p2, pr)
    # higher gap → (weakly) higher dump probability, as the synthetic truth says
    g = sub.iloc[[0] * 5].copy()
    g["gap_open"] = [-0.2, 0.0, 0.1, 0.2, 0.4]
    g.index = [f"x{i}" for i in range(5)]
    pg = M.predict(b, g, use_open=True)["prob_dump"].to_numpy()
    assert pg[-1] > pg[0]


def test_predict_edge_cases(trained):
    b = M.load(trained["out"])
    empty = M.predict(b, trained["panel"].iloc[0:0], use_open=False)
    assert empty.empty and set(M.OUT_COLUMNS) <= set(empty.columns)
    few = trained["panel"].iloc[:3][["date", "symbol", "r1"]]          # most features missing → NaN, still scores
    pr = M.predict(b, few, use_open=False)
    assert pr["prob_dump"].notna().all()


def test_load_missing_or_corrupt(tmp_path):
    assert M.load(tmp_path) is None
    (tmp_path / M.BUNDLE_NAME).write_bytes(b"not a pickle")
    assert M.load(tmp_path) is None
    with open(tmp_path / M.BUNDLE_NAME, "wb") as fh:
        pickle.dump({"something": 1}, fh)
    assert M.load(tmp_path) is None


# ── pieces ───────────────────────────────────────────────────────────────
def test_daily_metrics_and_sim_math():
    day = np.array(["2025-01-02"] * 3 + ["2025-01-03"] * 3 + ["2025-01-06"] * 3, dtype="datetime64[ns]")
    prob = np.array([0.9, 0.1, 0.2, 0.3, 0.8, 0.1, 0.5, 0.4, 0.6])
    oc = np.array([-0.10, 0.02, 0.01, 0.00, 0.05, -0.02, -0.01, 0.03, -0.20])
    y = (oc <= -0.05).astype(float)
    sq = np.zeros(9)
    daily = M._daily_table(day, prob, y, oc, sq)
    assert list(daily["top1_oc"]) == [-0.10, 0.05, -0.20]
    met = M._metrics(y, prob, daily)
    assert met["top1_hit"] == pytest.approx(2 / 3)
    assert met["top1_mean_oc"] == pytest.approx((-0.10 + 0.05 - 0.20) / 3)
    assert met["days"] == 3
    sim = M._sim(daily)
    assert sim["gross_total"] == pytest.approx(0.10 - 0.05 + 0.20)
    assert sim["net_total"] == pytest.approx(0.25 - 3 * 0.01)
    assert sim["win_rate"] == pytest.approx(2 / 3)
    assert sim["max_drawdown"] == pytest.approx(0.06)             # day 2: −0.05 − 0.01
    assert sim["worst_day"] == pytest.approx(-0.05)
    assert sim["daily"][0] == ["2025-01-02", -0.10, pytest.approx(np.mean([-0.10, 0.02, 0.01])),
                               pytest.approx(np.mean([-0.10, 0.02, 0.01])), -0.10, None]  # no rule mask → pub #1 = #1; no cost column
    assert M._max_drawdown(np.array([0.1, -0.3, 0.1, -0.1])) == pytest.approx(0.3)


def test_subsample_keeps_positives_and_base_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.1).astype(float)
    idx, w = M._subsample(y, 5_000, rng, classify=True)
    assert len(idx) == 5_000 and len(np.unique(idx)) == 5_000
    assert (y[idx] == 1).sum() == min((y == 1).sum(), 2_500)
    assert np.average(y[idx], weights=w) == pytest.approx(y.mean(), rel=1e-9)
    idx2, w2 = M._subsample(y, 50_000, rng, classify=True)
    assert len(idx2) == len(y) and w2 is None
    idx3, w3 = M._subsample(y, 1_000, rng, classify=False)
    assert len(idx3) == 1_000 and w3 is None


def test_fold_windows_monthly_and_quarterly():
    dates = pd.bdate_range("2023-01-02", periods=500).values
    s = M._settings(1, {"min_train_sessions": 252})
    mon = M._fold_windows(dates, s)
    qtr = M._fold_windows(dates, M._settings(3, {"min_train_sessions": 252}))
    assert len(mon) > len(qtr) >= 3
    for f in mon + qtr:
        assert f["test_start"] - f["train_end"] - 1 == 5
        assert f["cal_start"] - f["fit_end"] - 1 == 5
        assert f["train_end"] - f["cal_start"] + 1 == 63
    assert mon[0]["test_start"] == 252
    assert M._fold_windows(dates[:100], s) == []


def test_calibrated_ties_are_broken_but_probabilities_unchanged():
    class Const:
        def predict_proba(self, X):
            p = X[:, 0]
            return np.c_[1 - p, p]

    class Flat:
        def predict(self, raw):
            return np.full(len(raw), 0.2)

    X = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
    out = M._calibrated({"clf": Const(), "iso": Flat()}, X)
    assert np.all(np.diff(out) > 0)
    assert np.allclose(out, 0.2, atol=1e-6)


def test_ssr_and_publication_rule():
    import pandas as pd
    df = pd.DataFrame({"low": [8.9, 9.5, 5.0], "close": [9.5, 9.8, 5.5], "r1": [0.0, -0.02, 0.10],
                       "dvol20": [1e6, 1e5, 5e5]})
    # prev close = close/(1+r1): 9.5, 10.0, 5.0 → lows 8.9 (−6.3%), 9.5 (−5%), 5.0 (0%)
    assert list(M.ssr_next(df)) == [False, False, False]
    df.loc[0, "low"] = 8.5  # −10.5% intraday → SSR tomorrow
    assert list(M.ssr_next(df)) == [True, False, False]
    assert list(M.publishable(df)) == [False, False, True]  # SSR; illiquid; ok
    assert list(M.publishable(pd.DataFrame({"x": [1]}))) == [False]  # missing columns → not publishable


def test_cap_limits_calibrated_probability():
    class Clf:
        def predict_proba(self, X):
            return np.c_[1 - X[:, 0], X[:, 0]]
    class Iso:
        def predict(self, r):
            return r
    X = np.array([[0.2], [0.9], [0.99]])
    out = M._calibrated({"clf": Clf(), "iso": Iso(), "cap": 0.6}, X)
    assert out[0] == pytest.approx(0.2, abs=1e-5)
    assert out[1] <= 0.6 + 1e-5 and out[2] <= 0.6 + 1e-5
    assert out[2] > out[1]  # ranking survives the cap via the raw-score tie-break
