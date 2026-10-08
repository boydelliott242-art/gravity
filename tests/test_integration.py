"""Tests for the integration layer: score, twins, scorecard."""

from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from gravity import config, score, scorecard, twins
from gravity.util import ET


# ── shortability ─────────────────────────────────────────────────────────
def test_shortability_statuses():
    borrow = {"AAA": {"fee_rate": 0.3, "available": 5_000_000, "asof": "x"},
              "BBB": {"fee_rate": 98.8, "available": 35_000, "asof": "x"},
              "CCC": {"fee_rate": 12.0, "available": 0, "asof": "x"}}
    assert score.shortability("AAA", borrow, True)["status"] == "ETB"
    assert score.shortability("BBB", borrow, True)["status"] == "HTB"
    assert score.shortability("CCC", borrow, True)["status"] == "NONE"
    assert score.shortability("ZZZ", borrow, True)["status"] == "NONE"      # not in a healthy file
    assert score.shortability("ZZZ", borrow, False)["status"] == "UNKNOWN"  # file failed → don't guess


def test_squeeze_danger_renormalises_over_known_parts():
    s_all, parts = score.squeeze_danger(90, {"status": "HTB", "fee_rate": 100, "available": 1000}, 0.30, 5, 1e6, 0.70)
    assert s_all == 100 or s_all >= 95
    s_few, parts2 = score.squeeze_danger(10, {"status": "UNKNOWN", "fee_rate": None, "available": None}, None, None, None, None)
    assert s_few == 10  # only the model part is known
    assert parts2["fee"] is None and parts2["si"] is None
    none, _ = score.squeeze_danger(None, {"status": "UNKNOWN"}, None, None, None, None)
    assert none is None


def test_overlay_flags_but_never_moves_probability():
    p, note = score.overlay_catalysts(0.2, [{"category": "insider"}], 1.8)
    assert p == 0.2 and note is None
    p2, note2 = score.overlay_catalysts(0.2, [{"category": "offering", "form": "6-K", "date": "2026-09-30"}], 2.3)
    assert p2 == 0.2
    assert "not reflected in the probability" in note2 and "2.3×" in note2


def test_pct_never_rounds_to_minus_100():
    assert score.pct(-0.998) == "-99.8%"
    assert score.pct(-0.5) == "-50%"


def test_split_label():
    assert score.split_label(0.05) == "1:20"
    assert score.split_label(0.041667) == "1:24"
    assert score.split_label(2.0) == "2:1"


def test_build_reasons_dated_and_sourced():
    session = date(2026, 9, 30)
    m = {"r1": 1.4, "rvol1": 38, "dd_52w": -0.98, "close": 0.8, "asia": 1.0, "volume": 1e6}
    filings = [
        {"date": "2026-09-29", "form": "424B5", "category": "offering", "url": "https://sec/1"},
        {"date": "2026-09-10", "form": "S-1", "category": "registration", "url": "https://sec/2"},
        {"date": "2026-01-01", "form": "424B5", "category": "offering", "url": "https://sec/old"},  # too old for supply
    ]
    splits = [{"date": "2025-12-22", "ratio": 0.041667}, {"date": "2026-05-04", "ratio": 0.05}]
    reasons, flags = score.build_reasons(m, {"gap_pct": 0.35, "asof": "2026-09-30T08:12:00-04:00"},
                                         filings, splits, [], {"status": "HTB", "fee_rate": 98.8}, session, None)
    texts = " | ".join(r["text"] for r in reasons)
    assert "+140%" in texts and "38×" in texts
    assert "424B5 filed 2026-09-29" in texts
    assert "1:20 reverse split on 2026-05-04 (2 reverse splits in 2 years)" in texts
    assert "Pre-market +35%" in texts
    assert any(r.get("url") == "https://sec/1" for r in reasons)
    assert not any(r.get("url") == "https://sec/old" for r in reasons)
    for f in ("OFFERING", "R/S 1:20", "HTB 99%", "ASIA", "SUB-$1"):
        assert f in flags
    assert reasons == sorted(reasons, key=lambda r: -r["strength"])


def test_family_scores_orientation():
    rows = pd.DataFrame({"r1": [0.0, 0.5, 1.0], "dd_52w": [-0.1, -0.5, -0.9]}, index=["A", "B", "C"])
    docs = {"r1": ("exhaustion", ""), "dd_52w": ("decay", "")}
    fam = score.family_scores(rows, docs, {"r1": 1.0, "dd_52w": -1.0})
    assert fam.loc["C", "exhaustion"] > fam.loc["A", "exhaustion"]
    assert fam.loc["C", "decay"] > fam.loc["A", "decay"]  # deeper drawdown = more decay
    assert fam["dilution"].isna().all()


# ── twins ────────────────────────────────────────────────────────────────
def test_twins_prefers_structurally_similar_names():
    idx = ["INHD", "TWIN", "BIGCO", "MID"]
    rows = pd.DataFrame({
        "market_cap": [14e6, 18e6, 1.5e9, 300e6],
        "price_log": [0.5, 0.4, 1.8, 1.2],
        "rs_count_2y": [2, 2, 0, 0],
        "dd_52w": [-0.99, -0.97, -0.10, -0.40],
        "n_offer_365": [4, 3, 0, 1],
        "asia": [1.0, 1.0, 0.0, 0.0],
        "ipo_year": [2023, 2023, 1995, 2010],
        "vol60": [0.12, 0.11, 0.02, 0.05],
        "dvol20_log": [5.3, 5.2, 7.5, 6.3],
    }, index=idx)
    out = twins.find_twins(rows, ref="INHD", k=3, session_year=2026)
    assert out[0]["symbol"] == "TWIN"
    assert out[0]["similarity"] > out[-1]["similarity"]
    assert "Asia-linked issuer" in out[0]["reasons"]
    assert all(t["symbol"] != "INHD" for t in out)
    assert twins.find_twins(rows, ref="NOPE") == []
    assert twins.find_twins(rows) == []


# ── scorecard ────────────────────────────────────────────────────────────
@pytest.fixture
def pick_log(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PICK_LOG", tmp_path)
    return tmp_path


def _today(session="2026-10-01"):
    return {"session_date": session, "run": "morning", "generated_at": "2026-10-01T12:15:00+00:00",
            "model": {"version": "m1"},
            "top": {"symbol": "AAA", "prob_dump": 0.3, "score": 99, "premarket": None, "squeeze_danger": 20},
            "board": [{"rank": 1, "symbol": "AAA", "prob_dump": 0.3}, {"rank": 2, "symbol": "BBB", "prob_dump": 0.2}]}


def test_log_picks_refuses_after_open(pick_log, monkeypatch):
    monkeypatch.setattr(scorecard, "now_et", lambda: datetime(2026, 10, 1, 9, 31, tzinfo=ET))
    scorecard.log_picks(_today())
    assert not list(pick_log.glob("*.json"))
    monkeypatch.setattr(scorecard, "now_et", lambda: datetime(2026, 10, 1, 8, 15, tzinfo=ET))
    scorecard.log_picks(_today())
    assert (pick_log / "2026-10-01.json").exists()
    scorecard.log_picks({**_today(), "late": True, "top": {"symbol": "LATE"}})  # late runs never overwrite
    assert json.loads((pick_log / "2026-10-01.json").read_text())["top"]["symbol"] == "AAA"


def test_evening_never_replaces_morning_record(pick_log, monkeypatch):
    monkeypatch.setattr(scorecard, "now_et", lambda: datetime(2026, 10, 1, 8, 15, tzinfo=ET))
    assert scorecard.log_picks(_today()) is True                                   # morning list on record
    late_evening = {**_today(), "run": "evening", "top": {"symbol": "EVE"}}
    assert scorecard.log_picks(late_evening) is False                              # launchd fired a missed evening job
    assert json.loads((pick_log / "2026-10-01.json").read_text())["top"]["symbol"] == "AAA"
    assert scorecard.log_picks({**_today(), "top": {"symbol": "MOR2"}}) is True    # a later morning run may refresh


def test_grade_and_freeze(pick_log, monkeypatch):
    monkeypatch.setattr(scorecard, "now_et", lambda: datetime(2026, 10, 1, 8, 0, tzinfo=ET))
    scorecard.log_picks(_today())
    day = pd.Timestamp("2026-10-01")
    def bars(o, h, l, c, v=1e5):
        return pd.DataFrame({"open": [o], "high": [h], "low": [l], "close": [c], "volume": [v]}, index=[day])
    hist = {"AAA": bars(10, 10.5, 8, 8.5), "BBB": bars(5, 6.5, 4.9, 5.2)}
    uni = []
    for i in range(250):
        s = f"U{i}"
        hist[s] = bars(1, 1.1, 0.9, 1.0 + (0.02 if i % 2 else -0.06))
        uni.append(s)
    assert scorecard.grade(hist, uni + ["AAA", "BBB"]) == 1
    rec = json.loads((pick_log / "2026-10-01.json").read_text())
    top = rec["outcome"]["top"]
    assert top["oc"] == pytest.approx(-0.15) and top["dump"] is True and top["squeezed"] is False
    assert rec["outcome"]["board_n"] == 2
    assert 0.4 < rec["outcome"]["universe_dump_rate"] < 0.6
    # graded file is frozen: a later log for the same session changes nothing
    scorecard.log_picks({**_today(), "top": {"symbol": "ZZZ"}})
    assert json.loads((pick_log / "2026-10-01.json").read_text())["top"]["symbol"] == "AAA"
    s = scorecard.summary()
    assert s["live"]["n_days"] == 1 and s["live"]["top_dump_rate"] == 1.0


def test_grade_waits_for_bars(pick_log, monkeypatch):
    monkeypatch.setattr(scorecard, "now_et", lambda: datetime(2026, 10, 1, 8, 0, tzinfo=ET))
    scorecard.log_picks(_today())
    assert scorecard.grade({}, []) == 0  # no bars yet → stays ungraded
    assert json.loads((pick_log / "2026-10-01.json").read_text())["outcome"] is None


# ── coverage gate ────────────────────────────────────────────────────────
def test_coverage_gate():
    from gravity.cli import healthy
    good = {"universe": {"scored": 2850, "eligible": 2850}, "session_date": "2026-10-01",
            "features_asof": "2026-09-30", "board": [{"symbol": "A"}]}
    assert healthy(good)[0]
    assert not healthy({**good, "universe": {"scored": 900, "eligible": 2850}})[0]
    assert not healthy({**good, "universe": {"scored": 1600, "eligible": 2850}})[0]   # < 60% traded
    assert not healthy({**good, "features_asof": "2026-09-28"})[0]                    # stale
    assert healthy({**good, "session_date": "2026-10-05", "features_asof": "2026-10-02"})[0]  # Mon uses Fri
    assert not healthy({**good, "board": []})[0]


# ── point-in-time market cap (serial diluters must not be dropped) ───────
def test_pit_market_cap_uses_shares_as_reported_then():
    from gravity.cli import _pit_market_cap
    panel = pd.DataFrame({
        "symbol": ["DIL"] * 3 + ["RS"] * 2 + ["NOSEC"],
        "date": pd.to_datetime(["2025-01-10", "2025-07-10", "2026-03-10", "2025-05-01", "2025-07-01", "2025-05-01"]),
        "price": [2.0, 1.0, 0.5, 0.10, 2.00, 4.0],
    })
    panel["close"] = panel["price"]                  # no splits for these rows except RS (unused by the SEC branch)
    frames = [
        {"cik": 1, "end": "2024-12-31", "shares": 10e6},      # diluter: 10M → 1B shares
        {"cik": 1, "end": "2026-01-31", "shares": 1e9},
        {"cik": 2, "end": "2025-03-31", "shares": 400e6},     # 1:20 reverse split on 2025-06-01
    ]
    cmap = {"DIL": {"cik": 1}, "RS": {"cik": 2}}
    splits = {"RS": [{"date": "2025-06-01", "ratio": 0.05}]}
    shares_today = pd.Series({"DIL": 1e9, "RS": 20e6, "NOSEC": 1e6})
    cap = _pit_market_cap(panel, frames, cmap, splits, shares_today)
    assert cap.iloc[0] == pytest.approx(2.0 * 10e6)      # 2025: $20M (today's count would claim $2B)
    assert cap.iloc[1] == pytest.approx(1.0 * 10e6)
    assert cap.iloc[2] == pytest.approx(0.5 * 1e9)       # after the 2026 report
    assert cap.iloc[3] == pytest.approx(0.10 * 400e6)    # before the reverse split
    assert cap.iloc[4] == pytest.approx(2.00 * 20e6)     # after it: 400M × 0.05 = 20M shares
    assert cap.iloc[5] == pytest.approx(4.0 * 1e6)       # no SEC data → today's share count


def test_pit_market_cap_adr_and_fallback_split_basis():
    from gravity.cli import _pit_market_cap
    panel = pd.DataFrame({
        "symbol": ["ADR", "ADR", "NS", "NS"],
        "date": pd.to_datetime(["2025-03-03", "2026-03-02", "2025-03-03", "2026-03-02"]),
        "price": [0.50, 0.40, 6.60, 9.00],            # as traded
        "close": [0.50, 0.40, 132.0, 9.00],           # split-adjusted to today's basis (NS: 1:20 reverse split in 2025-06)
    })
    frames = [{"cik": 7, "end": "2024-12-31", "shares": 5e9},   # ORDINARY shares; 1 ADS = 500 ordinary
              {"cik": 7, "end": "2025-12-31", "shares": 6e9}]
    cmap = {"ADR": {"cik": 7}}
    shares_today = pd.Series({"ADR": 12e6, "NS": 30e6})          # Nasdaq: ADS count / post-split count
    cap = _pit_market_cap(panel, frames, cmap, {"NS": [{"date": "2025-06-02", "ratio": 0.05}]}, shares_today)
    assert cap.iloc[0] == pytest.approx(0.50 * 5e9 * (12e6 / 6e9))   # ≈ $5M, not $2.5B
    assert cap.iloc[1] == pytest.approx(0.40 * 12e6)
    assert panel.attrs["pit_adr_rescaled"] == 1
    assert cap.iloc[2] == pytest.approx(132.0 * 30e6)                 # $3.96B before the reverse split — not $198M
    assert cap.iloc[3] == pytest.approx(9.00 * 30e6)


def test_grade_records_a_vanished_pick_instead_of_waiting_forever(pick_log, monkeypatch):
    monkeypatch.setattr(scorecard, "now_et", lambda: datetime(2026, 10, 1, 8, 0, tzinfo=ET))
    scorecard.log_picks(_today())
    day, later = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-07")
    def bars(c):
        return pd.DataFrame({"open": [1.0, 1.0], "high": [1.1, 1.1], "low": [0.9, 0.9], "close": [c, 1.0],
                             "volume": [1e5, 1e5]}, index=[day, later])
    hist = {f"U{i}": bars(1.02 if i % 2 else 0.94) for i in range(250)}     # AAA (the #1) has no data at all
    uni = list(hist)
    monkeypatch.setattr(scorecard, "now_et", lambda: datetime(2026, 10, 2, 18, 0, tzinfo=ET))
    assert scorecard.grade(hist, uni) == 0                                   # 1 session later: still waits
    monkeypatch.setattr(scorecard, "now_et", lambda: datetime(2026, 10, 7, 18, 0, tzinfo=ET))
    assert scorecard.grade(hist, uni) == 1                                   # market moved on → recorded, not hidden
    top = json.loads((pick_log / "2026-10-01.json").read_text())["outcome"]["top"]
    assert top["missing"] is True and "no price data" in top["note"]
    assert scorecard.ungraded_symbols() == []
