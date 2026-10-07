"""Tests for history stores, the RSS feed, and the live tape."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET_

import pandas as pd
import pytest

from gravity import config, feed, history, live


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "PROBS", tmp_path / "probs")
    monkeypatch.setattr(history, "BORROW", tmp_path / "borrow")
    (tmp_path / "probs").mkdir()
    (tmp_path / "borrow").mkdir()
    history._recent.cache_clear()
    return tmp_path


def test_prob_and_borrow_history(stores):
    history.save_probs("2026-09-30", "2026-09-29", pd.Series({"AAA": 0.2, "BBB": float("nan")}))
    history.save_probs("2026-10-01", "2026-09-30", pd.Series({"AAA": 0.31234}))
    assert history.prob_history("AAA") == [["2026-09-30", 0.2], ["2026-10-01", 0.3123]]
    assert history.prob_history("BBB") == []
    history.save_borrow("2026-09-30", {"AAA": {"fee_rate": 12.345, "available": 500000}}, ["AAA", "ZZZ"])
    history.save_borrow("2026-10-01", {"AAA": {"fee_rate": 30.0, "available": 100000}}, ["AAA"])
    assert history.borrow_history("AAA") == [["2026-09-30", 12.35, 500000], ["2026-10-01", 30.0, 100000]]
    msg = history.borrow_tightening("AAA")
    assert msg and "500,000" in msg and "100,000" in msg


def test_borrow_fee_jump_warning(stores):
    history.save_borrow("2026-09-30", {"AAA": {"fee_rate": 10.0, "available": 100000}}, ["AAA"])
    history.save_borrow("2026-10-01", {"AAA": {"fee_rate": 45.0, "available": 90000}}, ["AAA"])
    assert "jumped from 10% to 45%" in history.borrow_tightening("AAA")
    assert history.borrow_tightening("NOPE") is None


def test_rss_feed_is_valid_xml(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PICK_LOG", tmp_path)
    (tmp_path / "2026-10-01.json").write_text(json.dumps({
        "session_date": "2026-10-01", "first_published_at": "2026-10-01T12:20:00+00:00",
        "top": {"symbol": "A&B", "prob_dump": 0.35}, "board": [{"symbol": "A&B"}, {"symbol": "CCC"}],
        "outcome": {"top": {"open": 1.0, "close": 0.9, "oc": -0.1, "oh": 0.05}}}))
    (tmp_path / "2026-10-02.json").write_text(json.dumps({"session_date": "2026-10-02", "top": None, "board": []}))
    xml = feed.build()
    root = ET_.fromstring(xml)
    items = root.findall("./channel/item")
    assert len(items) == 1
    assert "35%" in items[0].find("title").text and "A&B" in items[0].find("title").text
    assert "-10.0% open→close" in items[0].find("description").text


def test_live_row_math():
    r = live._row("AAA", {"open": 2.0, "last": 1.8, "high": 2.3, "low": 1.7, "asof": "x"})
    assert r["oc_now"] == pytest.approx(-0.1) and r["oh_now"] == pytest.approx(0.15) and r["ol_now"] == pytest.approx(-0.15)
    assert live._row("AAA", None) is None
    assert live._row("AAA", {"open": 0, "last": 1}) is None


def test_live_skips_outside_session(monkeypatch):
    monkeypatch.setattr(live, "market_phase", lambda: "closed")
    assert live.run(push=False) == 0


def test_live_rejects_quotes_from_another_session():
    q = {"open": 2.0, "last": 1.6, "high": 2.1, "low": 1.5, "session_date": "2026-09-30", "points_session": "regular"}
    assert live._row("AAA", q, "2026-10-01") is None            # yesterday's tape for a halted name
    assert live._row("AAA", dict(q, session_date="2026-10-01"), "2026-10-01")["oc_now"] == pytest.approx(-0.2)
    assert live._row("AAA", dict(q, session_date="2026-10-01", points_session="pre"), "2026-10-01") is None


def test_with_point_appends_or_replaces_today():
    from gravity.cli import _with_point
    assert _with_point([["2026-09-30", 0.2]], ["2026-10-01", 0.3]) == [["2026-09-30", 0.2], ["2026-10-01", 0.3]]
    assert _with_point([["2026-10-01", 0.1]], ["2026-10-01", 0.3]) == [["2026-10-01", 0.3]]
    assert _with_point([["2026-09-30", 0.2]], None) == [["2026-09-30", 0.2]]
    assert _with_point([["2026-09-30", 0.2]], ["2026-10-01", None]) == [["2026-09-30", 0.2]]


def test_rss_guid_changes_when_the_morning_pick_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PICK_LOG", tmp_path)
    rec = {"session_date": "2026-10-01", "run": "evening", "published_at": "2026-10-01T00:00:00+00:00",
           "top": {"symbol": "AAA", "prob_dump": 0.3}, "board": []}
    (tmp_path / "2026-10-01.json").write_text(json.dumps(rec))
    g1 = ET_.fromstring(feed.build()).find("./channel/item/guid").text
    (tmp_path / "2026-10-01.json").write_text(json.dumps(dict(rec, run="morning", top={"symbol": "BBB", "prob_dump": 0.3})))
    root = ET_.fromstring(feed.build())
    assert root.find("./channel/item/guid").text != g1
    assert "pre-market #1: BBB" in root.find("./channel/item/title").text


def test_quoted_spread_regular_session_only(monkeypatch):
    from gravity import net as _net
    payload = {"data": {"marketStatus": "Market Open", "primaryData": {"bidPrice": "$1.98", "askPrice": "$2.02"}}}
    monkeypatch.setattr(_net, "get_json", lambda *a, **k: payload)
    q = live.quoted_spread("AAA")
    assert q["bid"] == 1.98 and q["ask"] == 2.02 and q["spread"] == pytest.approx(0.04 / 2.0)
    payload["data"]["marketStatus"] = "After-Hours"
    assert live.quoted_spread("AAA") is None                 # after-hours quotes are not representative
    payload["data"]["marketStatus"] = "Market Open"
    payload["data"]["primaryData"]["askPrice"] = "N/A"
    assert live.quoted_spread("AAA") is None
