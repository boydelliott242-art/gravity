"""RSS feed of each session's #1 (docs/feed.xml) — subscribe on a phone."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from email.utils import format_datetime
from typing import List
from xml.sax.saxutils import escape

from . import config

SITE = "https://boydelliott242-art.github.io/gravity/"


def _rfc822(iso: str) -> str:
    try:
        d = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        d = datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return format_datetime(d)


def build(limit: int = 40) -> str:
    items: List[str] = []
    for f in sorted(config.PICK_LOG.glob("*.json"), reverse=True)[:limit]:
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        top = rec.get("top") or {}
        sym = top.get("symbol")
        if not sym:
            continue
        p = top.get("prob_dump")
        label = "pre-market #1" if rec.get("run") == "morning" else "watchlist #1 (evening before)"
        title = f"{rec['session_date']} · {label}: {sym}" + (f" — {p * 100:.0f}% odds of a 5%+ open→close drop" if p else "")
        out = (rec.get("outcome") or {}).get("top") or {}
        if out.get("missing"):
            result = "Result: halted / no regular-session trades."
        elif out.get("oc") is not None:
            result = (f"Result: open ${out['open']:.4g} → close ${out['close']:.4g} "
                      f"({out['oc'] * 100:+.1f}% open→close; high {out['oh'] * 100:+.1f}% above the open).")
        else:
            result = "Graded after the close."
        board = ", ".join(b["symbol"] for b in (rec.get("board") or [])[:10])
        desc = f"{result} Board top 10: {board}. Research, not advice."
        items.append(
            "<item>"
            f"<title>{escape(title)}</title>"
            f"<link>{escape(SITE)}#/{escape(sym)}</link>"
            f"<guid isPermaLink=\"false\">gravity-{escape(rec['session_date'])}-{escape(str(rec.get('run')))}-{escape(sym)}</guid>"
            f"<pubDate>{_rfc822(rec.get('published_at') or rec.get('first_published_at') or '')}</pubDate>"
            f"<description>{escape(desc)}</description>"
            "</item>"
        )
    now = format_datetime(datetime.now(timezone.utc))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0"><channel>'
        "<title>GRAVITY — daily #1</title>"
        f"<link>{SITE}</link>"
        "<description>The pre-market #1 from GRAVITY's dump-risk radar, published before the open and graded "
        "after the close. Research tool, not investment advice.</description>"
        f"<lastBuildDate>{now}</lastBuildDate>"
        + "".join(items)
        + "</channel></rss>\n"
    )


def write() -> None:
    p = config.SITE / "feed.xml"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(build())
    tmp.replace(p)
