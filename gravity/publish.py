"""Write the site's JSON feeds and push them to GitHub Pages (main:/docs)."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from . import config
from .util import clean

log = logging.getLogger(__name__)


def write_json(name: str, obj: Any) -> Path:
    """Atomically write docs/data/<name> (compact, NaN-free)."""
    p = config.SITE_DATA / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(obj), separators=(",", ":"), ensure_ascii=False))
    tmp.replace(p)
    log.info("wrote %s (%.0f KB)", p.relative_to(config.ROOT), p.stat().st_size / 1024)
    return p


def read_json(name: str) -> Optional[Any]:
    p = config.SITE_DATA / name
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(config.ROOT), *args],
        capture_output=True, text=True, check=check, timeout=180,
    )


def push(message: str) -> bool:
    """Commit docs/ and push. Returns True when the remote has the commit.

    Uses the macOS keychain credential helper (same as the user's other
    auto-publishing project), so it works from launchd."""
    try:
        _git("add", "docs")
        st = _git("status", "--porcelain", "docs")
        if not st.stdout.strip():
            log.info("publish: nothing changed")
            return True
        _git("commit", "-q", "-m", message)
        r = _git("push", "-q", "origin", "HEAD:main", check=False)
        if r.returncode != 0:
            log.warning("push rejected (%s) — rebasing and retrying", r.stderr.strip()[:200])
            _git("pull", "-q", "--rebase", "origin", "main", check=False)
            r = _git("push", "-q", "origin", "HEAD:main", check=False)
        if r.returncode != 0:
            log.error("push failed: %s", r.stderr.strip()[:300])
            return False
        log.info("published: %s", message)
        return True
    except (subprocess.SubprocessError, OSError) as e:
        log.error("publish error: %s", e)
        return False
