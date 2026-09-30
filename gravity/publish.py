"""Write the site's JSON feeds and push them to GitHub Pages (main:/docs).

The automated path is deliberately conservative: it only ever commits
``docs/data``, it never pulls or rebases (this machine is the only writer),
it refuses to run on any branch but ``main`` or in a half-finished
rebase/merge, and a push only counts once ``origin/main`` is confirmed to
equal the local commit.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from . import config
from .util import clean

log = logging.getLogger(__name__)

DATA_PATH = "docs/data"


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


def _notify(msg: str) -> None:
    """Best-effort macOS notification so a stuck publisher doesn't go unnoticed."""
    try:
        subprocess.run(["osascript", "-e", f'display notification "{msg[:180]}" with title "GRAVITY"'],
                       capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


def _unsafe_state() -> Optional[str]:
    """Why publishing would be unsafe right now, or None."""
    git_dir = config.ROOT / ".git"
    for marker in ("rebase-merge", "rebase-apply", "MERGE_HEAD", "CHERRY_PICK_HEAD"):
        if (git_dir / marker).exists():
            return f"repository is mid-operation ({marker}) — resolve it by hand"
    br = _git("symbolic-ref", "--short", "-q", "HEAD", check=False).stdout.strip()
    if br != "main":
        return f"checked-out branch is {br or 'detached HEAD'}, not main"
    return None


def push(message: str) -> bool:
    """Commit docs/data and push. True only when origin/main == local HEAD."""
    try:
        why = _unsafe_state()
        if why:
            log.error("publish refused: %s", why)
            _notify(f"Publish refused: {why}")
            return False
        _git("fetch", "-q", "origin", check=False)
        has_remote = _git("rev-parse", "-q", "--verify", "origin/main", check=False).returncode == 0
        if has_remote and _git("merge-base", "--is-ancestor", "origin/main", "HEAD", check=False).returncode != 0:
            log.error("publish refused: origin/main has commits this machine doesn't — pull them by hand")
            _notify("Publish refused: GitHub has commits this Mac doesn't. Needs a manual pull.")
            return False

        _git("add", "--", DATA_PATH)
        staged = _git("diff", "--cached", "--quiet", "--", DATA_PATH, check=False).returncode != 0
        if staged:
            _git("commit", "-q", "-m", message, "--", DATA_PATH)
        ahead = int(_git("rev-list", "--count", "origin/main..HEAD", check=False).stdout.strip() or 0) if has_remote else 1
        if not ahead:
            log.info("publish: nothing new to push")
            return True
        r = _git("push", "-q", "origin", "HEAD:main", check=False)
        if r.returncode != 0:
            log.error("push failed: %s", r.stderr.strip()[:300])
            _notify("Push to GitHub failed — the site was not updated.")
            return False
        _git("fetch", "-q", "origin", check=False)
        local = _git("rev-parse", "HEAD").stdout.strip()
        remote = _git("rev-parse", "origin/main", check=False).stdout.strip()
        if local != remote:
            log.error("push not confirmed (local %s, origin %s)", local[:8], remote[:8])
            return False
        log.info("published: %s (%s)", message, local[:8])
        return True
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        log.error("publish error: %s", e)
        return False
