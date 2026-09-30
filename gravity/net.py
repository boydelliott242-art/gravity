"""Shared HTTP layer: polite per-host rate limiting, retries with backoff,
a tiny JSON disk cache, and a source-status registry the site displays.

Every source module goes through here so throttling and failure reporting
behave the same everywhere. Nothing in this module raises on network
failure — callers get ``None`` and the failure is recorded in ``STATUS``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import requests

from . import config

log = logging.getLogger(__name__)

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# api.nasdaq.com hangs without an Origin/Referer pair.
NASDAQ_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}


def sec_headers() -> Dict[str, str]:
    """Headers for *.sec.gov. Empty UA → SEC answers 403, so callers should
    check ``sec_enabled()`` first and degrade gracefully."""
    return {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def sec_enabled() -> bool:
    return bool(config.SEC_USER_AGENT)


# ── Rate limiting ────────────────────────────────────────────────────────
_HOST_RPS = {
    "www.sec.gov": config.SEC_RPS,
    "data.sec.gov": config.SEC_RPS,
    "efts.sec.gov": config.SEC_RPS,
    "api.nasdaq.com": config.NASDAQ_RPS,
}
_last_hit: Dict[str, float] = {}
_rl_lock = threading.Lock()


def _throttle(host: str, rps: Optional[float] = None) -> None:
    rate = rps or _HOST_RPS.get(host, config.DEFAULT_RPS)
    gap = 1.0 / rate
    with _rl_lock:
        now = time.monotonic()
        wait = _last_hit.get(host, 0.0) + gap - now
        _last_hit[host] = max(now, _last_hit.get(host, 0.0) + gap)
    if wait > 0:
        time.sleep(wait)


_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        _tls.s = s
    return s


def get(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 25.0,
    retries: int = 3,
    rps: Optional[float] = None,
) -> Optional[requests.Response]:
    """GET with throttling + exponential backoff on 429/5xx/timeouts.
    Returns the Response on 2xx, else None (never raises)."""
    host = urlparse(url).netloc
    hdrs = {"User-Agent": BROWSER_UA}
    if headers:
        hdrs.update(headers)
    for attempt in range(retries + 1):
        _throttle(host, rps)
        try:
            r = _session().get(url, headers=hdrs, params=params, timeout=timeout)
        except requests.RequestException as e:
            log.debug("GET %s failed (%s), attempt %d", url, e, attempt)
            r = None
        if r is not None and 200 <= r.status_code < 300:
            return r
        status = r.status_code if r is not None else None
        if status is not None and status not in (403, 408, 425, 429) and status < 500:
            log.debug("GET %s → %s (not retried)", url, status)
            return None
        if attempt < retries:
            time.sleep(min(30.0, (2 ** attempt) * 1.5 + random.random()))
    return None


def get_json(url: str, **kw: Any) -> Any:
    r = get(url, **kw)
    if r is None:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def get_text(url: str, **kw: Any) -> Optional[str]:
    r = get(url, **kw)
    return None if r is None else r.text


# ── JSON disk cache ──────────────────────────────────────────────────────
def _cache_path(ns: str, key: str) -> Path:
    h = hashlib.sha1(key.encode()).hexdigest()[:20]
    d = config.CACHE / ns
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.json"


def cache_get(ns: str, key: str, max_age_s: float) -> Any:
    p = _cache_path(ns, key)
    if not p.exists() or (time.time() - p.stat().st_mtime) > max_age_s:
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def cache_set(ns: str, key: str, value: Any) -> None:
    p = _cache_path(ns, key)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, default=str))
    tmp.replace(p)


def cached(ns: str, key: str, max_age_s: float, fetch: Callable[[], Any]) -> Any:
    """Return a fresh-enough cached value, else fetch and cache it.
    ``None`` results are never cached (a failed fetch must not stick)."""
    hit = cache_get(ns, key, max_age_s)
    if hit is not None:
        return hit
    val = fetch()
    if val is not None:
        cache_set(ns, key, val)
    return val


# ── Source status registry (rendered on the site) ────────────────────────
STATUS: Dict[str, Dict[str, Any]] = {}
_status_lock = threading.Lock()


def record_status(name: str, ok: bool, detail: str = "") -> None:
    with _status_lock:
        STATUS[name] = {
            "name": name,
            "ok": bool(ok),
            "detail": detail,
            "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
