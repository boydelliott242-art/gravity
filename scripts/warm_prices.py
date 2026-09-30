#!/usr/bin/env python3
"""Warm the GRAVITY price cache: full-period daily history + split lists for
every symbol in today's universe, plus the IWM benchmark.

Polite by construction: batched ``yf.download`` chunks with pauses between
them, hard back-off on rate limits, and it stops (exit code 2) instead of
hammering Yahoo if the limit persists. Resumable: every symbol is written to
the disk cache as soon as its chunk lands, and a re-run skips anything that
is already current, so an interrupted warm simply continues where it left off.

    ./.venv/bin/python scripts/warm_prices.py              # whole universe
    ./.venv/bin/python scripts/warm_prices.py --limit 60   # smoke test
    ./.venv/bin/python scripts/warm_prices.py --symbols INHD,SOUN
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gravity import config, net  # noqa: E402
from gravity.sources import prices, universe  # noqa: E402


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbols", help="comma-separated symbols instead of the universe")
    ap.add_argument("--limit", type=int, default=0, help="only the first N universe symbols (0 = all)")
    ap.add_argument("--period", default=config.HISTORY_PERIOD, help="history period (default %(default)s)")
    ap.add_argument("--batch", type=int, default=500, help="symbols per load_history call / checkpoint")
    ap.add_argument("--chunk-size", type=int, default=prices.CHUNK_SIZE, help="symbols per yf.download")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between chunks")
    ap.add_argument("--batch-pause", type=float, default=10.0, help="seconds between batches")
    ap.add_argument("--workers", type=int, default=4, help="yfinance download threads")
    ap.add_argument("--no-benchmark", action="store_true", help="skip IWM")
    ap.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return ap.parse_args(argv)


def _cache_size_mb() -> float:
    d = config.CACHE / "prices"
    if not d.exists():
        return 0.0
    return sum(p.stat().st_size for p in d.glob("*.pkl")) / 1e6


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.monotonic()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        print(f"warm: {len(symbols)} symbols from --symbols")
    else:
        uni = universe.load_universe()
        if uni.empty:
            print("warm: universe unavailable —", net.STATUS.get(universe.SOURCE_NAME, {}).get("detail"))
            return 1
        symbols = uni["symbol"].tolist()
        print(f"warm: universe {len(symbols):,} symbols (listed {uni.attrs.get('listed'):,}, "
              f"fetched {uni.attrs.get('fetched_at')}{', STALE' if uni.attrs.get('stale') else ''})")
    if args.limit and args.limit > 0:
        symbols = symbols[: args.limit]
        print(f"warm: limited to first {len(symbols)}")

    totals: Dict[str, Any] = {}
    returned = 0
    rate_limited = False
    batches = [symbols[i:i + args.batch] for i in range(0, len(symbols), max(1, args.batch))]
    for i, batch in enumerate(batches, 1):
        got = prices.load_history(batch, period=args.period, refresh=True, max_workers=args.workers,
                                  chunk_size=args.chunk_size, pause_s=args.pause)
        returned += len(got)
        run = dict(prices.LAST_RUN)
        for k, v in run.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                totals[k] = totals.get(k, 0) + v
        totals.setdefault("failed_symbols", []).extend(run.get("failed_symbols", []))
        print(f"warm: batch {i}/{len(batches)} — {len(got)}/{len(batch)} ok "
              f"(current {run.get('current', 0)}, full {run.get('full', 0)}, "
              f"incremental {run.get('incremental', 0)}, nasdaq {run.get('nasdaq', 0)}, "
              f"stale {run.get('stale', 0)}, failed {run.get('failed', 0)}) — "
              f"{time.monotonic() - t0:,.0f}s elapsed")
        if run.get("yahoo_rate_limited"):
            rate_limited = True
            print("warm: Yahoo is rate-limiting — stopping here. Progress is cached; re-run later to resume.")
            break
        if i < len(batches):
            time.sleep(args.batch_pause)

    splits = prices.load_splits(symbols) if not rate_limited else {}
    n_split = sum(1 for v in splits.values() if v)
    n_reverse = sum(1 for v in splits.values() for sp in v if sp["ratio"] < 1)

    bench_rows = None
    if not args.no_benchmark and not rate_limited:
        bench = prices.benchmark_history(args.period)
        bench_rows = len(bench)

    elapsed = time.monotonic() - t0
    failed = len(symbols) - returned
    print("\n── warm summary ─────────────────────────────")
    print(f"requested        {len(symbols):,}")
    print(f"ok               {returned:,}")
    print(f"failed           {failed:,}")
    for k in ("current", "incremental", "full", "split_refetch", "drift_refetch",
              "individual", "nasdaq", "stale"):
        print(f"  {k:<15}{totals.get(k, 0):,}")
    if totals.get("failed_symbols"):
        print(f"failed symbols   {', '.join(totals['failed_symbols'][:30])}"
              f"{' …' if len(totals['failed_symbols']) > 30 else ''}")
    print(f"with splits      {n_split:,} symbols ({n_reverse:,} reverse splits)")
    if bench_rows is not None:
        print(f"benchmark IWM    {bench_rows:,} bars")
    print(f"cache            {config.CACHE / 'prices'} ({_cache_size_mb():,.1f} MB)")
    print(f"rate limited     {'YES' if rate_limited else 'no'}")
    print(f"elapsed          {elapsed:,.1f}s")
    if rate_limited:
        return 2
    return 0 if returned else 1


if __name__ == "__main__":
    sys.exit(main())
