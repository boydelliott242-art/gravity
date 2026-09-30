# GRAVITY — *What goes up.*

A daily pre-market radar of US small/micro-cap stocks with the highest
**evidence-based odds of a large open-to-close drop** — built for someone
who shorts names like INHD.

Every weekday before the 9:30 ET open it publishes:

- **Today's #1** — the shortable name with the highest calibrated
  probability of an open→close drop of 5 %+, next to the base rate.
- **The Board** — the top 25, each decomposed into dilution / exhaustion /
  decay / flow / street / news evidence with dated, linked sources.
- **Catalyst wire** — overnight SEC filings (offerings, ATMs, reverse
  splits, deficiency notices), bearish headlines, pre-market gaps.
- **Borrow desk** — IBKR availability and fee for every name, plus a
  squeeze-danger score, and a **Squeeze Zone** for names that look weak
  but are dangerous or impossible to short.
- **INHD twins** and a private **INHD position panel**.
- **Track record** — every pick is committed before the open and graded
  after the close against the whole universe. Plus the model's
  walk-forward backtest and an **Evidence Lab** of historical base rates.

## How it works

| Layer | Source |
|---|---|
| Universe, quotes, pre-market, short interest, analyst ratings, earnings | Nasdaq public API |
| Split-adjusted daily bars, splits, float | Yahoo Finance (yfinance) |
| Offerings, shelves, ATMs, resale S-1s, reverse splits, deficiency notices, going-concern, 144s | SEC EDGAR (submissions, full-text search, live feed, XBRL) |
| Daily short-sale volume | FINRA Reg SHO files |
| Borrow fee & availability | Interactive Brokers public short-stock file |
| Headlines | Google News RSS |
| Zacks · Bloomberg · Danelfin · WSJ · Finviz … | one-click research links (their ratings are proprietary); Danelfin API scores appear if `DANELFIN_API_KEY` is set |

A gradient-boosted model is trained walk-forward on ~3 years of every
eligible name (features as of the prior close; labels = what the next
session did), calibrated so "31 %" means roughly 31 in 100 similar setups
dropped 5 %+ open→close. The morning model also knows the opening gap
(via the pre-market price).

## Run it

```bash
./.venv/bin/python -m gravity.cli train     # full refresh + walk-forward training (~30 min)
./.venv/bin/python -m gravity.cli evening   # grade today, publish tomorrow's watchlist
./.venv/bin/python -m gravity.cli morning   # overnight catalysts + pre-market → today's #1
./scripts/install_launchd.sh                # schedule it (07:20 / 08:05 / 17:10 CT, Sat retrain)
```

Secrets (`SEC_USER_AGENT`, optional `DANELFIN_API_KEY`) live only in the
LaunchAgent environment — never in this repo. Module contracts:
[CONTRACTS.md](CONTRACTS.md).

## Limits, plainly

This is research software, not advice. Nobody can know which stock will
fall on a given day; the model finds setups that *historically* fell more
often than average, and it will be wrong a lot. The backtest uses
currently-listed names (survivorship bias), assumes opening-auction fills
and ignores locate availability. Shorting micro-caps risks unlimited
losses, squeezes, halts, borrow recalls and high fees.
