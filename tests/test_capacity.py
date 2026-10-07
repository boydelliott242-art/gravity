"""Capacity / impact model."""
import numpy as np
import pytest

from gravity import capacity as C


def test_expected_volume_buckets():
    assert C.exp_dvol(10e6, 5e6) == pytest.approx(10e6 * 0.54)        # quiet day (rvol 0.5)
    assert C.exp_dvol(10e6, 400e6) == pytest.approx(10e6 * 6.57)      # 40x spike → conservative 6.6x
    assert C.exp_dvol(0, 1e6) == 0


def test_capacity_respects_both_limits():
    dvx, sig = 20e6, 0.04
    cap = float(C.market_capacity(dvx, sig))
    assert cap <= 0.05 * dvx + 1
    assert float(C.impact(cap, dvx, sig)) <= C.MAX_IMPACT + 1e-9
    # a calmer stock with the same volume can take more
    assert float(C.market_capacity(dvx, 0.02)) >= cap


def test_impact_square_root_law():
    i1 = float(C.impact(1e5, 10e6, 0.05))
    i4 = float(C.impact(4e5, 10e6, 0.05))
    assert i4 == pytest.approx(2 * i1)                                # 4x size → 2x impact
    assert i1 == pytest.approx(0.7 * 0.05 * np.sqrt(0.01))


def test_round_trip_cost_and_borrow():
    base = float(C.round_trip_cost(5e5, 25e6, 0.04, 0.003))
    with_fee = float(C.round_trip_cost(5e5, 25e6, 0.04, 0.003, hold_days=20, borrow_fee_pct=25.0))
    assert with_fee - base == pytest.approx(0.25 * 20 / 252)
    assert float(C.round_trip_cost(5e5, 25e6, 0.04, np.nan)) > 0       # unknown spread → default, not NaN


def test_borrow_capacity_and_tiers():
    assert C.borrow_capacity(35000, 3.2) == pytest.approx(112000)
    assert C.borrow_capacity(None, 3.2) is None
    assert C.tier_of(2e6) == "$1M+" and C.tier_of(6e5) == "$500K+" and C.tier_of(5e4) == "under $100K"
    assert C.tier_of(None) is None


def test_ar_spread_and_breakeven():
    rng = np.random.default_rng(0)
    c = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, 40)))
    h, l = c * 1.02, c * 0.98
    s = C.ar_spread(h, l, c)
    assert s is not None and 0 <= s < 0.2
    assert C.ar_spread(h[:5], l[:5], c[:5]) is None
    be = C.breakeven_size(0.02, 10e6, 0.05, 0.005)
    assert float(C.round_trip_cost(be, 10e6, 0.05, 0.005)) == pytest.approx(0.02, rel=1e-6)
    assert C.breakeven_size(0.004, 10e6, 0.05, 0.005) == 0.0      # spread alone eats the move


def test_size_block_fields():
    c = np.linspace(2, 3, 40)
    b = C.size_block(3.0, 5e6, 2e7, 0.06, c * 1.03, c * 0.97, c, 35_000, 0.018)
    assert b["cap_borrow"] == pytest.approx(105_000) and b["capacity"] == pytest.approx(min(b["cap_market"], 105_000))
    assert b["limit"] == ("borrow" if 105_000 < b["cap_market"] else "impact")
    big = C.size_block(3.0, 5e7, 2e8, 0.03, c * 1.03, c * 0.97, c, 35_000, 0.018)
    assert big["limit"] == "borrow" and big["capacity"] == pytest.approx(105_000)
    assert set(b["costs"]) == {"10000", "50000", "100000", "500000", "1000000"}
    assert b["costs"]["1000000"] > b["costs"]["10000"]
    assert b["breakeven"] is not None


def test_spread_assumption_band():
    assert C.tick_floor(0.20) == pytest.approx(0.0005)        # $0.0001 tick under $1 → floor 0.05%
    assert C.tick_floor(0.50) == pytest.approx(0.0005)
    assert C.tick_floor(2.00) == pytest.approx(0.005)         # $0.01 / $2
    assert C.spread_cap(5e5) == 0.03 and C.spread_cap(2e6) == 0.02 and C.spread_cap(2e7) == 0.01
    assert C.spread_used(0.05, 3.0, 2e7) == pytest.approx(0.01)       # noisy 5% read on a liquid name → capped
    assert C.spread_used(0.002, 3.0, 2e7) == pytest.approx(0.01 / 3)   # readable but below one tick → tick
    assert C.spread_used(0.0002, 3.0, 2e7) == pytest.approx((0.01 / 3 + 0.01) / 2)  # ~0 = estimator failed → midpoint
    assert C.spread_used(None, 3.0, 2e7) == pytest.approx((0.01 / 3 + 0.01) / 2)  # unreadable → band midpoint
    wide = float(C.round_trip_cost(1e3, 1e9, 0.02, 0.06))
    assert wide >= 0.06                                   # round_trip_cost never caps the spread it is given
    assert C.breakeven_size(0.05, 1e7, None, 0.06) == 0.0  # spread alone eats the move; sigma None is fine
    flat = np.full(40, 2.0)                               # estimator can't read a spread here
    b = C.size_block(2.0, 5e6, 5e6, 0.03, flat, flat, flat, 1e6, 0.03)
    assert b["spread_estimated"] is False and b["spread"] == pytest.approx(C.spread_used(None, 2.0, 5e6))


def test_vectorised_spread_matches_scalar():
    est = [0.05, 0.002, np.nan, 0.0002]
    price = [3.0, 3.0, 0.4, 12.0]
    dv = [2e7, 2e6, 5e5, 8e6]
    v = C.spread_used_vec(est, price, dv)
    for i in range(4):
        assert v[i] == pytest.approx(C.spread_used(est[i], price[i], dv[i]))
