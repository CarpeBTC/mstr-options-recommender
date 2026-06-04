from __future__ import annotations
"""
models/marty.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Marty power-law model — mnav.marty-5b9.workers.dev/btc/power-law/

Log-log quantile regression of BTC price against block height,
fitted on empirical quantiles since 2010. 11 quantile bands.

Formula:  price = exp(intercept + slope × ln(block_height))

Coefficients last fetched: June 4, 2026 (updated quarterly).
"""

import math
from datetime import date
from typing import Optional

# ── Coefficients ──────────────────────────────────────────────────────────────
# Source: DATA.coefs from embedded payload at mnav.marty-5b9.workers.dev
# Format: [intercept, slope]  →  price = exp(intercept + slope × ln(height))
COEFS: dict[str, list[float]] = {
    "Q1":  [-71.492910, 5.991690],
    "Q5":  [-70.386493, 5.918261],
    "Q15": [-68.723755, 5.807605],
    "Q25": [-67.453237, 5.723190],
    "Q40": [-66.459790, 5.659477],
    "Q50": [-65.517327, 5.602862],   # median
    "Q60": [-64.215399, 5.519452],
    "Q75": [-62.051080, 5.375060],
    "Q85": [-59.601178, 5.212293],
    "Q95": [-56.596430, 5.017140],
    "Q99": [-52.794531, 4.752845],
    "OLS": [-64.588712, 5.539204],   # OLS (use Q50 for median scenarios)
}

_QTAUS = [0.01, 0.05, 0.15, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95, 0.99]
_QKEYS = ["Q1", "Q5", "Q15", "Q25", "Q40", "Q50", "Q60", "Q75", "Q85", "Q95", "Q99"]

# Block-height dead-reckoning reference (updated Jun 4, 2026)
REF_BLOCK      = 952_462
REF_DATE       = date(2026, 6, 4)
BLOCKS_PER_DAY = 144   # ~10-min average

# Scenario probability weights (midpoint rule on empirical distribution)
SCENARIO_PROBS: dict[str, float] = {
    "below_q01": 0.01,
    "q=0.01":    0.24,   # Q1%–Q25% range
    "q=0.25":    0.00,   # skipped — OLS covers the central scenario
    "OLS":       0.50,   # median (Q50)
    "q=0.75":    0.15,   # Q75%–Q95% range
    "q=0.99":    0.10,   # Q95%+ (absorbs upper tail)
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _block_height(target: date, ref_height: Optional[int] = None,
                  ref_date: Optional[date] = None) -> int:
    rh = ref_height if ref_height is not None else REF_BLOCK
    rd = ref_date   if ref_date   is not None else REF_DATE
    return rh + (target - rd).days * BLOCKS_PER_DAY


def _price_at_height(height: int, qkey: str) -> float:
    c = COEFS[qkey]
    return math.exp(c[0] + c[1] * math.log(height))


def _interpolate(tau: float, height: int) -> float:
    """Interpolate price for any quantile tau ∈ (0,1)."""
    for i in range(len(_QTAUS) - 1):
        if _QTAUS[i] <= tau <= _QTAUS[i + 1]:
            frac = (tau - _QTAUS[i]) / (_QTAUS[i + 1] - _QTAUS[i])
            p0 = _price_at_height(height, _QKEYS[i])
            p1 = _price_at_height(height, _QKEYS[i + 1])
            return p0 + frac * (p1 - p0)
    # Exact match
    idx = _QTAUS.index(tau)
    return _price_at_height(height, _QKEYS[idx])


# ── Public API ────────────────────────────────────────────────────────────────

def get_btc_price(
    target_date: date,
    ref_height: Optional[int] = None,
    ref_date: Optional[date] = None,
) -> dict[str, float]:
    """
    Return BTC prices at standard quantile labels for *target_date*.

    Keys: "q=0.01", "q=0.05", "q=0.10", "q=0.25", "OLS",
          "q=0.50", "q=0.75", "q=0.85", "q=0.95", "q=0.99"
    Also returns "OLS" = Q50 for display compatibility.
    """
    h = _block_height(target_date, ref_height, ref_date)
    prices: dict[str, float] = {}

    # Direct matches from model
    for qkey, tau in zip(_QKEYS, _QTAUS):
        label = f"q={tau:.2f}".replace("q=0.50", "OLS")
        prices[label] = _price_at_height(h, qkey)

    # Add the 5 standard labels used by the blend
    prices["q=0.01"]  = _price_at_height(h, "Q1")
    prices["q=0.25"]  = _price_at_height(h, "Q25")
    prices["OLS"]     = _price_at_height(h, "Q50")   # median as OLS alias
    prices["q=0.75"]  = _price_at_height(h, "Q75")
    prices["q=0.99"]  = _price_at_height(h, "Q99")

    # Cowen-compatible labels
    prices["q=0.10"]  = _interpolate(0.10, h)
    prices["q=0.95"]  = _price_at_height(h, "Q95")
    prices["q=0.50"]  = _price_at_height(h, "Q50")

    return prices


def get_scenario_prices(
    target_date: date,
    ref_height: Optional[int] = None,
    ref_date: Optional[date] = None,
) -> list[dict]:
    """Return scenario list compatible with analytics/kelly.py."""
    prices = get_btc_price(target_date, ref_height, ref_date)
    scenarios: list[dict] = []
    for label, prob in SCENARIO_PROBS.items():
        if prob == 0.0:
            continue
        if label == "below_q01":
            scenarios.append({"label": "below q=0.01", "prob": prob, "btc_price": 0.0})
        else:
            scenarios.append({"label": label, "prob": prob,
                               "btc_price": prices.get(label, 0.0)})
    return scenarios
