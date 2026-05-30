from __future__ import annotations  # enables dict[str, ...] syntax on Python 3.9
import math
from datetime import date
from typing import Dict, Tuple, List

# ── Cowen (2026) Asymmetric Quadratic Quantile Regression ──────────────────
# "Asymmetric Tail Curvature in Bitcoin Price Quantiles"
# Working Paper, May 29, 2026  |  benjamincowen.com
#
# Model:  log₁₀(P) = cτ + aτ·x + b(τ)·x²
#         x = ln(t) − μ,  t = days since Jan 1 2009,  μ = 7.9914
#
# Curvature groups (Table 4):
#   bᴸᴼ = −0.0241  (τ ≤ 0.25): near-linear lower tail, p = 0.258 (not sig.)
#   bᴹᴱᴰ = −0.1126 (τ = 0.50)
#   bᴴᴵ = −0.3259  (τ ≥ 0.75): significant upper-tail compression, p = 0.019
#
# Key finding: upper-tail quantiles curve DOWN in log-log space —
# future price ceilings are lower than linear power-law models predict.
# This corrects the ~+32% systematic optimistic bias of OLS power law (2019–2026).
#
# Rearrangement: predictions are sorted ascending at each date to guarantee
# quantile non-crossing per Chernozhukov, Fernández-Val & Galichon (2010).
# Without rearrangement, Q75% and Q95% would cross by ~Dec 2026.
# ────────────────────────────────────────────────────────────────────────────

# Time anchor: January 1, 2009 (Cowen's calendar anchor, not the Jan 3 genesis block)
GENESIS_DATE = date(2009, 1, 1)

# Centering constant: sample mean of ln(t) over the 2010–2026 calibration data.
# μ = 7.9914  ≈  ln(2955 days)  ≈  early February 2017
MU = 7.9914

# Shared curvature parameters (Table 4, full-sample estimates)
B_LO  = -0.0241   # lower tail  (τ ∈ {0.01, 0.10, 0.25})
B_MED = -0.1126   # median      (τ = 0.50)
B_HI  = -0.3259   # upper tail  (τ ∈ {0.75, 0.95, 0.99})

# Per-quantile coefficients: {label: (c_tau, a_tau, b_tau)} — Table 3
# All values as published; no estimation performed here.
QUANTILES: dict[str, tuple[float, float, float]] = {
    "q=0.01": (2.837, 2.578, B_LO),
    "q=0.10": (2.933, 2.552, B_LO),
    "q=0.25": (3.004, 2.554, B_LO),
    "q=0.50": (3.214, 2.482, B_MED),
    "q=0.75": (3.562, 2.283, B_HI),
    "q=0.95": (3.897, 1.964, B_HI),
    "q=0.99": (4.028, 1.904, B_HI),
}

# Ordered label list for rearrangement (must stay sorted by τ ascending)
_QUANTILE_ORDER = ["q=0.01", "q=0.10", "q=0.25", "q=0.50", "q=0.75", "q=0.95", "q=0.99"]

# Scenario probability mass per quantile band.
# Approximated from inter-quantile interval widths; sums to 1.00.
# Upper tail is intentionally more compressed than Jacobian/BHM because
# Cowen's bᴴᴵ bends the Q95%/Q99% curves downward — use these probabilities
# as-calibrated rather than forcing a match to other models' structures.
SCENARIO_PROBS: dict[str, float] = {
    "q=0.01":    0.04,   # 0th–5th percentile band
    "q=0.10":    0.10,   # 5th–17th percentile band
    "q=0.25":    0.20,   # 17th–37th percentile band
    "q=0.50":    0.25,   # 37th–62nd percentile band  (median / central)
    "q=0.75":    0.20,   # 62nd–85th percentile band
    "q=0.95":    0.15,   # 85th–97th percentile band
    "q=0.99":    0.05,   # 97th–100th percentile band
    "below_q01": 0.01,   # tail losses below Q1%
}


def get_btc_price(target_date: date) -> dict[str, float]:
    """
    Return rearranged asymmetric quadratic quantile BTC prices (USD) for target_date.

    Keys: "q=0.01", "q=0.10", "q=0.25", "q=0.50", "q=0.75", "q=0.95", "q=0.99"
    Also returns "OLS" as an alias for "q=0.50" for display compatibility
    with the Jacobian and Block Height models (which use "OLS" for their median).

    Rearrangement step: raw log₁₀ predictions are sorted ascending and
    re-assigned to quantile labels in order, preventing Q75%/Q95% crossing
    that would otherwise occur outside the calibration window.
    """
    t_days = (target_date - GENESIS_DATE).days
    if t_days <= 0:
        return {k: 0.0 for k in _QUANTILE_ORDER + ["OLS"]}

    x = math.log(t_days) - MU

    # Compute raw (un-rearranged) log₁₀ price for each quantile
    raw_log10 = [
        c + a * x + b * x * x
        for (c, a, b) in (QUANTILES[lbl] for lbl in _QUANTILE_ORDER)
    ]

    # Rearrangement: sort ascending, then zip back to ordered labels
    sorted_log10 = sorted(raw_log10)
    prices: dict[str, float] = {
        lbl: 10 ** lp for lbl, lp in zip(_QUANTILE_ORDER, sorted_log10)
    }
    prices["OLS"] = prices["q=0.50"]   # alias for display compatibility
    return prices


def get_scenario_prices(target_date: date) -> list[dict]:
    """
    Return a scenario list compatible with analytics/kelly.py.
    Each dict: {"label": str, "prob": float, "btc_price": float}
    """
    prices = get_btc_price(target_date)
    scenarios: list[dict] = []
    for label, prob in SCENARIO_PROBS.items():
        if label == "below_q01":
            scenarios.append({"label": "below q=0.01", "prob": prob, "btc_price": 0.0})
        else:
            scenarios.append({"label": label, "prob": prob, "btc_price": prices[label]})
    return scenarios
