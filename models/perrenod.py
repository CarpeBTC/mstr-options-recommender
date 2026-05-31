from __future__ import annotations
"""
models/perrenod.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Perrenod (2026) Bitcoin Power Law + Log-Periodic model, wired into the
same get_btc_price / get_scenario_prices interface as jacobian.py and
block_height.py.

Key differences vs Jacobian / BHM:
  • Single central trend (PL + Fundamental + Harmonic) + symmetric
    normally-distributed noise envelope (σ = 0.077 dex for full model).
  • Quantile prices are derived from sigma levels of the normal
    distribution, NOT from separate quantile regression fits.
  • Scenario probabilities are symmetric, matching the normal-distribution
    probability mass in each inter-quantile interval (midpoint rule).

Probability assignment (midpoint rule on normal distribution):
    below_q01:  0 – 0.5th pct   →  0.01   (total-loss tail)
    q=0.01:     0.5 – 13th pct  →  0.12   (Q1%  ≈ −2.33σ)
    q=0.25:    13 – 37.5th pct  →  0.25   (Q25% ≈ −0.67σ)
    OLS:       37.5 – 62.5th pct→  0.25   (Median / 0σ)
    q=0.75:    62.5 – 87th pct  →  0.25   (Q75% ≈ +0.67σ)
    q=0.99:    87 – 100th pct   →  0.12   (Q99% ≈ +2.33σ; absorbs upper tail)
    ─────────────────────────────────────────────────────────────────────
    Total:                          1.00
"""

import math
from datetime import date
from typing import Optional

# Lazy singleton — avoids re-importing heavy numpy / btc_powerlaw_model
# on every function call.
_model = None

def _get_model():
    global _model
    if _model is None:
        from btc_powerlaw_model import BitcoinPowerLawModel
        _model = BitcoinPowerLawModel()
    return _model


# ── Sigma-level mapping for each quantile label ──────────────────────────────
# Normal-distribution z-scores: scipy.stats.norm.ppf([0.01, 0.25, 0.50, 0.75, 0.99])
_SIGMA_FOR_QUANTILE: dict[str, float] = {
    "q=0.01": -2.3263,   # 1st percentile
    "q=0.25": -0.6745,   # 25th percentile
    "OLS":     0.0,      # median (50th percentile)
    "q=0.75": +0.6745,   # 75th percentile
    "q=0.99": +2.3263,   # 99th percentile
}

# Ordered list for consumers that need a consistent sequence
QUANTILE_ORDER = ["q=0.01", "q=0.25", "OLS", "q=0.75", "q=0.99"]

# Symmetric scenario probabilities (midpoint rule on normal distribution)
SCENARIO_PROBS: dict[str, float] = {
    "below_q01": 0.01,   # total-loss tail (price → 0)
    "q=0.01":    0.12,   # 0.5th to 13th percentile
    "q=0.25":    0.25,   # 13th to 37.5th percentile
    "OLS":       0.25,   # 37.5th to 62.5th percentile  (median)
    "q=0.75":    0.25,   # 62.5th to 87th percentile
    "q=0.99":    0.12,   # 87th to 100th percentile (absorbs upper 1% tail)
}


def get_btc_price(
    target_date: date,
    layer: str = "pl_fund_harmonic",
    model_fit: str = "qr_years",
) -> dict[str, float]:
    """
    Return Perrenod-model BTC prices (USD) for *target_date*.

    Uses the full PL + Fundamental + Harmonic layer (Perrenod argues the
    harmonic component is needed to capture price-surge behaviour that
    the plain power law misses).

    Sigma bands use the model's fitted RMS (0.077 dex for the full model)
    — the same value shown as ±σ bands on the BTC Power Law tab.

    Keys: "q=0.01", "q=0.25", "OLS", "q=0.75", "q=0.99"
    Also returns "OLS" as the canonical median key for display compatibility.
    """
    from btc_powerlaw_model import age_years
    m = _get_model()
    A = age_years(target_date)

    # Central (median) log10 price
    log10_central = m.log10_full(A, layer=layer, model=model_fit)

    # σ for this layer (fixed RMS from fitted model, not age-decaying formula)
    sigma = m.sigma_at(target_date, layer=layer)

    prices: dict[str, float] = {}
    for label, n_sigma in _SIGMA_FOR_QUANTILE.items():
        prices[label] = 10 ** (log10_central + n_sigma * sigma)

    return prices


def get_scenario_prices(target_date: date) -> list[dict]:
    """
    Return scenario list compatible with analytics/kelly.py.
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
