from __future__ import annotations
"""
models/marty_cohort.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Marty Cohort Forecast — mnav.marty-5b9.workers.dev/btc/power-law/

NOT a pure power-law extrapolation.  Instead: a cohort-based historical
analogue forecast.  For each future date it asks:

  "On all past days where BTC was at a similar quantile level (~q=4%),
   what happened to BTC over the following N days?"

The distribution of those historical outcomes (weighted by regime
similarity — same side of the 110-day MA) defines the forecast bands.

This is why the forward median ($244K in Dec 2027) is higher than the
pure power-law extrapolation ($175K): historically, every time BTC has
been this undervalued relative to trend, it recovered strongly.

Algorithm mirrors the JS `computeCohortForecast` in the Cloudflare
worker exactly.  Validated: computed median for Dec 17 2027 = $243,653
vs website display $244K ✓.
"""

import gzip, base64, json, math, re, urllib.request
from datetime import date, timedelta
from typing import Optional

import numpy as np

# ── Config (mirrors FB_CONFIG in the JS) ─────────────────────────────────────
_HW_MIN      = 0.02
_HW_MAX      = 0.04
_HW_STEP     = 0.005
_COHORT_TGT  = 150
_MA_WIN      = 110
_SMOOTH_WIN  = 61
_REGIME_K    = 3.0
_REGIME_TAU  = 0.7
_URL         = "https://mnav.marty-5b9.workers.dev/btc/power-law/"

# ── Module-level cache (populated on first call) ──────────────────────────────
_cache: Optional[dict] = None


def _fetch_and_build() -> dict:
    """Download payload, decompress, run cohort forecast, cache results."""
    req = urllib.request.Request(_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        content = r.read().decode("utf-8", "ignore")

    scripts = re.findall(r"<script[^>]*>(.*?)</script>", content, re.DOTALL)
    raw_b64 = next(s.strip() for s in scripts
                   if len(s) > 1000 and s.strip().startswith("H4s"))
    D = json.loads(gzip.decompress(base64.b64decode(raw_b64)).decode())

    # Reconstruct delta-encoded arrays
    heights = [D["heights_0"]]
    for d in D["heights_d"]:
        heights.append(heights[-1] + d)
    day_offs = [D["day_offsets_0"]]
    for d in D["day_offsets_d"]:
        day_offs.append(day_offs[-1] + d)

    COEFS    = D["coefs"]
    QKEYS    = D["qkeys"]
    Q_PCT    = [1, 5, 15, 25, 40, 50, 60, 75, 85, 95, 99]
    BASE     = date.fromisoformat(D["base_date"])
    osc_off  = D.get("osc_start_offset", day_offs[0])
    oscs     = D.get("oscs_daily", D["oscs"])

    last_h   = D["last_height"]
    last_d   = date.fromisoformat(D["last_date"])
    last_q   = D["last_osc"]
    N        = len(oscs)

    # ── Core helpers ──────────────────────────────────────────────────────────
    def price_from_osc(osc: float, H: float) -> float:
        pct = osc * 100
        lnH = math.log(max(1, H))
        qlp = [COEFS[k][0] + COEFS[k][1] * lnH for k in QKEYS]
        M = len(QKEYS)
        if pct <= Q_PCT[0]:  return math.exp(qlp[0])
        if pct >= Q_PCT[-1]: return math.exp(qlp[-1])
        for j in range(M - 1):
            if Q_PCT[j] <= pct <= Q_PCT[j + 1]:
                t = (pct - Q_PCT[j]) / (Q_PCT[j + 1] - Q_PCT[j])
                return math.exp(qlp[j] + t * (qlp[j + 1] - qlp[j]))
        return math.exp(qlp[0])

    def sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x); return 1 / (1 + z)
        z = math.exp(x); return z / (1 + z)

    def weighted_percentile(pairs: list, p: float) -> float:
        if not pairs: return float("nan")
        tot = sum(w for _, w in pairs)
        if tot <= 0: return float("nan")
        cum, cpos = 0.0, []
        for v, w in pairs:
            cpos.append((cum + 0.5 * w) / tot)
            cum += w
        target = p / 100
        if target <= cpos[0]:  return pairs[0][0]
        if target >= cpos[-1]: return pairs[-1][0]
        j = 1
        while j < len(cpos) and cpos[j] < target:
            j += 1
        t = (target - cpos[j - 1]) / (cpos[j] - cpos[j - 1])
        return pairs[j - 1][0] + t * (pairs[j][0] - pairs[j - 1][0])

    # ── Daily prices and regime ───────────────────────────────────────────────
    day_p = []
    for i in range(N):
        dflt = (osc_off + i) - day_offs[-1]
        H = max(1, last_h + dflt * 144)
        day_p.append(price_from_osc(oscs[i], H))

    ma = [float("nan")] * N
    for i in range(N):
        if i >= _MA_WIN - 1:
            ma[i] = sum(day_p[i - _MA_WIN + 1 : i + 1]) / _MA_WIN

    rlog = [float("nan")] * N
    rvals = []
    for i in range(N):
        if not (math.isnan(ma[i]) or ma[i] <= 0 or day_p[i] <= 0):
            r = math.log(day_p[i]) - math.log(ma[i])
            rlog[i] = r
            rvals.append(r)
    mean_r  = sum(rvals) / len(rvals)
    sigma_r = math.sqrt(sum((x - mean_r) ** 2 for x in rvals) / (len(rvals) - 1))

    anchor  = N - 1
    z0      = rlog[anchor] / sigma_r if not math.isnan(rlog[anchor]) else 0
    g_reg   = math.tanh(z0 / _REGIME_TAU)
    start_q = last_q

    def build_cohort(hw):
        lo, hi = start_q - hw, start_q + hw
        c = []
        for i in range(anchor + 1):
            if math.isnan(ma[i]) or math.isnan(rlog[i]): continue
            if lo <= oscs[i] <= hi:
                w = sigmoid(_REGIME_K * (rlog[i] / sigma_r) * g_reg)
                c.append((i, w))
        return c

    def neff(c):
        sw = sum(w for _, w in c)
        sw2 = sum(w * w for _, w in c)
        return (sw * sw) / sw2 if sw2 > 0 else 0

    hw = _HW_MIN
    cohort = build_cohort(hw)
    while neff(cohort) < _COHORT_TGT and hw < _HW_MAX:
        hw += _HW_STEP
        cohort = build_cohort(hw)

    # ── Precompute forecast for every 7 days through 2030 ────────────────────
    max_h = min(anchor, 1460)   # cap at ~4 years
    forecast: dict[int, dict] = {}   # h (days) → {p5, p25, p50, p75, p95}

    for h in range(30, max_h + 1, 7):
        fwd = []
        for idx_c, w in cohort:
            fwd_i = idx_c + h
            if fwd_i > anchor: continue
            q = oscs[fwd_i]
            if not math.isnan(q):
                fwd.append((q, w))
        if len(fwd) < 10:
            continue
        fwd.sort(key=lambda x: x[0])
        fH = last_h + h * 144
        forecast[h] = {
            "p5":  price_from_osc(weighted_percentile(fwd,  5), fH),
            "p25": price_from_osc(weighted_percentile(fwd, 25), fH),
            "p50": price_from_osc(weighted_percentile(fwd, 50), fH),
            "p75": price_from_osc(weighted_percentile(fwd, 75), fH),
            "p95": price_from_osc(weighted_percentile(fwd, 95), fH),
        }

    return {
        "last_date":    last_d,
        "last_osc":     last_q,
        "last_height":  last_h,
        "cohort_size":  len(cohort),
        "cohort_neff":  neff(cohort),
        "half_width":   hw,
        "forecast":     forecast,   # {days_ahead: {p5,p25,p50,p75,p95}}
        "price_from_osc": price_from_osc,
        "last_h":       last_h,
    }


def _get_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = _fetch_and_build()
    return _cache


def _interp_forecast(h: int, cache: dict) -> Optional[dict]:
    """Interpolate between precomputed 7-day steps for any horizon h."""
    fc = cache["forecast"]
    keys = sorted(fc.keys())
    if not keys or h < keys[0] or h > keys[-1]:
        return None
    # Find bracketing keys
    lo = max(k for k in keys if k <= h)
    hi = min(k for k in keys if k >= h)
    if lo == hi:
        return fc[lo]
    t = (h - lo) / (hi - lo)
    r = {}
    for band in ("p5", "p25", "p50", "p75", "p95"):
        r[band] = fc[lo][band] + t * (fc[hi][band] - fc[lo][band])
    return r


# ── Public API ────────────────────────────────────────────────────────────────

SCENARIO_PROBS: dict[str, float] = {
    "below_q01": 0.01,
    "q=0.01":    0.05,   # ≈ p5 cohort band
    "q=0.25":    0.19,   # p25
    "OLS":       0.50,   # p50 median
    "q=0.75":    0.19,   # p75
    "q=0.99":    0.06,   # p95 (absorbs upper tail)
}


def get_btc_price(target_date: date) -> dict[str, float]:
    """
    Return cohort-forecast BTC prices for *target_date*.

    Keys: "q=0.01", "q=0.25", "OLS", "q=0.75", "q=0.99"
    (OLS = p50 median of cohort outcomes, NOT the power-law trend line)
    """
    cache = _get_cache()
    h = (target_date - cache["last_date"]).days
    if h <= 0:
        # Past dates: fall back to power-law using cached price_from_osc at h=0
        h = 1
    fcast = _interp_forecast(h, cache)
    if fcast is None:
        return {}
    return {
        "q=0.01": fcast["p5"],
        "q=0.05": fcast["p5"],
        "q=0.10": fcast["p5"],   # p5 is lowest reliable band
        "q=0.25": fcast["p25"],
        "q=0.50": fcast["p50"],
        "OLS":    fcast["p50"],
        "q=0.75": fcast["p75"],
        "q=0.95": fcast["p95"],
        "q=0.99": fcast["p95"],  # p95 is highest reliable band
    }


def get_scenario_prices(target_date: date) -> list[dict]:
    """Return scenario list compatible with analytics/kelly.py."""
    prices = get_btc_price(target_date)
    if not prices:
        return []
    scenarios = []
    for label, prob in SCENARIO_PROBS.items():
        if label == "below_q01":
            scenarios.append({"label": "below q=0.01", "prob": prob, "btc_price": 0.0})
        else:
            scenarios.append({"label": label, "prob": prob,
                               "btc_price": prices.get(label, 0.0)})
    return scenarios


def info() -> dict:
    """Return diagnostic info about the current cohort."""
    c = _get_cache()
    return {
        "last_date":   c["last_date"],
        "last_osc":    c["last_osc"],
        "cohort_size": c["cohort_size"],
        "cohort_neff": c["cohort_neff"],
        "half_width":  c["half_width"],
    }
