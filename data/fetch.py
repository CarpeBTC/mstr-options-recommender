import re as _re
import json as _json
import urllib.request as _urlreq
from pathlib import Path as _Path
from datetime import datetime, date as _date
from typing import Optional
import time as _time

import yfinance as yf
import pandas as pd
import streamlit as st

# ── File-based last-known-good caches ────────────────────────────────────────
_HOLDINGS_CACHE_FILE      = _Path(__file__).parent / ".holdings_cache.json"
_ASST_HOLDINGS_CACHE_FILE = _Path(__file__).parent / ".asst_holdings_cache.json"

# ── ASST (Strive) hardcoded fallbacks ────────────────────────────────────────
ASST_BTC_HOLDINGS         = 13_628
ASST_FULLY_DILUTED_SHARES_K = 70_313   # thousands (~70.3M, derived from treasury.strive.com)
ASST_REF_DATE             = _date(2026, 3, 17)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _with_retry(fn, retries: int = 3, base_delay: float = 5.0):
    """Call fn(), retrying on Yahoo 429 / rate-limit errors with exponential backoff."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("too many requests", "rate limit", "429")):
                if attempt < retries - 1:
                    _time.sleep(base_delay * (3 ** attempt))   # 5s, 15s, 45s
                    continue
            raise
    raise RuntimeError("Yahoo Finance rate limit exceeded after retries — try again shortly")


@st.cache_data(ttl=480)   # 8 min — staggered from chain (10 min) to avoid simultaneous expiry
def get_equity_data(ticker: str) -> dict:
    """Fetch equity price and option expiries in a single Ticker instantiation.
    Combining these reduces Yahoo Finance API calls from 2 → 1 on cold start."""
    def _fetch():
        t = yf.Ticker(ticker)
        price    = float(t.fast_info.last_price)
        expiries = list(t.options)
        return {"price": price, "expiries": expiries}
    return _with_retry(_fetch)


@st.cache_data(ttl=600)   # 10 min
def get_option_chain(ticker: str, expiry_str: str) -> pd.DataFrame:
    def _fetch():
        chain = yf.Ticker(ticker).option_chain(expiry_str)
        calls = chain.calls.copy()
        calls["mid"] = (calls["bid"] + calls["ask"]) / 2
        return calls[["strike", "lastPrice", "bid", "ask", "mid", "volume", "openInterest", "impliedVolatility"]].copy()
    return _with_retry(_fetch)


@st.cache_data(ttl=360)   # 6 min — staggered from equity data (8 min)
def get_btc_price_live() -> Optional[float]:
    """Fetch live BTC-USD price from yfinance. Returns None on failure."""
    try:
        return _with_retry(lambda: float(yf.Ticker("BTC-USD").fast_info.last_price))
    except Exception:
        return None


@st.cache_data(ttl=3600)  # 1 hour — yield moves slowly
def get_treasury_yield_10y() -> Optional[float]:
    """Fetch current 10-year US Treasury yield (^TNX).
    Returns decimal fraction (e.g. 0.0435 for 4.35%). Returns None on failure."""
    try:
        raw = _with_retry(lambda: float(yf.Ticker("^TNX").fast_info.last_price), retries=2, base_delay=3.0)
        return raw / 100.0   # ^TNX quotes in percent (4.35), convert to decimal
    except Exception:
        return None


@st.cache_data(ttl=480)   # 8 min — preferred stocks trade less actively
def get_preferred_price(ticker: str) -> Optional[float]:
    """Fetch last traded price for a preferred stock (e.g. STRK, STRF, STRC).
    Uses fast_info.last_price; falls back to regularMarketPreviousClose.
    Returns None on failure so callers can use hardcoded CSV fallbacks."""
    try:
        def _fetch():
            t    = yf.Ticker(ticker)
            fi   = t.fast_info
            price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_previous_close", None)
            if price and float(price) > 0:
                return float(price)
            raise ValueError(f"No valid price for {ticker}")
        return _with_retry(_fetch, retries=2, base_delay=3.0)
    except Exception:
        return None


@st.cache_data(ttl=3600)  # refresh hourly — strategy.com updates daily
def get_strategy_holdings() -> Optional[dict]:
    """Fetch latest BTC holdings and assumed diluted shares from strategy.com/shares.

    Returns dict with keys:
        btc_holdings (int)     — total BTC held
        diluted_shares_k (int) — assumed diluted shares outstanding (thousands)
        as_of (str)            — date of the data (YYYY-MM-DD)
        source (str)           — "live" | "cached"

    On live-fetch failure, falls back to the last successfully cached values written
    to .holdings_cache.json.  Returns None only if both the network AND the file
    cache are unavailable — callers should then fall back to hardcoded defaults.
    """
    try:
        req = _urlreq.Request(
            "https://www.strategy.com/shares",
            headers={"User-Agent": _UA},
        )
        with _urlreq.urlopen(req, timeout=10) as r:
            text = r.read().decode("utf-8", errors="ignore")
        match = _re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            text, _re.DOTALL,
        )
        if not match:
            raise ValueError("__NEXT_DATA__ not found")
        rows = _json.loads(match.group(1))["props"]["pageProps"]["shares"]
        latest = max(rows, key=lambda x: x.get("date", ""))

        # ── Annualised YTD BTC yield (BTC per diluted share, linear extrapolation) ──
        # Strategy measures BTC Yield from Dec 31 of the prior year.
        # Use the LAST entry before Jan 1 of the current year as the start point.
        def _bps(row):
            btc = float(row.get("total_bitcoin_holdings", 0) or 0)
            shrs = float(row.get("assumed_diluted_shares_outstanding", 1) or 1)
            return btc / shrs if shrs > 0 else 0.0

        from datetime import datetime as _dt
        year_start = f"{_dt.now().year}-01-01"
        dated = sorted(
            [(r.get("date", "")[:10], r) for r in rows if r.get("date")],
            key=lambda x: x[0],
        )
        # Pre-year rows: Strategy's Dec 31 prior-year reference point
        pre_year  = [(d, r) for d, r in dated if d < year_start]
        post_year = [(d, r) for d, r in dated if d >= year_start]
        btc_yield_ytd_ann = None
        if pre_year and post_year:
            start_date, start_row = pre_year[-1]   # last 2025 entry = Strategy's YTD basis
            end_date,   end_row   = dated[-1]       # most recent entry
            bps_start = _bps(start_row)
            bps_end   = _bps(end_row)
            days = (_dt.fromisoformat(end_date) - _dt.fromisoformat(start_date)).days
            if bps_start > 0 and days > 0:
                ytd_raw = bps_end / bps_start - 1.0
                btc_yield_ytd_ann = ytd_raw * (365.0 / days)   # linear annualisation

        data = {
            "btc_holdings":          int(latest["total_bitcoin_holdings"]),
            "diluted_shares_k":      int(latest["assumed_diluted_shares_outstanding"]),
            "as_of":                 latest["date"],
            "btc_yield_ytd_ann":     btc_yield_ytd_ann,   # float | None
            "source":                "live",
        }
        try:
            _HOLDINGS_CACHE_FILE.write_text(_json.dumps(data))
        except Exception:
            pass
        return data
    except Exception:
        try:
            if _HOLDINGS_CACHE_FILE.exists():
                data = _json.loads(_HOLDINGS_CACHE_FILE.read_text())
                data["source"] = "cached"
                return data
        except Exception:
            pass
        return None


@st.cache_data(ttl=3600)  # refresh hourly — treasury.strive.com updates daily
def get_asst_holdings() -> Optional[dict]:
    """Fetch latest BTC holdings and diluted shares for Strive (ASST).

    Data source: data.strategytracker.com — the backend API powering treasury.strive.com.
    treasury.strive.com is a React SPA; the raw data lives at the API endpoint.

    Returns dict with keys:
        btc_holdings (int)     — total BTC held
        diluted_shares_k (int) — diluted shares (thousands), derived from satsPerShare
        as_of (str)            — data timestamp date (YYYY-MM-DD)
        source (str)           — "live" | "cached"

    On live-fetch failure, falls back to last cached values in .asst_holdings_cache.json.
    Returns None only if both fail — callers should use ASST hardcoded defaults.
    """
    try:
        _headers = {"User-Agent": _UA, "Referer": "https://treasury.strive.com/"}

        # Step 1: resolve current versioned filename
        req = _urlreq.Request("https://data.strategytracker.com/latest.json", headers=_headers)
        with _urlreq.urlopen(req, timeout=10) as r:
            latest_meta = _json.loads(r.read().decode())

        light_file = latest_meta["files"]["light"]
        timestamp  = latest_meta.get("timestamp", "")

        # Step 2: fetch the light data file
        req2 = _urlreq.Request(
            f"https://data.strategytracker.com/{light_file}",
            headers=_headers,
        )
        with _urlreq.urlopen(req2, timeout=10) as r:
            payload = _json.loads(r.read().decode())

        asst           = payload["companies"]["ASST"]
        btc_holdings   = float(asst["holdings"])
        sats_per_share = int(asst["satsPerShare"])
        btc_per_share  = sats_per_share / 1e8
        diluted_shares = int(btc_holdings / btc_per_share)
        diluted_shares_k = diluted_shares // 1000
        as_of = timestamp[:10] if timestamp else ""

        # btcYieldYtd is the raw YTD BTC/share growth (same metric as MSTR's YTD yield).
        # Annualise using days elapsed since Jan 1 of current year (linear, same method as MSTR).
        from datetime import datetime as _dt2
        year_start = _dt2(_dt2.now().year, 1, 1)
        days_elapsed = (_dt2.now() - year_start).days
        btc_yield_ytd_raw = asst.get("btcYieldYtd")   # e.g. 36.66 (percent)
        btc_yield_ytd_ann = None
        if btc_yield_ytd_raw is not None and days_elapsed > 0:
            raw_frac = float(btc_yield_ytd_raw) / 100.0
            btc_yield_ytd_ann = raw_frac * (365.0 / days_elapsed)

        data = {
            "btc_holdings":     int(btc_holdings),
            "diluted_shares_k": diluted_shares_k,
            "as_of":            as_of,
            "btc_yield_ytd_ann": btc_yield_ytd_ann,   # annualised, same methodology as MSTR
            "nav_premium":      asst.get("navPremium"),
            "source":           "live",
        }
        try:
            _ASST_HOLDINGS_CACHE_FILE.write_text(_json.dumps(data))
        except Exception:
            pass
        return data
    except Exception:
        try:
            if _ASST_HOLDINGS_CACHE_FILE.exists():
                data = _json.loads(_ASST_HOLDINGS_CACHE_FILE.read_text())
                data["source"] = "cached"
                return data
        except Exception:
            pass
        return None


@st.cache_data(ttl=600)  # refresh every 10 minutes — ~1 block interval
def get_block_height_live() -> Optional[int]:
    """Fetch current Bitcoin block height.
    Tries mempool.space first, falls back to blockchain.info.
    Returns None if both sources fail — callers should fall back to dead-reckoning.
    """
    for url in [
        "https://mempool.space/api/blocks/tip/height",
        "https://blockchain.info/q/getblockcount",
    ]:
        try:
            req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with _urlreq.urlopen(req, timeout=5) as r:
                return int(r.read().decode().strip())
        except Exception:
            continue
    return None


def get_last_updated() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
