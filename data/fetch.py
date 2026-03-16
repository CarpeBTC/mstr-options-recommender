import re as _re
import json as _json
import urllib.request as _urlreq
from pathlib import Path as _Path
import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Optional

_HOLDINGS_CACHE_FILE = _Path(__file__).parent / ".holdings_cache.json"


@st.cache_data(ttl=300)
def get_mstr_price() -> float:
    ticker = yf.Ticker("MSTR")
    info = ticker.fast_info
    return float(info.last_price)


@st.cache_data(ttl=300)
def get_available_expiries() -> list[str]:
    ticker = yf.Ticker("MSTR")
    return list(ticker.options)


@st.cache_data(ttl=300)
def get_option_chain(expiry_str: str) -> pd.DataFrame:
    ticker = yf.Ticker("MSTR")
    chain = ticker.option_chain(expiry_str)
    calls = chain.calls.copy()
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2
    return calls[["strike", "lastPrice", "bid", "ask", "mid", "volume", "openInterest", "impliedVolatility"]].copy()


@st.cache_data(ttl=300)
def get_btc_price_live() -> Optional[float]:
    """Fetch live BTC-USD price from yfinance. Returns None on failure."""
    try:
        ticker = yf.Ticker("BTC-USD")
        return float(ticker.fast_info.last_price)
    except Exception:
        return None


@st.cache_data(ttl=3600)  # refresh hourly — strategy.com updates daily
def get_strategy_holdings() -> Optional[dict]:
    """Fetch latest BTC holdings and assumed diluted shares from strategy.com/shares.

    Returns dict with keys:
        btc_holdings (int)    — total BTC held
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
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
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
        data = {
            "btc_holdings":     int(latest["total_bitcoin_holdings"]),
            "diluted_shares_k": int(latest["assumed_diluted_shares_outstanding"]),
            "as_of":            latest["date"],
            "source":           "live",
        }
        # Persist last-known-good so future failures can fall back here
        try:
            _HOLDINGS_CACHE_FILE.write_text(_json.dumps(data))
        except Exception:
            pass
        return data
    except Exception:
        # Live fetch failed — return last cached values rather than hardcoded defaults
        try:
            if _HOLDINGS_CACHE_FILE.exists():
                data = _json.loads(_HOLDINGS_CACHE_FILE.read_text())
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
