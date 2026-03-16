import yfinance as yf
import pandas as pd
import streamlit as st
import requests
from datetime import datetime
from typing import Optional

# Custom session with browser-like headers to avoid rate limiting on cloud servers
def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    })
    return session


@st.cache_data(ttl=300)
def get_mstr_price() -> float:
    ticker = yf.Ticker("MSTR", session=_make_session())
    info = ticker.fast_info
    return float(info.last_price)


@st.cache_data(ttl=300)
def get_available_expiries() -> list[str]:
    ticker = yf.Ticker("MSTR", session=_make_session())
    return list(ticker.options)


@st.cache_data(ttl=300)
def get_option_chain(expiry_str: str) -> pd.DataFrame:
    ticker = yf.Ticker("MSTR", session=_make_session())
    chain = ticker.option_chain(expiry_str)
    calls = chain.calls.copy()
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2
    return calls[["strike", "bid", "ask", "mid", "volume", "openInterest", "impliedVolatility"]].copy()


@st.cache_data(ttl=300)
def get_btc_price_live() -> Optional[float]:
    """Fetch live BTC-USD price from yfinance. Returns None on failure."""
    try:
        ticker = yf.Ticker("BTC-USD", session=_make_session())
        return float(ticker.fast_info.last_price)
    except Exception:
        return None


def get_last_updated() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
