from datetime import date
from typing import Optional

# Hardcoded fallbacks — used only when live fetch from strategy.com fails
BTC_HOLDINGS = 738_731
FULLY_DILUTED_SHARES_K = 374_506  # thousands
REF_DATE = date(2026, 3, 4)


def _btc_per_share(
    target_date: date,
    btc_yield_yr1: float = 0.10,
    btc_holdings: Optional[int] = None,
    diluted_shares_k: Optional[int] = None,
    ref_date: Optional[date] = None,
) -> float:
    """BTC per fully diluted share, growing at BTC yield rate (declining 1%/yr from btc_yield_yr1)."""
    years = (target_date - (ref_date or REF_DATE)).days / 365.25
    btc = float(btc_holdings if btc_holdings is not None else BTC_HOLDINGS)
    shares_k = float(diluted_shares_k if diluted_shares_k is not None else FULLY_DILUTED_SHARES_K)
    year = 0
    while year < years:
        fraction = min(1.0, years - year)
        annual_yield = max(0.0, btc_yield_yr1 - 0.01 * year)
        btc *= (1 + annual_yield * fraction)
        year += 1
    return btc / (shares_k * 1000)


def btc_to_mstr_1x_mnav(
    btc_price: float,
    target_date: date,
    btc_yield_yr1: float = 0.10,
    btc_holdings: Optional[int] = None,
    diluted_shares_k: Optional[int] = None,
    ref_date: Optional[date] = None,
) -> float:
    """Convert BTC price to MSTR price at 1x mNAV."""
    if btc_price <= 0:
        return 0.0
    return btc_price * _btc_per_share(
        target_date, btc_yield_yr1, btc_holdings, diluted_shares_k, ref_date
    )


def btc_to_mstr(
    btc_price: float,
    target_date: date,
    mnav_multiplier: float = 1.5,
    btc_yield_yr1: float = 0.10,
    btc_holdings: Optional[int] = None,
    diluted_shares_k: Optional[int] = None,
    ref_date: Optional[date] = None,
) -> float:
    """Convert BTC price to MSTR market price at given mNAV multiplier."""
    return btc_to_mstr_1x_mnav(
        btc_price, target_date, btc_yield_yr1, btc_holdings, diluted_shares_k, ref_date
    ) * mnav_multiplier


def apply_mnav(
    scenarios: list,
    target_date: date,
    mnav_multiplier: float = 1.5,
    btc_yield_yr1: float = 0.10,
    btc_holdings: Optional[int] = None,
    diluted_shares_k: Optional[int] = None,
    ref_date: Optional[date] = None,
) -> list:
    """Add mstr_price to each scenario dict."""
    result = []
    for s in scenarios:
        s = s.copy()
        s["mstr_price"] = btc_to_mstr(
            s["btc_price"], target_date, mnav_multiplier, btc_yield_yr1,
            btc_holdings, diluted_shares_k, ref_date,
        )
        result.append(s)
    return result
