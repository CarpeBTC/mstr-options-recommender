from datetime import date

# From spreadsheet (March 2026 snapshot)
BTC_HOLDINGS = 738_731
FULLY_DILUTED_SHARES_K = 374_506  # thousands
REF_DATE = date(2026, 3, 4)


def _btc_per_share(target_date: date, btc_yield_yr1: float = 0.15) -> float:
    """BTC per fully diluted share, growing at BTC yield rate (declining 1%/yr from btc_yield_yr1)."""
    years = (target_date - REF_DATE).days / 365.25
    btc = BTC_HOLDINGS
    shares_k = FULLY_DILUTED_SHARES_K
    year = 0
    while year < years:
        fraction = min(1.0, years - year)
        annual_yield = max(0.0, btc_yield_yr1 - 0.01 * year)
        btc *= (1 + annual_yield * fraction)
        year += 1
    return btc / (shares_k * 1000)


def btc_to_mstr_1x_mnav(btc_price: float, target_date: date, btc_yield_yr1: float = 0.15) -> float:
    """Convert BTC price to MSTR price at 1x mNAV."""
    if btc_price <= 0:
        return 0.0
    return btc_price * _btc_per_share(target_date, btc_yield_yr1)


def btc_to_mstr(btc_price: float, target_date: date, mnav_multiplier: float = 1.5,
                btc_yield_yr1: float = 0.15) -> float:
    """Convert BTC price to MSTR market price at given mNAV multiplier."""
    return btc_to_mstr_1x_mnav(btc_price, target_date, btc_yield_yr1) * mnav_multiplier


def apply_mnav(scenarios: list[dict], target_date: date, mnav_multiplier: float = 1.5,
               btc_yield_yr1: float = 0.15) -> list[dict]:
    """Add mstr_price to each scenario dict."""
    result = []
    for s in scenarios:
        s = s.copy()
        s["mstr_price"] = btc_to_mstr(s["btc_price"], target_date, mnav_multiplier, btc_yield_yr1)
        result.append(s)
    return result
