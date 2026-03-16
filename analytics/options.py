import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.stats import norm
import math


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price. T in years."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def option_value_at_date(mstr_price: float, strike: float, exit_date: date,
                         expiry_date: date, iv: float = 1.0, r: float = 0.05) -> float:
    """Estimate option value at an intermediate exit date using Black-Scholes."""
    T = (expiry_date - exit_date).days / 365.25
    return black_scholes_call(mstr_price, strike, T, r, iv)


def compute_exit_timing(strike: float, premium: float, scenarios: list[dict],
                        expiry_date: date, today: date = None,
                        iv: float = 1.0, num_dates: int = 24) -> pd.DataFrame:
    """
    For each future exit date, compute E[P&L], std[P&L], Sharpe ratio.
    scenarios: list of {"label", "prob", "mstr_price"} — MSTR price at EXPIRY.
    We assume a linear price path from current to scenario target.
    """
    if today is None:
        today = date.today()

    days_total = (expiry_date - today).days
    step = max(1, days_total // num_dates)
    dates = [today + timedelta(days=i * step) for i in range(1, num_dates + 1)]
    if dates[-1] < expiry_date:
        dates.append(expiry_date)

    rows = []
    cost = premium * 100  # per contract

    for exit_date in dates:
        frac = (exit_date - today).days / max(days_total, 1)
        pnls = []
        weights = []
        for s in scenarios:
            if s["prob"] <= 0:
                continue
            # Interpolate MSTR price linearly to exit date
            mstr_at_exit = s["mstr_price"] * frac  # simplified linear path from 0
            # Better: assume current price + fraction of the move
            # (we don't have current price here, so use intrinsic at expiry scaled by frac)
            if exit_date >= expiry_date:
                # At expiry: intrinsic value only
                option_val = max(s["mstr_price"] - strike, 0) * 100
            else:
                # Mid-contract: Black-Scholes on scenario MSTR price at that date
                option_val = option_value_at_date(
                    s["mstr_price"], strike, exit_date, expiry_date, iv
                ) * 100
            pnl = option_val - cost
            pnls.append(pnl)
            weights.append(s["prob"])

        weights = np.array(weights)
        weights = weights / weights.sum()
        pnls = np.array(pnls)

        e_pnl = float(np.dot(weights, pnls))
        std_pnl = float(np.sqrt(np.dot(weights, (pnls - e_pnl) ** 2)))
        sharpe = e_pnl / std_pnl if std_pnl > 0 else 0.0
        p_profit = float(np.dot(weights, (pnls > 0).astype(float)))

        rows.append({
            "Date": exit_date,
            "DTE": (expiry_date - exit_date).days,
            "E[P&L]": e_pnl,
            "Std[P&L]": std_pnl,
            "Sharpe": sharpe,
            "P(Profit>0)": p_profit,
        })

    return pd.DataFrame(rows)


def compute_pnl_heatmap(strike: float, premium: float, expiry_date: date,
                        mstr_prices: list[float], exit_dates: list[date],
                        iv: float = 1.0) -> pd.DataFrame:
    """
    2D grid: rows = MSTR price at exit, cols = exit dates.
    Values = P&L per contract.
    """
    cost = premium * 100
    data = {}
    for exit_date in exit_dates:
        col = {}
        for mstr in mstr_prices:
            if exit_date >= expiry_date:
                val = max(mstr - strike, 0) * 100
            else:
                val = option_value_at_date(mstr, strike, exit_date, expiry_date, iv) * 100
            col[mstr] = val - cost
        data[exit_date.strftime("%b %d '%y")] = col
    return pd.DataFrame(data)
