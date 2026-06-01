"""
Portfolio Performance & Forecast Tab

Tracks MSTR / ASST equity, options, preferred stock, and direct BTC positions
sourced from the Fidelity export dated 2026-05-30.  Forecasts the distribution
of returns at each calendar quarter-end through 2028 using the active BTC price
model blend.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.options import black_scholes_call
from models import block_height, cowen, jacobian

# ── Reference prices from Fidelity export 2026-05-30 ─────────────────────────
# These are used as fallback when Yahoo Finance live prices are unavailable.
# Source: "Portfolio Value-Brokerage-2026-05-30.csv" Quote/Price column.
BTC_REF_PRICE  = 73_301.53   # INDEX:NQBT quote (BTC spot)
MSTR_REF_PRICE = 159.09
ASST_REF_PRICE = 17.67

# ── Position data (Fidelity export 2026-05-30) ───────────────────────────────
#
# ptype:
#   "equity"    — common shares; value = shares × projected_equity_price
#   "preferred" — fixed-rate preferred; held flat at ref_mv in forecast
#   "call"      — long call; value via Black-Scholes (pre-expiry) / intrinsic
#   "btc_etf"   — BTC spot ETF (FBTC, IBIT); value = btc_equiv × projected_btc
#   "btc_cold"  — cold-storage BTC (INDEX:NQBT); value = btc_qty × projected_btc
#
# quote_price: exact price from CSV "Quote / Price" column
# contracts:   Fidelity "Shares" column ÷ 100
#   C250:     400 ÷ 100 = 4  @ $40.55/share
#   C350:   1,200 ÷ 100 = 12 @ $27.95/share
#   C400:     600 ÷ 100 = 6  @ $24.00/share
#   ASST C25: 400 ÷ 100 = 4  @ $7.90/share

POSITIONS: List[dict] = [
    # ── Strategy / MSTR ──────────────────────────────────────────────────────
    # shares / contracts sourced from Fidelity portfolio export 2026-05-30
    dict(symbol="MSTR",           name="Strategy Cl A (MSTR)",        category="Strategy",
         ptype="equity",    underlying="MSTR",  shares=327.441,  quote_price=159.09,
         strike=None, expiry=None, contracts=None,
         cost_basis=54_191.25,  ref_mv=52_092.59),

    dict(symbol="STRK",           name="Strategy 10% Perp Pfd (STRK)",category="Strategy",
         ptype="preferred", underlying=None,    shares=930,      quote_price=70.27,
         strike=None, expiry=None, contracts=None,
         cost_basis=72_958.41,  ref_mv=65_351.10),

    dict(symbol="STRF",           name="Strategy 10% Pfd (STRF)",     category="Strategy",
         ptype="preferred", underlying=None,    shares=300,      quote_price=98.50,
         strike=None, expiry=None, contracts=None,
         cost_basis=30_217.50,  ref_mv=29_550.00),

    dict(symbol="STRC",           name="Strategy 11% Var Pfd (STRC)", category="Strategy",
         ptype="preferred", underlying=None,    shares=500,      quote_price=98.99,
         strike=None, expiry=None, contracts=None,
         cost_basis=49_999.08,  ref_mv=49_495.00),

    dict(symbol="MSTR271217C250", name="MSTR Dec'27 $250 Call",       category="Strategy",
         ptype="call",      underlying="MSTR",  quote_price=40.55,
         strike=250.0, expiry=date(2027, 12, 17), contracts=4,
         cost_basis=17_742.69,  ref_mv=16_220.00),

    dict(symbol="MSTR271217C350", name="MSTR Dec'27 $350 Call",       category="Strategy",
         ptype="call",      underlying="MSTR",  quote_price=27.95,
         strike=350.0, expiry=date(2027, 12, 17), contracts=12,
         cost_basis=35_948.08,  ref_mv=33_540.00),

    dict(symbol="MSTR271217C400", name="MSTR Dec'27 $400 Call",       category="Strategy",
         ptype="call",      underlying="MSTR",  quote_price=24.00,
         strike=400.0, expiry=date(2027, 12, 17), contracts=6,
         cost_basis=15_604.04,  ref_mv=14_400.00),

    # ── Strive / ASST ─────────────────────────────────────────────────────────
    dict(symbol="ASST",           name="Strive Inc Cl A (ASST)",      category="Strive",
         ptype="equity",    underlying="ASST",  shares=1_200,    quote_price=17.67,
         strike=None, expiry=None, contracts=None,
         cost_basis=18_600.00,  ref_mv=21_204.00),

    dict(symbol="SATA",           name="Strive 12.25% Var Pfd (SATA)",category="Strive",
         ptype="preferred", underlying=None,    shares=50,       quote_price=100.01,
         strike=None, expiry=None, contracts=None,
         cost_basis=4_900.74,   ref_mv=5_000.50),

    dict(symbol="ASST280121C25",  name="ASST Jan'28 $25 Call",        category="Strive",
         ptype="call",      underlying="ASST",  quote_price=7.90,
         strike=25.0, expiry=date(2028, 1, 21), contracts=4,
         cost_basis=3_394.69,   ref_mv=3_160.00),

    # ── Bitcoin (direct / ETF) ────────────────────────────────────────────────
    dict(symbol="FBTC",           name="Fidelity Bitcoin Fund (FBTC)", category="Bitcoin",
         ptype="btc_etf",   underlying="BTC",  shares=37.392,   quote_price=63.90,
         strike=None, expiry=None, contracts=None,
         cost_basis=2_383.55,   ref_mv=2_389.35),

    dict(symbol="IBIT",           name="iShares Bitcoin Trust (IBIT)", category="Bitcoin",
         ptype="btc_etf",   underlying="BTC",  shares=127.766,  quote_price=41.63,
         strike=None, expiry=None, contracts=None,
         cost_basis=5_549.92,   ref_mv=5_318.90),

    dict(symbol="INDEX:NQBT",     name="BTC Cold Storage (NQBT)",      category="Bitcoin",
         ptype="btc_cold",  underlying="BTC",  quote_price=BTC_REF_PRICE,
         strike=None, expiry=None, contracts=None,
         btc_qty=1.20595,          # exact BTC in cold storage (user-confirmed)
         cost_basis=86_965.69,  ref_mv=88_572.49),
]

_TOTAL_COST = sum(p["cost_basis"] for p in POSITIONS)
_TOTAL_MV   = sum(p["ref_mv"]    for p in POSITIONS)

# Display quantiles — labels must exist in all three BTC model outputs
_QUANTILES = ["q=0.01", "q=0.25", "OLS", "q=0.75", "q=0.99"]
_Q_LABELS  = {
    "q=0.01": "Bear (Q1%)",
    "q=0.25": "Low (Q25%)",
    "OLS":    "Median",
    "q=0.75": "Bull (Q75%)",
    "q=0.99": "Euphoria (Q99%)",
}
_Q_COLORS = {
    "q=0.01": "#d62728",
    "q=0.25": "#ff7f0e",
    "OLS":    "#F7931A",
    "q=0.75": "#2ca02c",
    "q=0.99": "#1f77b4",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quarter_end_dates() -> List[date]:
    """Quarter-end dates from today through Dec 31 2028."""
    today = date.today()
    qends = []
    for year in (2026, 2027, 2028):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            qdate = date(year, m, d)
            if qdate > today:
                qends.append(qdate)
    return qends


def _blend_btc(
    target: date,
    quantile: str,
    use_j: bool,
    use_b: bool,
    use_c: bool,
    bhm_fn: Callable,
    use_p: bool = False,
) -> float:
    """Average BTC price at *target* across checked models for *quantile*."""
    from models import perrenod as _perrenod
    vals = []
    if use_j:
        vals.append(jacobian.get_btc_price(target).get(quantile, 0.0))
    if use_b:
        vals.append(bhm_fn(target).get(quantile, 0.0))
    if use_c:
        vals.append(cowen.get_btc_price(target).get(quantile, 0.0))
    if use_p:
        vals.append(_perrenod.get_btc_price(target).get(quantile, 0.0))
    return float(np.mean(vals)) if vals else 0.0


def _option_value(
    S: float,
    strike: float,
    forecast_date: date,
    expiry: date,
    contracts: int,
    iv: float,
    r: float = 0.05,
) -> float:
    """Option market value at *forecast_date* given underlying price *S*."""
    T = (expiry - forecast_date).days / 365.25
    if T <= 0:
        return max(S - strike, 0.0) * 100 * contracts
    return black_scholes_call(S, strike, T, r, iv) * 100 * contracts


# ── Main render function ───────────────────────────────────────────────────────

def render_portfolio_tab(
    mstr_price_live: float,
    asst_price_live: float,
    btc_price_live: float,
    strk_price_live: Optional[float],
    use_jacobian: bool,
    use_bhm: bool,
    use_cowen: bool,
    use_perrenod: bool,
    mnav: float,
    asst_mnav: float,
    btc_yield: float,
    btc_to_mstr_fn: Callable,   # partial(btc_to_mstr, MSTR holdings baked in)
    btc_to_asst_fn: Callable,   # partial(btc_to_mstr, ASST holdings baked in)
    bhm_price_fn: Callable,     # partial(block_height.get_btc_price, ref_height=…, ref_date=…)
) -> None:
    today = date.today()

    # ── Safe today prices (must be defined before _live_mv closure) ───────────
    # Preference order: (1) live Yahoo Finance, (2) CSV quote_price, (3) hardcoded default.
    def _ref_price(symbol: str, default: float) -> float:
        p = next((p for p in POSITIONS if p["symbol"] == symbol), None)
        return p["quote_price"] if p and "quote_price" in p else default

    mstr_p_today  = mstr_price_live  if mstr_price_live  else _ref_price("MSTR", MSTR_REF_PRICE)
    asst_p_today  = asst_price_live  if asst_price_live  else _ref_price("ASST", ASST_REF_PRICE)
    btc_p_today   = btc_price_live   if btc_price_live   else BTC_REF_PRICE
    # STRK live price must be defined here — used by both caption and _live_mv closure
    _strk_price_today = (strk_price_live if strk_price_live and strk_price_live > 0
                         else _ref_price("STRK", 70.27))

    st.subheader("Portfolio — Strategy & Strive Holdings")

    # Live price status bar
    _btc_str  = f"${btc_price_live:,.0f}" if btc_price_live else "unavailable"
    _mstr_str = f"${mstr_p_today:,.0f}" + (" (live)" if mstr_price_live else " (CSV)")
    _asst_str = f"${asst_p_today:,.0f}" + (" (live)" if asst_price_live else " (CSV)")
    _strk_str = f"${_strk_price_today:,.2f}" + (" (live)" if strk_price_live and strk_price_live > 0 else " (CSV)")
    st.caption(
        f"Live prices: BTC {_btc_str} · MSTR {_mstr_str} · ASST {_asst_str} · "
        f"STRK {_strk_str} · Forecast uses active BTC model blend · Options via Black-Scholes"
    )

    # ── Holdings Summary Table ────────────────────────────────────────────────

    def _live_mv(p: dict) -> float:
        """Compute live market value from exact quantities where available."""
        if "btc_qty" in p:
            return p["btc_qty"] * btc_p_today
        if p["ptype"] == "equity" and "shares" in p:
            price = mstr_p_today if p["underlying"] == "MSTR" else asst_p_today
            return p["shares"] * price if price else p["ref_mv"]
        if p["symbol"] == "STRK":
            return p["shares"] * _strk_price_today
        return p["ref_mv"]

    rows = []
    running_cost = 0.0
    running_mv   = 0.0
    for p in POSITIONS:
        mv   = _live_mv(p)
        gl   = mv - p["cost_basis"]
        pct  = gl / p["cost_basis"] * 100 if p["cost_basis"] else 0.0
        if p["ptype"] == "call":
            extra = f"  ×{p['contracts']} contracts"
        elif "btc_qty" in p:
            extra = f"  ({p['btc_qty']:.5f} BTC)"
        elif p.get("shares"):
            extra = f"  ({p['shares']:,g} shares)"
        else:
            extra = ""
        rows.append({
            "Position":     p["name"] + extra,
            "Category":     p["category"],
            "Type":         p["ptype"].capitalize(),
            "Cost Basis":   f"${round(p['cost_basis']):,}",
            "Market Value": f"${round(mv):,}",
            "Gain / Loss":  f"${round(gl):,}",
            "G/L %":        f"{pct:.1f}%",
        })
        running_cost += p["cost_basis"]
        running_mv   += mv

    rows.append({
        "Position": "── TOTAL ──",
        "Category": "",
        "Type": "",
        "Cost Basis": f"${round(running_cost):,}",
        "Market Value": f"${round(running_mv):,}",
        "Gain / Loss": f"${round(running_mv - running_cost):,}",
        "G/L %": f"{(running_mv - running_cost) / running_cost * 100:.1f}%",
    })

    holdings_df = pd.DataFrame(rows)
    st.dataframe(holdings_df.set_index("Position"), use_container_width=True)

    st.markdown("---")
    st.subheader("Quarterly Return Forecast")

    # ── Sidebar-style controls within the tab ─────────────────────────────────
    col_iv1, col_iv2 = st.columns(2)
    with col_iv1:
        iv_mstr = st.slider(
            "MSTR Option IV (for BS pricing)",
            min_value=0.3, max_value=2.5, value=0.80, step=0.05,
            help="Implied volatility for Black-Scholes MSTR call forecasting. "
                 "Current market-implied IV for Dec'27 MSTR options is ~75–100% "
                 "(varies by strike — lower for deeper OTM calls).",
        )
    with col_iv2:
        iv_asst = st.slider(
            "ASST Option IV (for BS pricing)",
            min_value=0.3, max_value=2.5, value=1.00, step=0.05,
            help="Implied volatility for Black-Scholes ASST call forecasting. "
                 "Current market-implied IV for Jan'28 ASST C25 is ~100%.",
        )

    # ── Build forecast grid: dates × quantiles ────────────────────────────────
    q_dates = _quarter_end_dates()

    # Exact share counts from Fidelity portfolio export 2026-05-30
    mstr_shares = next(p["shares"] for p in POSITIONS if p["symbol"] == "MSTR")
    asst_shares = next(p["shares"] for p in POSITIONS if p["symbol"] == "ASST")

    # BTC-equivalent quantity for each BTC-denominated position (ETF or cold storage).
    # For cold storage, use the exact known BTC quantity (btc_qty field).
    # For ETFs, derive from today's market value ÷ live BTC price.
    btc_equiv: dict[str, float] = {}
    for p in POSITIONS:
        if p["ptype"] not in ("btc_etf", "btc_cold"):
            continue
        if "btc_qty" in p:                         # exact quantity known (cold storage)
            btc_equiv[p["symbol"]] = p["btc_qty"]
        elif btc_p_today:                          # ETF: shares × quote_price ÷ BTC price
            btc_equiv[p["symbol"]] = p.get("shares", 0) * p.get("quote_price", 0) / btc_p_today
        else:
            btc_equiv[p["symbol"]] = 0.0

    # Calls list for easy iteration
    calls = [p for p in POSITIONS if p["ptype"] == "call"]

    # ── Preferred stock constants ─────────────────────────────────────────────
    # STRF: fixed 10% pfd, $2.50/share/qtr dividend reinvested at $98.50 par
    _strf_pos    = next(p for p in POSITIONS if p["symbol"] == "STRF")
    _STRF_PRICE  = _strf_pos["quote_price"]   # $98.50 (assumed stable)
    _STRF_DIV_Q  = 2.50                        # per share per quarter
    _strf_shares = float(_strf_pos["shares"])  # 300 — running count

    # STRK: perpetual pfd with 10:1 MSTR conversion option
    #   STRK price  = bond_floor + 0.10 × MSTR_price
    #   bond_floor  = STRK_today − 0.10 × MSTR_today  (live price preferred over CSV)
    _strk_pos    = next(p for p in POSITIONS if p["symbol"] == "STRK")
    _STRK_BOND   = _strk_price_today - 0.10 * mstr_p_today   # derived from live STRK price
    _STRK_CONV   = 0.10                        # MSTR shares per STRK share
    _STRK_DIV_Q  = 2.00                        # per share per quarter
    _strk_shares = {q: float(_strk_pos["shares"]) for q in _QUANTILES}  # 930, per scenario

    # STRC: variable 11% pfd, dividends NOT reinvested — held flat
    _strc_pos   = next(p for p in POSITIONS if p["symbol"] == "STRC")
    _STRC_FLAT  = _strc_pos["shares"] * _strc_pos["quote_price"]   # $49,495

    # SATA: variable 12.25% pfd, held flat
    _sata_pos   = next(p for p in POSITIONS if p["symbol"] == "SATA")
    _SATA_FLAT  = _sata_pos["shares"] * _sata_pos["quote_price"]   # $5,001

    # ── Pre-compute option expiry values (per symbol × quantile) ─────────────
    # After an option expires it is assumed converted to cash at the expiry-date
    # intrinsic value.  All forecast quarters AFTER the expiry date use this
    # frozen value rather than continuing to project the option value upward.
    _expiry_vals: dict = {}   # key: (symbol, quantile) → float
    for quant in _QUANTILES:
        for c in calls:
            btc_e = _blend_btc(c["expiry"], quant, use_jacobian, use_bhm, use_cowen, bhm_price_fn, use_p=use_perrenod)
            if c["underlying"] == "MSTR":
                S_e = btc_to_mstr_fn(btc_e, c["expiry"], mnav, btc_yield) if btc_e > 0 else 0.0
            else:
                S_e = btc_to_asst_fn(btc_e, c["expiry"], asst_mnav, btc_yield) if btc_e > 0 else 0.0
            _expiry_vals[(c["symbol"], quant)] = max(S_e - c["strike"], 0.0) * 100 * c["contracts"]

    # ── Forecast grid ─────────────────────────────────────────────────────────
    forecast: dict[str, list[float]] = {q: [] for q in _QUANTILES}

    for qdate in q_dates:
        # STRF: reinvest quarterly dividend at constant $98.50 price.
        # Same for all quantile scenarios (price doesn't depend on BTC model).
        _strf_div     = _strf_shares * _STRF_DIV_Q
        _strf_shares += _strf_div / _STRF_PRICE
        _strf_value   = _strf_shares * _STRF_PRICE

        for quant in _QUANTILES:
            # Projected BTC and MSTR
            btc = _blend_btc(qdate, quant, use_jacobian, use_bhm, use_cowen, bhm_price_fn, use_p=use_perrenod)
            mstr_proj = btc_to_mstr_fn(btc, qdate, mnav, btc_yield) if btc > 0 else 0.0

            # ASST: scale proportionally to BTC
            asst_proj = btc_to_asst_fn(btc, qdate, asst_mnav, btc_yield) if btc > 0 else 0.0

            # STRK: price = bond floor + MSTR conversion option value.
            # Reinvest quarterly dividend at this quarter's STRK price.
            _strk_price   = _STRK_BOND + _STRK_CONV * mstr_proj
            _strk_div     = _strk_shares[quant] * _STRK_DIV_Q
            _strk_shares[quant] += _strk_div / _strk_price if _strk_price > 0 else 0.0
            _strk_value   = _strk_shares[quant] * _strk_price

            pref_total = _strf_value + _strk_value + _STRC_FLAT + _SATA_FLAT

            # Equity values
            eq_mstr = mstr_shares * mstr_proj
            eq_asst = asst_shares * asst_proj

            # Option values — frozen at expiry intrinsic value for post-expiry quarters
            opt_total = 0.0
            for c in calls:
                if qdate >= c["expiry"]:
                    opt_total += _expiry_vals[(c["symbol"], quant)]
                else:
                    S  = mstr_proj if c["underlying"] == "MSTR" else asst_proj
                    iv = iv_mstr   if c["underlying"] == "MSTR" else iv_asst
                    opt_total += _option_value(S, c["strike"], qdate, c["expiry"],
                                               c["contracts"], iv)

            # BTC-denominated (ETFs + cold storage)
            btc_total = sum(qty * btc for qty in btc_equiv.values())

            forecast[quant].append(
                eq_mstr + eq_asst + opt_total + pref_total + btc_total
            )

    # ── Total portfolio forecast chart ────────────────────────────────────────
    fig = go.Figure()

    date_labels = [d.strftime("Q%q '%y").replace(
        "Q1", "Q1").replace("Q2", "Q2").replace("Q3", "Q3").replace("Q4", "Q4")
        for d in q_dates]
    # Simpler quarter labels
    def _qlabel(d: date) -> str:
        q = (d.month - 1) // 3 + 1
        return f"Q{q} '{str(d.year)[2:]}"
    date_labels = [_qlabel(d) for d in q_dates]

    # ── Today's actual portfolio MV (from CSV prices) ──────────────────────
    today_total_mv = sum(_live_mv(p) for p in POSITIONS)

    # Cost-basis reference line
    fig.add_hline(
        y=_TOTAL_COST,
        line_dash="dot", line_color="rgba(255,255,255,0.35)",
        annotation_text=f"Cost basis  ${_TOTAL_COST:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color="rgba(255,255,255,0.5)", size=10),
    )
    # Today's MV reference line (computed from live/CSV prices)
    fig.add_hline(
        y=today_total_mv,
        line_dash="dash", line_color="rgba(247,147,26,0.5)",
        annotation_text=f"Today  ${today_total_mv:,.0f}",
        annotation_position="bottom right",
        annotation_font=dict(color="rgba(247,147,26,0.7)", size=10),
    )
    # "Today" scatter marker on each quantile line — prepend to x-axis
    today_label = f"Today\n(${mstr_p_today:.0f} MSTR\n${btc_p_today:,.0f} BTC)"

    # Prepend "Today" to all traces so the chart starts at actual current value
    all_labels = ["Today"] + date_labels
    for quant in _QUANTILES:
        fig.add_trace(go.Scatter(
            x=all_labels,
            y=[today_total_mv] + forecast[quant],
            name=_Q_LABELS[quant],
            mode="lines+markers",
            line=dict(color=_Q_COLORS[quant], width=2),
            marker=dict(size=7),
            hovertemplate=(
                f"<b>{_Q_LABELS[quant]}</b><br>"
                "%{x}<br>"
                "Portfolio: <b>$%{y:,.0f}</b><extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="Total Portfolio Value — MSTR + ASST (BTC-Sensitive + Preferreds)<br>"
                 f"<sup>mNAV {mnav:.1f}x · Model: {'+'.join([n for f, n in [(use_jacobian,'Jac'),(use_bhm,'BHM'),(use_cowen,'Cowen')] if f])}"
                 f" · STRC/SATA flat; STRF/STRK dividends reinvested</sup>",
            font=dict(size=15),
        ),
        xaxis=dict(title="Quarter", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(
            title="Portfolio Value ($)",
            tickprefix="$", tickformat=",.0f",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, traceorder="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-position breakdown table at each quarter ──────────────────────────
    st.markdown("---")
    st.subheader("Per-Position Forecast")
    st.caption("Select a quantile scenario to see the breakdown by position.")

    sel_quant = st.selectbox(
        "Scenario",
        options=_QUANTILES,
        index=2,  # default = Median
        format_func=lambda q: _Q_LABELS[q],
    )

    # Re-run STRF/STRK reinvestment from scratch for the selected scenario
    _bd_strf_shares = float(_strf_pos["shares"])
    _bd_strk_shares = float(_strk_pos["shares"])

    breakdown_rows = []
    for qdate in q_dates:
        btc = _blend_btc(qdate, sel_quant, use_jacobian, use_bhm, use_cowen, bhm_price_fn, use_p=use_perrenod)
        mstr_proj  = btc_to_mstr_fn(btc, qdate, mnav, btc_yield) if btc > 0 else 0.0
        asst_proj  = btc_to_asst_fn(btc, qdate, asst_mnav, btc_yield) if btc > 0 else 0.0

        # STRF reinvestment
        _bd_strf_shares += (_bd_strf_shares * _STRF_DIV_Q) / _STRF_PRICE
        _bd_strf_val = _bd_strf_shares * _STRF_PRICE

        # STRK reinvestment
        _bd_strk_price = _STRK_BOND + _STRK_CONV * mstr_proj
        _bd_strk_shares += (_bd_strk_shares * _STRK_DIV_Q) / _bd_strk_price if _bd_strk_price > 0 else 0.0
        _bd_strk_val = _bd_strk_shares * _bd_strk_price

        row: dict[str, object] = {"Quarter": _qlabel(qdate)}
        for p in POSITIONS:
            if p["ptype"] == "equity":
                val = mstr_shares * mstr_proj if p["underlying"] == "MSTR" else asst_shares * asst_proj
            elif p["symbol"] == "STRF":
                val = _bd_strf_val
            elif p["symbol"] == "STRK":
                val = _bd_strk_val
            elif p["symbol"] == "STRC":
                val = _STRC_FLAT
            elif p["ptype"] == "preferred":   # SATA
                val = _SATA_FLAT
            elif p["ptype"] in ("btc_etf", "btc_cold"):
                val = btc_equiv.get(p["symbol"], 0) * btc
            else:  # call — frozen at expiry value after expiration
                if qdate >= p["expiry"]:
                    val = _expiry_vals[(p["symbol"], sel_quant)]
                else:
                    S  = mstr_proj if p["underlying"] == "MSTR" else asst_proj
                    iv = iv_mstr   if p["underlying"] == "MSTR" else iv_asst
                    val = _option_value(S, p["strike"], qdate, p["expiry"], p["contracts"], iv)
            row[p["symbol"]] = val

        row["TOTAL"] = sum(row[p["symbol"]] for p in POSITIONS)
        breakdown_rows.append(row)

    bdf = pd.DataFrame(breakdown_rows).set_index("Quarter")
    bdf = bdf.applymap(lambda x: f"${round(x):,}" if isinstance(x, float) else x)
    st.dataframe(bdf, use_container_width=True)

    # ── Option-specific forecast (P&L vs cost basis) ──────────────────────────
    st.markdown("---")
    st.subheader("Options P&L Forecast")

    opt_quant = st.selectbox(
        "Scenario",
        options=_QUANTILES,
        index=2,  # default = Median
        format_func=lambda q: _Q_LABELS[q],
        key="opt_scenario",
    )

    option_colors = {
        "MSTR271217C250": "#F7931A",
        "MSTR271217C350": "#2ca02c",
        "MSTR271217C400": "#d62728",
        "ASST280121C25":  "#00ced1",
    }

    opt_fig = go.Figure()

    # Accumulate per-quarter P&L totals across all options
    total_today_pnl = 0.0
    total_pnls = [0.0] * len(q_dates)

    for opt_pos in calls:
        sym = opt_pos["symbol"]
        color = option_colors.get(sym, "#aaa")
        opt_pnls = []
        for i, qdate in enumerate(q_dates):
            if qdate >= opt_pos["expiry"]:
                # Frozen at expiry intrinsic value (cash conversion assumed)
                val = _expiry_vals[(opt_pos["symbol"], opt_quant)]
            else:
                btc = _blend_btc(qdate, opt_quant, use_jacobian, use_bhm, use_cowen, bhm_price_fn, use_p=use_perrenod)
                if opt_pos["underlying"] == "MSTR":
                    S = btc_to_mstr_fn(btc, qdate, mnav, btc_yield) if btc > 0 else 0.0
                else:
                    S = btc_to_asst_fn(btc, qdate, asst_mnav, btc_yield) if btc > 0 else 0.0
                iv = iv_mstr if opt_pos["underlying"] == "MSTR" else iv_asst
                val = _option_value(S, opt_pos["strike"], qdate, opt_pos["expiry"],
                                    opt_pos["contracts"], iv)
            pnl = val - opt_pos["cost_basis"]
            opt_pnls.append(pnl)
            total_pnls[i] += pnl

        today_opt_pnl = opt_pos["ref_mv"] - opt_pos["cost_basis"]
        total_today_pnl += today_opt_pnl

        opt_fig.add_trace(go.Scatter(
            x=["Today"] + all_labels[1:],
            y=[today_opt_pnl] + opt_pnls,
            name=opt_pos["name"],
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=8),
            hovertemplate=(
                f"<b>{opt_pos['name']}</b><br>"
                "%{x}<br>"
                "P&L vs cost: <b>$%{y:,.0f}</b><extra></extra>"
            ),
        ))

    # Total line — all options combined
    opt_fig.add_trace(go.Scatter(
        x=["Today"] + all_labels[1:],
        y=[total_today_pnl] + total_pnls,
        name="── Total ──",
        mode="lines+markers",
        line=dict(color="white", width=3, dash="dash"),
        marker=dict(size=9, symbol="diamond"),
        hovertemplate=(
            "<b>Total Options P&L</b><br>"
            "%{x}<br>"
            "P&L vs cost: <b>$%{y:,.0f}</b><extra></extra>"
        ),
    ))

    opt_fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.45)",
                      annotation_text="Break-even", annotation_position="top left",
                      annotation_font=dict(color="rgba(255,255,255,0.5)", size=10))
    opt_fig.update_layout(
        title=f"Options P&L vs Cost Basis — {_Q_LABELS[opt_quant]} scenario",
        xaxis=dict(title="Quarter", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(
            title="P&L ($)",
            tickprefix="$", tickformat=",.0f",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(opt_fig, use_container_width=True)
    st.caption(
        "Option values estimated via Black-Scholes. IV sliders above control "
        "the volatility assumption. Options with passed expiry shown at intrinsic value."
    )

    # ── mNAV × Scenario Portfolio Value Heatmap ──────────────────────────────
    st.markdown("---")
    st.subheader("Portfolio Value: mNAV × Scenario Matrix")
    st.caption(
        "Each cell = total portfolio value at the selected date under a given "
        "BTC quantile scenario (row) and MSTR mNAV assumption (column). "
        "ASST mNAV held at the sidebar slider value."
    )

    heat_date = st.date_input(
        "Forecast Date",
        value=date(2027, 12, 17),
        min_value=date.today(),
        max_value=date(2030, 12, 31),
        key="heatmap_date",
        help="Portfolio value is computed for this date under every Q × mNAV combination.",
    )

    # mNAV columns: 0.6 to 3.0 in 0.2 steps (13 columns)
    _mnav_range = [round(0.6 + i * 0.2, 1) for i in range(13)]

    # Row scenario order: best on top, bear on bottom
    _H_QUANTS  = ["q=0.99", "q=0.75", "OLS", "q=0.25", "q=0.01"]
    _H_LABELS  = {
        "q=0.99": "🚀 Euphoria (Q99%)",
        "q=0.75": "🐂 Bull (Q75%)",
        "OLS":    "📊 Median",
        "q=0.25": "🔻 Low (Q25%)",
        "q=0.01": "🐻 Bear (Q1%)",
    }
    # Row fill colors (50% transparent): best=green → bear=red
    _H_COLORS  = {
        "q=0.99": "rgba(30,160,30,0.50)",
        "q=0.75": "rgba(80,195,80,0.50)",
        "OLS":    "rgba(210,190,50,0.50)",
        "q=0.25": "rgba(215,130,50,0.50)",
        "q=0.01": "rgba(210,50,50,0.50)",
    }
    # Typical mNAV range per scenario row (from sidebar mNAV guidance table)
    _TYPICAL_MNAV = {
        "q=0.99": (2.5, 3.0),   # Euphoria: mNAV 2.5–3.0×
        "q=0.75": (1.8, 2.5),   # Bull:     mNAV 1.8–2.5×
        "OLS":    (1.3, 1.6),   # Base:     mNAV 1.3–1.6×
        "q=0.25": (0.8, 1.2),   # Low:      mNAV 0.8–1.2×
        "q=0.01": (0.7, 1.0),   # Bear:     mNAV 0.7–1.0×
    }

    # ── Compute matrix values ────────────────────────────────────────────────
    # For each quantile, blend BTC price once; then vary mNAV across columns.
    # Option expiry intrinsic values are recomputed per mNAV since MSTR call
    # payoffs depend directly on MSTR price (= f(mNAV)).

    _h_matrix: dict[str, list[float]] = {}

    for quant in _H_QUANTS:
        # BTC price at heat_date for this quantile
        btc_h = _blend_btc(heat_date, quant,
                           use_jacobian, use_bhm, use_cowen, bhm_price_fn,
                           use_p=use_perrenod)
        # BTC at each option expiry (needed for post-expiry intrinsic)
        btc_exp_cache: dict[date, float] = {}
        for c in calls:
            if c["expiry"] not in btc_exp_cache:
                btc_exp_cache[c["expiry"]] = _blend_btc(
                    c["expiry"], quant,
                    use_jacobian, use_bhm, use_cowen, bhm_price_fn,
                    use_p=use_perrenod,
                )

        row_vals: list[float] = []
        for mv in _mnav_range:
            # Equity at this mNAV
            mstr_h = btc_to_mstr_fn(btc_h, heat_date, mv, btc_yield) if btc_h > 0 else 0.0
            asst_h = btc_to_asst_fn(btc_h, heat_date, asst_mnav, btc_yield) if btc_h > 0 else 0.0
            eq_m   = mstr_shares * mstr_h
            eq_a   = asst_shares * asst_h

            # Preferred: STRK with this mNAV (no quarterly compounding in heatmap)
            strk_h_price = _STRK_BOND + _STRK_CONV * mstr_h
            strk_h_val   = float(_strk_pos["shares"]) * strk_h_price
            strf_h_val   = float(_strf_pos["shares"]) * _STRF_PRICE
            pref_h       = strf_h_val + strk_h_val + _STRC_FLAT + _SATA_FLAT

            # Options: use this mNAV for both pre-expiry BS and post-expiry intrinsic
            opt_h = 0.0
            for c in calls:
                if heat_date >= c["expiry"]:
                    btc_e = btc_exp_cache[c["expiry"]]
                    if c["underlying"] == "MSTR":
                        S_e = btc_to_mstr_fn(btc_e, c["expiry"], mv, btc_yield) if btc_e > 0 else 0.0
                    else:
                        S_e = btc_to_asst_fn(btc_e, c["expiry"], asst_mnav, btc_yield) if btc_e > 0 else 0.0
                    opt_h += max(S_e - c["strike"], 0.0) * 100 * c["contracts"]
                else:
                    S  = mstr_h if c["underlying"] == "MSTR" else asst_h
                    iv = iv_mstr if c["underlying"] == "MSTR" else iv_asst
                    opt_h += _option_value(S, c["strike"], heat_date, c["expiry"],
                                           c["contracts"], iv)

            # BTC positions (insensitive to mNAV)
            btc_h_total = sum(qty * btc_h for qty in btc_equiv.values())

            row_vals.append(eq_m + eq_a + pref_h + opt_h + btc_h_total)

        _h_matrix[quant] = row_vals

    # ── Build heatmaps (go.Heatmap + shapes for clean 4-sided outlines) ─────
    # Rows ordered BOTTOM → TOP for heatmap (bear=0, euphoria=4)
    _H_QUANTS_BT = ["q=0.01", "q=0.25", "OLS", "q=0.75", "q=0.99"]
    _H_YLABELS   = {
        "q=0.01": "🐻 Bear Q1%",
        "q=0.25": "🔻 Low Q25%",
        "OLS":    "📊 Median",
        "q=0.75": "🐂 Bull Q75%",
        "q=0.99": "🚀 Euphoria Q99%",
    }

    # z matrix: row index 0–4 drives background color via colorscale
    _z_vals = [[i] * len(_mnav_range) for i in range(5)]   # row 0=bear at bottom

    # Custom colorscale: maps 0/4 → 4/4 to bear-red → euphoria-green
    _hm_colorscale = [
        [0.00, "rgba(210, 50, 50,0.60)"],
        [0.25, "rgba(215,130, 50,0.60)"],
        [0.50, "rgba(210,190, 50,0.60)"],
        [0.75, "rgba( 80,195, 80,0.60)"],
        [1.00, "rgba( 30,160, 30,0.60)"],
    ]

    _x_labels = [f"{mv:.1f}×" for mv in _mnav_range]
    _y_labels  = [_H_YLABELS[q] for q in _H_QUANTS_BT]

    # Text cells for portfolio-value table ($X.XXXM)
    _text_pv = [
        [f"${_h_matrix[q][j] / 1_000_000:.3f}M" for j in range(len(_mnav_range))]
        for q in _H_QUANTS_BT
    ]

    # Text cells for return-multiple table (value ÷ cost basis)
    _text_mult = [
        [f"{_h_matrix[q][j] / _TOTAL_COST:.2f}×" for j in range(len(_mnav_range))]
        for q in _H_QUANTS_BT
    ]

    # Rectangle shapes for typical-scenario outlines (all 4 sides, clean white)
    def _typical_shapes():
        shapes = []
        for row_i, q in enumerate(_H_QUANTS_BT):
            lo, hi = _TYPICAL_MNAV[q]
            cols = [j for j, mv in enumerate(_mnav_range) if lo <= mv <= hi]
            if cols:
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=min(cols) - 0.5, x1=max(cols) + 0.5,
                    y0=row_i - 0.5,     y1=row_i + 0.5,
                    line=dict(color="white", width=3),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                ))
        return shapes

    _hm_layout = dict(
        height=280,
        margin=dict(l=130, r=10, t=40, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(side="bottom", tickfont=dict(color="white", size=10),
                   tickangle=0, showgrid=False),
        yaxis=dict(tickfont=dict(color="white", size=10), showgrid=False),
    )

    # ── Table 1: Portfolio Value ($M) ─────────────────────────────────────────
    fig_pv = go.Figure(go.Heatmap(
        z=_z_vals, text=_text_pv,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        x=_x_labels, y=_y_labels,
        colorscale=_hm_colorscale, zmin=0, zmax=4,
        showscale=False, xgap=2, ygap=2,
    ))
    for _s in _typical_shapes():
        fig_pv.add_shape(**_s)
    fig_pv.update_layout(title="Portfolio Value ($M)", **_hm_layout)
    st.plotly_chart(fig_pv, use_container_width=True)

    # ── Table 2: Return Multiple (Value ÷ Cost Basis) ─────────────────────────
    fig_mult = go.Figure(go.Heatmap(
        z=_z_vals, text=_text_mult,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        x=_x_labels, y=_y_labels,
        colorscale=_hm_colorscale, zmin=0, zmax=4,
        showscale=False, xgap=2, ygap=2,
    ))
    for _s in _typical_shapes():
        fig_mult.add_shape(**_s)
    fig_mult.update_layout(
        title=f"Return Multiple (Value ÷ Cost Basis ${_TOTAL_COST/1_000_000:.2f}M)",
        **_hm_layout,
    )
    st.plotly_chart(fig_mult, use_container_width=True)
    st.caption(
        "White outlines = likely scenario for that BTC quantile row based on "
        "historical mNAV ranges · STRF/STRK at current share count (no compounding)."
    )
