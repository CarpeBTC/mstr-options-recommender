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

# ── Position data (Fidelity export 2026-05-30) ───────────────────────────────
#
# ptype:
#   "equity"    — common shares; value = shares × projected_equity_price
#   "preferred" — fixed-rate preferred; held flat at ref_mv in forecast
#   "call"      — long call; value via Black-Scholes (pre-expiry) / intrinsic
#   "btc_etf"   — BTC spot ETF (FBTC, IBIT); value = btc_equiv × projected_btc
#   "btc_cold"  — cold-storage BTC (INDEX:NQBT); value = btc_qty × projected_btc
#
# contracts for options: derived from ref_mv ÷ (ref_price_per_share × 100)
#   C250: $16,220 ÷ (162.20 × 100) ≈ 1 contract
#   C350: $33,540 ÷ (111.80 × 100) ≈ 3 contracts
#   C400: $14,400 ÷ ( 72.00 × 100) ≈ 2 contracts
#   ASST28C25: $3,160 ÷ (  6.32 × 100) ≈ 5 contracts

POSITIONS: List[dict] = [
    # ── Strategy / MSTR ──────────────────────────────────────────────────────
    dict(symbol="MSTR",           name="Strategy Cl A (MSTR)",        category="Strategy",
         ptype="equity",    underlying="MSTR",
         strike=None, expiry=None, contracts=None,
         cost_basis=54_191.25,  ref_mv=52_092.59),

    dict(symbol="STRK",           name="Strategy 10% Perp Pfd (STRK)",category="Strategy",
         ptype="preferred", underlying=None,
         strike=None, expiry=None, contracts=None,
         cost_basis=72_958.41,  ref_mv=65_351.10),

    dict(symbol="STRF",           name="Strategy 10% Pfd (STRF)",     category="Strategy",
         ptype="preferred", underlying=None,
         strike=None, expiry=None, contracts=None,
         cost_basis=30_217.50,  ref_mv=29_550.00),

    dict(symbol="STRC",           name="Strategy 11% Var Pfd (STRC)", category="Strategy",
         ptype="preferred", underlying=None,
         strike=None, expiry=None, contracts=None,
         cost_basis=49_999.08,  ref_mv=49_495.00),

    dict(symbol="MSTR271217C250", name="MSTR Dec'27 $250 Call",       category="Strategy",
         ptype="call",      underlying="MSTR",
         strike=250.0, expiry=date(2027, 12, 17), contracts=1,
         cost_basis=17_742.69,  ref_mv=16_220.00),

    dict(symbol="MSTR271217C350", name="MSTR Dec'27 $350 Call",       category="Strategy",
         ptype="call",      underlying="MSTR",
         strike=350.0, expiry=date(2027, 12, 17), contracts=3,
         cost_basis=35_948.08,  ref_mv=33_540.00),

    dict(symbol="MSTR271217C400", name="MSTR Dec'27 $400 Call",       category="Strategy",
         ptype="call",      underlying="MSTR",
         strike=400.0, expiry=date(2027, 12, 17), contracts=2,
         cost_basis=15_604.04,  ref_mv=14_400.00),

    # ── Strive / ASST ─────────────────────────────────────────────────────────
    dict(symbol="ASST",           name="Strive Inc Cl A (ASST)",      category="Strive",
         ptype="equity",    underlying="ASST",
         strike=None, expiry=None, contracts=None,
         cost_basis=18_600.00,  ref_mv=21_204.00),

    dict(symbol="SATA",           name="Strive 12.25% Var Pfd (SATA)",category="Strive",
         ptype="preferred", underlying=None,
         strike=None, expiry=None, contracts=None,
         cost_basis=4_900.74,   ref_mv=5_000.50),

    dict(symbol="ASST280121C25",  name="ASST Jan'28 $25 Call",        category="Strive",
         ptype="call",      underlying="ASST",
         strike=25.0, expiry=date(2028, 1, 21), contracts=5,
         cost_basis=3_394.69,   ref_mv=3_160.00),

    # ── Bitcoin (direct / ETF) ────────────────────────────────────────────────
    # Projected as: btc_equivalent × projected_btc_price
    # btc_equivalent = ref_mv / btc_live_price  (computed at render time)
    dict(symbol="FBTC",           name="Fidelity Bitcoin Fund (FBTC)", category="Bitcoin",
         ptype="btc_etf",   underlying="BTC",
         strike=None, expiry=None, contracts=None,
         cost_basis=2_383.55,   ref_mv=2_389.35),

    dict(symbol="IBIT",           name="iShares Bitcoin Trust (IBIT)", category="Bitcoin",
         ptype="btc_etf",   underlying="BTC",
         strike=None, expiry=None, contracts=None,
         cost_basis=5_549.92,   ref_mv=5_318.90),

    dict(symbol="INDEX:NQBT",     name="BTC Cold Storage (NQBT)",      category="Bitcoin",
         ptype="btc_cold",  underlying="BTC",
         strike=None, expiry=None, contracts=None,
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
) -> float:
    """Average BTC price at *target* across checked models for *quantile*."""
    vals = []
    if use_j:
        vals.append(jacobian.get_btc_price(target).get(quantile, 0.0))
    if use_b:
        vals.append(bhm_fn(target).get(quantile, 0.0))
    if use_c:
        vals.append(cowen.get_btc_price(target).get(quantile, 0.0))
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
    use_jacobian: bool,
    use_bhm: bool,
    use_cowen: bool,
    mnav: float,
    btc_yield: float,
    btc_to_mstr_fn: Callable,   # partial(btc_to_mstr, btc_holdings=…, diluted_shares_k=…, ref_date=…)
    bhm_price_fn: Callable,     # partial(block_height.get_btc_price, ref_height=…, ref_date=…)
) -> None:
    today = date.today()

    st.subheader("Portfolio — Strategy & Strive Holdings")
    st.caption(
        f"Positions from Fidelity export {today.strftime('%Y-%m-%d')} · "
        f"Forecast uses active BTC model blend · "
        f"Options valued via Black-Scholes"
    )

    # ── Holdings Summary Table ────────────────────────────────────────────────
    rows = []
    for p in POSITIONS:
        gl  = p["ref_mv"] - p["cost_basis"]
        pct = gl / p["cost_basis"] * 100 if p["cost_basis"] else 0.0
        extra = ""
        if p["ptype"] == "call":
            extra = f"  ×{p['contracts']} contract{'s' if p['contracts'] != 1 else ''}"
        rows.append({
            "Position":     p["name"] + extra,
            "Category":     p["category"],
            "Type":         p["ptype"].capitalize(),
            "Cost Basis":   p["cost_basis"],
            "Market Value": p["ref_mv"],
            "Gain / Loss":  gl,
            "G/L %":        pct,
        })

    rows.append({
        "Position": "── TOTAL ──",
        "Category": "",
        "Type": "",
        "Cost Basis": _TOTAL_COST,
        "Market Value": _TOTAL_MV,
        "Gain / Loss": _TOTAL_MV - _TOTAL_COST,
        "G/L %": (_TOTAL_MV - _TOTAL_COST) / _TOTAL_COST * 100,
    })

    holdings_df = pd.DataFrame(rows)
    st.dataframe(
        holdings_df.set_index("Position"),
        use_container_width=True,
        column_config={
            "Cost Basis":   st.column_config.NumberColumn("Cost Basis",   format="$%.2f"),
            "Market Value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
            "Gain / Loss":  st.column_config.NumberColumn("Gain / Loss",  format="$%.2f"),
            "G/L %":        st.column_config.NumberColumn("G/L %",        format="%.2f%%"),
        },
    )

    st.markdown("---")
    st.subheader("Quarterly Return Forecast")

    # ── Sidebar-style controls within the tab ─────────────────────────────────
    col_iv1, col_iv2 = st.columns(2)
    with col_iv1:
        iv_mstr = st.slider(
            "MSTR Option IV (for BS pricing)",
            min_value=0.5, max_value=3.0, value=1.4, step=0.05,
            help="Implied volatility used in Black-Scholes for MSTR call forecasting. "
                 "Historical MSTR IV is typically 100–180%.",
        )
    with col_iv2:
        iv_asst = st.slider(
            "ASST Option IV (for BS pricing)",
            min_value=0.5, max_value=3.0, value=1.8, step=0.05,
            help="Implied volatility used in Black-Scholes for ASST call forecasting.",
        )

    # ── Build forecast grid: dates × quantiles ────────────────────────────────
    q_dates = _quarter_end_dates()

    # Derive implied shares / BTC quantities from today's live prices and ref MV
    mstr_shares = (
        next(p["ref_mv"] for p in POSITIONS if p["symbol"] == "MSTR")
        / mstr_price_live if mstr_price_live else 0
    )
    asst_shares = (
        next(p["ref_mv"] for p in POSITIONS if p["symbol"] == "ASST")
        / asst_price_live if asst_price_live else 0
    )
    # BTC-equivalent quantity for each BTC-denominated position (ETF or cold storage)
    # btc_equiv = ref_mv / btc_price_today  →  projected_value = btc_equiv × btc_proj
    btc_equiv: dict[str, float] = {
        p["symbol"]: (p["ref_mv"] / btc_price_live if btc_price_live else 0)
        for p in POSITIONS if p["ptype"] in ("btc_etf", "btc_cold")
    }

    # Stable preferred MV (fixed income, held at today's value throughout)
    preferred_mv = sum(p["ref_mv"] for p in POSITIONS if p["ptype"] == "preferred")

    # Calls list for easy iteration
    calls = [p for p in POSITIONS if p["ptype"] == "call"]

    forecast: dict[str, list[float]] = {q: [] for q in _QUANTILES}

    for qdate in q_dates:
        for quant in _QUANTILES:
            # Projected BTC price
            btc = _blend_btc(qdate, quant, use_jacobian, use_bhm, use_cowen, bhm_price_fn)

            # Projected MSTR price (use btc_to_mstr_fn with slider mNAV)
            mstr_proj = btc_to_mstr_fn(btc, qdate, mnav, btc_yield) if btc > 0 else 0.0

            # ASST projected proportionally to MSTR (both BTC treasury companies)
            asst_scale  = asst_price_live / mstr_price_live if mstr_price_live else 1.0
            asst_proj   = mstr_proj * asst_scale

            # Equity values
            eq_mstr = mstr_shares * mstr_proj
            eq_asst = asst_shares * asst_proj

            # Option values
            opt_total = 0.0
            for c in calls:
                if c["underlying"] == "MSTR":
                    S  = mstr_proj
                    iv = iv_mstr
                else:
                    S  = asst_proj
                    iv = iv_asst
                opt_total += _option_value(
                    S, c["strike"], qdate, c["expiry"], c["contracts"], iv
                )

            # BTC-denominated positions (ETFs + cold storage): qty × projected BTC price
            btc_total = sum(qty * btc for qty in btc_equiv.values())

            forecast[quant].append(
                eq_mstr + eq_asst + opt_total + preferred_mv + btc_total
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

    # Cost-basis reference line
    fig.add_hline(
        y=_TOTAL_COST,
        line_dash="dot", line_color="rgba(255,255,255,0.35)",
        annotation_text=f"Cost basis  ${_TOTAL_COST:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color="rgba(255,255,255,0.5)", size=10),
    )
    # Today's MV reference line
    fig.add_hline(
        y=_TOTAL_MV,
        line_dash="dash", line_color="rgba(247,147,26,0.5)",
        annotation_text=f"Today  ${_TOTAL_MV:,.0f}",
        annotation_position="bottom right",
        annotation_font=dict(color="rgba(247,147,26,0.7)", size=10),
    )

    for quant in _QUANTILES:
        fig.add_trace(go.Scatter(
            x=date_labels,
            y=forecast[quant],
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
                 f" · Preferreds held at current MV ${preferred_mv:,.0f}</sup>",
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
        legend=dict(orientation="h", y=-0.15),
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

    breakdown_rows = []
    for qdate in q_dates:
        btc = _blend_btc(qdate, sel_quant, use_jacobian, use_bhm, use_cowen, bhm_price_fn)
        mstr_proj = btc_to_mstr_fn(btc, qdate, mnav, btc_yield) if btc > 0 else 0.0
        asst_scale = asst_price_live / mstr_price_live if mstr_price_live else 1.0
        asst_proj  = mstr_proj * asst_scale

        row: dict[str, object] = {"Quarter": _qlabel(qdate)}
        for p in POSITIONS:
            if p["ptype"] == "equity":
                val = mstr_shares * mstr_proj if p["underlying"] == "MSTR" else asst_shares * asst_proj
            elif p["ptype"] == "preferred":
                val = p["ref_mv"]
            elif p["ptype"] in ("btc_etf", "btc_cold"):
                val = btc_equiv.get(p["symbol"], 0) * btc
            else:  # call
                S  = mstr_proj if p["underlying"] == "MSTR" else asst_proj
                iv = iv_mstr   if p["underlying"] == "MSTR" else iv_asst
                val = _option_value(S, p["strike"], qdate, p["expiry"], p["contracts"], iv)
            row[p["symbol"]] = val

        row["TOTAL"] = sum(row[p["symbol"]] for p in POSITIONS)
        breakdown_rows.append(row)

    bdf = pd.DataFrame(breakdown_rows).set_index("Quarter")
    col_cfg = {
        col: st.column_config.NumberColumn(col, format="$%.0f")
        for col in bdf.columns
    }
    st.dataframe(bdf, use_container_width=True, column_config=col_cfg)

    # ── Option-specific forecast (P&L vs cost basis) ──────────────────────────
    st.markdown("---")
    st.subheader("Options P&L Forecast")

    opt_fig = go.Figure()
    option_colors = {
        "MSTR271217C250": "#F7931A",
        "MSTR271217C350": "#ff7f0e",
        "MSTR271217C400": "#d62728",
        "ASST280121C25":  "#00ced1",
    }
    dash_map = {
        "q=0.25": "dot",
        "OLS":    "solid",
        "q=0.75": "dash",
    }

    for opt_pos in calls:
        sym = opt_pos["symbol"]
        base_color = option_colors.get(sym, "#aaa")
        for quant in ["q=0.25", "OLS", "q=0.75"]:
            opt_pnls = []
            for qdate in q_dates:
                btc = _blend_btc(qdate, quant, use_jacobian, use_bhm, use_cowen, bhm_price_fn)
                mstr_proj = btc_to_mstr_fn(btc, qdate, mnav, btc_yield) if btc > 0 else 0.0
                asst_scale = asst_price_live / mstr_price_live if mstr_price_live else 1.0
                S  = mstr_proj if opt_pos["underlying"] == "MSTR" else mstr_proj * asst_scale
                iv = iv_mstr   if opt_pos["underlying"] == "MSTR" else iv_asst
                val = _option_value(S, opt_pos["strike"], qdate, opt_pos["expiry"],
                                    opt_pos["contracts"], iv)
                opt_pnls.append(val - opt_pos["cost_basis"])

            opt_fig.add_trace(go.Scatter(
                x=date_labels,
                y=opt_pnls,
                name=f"{opt_pos['name']} ({_Q_LABELS[quant]})",
                mode="lines+markers",
                line=dict(color=base_color, dash=dash_map[quant], width=1.8),
                marker=dict(size=5),
                hovertemplate=(
                    f"<b>{opt_pos['name']} — {_Q_LABELS[quant]}</b><br>"
                    "%{x}<br>"
                    "P&L: <b>$%{y:,.0f}</b><extra></extra>"
                ),
            ))

    opt_fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    opt_fig.update_layout(
        title="Options P&L vs Cost Basis  (Q25% / Median / Q75% scenarios)",
        xaxis=dict(title="Quarter", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(
            title="P&L ($)",
            tickprefix="$", tickformat=",.0f",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(opt_fig, use_container_width=True)
    st.caption(
        "Option values estimated via Black-Scholes. IV sliders above control "
        "the volatility assumption. Options with passed expiry shown at intrinsic value."
    )
