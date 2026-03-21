import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta

from functools import partial
from data.fetch import get_mstr_price, get_available_expiries, get_option_chain, get_last_updated, get_btc_price_live, get_strategy_holdings, get_block_height_live
from models import jacobian, block_height
from models.mstr import apply_mnav, btc_to_mstr
from analytics.kelly import build_portfolio_metrics
from analytics.options import compute_exit_timing, compute_pnl_heatmap

st.set_page_config(
    page_title="MSTR Options Recommender",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("₿ MSTR Options Recommender")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()

st.sidebar.markdown("---")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=1000, max_value=10_000_000,
                                    value=10_000, step=1_000)
# Expiry selectbox is filled after live data loads (needs available expiry list)
expiry_placeholder = st.sidebar.empty()
max_spread_pct = st.sidebar.slider("Max Spread % Filter", 5, 100, 50, 5,
                                    help="Exclude strikes where bid-ask spread exceeds "
                                         "this % of the option premium (liquidity filter)")
alt_return = st.sidebar.slider("Alt. Return (Annual %)", -10.0, 15.0, 4.3, 0.1,
                                help="Expected annual return on non-invested capital "
                                     "(e.g. 4.3% = 10Y Treasury, 11.5% = STRC, "
                                     "-10% = cost of a loan)") / 100
mnav = st.sidebar.slider(
    "mNAV — Forward Estimate (Market Cap ÷ BTC NAV)",
    0.5, 3.0, 1.5, 0.1,
    help=(
        "**mNAV = Market Cap ÷ BTC NAV** (not Enterprise Value — this excludes debt, "
        "unlike the EV/NAV ratio shown on strategy.com).\n\n"
        "**Historical range:** 0.75x – 2.7x\n\n"
        "**Scenario guidance:**\n"
        "- 🐻 Bear: 0.8x – 1.0x\n"
        "- 📊 Base: 1.3x – 1.5x\n"
        "- 🐂 Bull: 2.0x – 3.0x"
    ),
)
_mnav_live_caption = st.sidebar.empty()
kelly_frac = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, 0.5, 0.05,
                                help="0.5 = half-Kelly (recommended)")
model_choice = st.sidebar.selectbox("Price Model", ["Blended", "Jacobian", "Block Height"])
st.sidebar.markdown("---")
btc_yield = st.sidebar.slider("MSTR BTC Yield Yr 1 (%)", 0, 30, 10, 1) / 100

# ── Load Live Data ────────────────────────────────────────────────────────────

with st.spinner("Fetching MSTR data..."):
    try:
        mstr_price = get_mstr_price()
        expiries = get_available_expiries()
        btc_live = get_btc_price_live()
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

# Fetch live Strategy holdings (BTC + diluted shares) from strategy.com/shares
# Falls back to hardcoded defaults in models/mstr.py if the fetch fails
_holdings = get_strategy_holdings()
_live_btc   = _holdings["btc_holdings"]     if _holdings else None
_live_shrs  = _holdings["diluted_shares_k"] if _holdings else None
_live_date  = _holdings["as_of"]            if _holdings else None
_live_rdate = date.fromisoformat(_live_date) if _live_date else None

# Partial wrappers that bake in the live holdings — all model calls use these
_btc_to_mstr = partial(btc_to_mstr,  btc_holdings=_live_btc, diluted_shares_k=_live_shrs, ref_date=_live_rdate)
_apply_mnav  = partial(apply_mnav,   btc_holdings=_live_btc, diluted_shares_k=_live_shrs, ref_date=_live_rdate)

# Fill the mNAV live caption now that we have price + holdings data
if btc_live and _live_btc and _live_shrs and mstr_price:
    _current_mnav = (mstr_price * _live_shrs * 1000) / (btc_live * _live_btc)
    _mnav_live_caption.caption(f"📍 Current: **{_current_mnav:.2f}x** (live)")
else:
    _current_mnav = mnav  # fall back to slider value if live data unavailable
    _mnav_live_caption.caption("📍 Current mNAV: unavailable")

# Fetch live Bitcoin block height (mempool.space → blockchain.info → dead-reckoning fallback)
_live_block_height = get_block_height_live()
_block_ref_date    = date.today() if _live_block_height is not None else None
_get_bhm_price     = partial(block_height.get_btc_price,      ref_height=_live_block_height, ref_date=_block_ref_date)
_get_bhm_scenarios = partial(block_height.get_scenario_prices, ref_height=_live_block_height, ref_date=_block_ref_date)

expiry_options = [e for e in expiries if e >= "2026-06-01"]
if not expiry_options:
    expiry_options = expiries

default_expiry = next((e for e in expiry_options if "2027-12" in e), expiry_options[-1])
selected_expiry = expiry_placeholder.selectbox("Option Expiry", expiry_options,
                                               index=expiry_options.index(default_expiry) if default_expiry in expiry_options else 0)

try:
    expiry_date = datetime.strptime(selected_expiry, "%Y-%m-%d").date()
except Exception:
    expiry_date = date(2027, 12, 17)

with st.spinner("Fetching option chain..."):
    try:
        chain_df = get_option_chain(selected_expiry)
    except Exception as e:
        st.error(f"Failed to fetch option chain for {selected_expiry}: {e}")
        st.stop()

# ── Compute Model Prices ──────────────────────────────────────────────────────

j_scenarios_raw = jacobian.get_scenario_prices(expiry_date)
b_scenarios_raw = _get_bhm_scenarios(expiry_date)

j_scenarios = _apply_mnav(j_scenarios_raw, expiry_date, mnav, btc_yield)
b_scenarios = _apply_mnav(b_scenarios_raw, expiry_date, mnav, btc_yield)

# ── Spread % and Entry Price ──────────────────────────────────────────────────

# Detect stale quotes: bid/ask = 0 but lastPrice exists (markets closed / after-hours)
chain_df["is_stale"] = (chain_df["bid"] == 0) | (chain_df["ask"] == 0)

# Mid price: use live bid/ask average when available; fall back to lastPrice when markets closed
chain_df["mid"] = np.where(
    ~chain_df["is_stale"] & (chain_df["bid"] > 0) & (chain_df["ask"] > 0),
    (chain_df["bid"] + chain_df["ask"]) / 2,
    chain_df["lastPrice"],
)

# Spread % = (ask − bid) / mid — only meaningful for live quotes
# Negative spreads (bid > ask) = crossed/stale quotes → NaN; stale quotes → NaN
_raw_spread = np.where(
    ~chain_df["is_stale"] & (chain_df["mid"] > 0),
    (chain_df["ask"] - chain_df["bid"]) / chain_df["mid"] * 100,
    np.nan,
)
chain_df["spread_pct"] = np.where(_raw_spread > 0, _raw_spread, np.nan)

# Entry price = ask (live fill); fall back to lastPrice for stale quotes
chain_df["entry_price"] = np.where(
    chain_df["ask"] > 0, chain_df["ask"], chain_df["lastPrice"]
)

# Apply liquidity filter:
#   • Live quotes:  spread must be positive and within max_spread_pct threshold
#   • Stale quotes: OI ≥ 10 (real accumulated interest) OR volume > 0 (traded today)
#     — eliminates phantom data-artifact strikes (OI=0, volume≈2) while keeping
#       genuinely traded strikes that haven't yet settled their OI
_stale_liquid = (
    chain_df["is_stale"] &
    ((chain_df["openInterest"] >= 10) | (chain_df["volume"] > 0))
)
chain_liq = chain_df[
    (chain_df["mid"] > 0) &
    (
        _stale_liquid |
        ((~chain_df["is_stale"]) & (chain_df["spread_pct"] > 0) & (chain_df["spread_pct"] <= max_spread_pct))
    )
].copy()

# ── Build Option Premiums Dict (ask price, liquid strikes only) ───────────────

strikes = sorted(chain_liq["strike"].tolist())
premiums = dict(zip(chain_liq["strike"], chain_liq["entry_price"]))

# ── Opportunity Cost: holding-period rate for the alternative investment ───────

T_years = max((expiry_date - date.today()).days / 365.25, 1 / 365)
r_period = (1 + alt_return) ** T_years - 1   # e.g. 4.3%/yr × ~1.75yr → ~7.7%

# ── Shared Computations ───────────────────────────────────────────────────────

# Build portfolio metrics once — reused by Tab 1, Tab 3, Tab 4
# r_period drives excess-return Kelly; E[R] columns remain raw for display
metrics_df = build_portfolio_metrics(
    strikes, premiums, j_scenarios, b_scenarios, kelly_frac, bankroll, r_period
)

# Attach Spread %, stale flag, and Open Interest from full chain
_spread_map = dict(zip(chain_df["strike"], chain_df["spread_pct"]))
_stale_map  = dict(zip(chain_df["strike"], chain_df["is_stale"]))
_oi_map     = dict(zip(chain_df["strike"], chain_df["openInterest"]))
metrics_df["Spread %"]      = metrics_df.index.map(_spread_map)
metrics_df["_is_stale"]     = metrics_df.index.map(_stale_map).fillna(False)
metrics_df["Open Interest"] = metrics_df.index.map(_oi_map).fillna(0).astype(int)

# Attach Marginal Return Efficiency column (Blended E[R] basis, strikes sorted ascending)
_mdf_s = metrics_df["Blended E[R]"].sort_index()
_ks_m, _ers_m = _mdf_s.index.tolist(), _mdf_s.tolist()
_marg_map = {}
for _i in range(1, len(_ks_m)):
    _k0, _k1 = _ks_m[_i - 1], _ks_m[_i]
    _er0, _er1 = _ers_m[_i - 1], _ers_m[_i]
    if _k0 > 0 and _er0 > 0 and (_k1 - _k0) != 0:
        _marg_map[_k1] = ((_er1 - _er0) / _er0) / ((_k1 - _k0) / _k0)
# Apply same 3×IQR outlier fence used in the Marginal Return chart —
# sparse/mispriced option strikes can produce extreme ratios that distort the ranking
if _marg_map:
    _mv = np.array(list(_marg_map.values()))
    _mq1, _mq3 = np.percentile(_mv, 25), np.percentile(_mv, 75)
    _miqr = _mq3 - _mq1
    _mlo, _mhi = _mq1 - 3.0 * _miqr, _mq3 + 3.0 * _miqr
    _marg_map = {k: v for k, v in _marg_map.items() if _mlo <= v <= _mhi}
metrics_df["Marg. Efficiency"] = pd.Series(_marg_map)


def _build_blended_scenarios(j_scenarios: list[dict], b_scenarios: list[dict]) -> list[dict]:
    """Merge Jacobian and BHM scenario lists, normalizing probabilities."""
    all_s = []
    for s in j_scenarios:
        all_s.append({"label": f"J:{s['label']}", "prob": s["prob"] * 0.5,
                       "mstr_price": s["mstr_price"]})
    for s in b_scenarios:
        all_s.append({"label": f"B:{s['label']}", "prob": s["prob"] * 0.5,
                       "mstr_price": s["mstr_price"]})
    total = sum(s["prob"] for s in all_s)
    for s in all_s:
        s["prob"] /= total
    return all_s


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Recommendations", "Price Projections", "Strike Detail", "Marginal Return"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSTR Price", f"${mstr_price:,.2f}")
    col2.metric("Option Expiry", selected_expiry)
    col3.metric("mNAV Multiplier", f"{mnav:.1f}x")
    col4.metric("Bankroll", f"${bankroll:,.0f}")

    stale_count = int(chain_liq["is_stale"].sum()) if "is_stale" in chain_liq.columns else 0
    live_count  = len(chain_liq) - stale_count
    if stale_count > 0 and live_count == 0:
        st.warning(
            f"⚠️ **Markets closed** — no live bid/ask quotes available. "
            f"All {stale_count} strikes are showing **last traded price** (stale). "
            f"Recommendations are based on last-trade data; check again during market hours for live spreads."
        )
    elif stale_count > 0:
        st.info(
            f"ℹ️ {stale_count} of {len(chain_liq)} strikes are using **last traded price** "
            f"(no live bid/ask). Spread filter applied only to {live_count} live quotes."
        )
    if _holdings and _holdings.get("source") == "live":
        _src = f"Strategy BTC: {_live_btc:,} · Diluted Shares: {_live_shrs:,}K (as of {_live_date}, strategy.com)"
    elif _holdings and _holdings.get("source") == "cached":
        _src = f"Strategy BTC: {_live_btc:,} · Diluted Shares: {_live_shrs:,}K (as of {_live_date}, last cached — strategy.com unavailable)"
    else:
        from models.mstr import BTC_HOLDINGS, FULLY_DILUTED_SHARES_K
        _src = f"Strategy BTC: {BTC_HOLDINGS:,} · Diluted Shares: {FULLY_DILUTED_SHARES_K:,}K (hardcoded fallback — strategy.com unavailable)"
    _blk = f"Block Height: {_live_block_height:,} (live)" if _live_block_height else "Block Height: estimated (API unavailable)"
    st.caption(f"Data as of {get_last_updated()} | Model: Blended (Jacobian + Block Height) | {_src} | {_blk}")
    st.markdown("---")

    # ── Price Targets Summary ──
    st.subheader("MSTR Price Targets at Expiry")

    today = date.today()
    j_btc_today   = jacobian.get_btc_price(today)
    b_btc_today   = _get_bhm_price(today)
    j_btc_expiry  = jacobian.get_btc_price(expiry_date)
    b_btc_expiry  = _get_bhm_price(expiry_date)
    # btc_live already fetched at startup

    display_quantiles = ["q=0.01", "q=0.25", "OLS", "q=0.75", "q=0.99"]
    _q_nums = {"q=0.01": 0.01, "q=0.25": 0.25, "OLS": 0.50, "q=0.75": 0.75, "q=0.99": 0.99}
    # Short label for column headers
    _model_lbl = {"Jacobian": "Jacobian", "Block Height": "BHM", "Blended": "Blended"}[model_choice]

    def _model_btc(btc_j, btc_b, q_label):
        """Return BTC price for the active model at a given quantile label."""
        if model_choice == "Jacobian":
            return btc_j.get(q_label)
        elif model_choice == "Block Height":
            return btc_b.get(q_label)
        else:  # Blended
            vals = [x for x in [btc_j.get(q_label), btc_b.get(q_label)] if x]
            return float(np.mean(vals)) if vals else None

    # Estimate today's quantile position from live BTC vs model band
    today_q_str = None
    today_q_num = -1.0   # fallback: sorts before all quantile rows
    if btc_live:
        _qp = [(q, _model_btc(j_btc_today, b_btc_today, q)) for q in display_quantiles]
        _qp = [(q, p) for q, p in _qp if p]
        _qp.sort(key=lambda x: _q_nums[x[0]])
        if _qp:
            if btc_live <= _qp[0][1]:
                today_q_str = f"≤ {_qp[0][0]}"
                today_q_num = _q_nums[_qp[0][0]] - 0.001   # just before first quantile
            elif btc_live >= _qp[-1][1]:
                today_q_str = f"≥ {_qp[-1][0]}"
                today_q_num = _q_nums[_qp[-1][0]] + 0.001  # just after last quantile
            else:
                for _i in range(len(_qp) - 1):
                    _q0, _p0 = _q_nums[_qp[_i][0]], _qp[_i][1]
                    _q1, _p1 = _q_nums[_qp[_i + 1][0]], _qp[_i + 1][1]
                    if _p0 <= btc_live <= _p1:
                        _frac = (btc_live - _p0) / (_p1 - _p0)
                        today_q_num = _q0 + _frac * (_q1 - _q0)
                        today_q_str = f"q≈{today_q_num:.2f}"
                        break

    today_row_label = f"Today ({today_q_str})" if today_q_str else "Today"
    col_btc_e  = f"{_model_lbl} BTC @ {selected_expiry}"
    col_mstr_e = f"{_model_lbl} MSTR @ {selected_expiry}"

    target_rows = []
    # "Today" row — live market prices, no model expiry forecast
    target_rows.append({
        "Scenario":  today_row_label,
        "_sort_q":   today_q_num,
        "Today BTC":  f"${btc_live:,.0f}" if btc_live else "—",
        "Today MSTR": f"${mstr_price:,.0f}",
        col_btc_e:    "—",
        col_mstr_e:   "—",
    })
    # Quantile rows
    for q in display_quantiles:
        btc_t  = _model_btc(j_btc_today,  b_btc_today,  q)
        mstr_t = _btc_to_mstr(btc_t,  today,       _current_mnav, btc_yield) if btc_t  else None
        btc_e  = _model_btc(j_btc_expiry, b_btc_expiry, q)
        mstr_e = _btc_to_mstr(btc_e,  expiry_date, mnav, btc_yield) if btc_e  else None
        target_rows.append({
            "Scenario":   q,
            "_sort_q":    _q_nums[q],
            "Today BTC":  f"${btc_t:,.0f}"  if btc_t  else "—",
            "Today MSTR": f"${mstr_t:,.0f}" if mstr_t else "—",
            col_btc_e:    f"${btc_e:,.0f}"  if btc_e  else "—",
            col_mstr_e:   f"${mstr_e:,.0f}" if mstr_e else "—",
        })
    _tdf = (pd.DataFrame(target_rows)
              .sort_values("_sort_q")
              .drop(columns="_sort_q")
              .set_index("Scenario"))
    _today_hdr  = f"mNAV = {_current_mnav:.2f}x"
    _expiry_hdr = f"mNAV = {mnav:.2f}x"
    _tdf.columns = pd.MultiIndex.from_tuples([
        (_today_hdr,  "Today BTC"),
        (_today_hdr,  "Today MSTR"),
        (_expiry_hdr, col_btc_e),
        (_expiry_hdr, col_mstr_e),
    ])
    st.dataframe(_tdf, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Portfolio Growth Metrics — {selected_expiry} Calls")

    # Format for display — numeric columns stay numeric so column sorting works correctly
    display_df = metrics_df.copy()

    # ── Replace unselected-model E[R] cols with q=0.25 / selected avg / q=0.75 ──────
    # Point return: what you'd earn IF the MSTR price hits exactly the q=X target
    _btc_q25_d = _model_btc(j_btc_expiry, b_btc_expiry, "q=0.25")
    _btc_q75_d = _model_btc(j_btc_expiry, b_btc_expiry, "q=0.75")
    _mstr_q25_d = _btc_to_mstr(_btc_q25_d, expiry_date, mnav, btc_yield) if _btc_q25_d else None
    _mstr_q75_d = _btc_to_mstr(_btc_q75_d, expiry_date, mnav, btc_yield) if _btc_q75_d else None

    def _pt_r(mstr_tgt, strike, prem):
        return (mstr_tgt - strike) / prem - 1 if (mstr_tgt and mstr_tgt > strike and prem > 0) else -1.0

    display_df["R @ q=0.25"] = (
        display_df.index.map(lambda k: _pt_r(_mstr_q25_d, k, premiums.get(k, 1)))
        if _mstr_q25_d else np.nan
    )
    display_df["R @ q=0.75"] = (
        display_df.index.map(lambda k: _pt_r(_mstr_q75_d, k, premiums.get(k, 1)))
        if _mstr_q75_d else np.nan
    )
    # Keep only the selected model's probability-weighted E[R]; drop the other two
    _sel_er = {"Jacobian": "Jac E[R]", "Block Height": "BHM E[R]", "Blended": "Blended E[R]"}[model_choice]
    display_df = display_df.rename(columns={_sel_er: f"{_model_lbl} E[R]"})
    display_df = display_df.drop(columns=[c for c in ["Jac E[R]", "BHM E[R]", "Blended E[R]"] if c in display_df.columns and c != _sel_er])
    # ─────────────────────────────────────────────────────────────────────────────────

    # Strike index stays numeric so Streamlit sorts it correctly; formatted via column_config below

    # Spread % stays a string (emoji prefix makes it inherently non-numeric)
    def _fmt_spread(row):
        x, stale = row["Spread %"], row["_is_stale"]
        if stale:
            return "⚪ Stale"
        if pd.notna(x):
            return ("🟢 " if x < 5 else "🟡 " if x < 15 else "🔴 ") + f"{x:.1f}%"
        return "—"
    display_df["Spread %"] = display_df.apply(_fmt_spread, axis=1)
    display_df = display_df.drop(columns=["_is_stale"], errors="ignore")

    # Keep only the selected model's Kelly f*; drop the other two
    _sel_kf = {"Jacobian": "Jac Kelly f*", "Block Height": "BHM Kelly f*", "Blended": "Blended Kelly f*"}[model_choice]
    display_df = display_df.rename(columns={_sel_kf: "Kelly f*"})
    display_df = display_df.drop(columns=[c for c in ["Jac Kelly f*", "BHM Kelly f*", "Blended Kelly f*"] if c in display_df.columns and c != _sel_kf])

    # Scale Kelly columns from fraction to percentage points (0.123 → 12.3) for display
    display_df["Kelly f*"]   = display_df["Kelly f*"]   * 100
    display_df["Adj. Kelly"] = display_df["Adj. Kelly"] * 100
    display_df["Contracts"]  = display_df["Contracts"].astype(int)

    # Reorder: Premium → Marg. Efficiency → Spread % → Open Interest → R @ q=0.25 → {model} E[R] → R @ q=0.75 → rest
    _er_cols    = ["R @ q=0.25", f"{_model_lbl} E[R]", "R @ q=0.75"]
    _front      = ["Premium", "Marg. Efficiency", "Spread %", "Open Interest"]
    _other_cols = [c for c in display_df.columns if c not in _front + _er_cols]
    display_df  = display_df[_front + _er_cols + _other_cols]

    hurdle_er = 1 + r_period
    _col_cfg = {
        "Strike":               st.column_config.NumberColumn("Strike",               format="$%,.0f"),
        "Premium":              st.column_config.NumberColumn("Premium",              format="$%.2f"),
        "Marg. Efficiency":     st.column_config.NumberColumn("Marg. Efficiency",     format="%.3f"),
        "Open Interest":        st.column_config.NumberColumn("Open Interest",         format="%d"),
        "R @ q=0.25":           st.column_config.NumberColumn("R @ q=0.25",           format="%.2fx"),
        f"{_model_lbl} E[R]":   st.column_config.NumberColumn(f"{_model_lbl} E[R]",  format="%.2fx"),
        "R @ q=0.75":           st.column_config.NumberColumn("R @ q=0.75",           format="%.2fx"),
        "Kelly f*":             st.column_config.NumberColumn("Kelly f*",              format="%.1f%%"),
        "Adj. Kelly":           st.column_config.NumberColumn("Adj. Kelly",            format="%.1f%%"),
        "$ Allocated":          st.column_config.NumberColumn("$ Allocated",           format="$%.0f"),
        "Contracts":            st.column_config.NumberColumn("Contracts",             format="%d"),
    }
    st.dataframe(display_df, use_container_width=True, height=520, column_config=_col_cfg)
    st.caption(
        f"Kelly f\\* uses excess returns (R − alt. hurdle) · "
        f"Alt. hurdle over holding period: {hurdle_er:.3f}x ({alt_return*100:.1f}%/yr × {T_years:.2f}yr) · "
        f"Entry price = ask · Spread %: 🟢<5% 🟡5–15% 🔴>15% · "
        f"Click any column header to sort"
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: PRICE PROJECTIONS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("BTC Price Projections")

    # Build time series from today to expiry + 1 year
    today = date.today()
    proj_dates = [today + timedelta(days=90 * i) for i in range(
        0, int((expiry_date - today).days / 90) + 6
    )]
    proj_dates = [d for d in proj_dates if d <= expiry_date + timedelta(days=365)]

    j_quants_to_plot = ["q=0.01", "q=0.25", "OLS", "q=0.75", "q=0.99"]
    b_quants_to_plot = ["q=0.01", "q=0.25", "OLS", "q=0.75", "q=0.99"]

    fig_btc = go.Figure()
    colors_j = ["#d62728", "#ff7f0e", "#F7931A", "#2ca02c", "#1f77b4"]
    colors_b = ["#9467bd", "#c5b0d5", "#aec7e8", "#98df8a", "#ffbb78"]

    for q, color in zip(j_quants_to_plot, colors_j):
        prices = [jacobian.get_btc_price(d).get(q, 0) for d in proj_dates]
        fig_btc.add_trace(go.Scatter(
            x=proj_dates, y=prices, name=f"Jacobian {q}",
            line=dict(color=color, dash="solid"), mode="lines"
        ))

    for q, color in zip(b_quants_to_plot, colors_b):
        prices_b = []
        for d in proj_dates:
            bp = _get_bhm_price(d)
            prices_b.append(bp.get(q, 0))
        fig_btc.add_trace(go.Scatter(
            x=proj_dates, y=prices_b, name=f"BHM {q}",
            line=dict(color=color, dash="dash"), mode="lines"
        ))

    # add_vline needs string dates when x-axis uses date objects
    expiry_str_for_plot = expiry_date.strftime("%Y-%m-%d")
    fig_btc.add_shape(type="line", x0=expiry_str_for_plot, x1=expiry_str_for_plot,
                      y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color="white", dash="dot", width=1))
    fig_btc.add_annotation(x=expiry_str_for_plot, y=1, xref="x", yref="paper",
                            text=f"Expiry {selected_expiry}", showarrow=False,
                            xanchor="left", font=dict(color="white", size=11))
    fig_btc.update_layout(
        title="BTC Price Projection by Quantile",
        yaxis_title="BTC Price (USD)",
        xaxis_title="Date",
        legend=dict(orientation="v", x=1.01),
        height=450,
        hovermode="x unified",
    )
    fig_btc.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_btc, use_container_width=True)

    # ── MSTR Price at Expiry ──
    st.subheader(f"MSTR Price Targets at Expiry ({selected_expiry})")

    j_mstr_targets = {q: _btc_to_mstr(jacobian.get_btc_price(expiry_date)[q], expiry_date, mnav, btc_yield)
                      for q in j_quants_to_plot}
    b_mstr_targets = {q: _btc_to_mstr(_get_bhm_price(expiry_date).get(q, 0), expiry_date, mnav, btc_yield)
                      for q in b_quants_to_plot if q in _get_bhm_price(expiry_date)}

    fig_mstr = go.Figure()
    fig_mstr.add_trace(go.Bar(
        x=list(j_mstr_targets.keys()), y=list(j_mstr_targets.values()),
        name="Jacobian", marker_color="#F7931A", opacity=0.85
    ))
    fig_mstr.add_trace(go.Bar(
        x=list(b_mstr_targets.keys()), y=list(b_mstr_targets.values()),
        name="Block Height", marker_color="#7B2D8B", opacity=0.85
    ))
    fig_mstr.add_hline(y=mstr_price, line_dash="dot", line_color="white",
                       annotation_text=f"Current ${mstr_price:.0f}")
    fig_mstr.update_layout(
        barmode="group", title=f"MSTR Price at {mnav:.1f}x mNAV by Quantile",
        yaxis_title="MSTR Price (USD)", yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        height=400
    )
    st.plotly_chart(fig_mstr, use_container_width=True)

    # ── Option Chain Summary ──
    st.subheader("Live Option Chain (Calls)")
    display_chain = chain_df[chain_df["mid"] > 0].copy()
    display_chain = display_chain[display_chain["strike"].between(
        mstr_price * 0.5, mstr_price * 8
    )]
    display_chain["spread_pct_fmt"] = display_chain.apply(
        lambda r: "⚪ Stale" if r.get("is_stale", False)
        else (("🟢 " if r["spread_pct"] < 5 else "🟡 " if r["spread_pct"] < 15 else "🔴 ") + f"{r['spread_pct']:.1f}%")
        if pd.notna(r["spread_pct"]) else "—",
        axis=1,
    )
    display_chain["strike"] = display_chain["strike"].map("${:,.0f}".format)
    display_chain["bid"] = display_chain["bid"].map(lambda x: f"${x:.2f}" if x > 0 else "—")
    display_chain["mid"] = display_chain["mid"].map("${:.2f}".format)
    display_chain["ask"] = display_chain["ask"].map(lambda x: f"${x:.2f}" if x > 0 else "—")
    display_chain["impliedVolatility"] = (display_chain["impliedVolatility"] * 100).map("{:.1f}%".format)
    display_chain = display_chain.rename(columns={
        "strike": "Strike", "bid": "Bid", "mid": "Mid", "ask": "Ask",
        "volume": "Volume", "openInterest": "Open Interest",
        "impliedVolatility": "IV", "spread_pct_fmt": "Spread %",
    })
    st.dataframe(
        display_chain.set_index("Strike")[["Bid", "Mid", "Ask", "Spread %", "IV", "Volume", "Open Interest"]],
        use_container_width=True,
    )
    st.caption(f"Spread %: 🟢 < 5% (liquid) · 🟡 5–15% (moderate) · 🔴 > 15% (illiquid) · "
               f"Showing all strikes; Max Spread filter ({max_spread_pct}%) applied to recommendations only.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: STRIKE DETAIL — Investment vs Expected Return
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Investment vs. Expected Return Multiple — {model_choice} Model")
    st.caption(f"Option Expiry: {selected_expiry} · mNAV: {mnav:.1f}x · Kelly Fraction: {kelly_frac:.0%}")

    # Re-use the same metrics_df computed in Tab 1
    er_col = {"Jacobian": "Jac E[R]", "Block Height": "BHM E[R]", "Blended": "Blended E[R]"}[model_choice]

    # Filter to strikes with positive allocation and E[R], within a sensible price range
    plot_df = metrics_df[
        (metrics_df["$ Allocated"] > 0) &
        (metrics_df[er_col] > 0) &
        (metrics_df.index >= mstr_price * 0.3) &
        (metrics_df.index <= mstr_price * 10)
    ].copy()
    plot_df = plot_df.sort_values("$ Allocated", ascending=False).head(60)

    if plot_df.empty:
        st.warning("No strikes with positive allocation found for the selected parameters.")
    else:
        strike_labels = [f"${k:,.0f}" for k in plot_df.index]

        # Color by moneyness: ITM (strike < current) = green, OTM = orange → red
        def moneyness_color(strike):
            ratio = strike / mstr_price
            if ratio <= 1.0:
                return "#2ca02c"   # green — in the money
            elif ratio <= 2.0:
                return "#F7931A"   # orange — near / moderate OTM
            elif ratio <= 4.0:
                return "#ff7f0e"   # amber — OTM
            else:
                return "#d62728"   # red — deep OTM

        colors = [moneyness_color(k) for k in plot_df.index]

        # Marker size proportional to contracts (at least 8px, max 28px)
        max_contracts = max(plot_df["Contracts"].max(), 1)
        marker_sizes = [
            max(8, min(28, 8 + 20 * (c / max_contracts)))
            for c in plot_df["Contracts"]
        ]

        custom_data = list(zip(
            plot_df.index,
            plot_df["Premium"],
            plot_df["Contracts"],
            plot_df["Adj. Kelly"] * 100,
            plot_df["Blended E[R]"] if "Blended E[R]" in plot_df.columns else plot_df[er_col],
        ))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df["$ Allocated"],
            y=plot_df[er_col],
            mode="markers+text",
            text=strike_labels,
            textposition="top center",
            textfont=dict(size=9, color="rgba(250,250,250,0.75)"),
            marker=dict(
                size=marker_sizes,
                color=colors,
                line=dict(color="white", width=0.5),
            ),
            hovertemplate=(
                "<b>Strike: %{customdata[0]:$,.0f}</b><br>"
                "Investment: %{x:$,.0f}<br>"
                "E[Return] (" + model_choice + "): <b>%{y:.2f}x</b><br>"
                "Mid Premium: %{customdata[1]:$,.2f}<br>"
                "Contracts: %{customdata[2]}<br>"
                "Adj. Kelly: %{customdata[3]:.1f}%<br>"
                "<extra></extra>"
            ),
            customdata=custom_data,
        ))

        # Reference line at E[R] = 0
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")

        # Opportunity cost hurdle line at 1 + r_period
        hurdle_er = 1 + r_period
        fig.add_hline(
            y=hurdle_er,
            line_dash="dash",
            line_color="rgba(100,220,100,0.55)",
            annotation_text=f"Alt. hurdle {hurdle_er:.2f}x ({alt_return*100:.1f}%/yr)",
            annotation_position="top right",
            annotation_font=dict(color="rgba(100,220,100,0.8)", size=10),
        )

        # Vertical line at current MSTR price mapped to typical ATM allocation
        atm_alloc = plot_df.loc[
            plot_df.index[np.argmin(np.abs(plot_df.index.values - mstr_price))], "$ Allocated"
        ] if len(plot_df) > 0 else None

        fig.update_layout(
            title=dict(
                text=f"Investment vs. Expected Return — {model_choice} Model<br>"
                     f"<sup>Expiry {selected_expiry} · mNAV {mnav:.1f}x · "
                     f"Bubble size = contracts · Color = moneyness</sup>",
                font=dict(size=16),
            ),
            xaxis=dict(
                title="Kelly-Recommended Investment ($)",
                tickprefix="$",
                tickformat=",.0f",
                gridcolor="rgba(255,255,255,0.08)",
            ),
            yaxis=dict(
                title=f"Expected Return Multiple — {model_choice}",
                ticksuffix="x",
                gridcolor="rgba(255,255,255,0.08)",
                autorange=True,
            ),
            height=580,
            hovermode="closest",
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        # Moneyness legend as annotations
        fig.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper",
                           text="● In the money", font=dict(color="#2ca02c", size=11),
                           showarrow=False, xanchor="left", yanchor="top")
        fig.add_annotation(x=0.01, y=0.93, xref="paper", yref="paper",
                           text="● Near / moderate OTM", font=dict(color="#F7931A", size=11),
                           showarrow=False, xanchor="left", yanchor="top")
        fig.add_annotation(x=0.01, y=0.87, xref="paper", yref="paper",
                           text="● Out of the money", font=dict(color="#ff7f0e", size=11),
                           showarrow=False, xanchor="left", yanchor="top")
        fig.add_annotation(x=0.01, y=0.81, xref="paper", yref="paper",
                           text="● Deep OTM", font=dict(color="#d62728", size=11),
                           showarrow=False, xanchor="left", yanchor="top")

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Each point is one strike. X = Kelly-recommended dollar allocation "
            f"(Blended f* × {kelly_frac:.0%} × \\${bankroll:,.0f}). "
            f"Y = E[R] from {model_choice} model scenarios. "
            f"Bubble size scales with number of contracts. "
            f"Current MSTR: \\${mstr_price:,.2f}."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: MARGINAL RETURN EFFICIENCY
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"Marginal Return Efficiency — {model_choice} Model")
    st.caption(
        f"Option Expiry: {selected_expiry} · mNAV: {mnav:.1f}x · Kelly Fraction: {kelly_frac:.0%}"
    )

    er_col4 = {"Jacobian": "Jac E[R]", "Block Height": "BHM E[R]", "Blended": "Blended E[R]"}[model_choice]

    # Sort by strike ascending, keep only positive E[R] strikes in a sensible range
    marg_df = metrics_df[[er_col4, "$ Allocated", "Premium", "Contracts"]].copy()
    marg_df = marg_df[
        (marg_df[er_col4] > 0) &
        (marg_df.index >= mstr_price * 0.3) &
        (marg_df.index <= mstr_price * 10)
    ].sort_index()

    if len(marg_df) < 2:
        st.warning("Not enough strikes with positive E[R] to compute marginal efficiency.")
    else:
        strikes_s = marg_df.index.tolist()
        er_s = marg_df[er_col4].tolist()

        # Compute (% change in E[R]) / (% change in strike) for each consecutive pair
        ratio_strikes_raw, ratios_raw = [], []
        for i in range(1, len(strikes_s)):
            k0, k1 = strikes_s[i - 1], strikes_s[i]
            er0, er1 = er_s[i - 1], er_s[i]
            if k0 <= 0 or er0 <= 0:
                continue
            delta_er_pct = (er1 - er0) / er0
            delta_k_pct = (k1 - k0) / k0
            if delta_k_pct == 0:
                continue
            ratios_raw.append(delta_er_pct / delta_k_pct)
            ratio_strikes_raw.append(k1)

        if not ratios_raw:
            st.warning("Could not compute marginal efficiency ratios.")
        else:
            # Remove statistical outliers using 3×IQR fences before finding peak
            ratios_arr = np.array(ratios_raw)
            q1, q3 = np.percentile(ratios_arr, 25), np.percentile(ratios_arr, 75)
            iqr = q3 - q1
            fence_lo = q1 - 3.0 * iqr
            fence_hi = q3 + 3.0 * iqr
            mask = (ratios_arr >= fence_lo) & (ratios_arr <= fence_hi)

            ratio_strikes = [ratio_strikes_raw[i] for i in range(len(ratio_strikes_raw)) if mask[i]]
            ratios = ratios_arr[mask].tolist()

            if not ratios:
                st.warning("All ratios were outliers — try adjusting the expiry or model.")
            else:
                plot_marg = pd.DataFrame(
                    {"Strike": ratio_strikes, "Ratio": ratios}
                ).set_index("Strike")

                # Peak = highest ratio after outlier removal
                peak_idx = plot_marg["Ratio"].idxmax()
                peak_val = plot_marg.loc[peak_idx, "Ratio"]

                # Color: highlight peak and near-peak bars
                bar_colors = []
                for k in plot_marg.index:
                    if k == peak_idx:
                        bar_colors.append("#F7931A")
                    elif abs(k - peak_idx) / peak_idx < 0.12:
                        bar_colors.append("rgba(247,147,26,0.55)")
                    else:
                        bar_colors.append("rgba(150,150,200,0.45)")

                # Hover: E[R] and $ Allocated at each strike
                hover_er = [
                    marg_df.loc[k, er_col4] if k in marg_df.index else float("nan")
                    for k in plot_marg.index
                ]
                hover_alloc = [
                    marg_df.loc[k, "$ Allocated"] if k in marg_df.index else float("nan")
                    for k in plot_marg.index
                ]

                fig_marg = go.Figure()
                fig_marg.add_trace(go.Bar(
                    x=plot_marg.index,
                    y=plot_marg["Ratio"],
                    marker_color=bar_colors,
                    customdata=list(zip(hover_er, hover_alloc)),
                    hovertemplate=(
                        "<b>Strike: $%{x:,.0f}</b><br>"
                        "Marginal Efficiency: <b>%{y:.3f}</b><br>"
                        f"E[R] ({model_choice}): %{{customdata[0]:.2f}}x<br>"
                        "Kelly Allocation: $%{customdata[1]:,.0f}<br>"
                        "<extra></extra>"
                    ),
                ))

                # Reference line at ratio = 1.0
                fig_marg.add_hline(
                    y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.35)",
                    annotation_text="ratio = 1  (E[R] ∝ strike)",
                    annotation_position="top right",
                    annotation_font=dict(color="rgba(255,255,255,0.5)", size=10),
                )

                # Current MSTR price marker — numeric x-axis, no markdown escaping needed
                fig_marg.add_vline(
                    x=mstr_price, line_dash="dash", line_color="rgba(255,255,255,0.45)",
                    annotation_text=f"MSTR ${mstr_price:.0f}",
                    annotation_position="top left",
                    annotation_font=dict(color="rgba(255,255,255,0.6)", size=10),
                )

                # Peak annotation — Plotly uses HTML, plain $ is fine
                fig_marg.add_annotation(
                    x=peak_idx, y=peak_val,
                    text=f"<b>Peak</b><br>${peak_idx:,.0f}<br>{peak_val:.2f}",
                    showarrow=True, arrowhead=2, arrowcolor="#F7931A", arrowwidth=1.5,
                    bgcolor="rgba(247,147,26,0.18)", bordercolor="#F7931A", borderwidth=1,
                    font=dict(color="white", size=11),
                    yshift=12,
                )

                fig_marg.update_layout(
                    title=dict(
                        text=(
                            f"Marginal E[R] Efficiency by Strike — {model_choice} Model<br>"
                            f"<sup>(% change in E[R]) ÷ (% change in strike) · "
                            f"Peak = best return improvement per unit of additional strike</sup>"
                        ),
                        font=dict(size=15),
                    ),
                    xaxis=dict(
                        title="Strike Price ($)",
                        tickprefix="$",
                        tickformat=",.0f",
                        gridcolor="rgba(255,255,255,0.07)",
                    ),
                    yaxis=dict(
                        title="Marginal Return Efficiency  (ΔE[R]% / ΔStrike%)",
                        gridcolor="rgba(255,255,255,0.07)",
                    ),
                    height=520,
                    bargap=0.15,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )

                st.plotly_chart(fig_marg, use_container_width=True)

                # Summary metrics — st.metric uses plain text, no markdown escaping
                peak_er = marg_df.loc[peak_idx, er_col4] if peak_idx in marg_df.index else float("nan")
                peak_alloc = marg_df.loc[peak_idx, "$ Allocated"] if peak_idx in marg_df.index else float("nan")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Peak Efficiency Strike", f"${peak_idx:,.0f}")
                c2.metric("Efficiency Ratio at Peak", f"{peak_val:.3f}")
                c3.metric(f"E[R] at Peak ({model_choice})", f"{peak_er:.2f}x")
                c4.metric("Kelly Allocation at Peak", f"${peak_alloc:,.0f}")

                n_clipped = len(ratios_raw) - len(ratios)
                clip_note = f" ({n_clipped} outlier strikes removed via 3×IQR filter.)" if n_clipped else ""
                st.caption(
                    f"Each bar = (% change in E[R]) ÷ (% change in strike price) between consecutive strikes. "
                    f"Ratio > 1: E[R] grows faster than the strike price increase. "
                    f"Peak bar (orange) = best marginal risk/reward — the strike where moving one step higher "
                    f"earns the most additional expected return per unit of strike premium paid.{clip_note}"
                )
