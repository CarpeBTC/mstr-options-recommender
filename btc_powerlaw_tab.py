"""
btc_powerlaw_tab.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Drop-in Streamlit tab/page for the Bitcoin power law forecast model.
Paste into your existing app with:

    from btc_powerlaw_tab import render_powerlaw_tab
    render_powerlaw_tab()

Or wire into your tab structure:
    tab1, tab2, tab3 = st.tabs(["...", "BTC Power Law", "..."])
    with tab2:
        render_powerlaw_tab()

Requires: btc_powerlaw_model.py in the same directory.
Pip deps:  streamlit, pandas, numpy, plotly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

from btc_powerlaw_model import (
    BitcoinPowerLawModel,
    PowerLawParams,
    LogPeriodicParams,
    NoiseParams,
    age_years,
    ModelType,
    LayerType,
)

# ─────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (dark background, white text — your preference)
# ─────────────────────────────────────────────────────────────────────

COLORS = {
    "bg":           "#0e0e0e",
    "grid":         "#1e1e1e",
    "text":         "#e0e0e0",
    "btc":          "#f7931a",   # Bitcoin orange
    "trend":        "#ff4b4b",   # power-law dotted line
    "fundamental":  "#4caf50",   # green — PL + fundamental
    "full":         "#2196f3",   # blue  — full model
    "band_fill":    "rgba(33, 150, 243, 0.10)",
    "band_fill_1s": "rgba(33, 150, 243, 0.18)",
    "warn":         "#ffc107",
}

LAYER_LABELS = {
    "power_law":        "Power Law Only",
    "pl_fundamental":   "PL + Fundamental",
    "pl_fund_harmonic": "PL + Fundamental + Harmonic",
}

MODEL_LABELS = {
    "ols_days":  "OLS (Santostasi & Perrenod 2026, β=5.690)",
    "qr_years":  "QR Median (Perrenod, β=5.865)  ← recommended",
    "bayes":     "Bayesian Posterior (β=5.729)",
}


# ─────────────────────────────────────────────────────────────────────
# CACHED MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_model() -> BitcoinPowerLawModel:
    return BitcoinPowerLawModel()


# ─────────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────

def render_powerlaw_tab():
    model = get_model()

    st.title("₿ Bitcoin Power Law Forecast")
    st.caption(
        "Model: Santostasi & Perrenod (2026) + Perrenod log-periodic DSI layers. "
        "Not financial advice."
    )

    # ── Calibration warning ──────────────────────────────────────────
    if model.lp.NEEDS_CALIBRATION:
        st.warning(
            "⚠️ **Log-periodic amplitude/phase parameters are estimated** from "
            "published charts. Perrenod has not released exact A0/φ0/A1/φ1 values. "
            "The power law spine (Layer 1) is fully calibrated from the paper. "
            "Re-calibrate lp_params once exact values are published.",
            icon="⚠️",
        )

    # ─────────────────────────────────────────────────────────────────
    # SIDEBAR CONTROLS
    # ─────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model Controls")

        model_choice: ModelType = st.selectbox(
            "Power Law Fit Method",
            options=list(MODEL_LABELS.keys()),
            format_func=lambda k: MODEL_LABELS[k],
            index=1,   # default: qr_years
        )

        layer_choice: LayerType = st.selectbox(
            "Model Layer",
            options=list(LAYER_LABELS.keys()),
            format_func=lambda k: LAYER_LABELS[k],
            index=2,   # default: full model
        )

        st.divider()
        st.subheader("Forecast Window")
        forecast_start = st.date_input(
            "Start Date", value=date(2024, 1, 1),
            min_value=date(2010, 7, 18),
            max_value=date(2040, 1, 1),
        )
        forecast_end = st.date_input(
            "End Date", value=date(2030, 1, 1),
            min_value=date(2010, 7, 18),
            max_value=date(2040, 1, 1),
        )

        show_bands = st.checkbox("Show ±1σ / ±2σ bands", value=True)
        log_y_axis = st.checkbox("Log Y-axis", value=True)

        st.divider()
        st.subheader("Key Date Analysis")
        analysis_date = st.date_input(
            "Analyze Specific Date",
            value=date(2029, 1, 1),
            min_value=date(2020, 1, 1),
            max_value=date(2040, 1, 1),
        )

    # ─────────────────────────────────────────────────────────────────
    # FORECAST CHART
    # ─────────────────────────────────────────────────────────────────
    st.subheader("Price Forecast")

    # Generate series for all three layers (so we can show comparison)
    df_pl   = model.forecast_series(forecast_start, forecast_end, layer="power_law",        model=model_choice)
    df_fund = model.forecast_series(forecast_start, forecast_end, layer="pl_fundamental",   model=model_choice)
    df_full = model.forecast_series(forecast_start, forecast_end, layer="pl_fund_harmonic", model=model_choice)

    fig = go.Figure()

    # ±2σ outer band (full model)
    if show_bands:
        fig.add_trace(go.Scatter(
            x=list(df_full.index) + list(df_full.index[::-1]),
            y=list(df_full["price_p2s"]) + list(df_full["price_m2s"][::-1]),
            fill="toself",
            fillcolor=COLORS["band_fill"],
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="±2σ band",
        ))
        # ±1σ inner band
        fig.add_trace(go.Scatter(
            x=list(df_full.index) + list(df_full.index[::-1]),
            y=list(df_full["price_p1s"]) + list(df_full["price_m1s"][::-1]),
            fill="toself",
            fillcolor=COLORS["band_fill_1s"],
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="±1σ band",
        ))

    # Power law baseline (dotted)
    fig.add_trace(go.Scatter(
        x=df_pl.index, y=df_pl["price_median"],
        mode="lines",
        line=dict(color=COLORS["trend"], width=1.5, dash="dot"),
        name="Power Law",
    ))

    # PL + Fundamental (dashed green)
    fig.add_trace(go.Scatter(
        x=df_fund.index, y=df_fund["price_median"],
        mode="lines",
        line=dict(color=COLORS["fundamental"], width=1.5, dash="dash"),
        name="PL + Fundamental",
    ))

    # Full model (solid blue)
    fig.add_trace(go.Scatter(
        x=df_full.index, y=df_full["price_median"],
        mode="lines",
        line=dict(color=COLORS["full"], width=2.5),
        name="PL + Fund + Harmonic",
    ))

    # Vertical line for analysis date
    fig.add_vline(
        x=str(analysis_date),
        line=dict(color=COLORS["warn"], width=1.5, dash="dash"),
        annotation_text=str(analysis_date),
        annotation_font_color=COLORS["warn"],
    )

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["grid"],
        font=dict(color=COLORS["text"], size=12),
        yaxis_type="log" if log_y_axis else "linear",
        yaxis=dict(
            title="BTC Price (USD)",
            gridcolor="#2a2a2a",
            tickformat="$,.0f",
        ),
        xaxis=dict(title="Date", gridcolor="#2a2a2a"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#444",
            borderwidth=1,
        ),
        hovermode="x unified",
        height=500,
        margin=dict(l=60, r=20, t=30, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # LAYER COMPARISON (active layer — RMS info)
    # ─────────────────────────────────────────────────────────────────
    rms_vals = {
        "Power Law Only":                model.noise.rms_powerlaw_only,
        "PL + Fundamental":              model.noise.rms_pl_fundamental,
        "PL + Fundamental + Harmonic":   model.noise.rms_pl_fund_harmonic,
    }
    c1, c2, c3 = st.columns(3)
    for col, (lbl, rms) in zip([c1, c2, c3], rms_vals.items()):
        active = LAYER_LABELS[layer_choice] == lbl
        border = f"2px solid {COLORS['full']}" if active else "1px solid #333"
        factor = 10**rms
        col.markdown(
            f"""<div style="background:#111;border:{border};border-radius:8px;
                padding:12px;text-align:center;">
                <div style="color:{COLORS['text']};font-size:11px;
                    margin-bottom:4px;">{'▶ ' if active else ''}{lbl}</div>
                <div style="color:{COLORS['full'] if active else '#888'};
                    font-size:20px;font-weight:bold;">±{rms:.3f} dex</div>
                <div style="color:#888;font-size:11px;">
                    ≈ ×{factor:.2f} factor @ 1σ</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────
    # KEY DATE ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    st.subheader(f"Distribution Analysis — {analysis_date}")

    dist  = model.price_distribution(analysis_date, layer=layer_choice, model=model_choice)
    phase = model.cycle_phase(analysis_date)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Median (Trend)", f"${dist['median']:,.0f}")
    col2.metric("+1σ Price",      f"${dist['+1σ']:,.0f}")
    col3.metric("-1σ Price",      f"${dist['-1σ']:,.0f}")
    col4.metric("Cycle Phase",    phase["phase_label"].title(),
                delta=f"{phase['lp_contribution']:+.3f} dex LP offset")

    # Distribution bar chart
    labels = ["-2σ", "-1σ", "median", "+1σ", "+2σ"]
    prices = [dist[k] for k in labels]
    bar_colors = [
        "#e53935", "#ef9a9a", COLORS["full"], "#a5d6a7", "#43a047"
    ]

    fig2 = go.Figure(go.Bar(
        x=labels,
        y=prices,
        marker_color=bar_colors,
        text=[f"${p:,.0f}" for p in prices],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
    ))
    fig2.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["grid"],
        font=dict(color=COLORS["text"]),
        yaxis=dict(type="log", title="BTC Price (USD, log scale)", gridcolor="#2a2a2a"),
        xaxis=dict(title="Sigma Level"),
        height=320,
        margin=dict(l=60, r=20, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # MILESTONE TABLE
    # ─────────────────────────────────────────────────────────────────
    st.subheader("Milestone Price Targets")

    milestone_dates = [
        "2025-01-01", "2026-01-01", "2026-07-01",
        "2027-01-01", "2028-01-01", "2029-01-01",
        "2030-01-01", "2033-01-01",
    ]
    df_table = model.summary_table(milestone_dates, layer=layer_choice, model=model_choice)
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
    )

    # ─────────────────────────────────────────────────────────────────
    # LOG-PERIODIC PHASE DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────
    with st.expander("🔬 Log-Periodic Cycle Diagnostics", expanded=False):
        st.markdown("""
        **Model equation** (slide deck, Perrenod April 2026):

        ```
        log₁₀ P(A) = a + k·log₁₀(A)
                   + [1/(A + 2.0)] × [A₀·cos(ω₀·ln(A) + φ₀)
                                    + A₁·cos(ω₁·ln(A) + φ₁)]
        ```

        | Parameter | Value | Source |
        |-----------|-------|--------|
        | λ (bubble spacing) | 2.07 | Fourier/wavelet analysis |
        | ω₀ = 2π/ln(λ) | **8.63** | Derived (not free parameter) |
        | ω₁ ≈ 3.5 × ω₀ | **~30.2** | Slide deck |
        | Decay form | 1/(A + 2.0) | Perrenod Substack Mar 2026 |
        | Noise decay | 6.1/(A + 27.1) | Perrenod Substack Mar 2026 |
        | A₀, φ₀, A₁, φ₁ | *estimated* | ⚠️ Needs calibration |
        """)

        # Show LP contribution over time
        dates_lp = pd.date_range("2020-01-01", "2030-01-01", freq="ME")
        lp_vals  = [model.log_periodic_signal(age_years(d)) for d in dates_lp]
        fig3 = go.Figure(go.Scatter(
            x=dates_lp, y=lp_vals,
            mode="lines",
            line=dict(color=COLORS["fundamental"], width=2),
            fill="tozeroy",
            fillcolor="rgba(76,175,80,0.12)",
        ))
        fig3.add_hline(y=0, line=dict(color="#555", width=1))
        fig3.update_layout(
            paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["grid"],
            font=dict(color=COLORS["text"]),
            yaxis=dict(title="LP Contribution (dex)", gridcolor="#2a2a2a"),
            xaxis=dict(gridcolor="#2a2a2a"),
            height=280,
            margin=dict(l=60, r=20, t=20, b=40),
            title=dict(
                text="Log-Periodic Oscillation Component (dex offset from trend)",
                font=dict(size=13),
            ),
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "Positive values = price above trend; negative = below trend. "
            "⚠️ Shape is indicative only until A₀/φ₀/A₁/φ₁ are calibrated."
        )

    # ─────────────────────────────────────────────────────────────────
    # MODEL COMPARISON TABLE
    # ─────────────────────────────────────────────────────────────────
    with st.expander("📊 Model Method Comparison at Selected Date", expanded=False):
        rows = []
        for m in ["ols_days", "qr_years", "bayes"]:
            for lyr in ["power_law", "pl_fundamental", "pl_fund_harmonic"]:
                dist_c = model.price_distribution(analysis_date, layer=lyr, model=m)
                rows.append({
                    "Fit Method": MODEL_LABELS[m].split(" ←")[0],
                    "Layer":      LAYER_LABELS[lyr],
                    "-1σ":       f"${dist_c['-1σ']:,.0f}",
                    "Median":     f"${dist_c['median']:,.0f}",
                    "+1σ":       f"${dist_c['+1σ']:,.0f}",
                    "+2σ":       f"${dist_c['+2σ']:,.0f}",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────
    # DATA EXPORT
    # ─────────────────────────────────────────────────────────────────
    with st.expander("⬇️ Export Forecast Data", expanded=False):
        df_export = model.forecast_series(
            forecast_start, forecast_end,
            freq="W",
            layer=layer_choice,
            model=model_choice,
            include_bands=True,
        ).reset_index()

        csv = df_export.to_csv(index=False)
        st.download_button(
            "Download Forecast CSV",
            data=csv,
            file_name=f"btc_powerlaw_forecast_{forecast_start}_{forecast_end}.csv",
            mime="text/csv",
        )
        st.dataframe(df_export.head(10), use_container_width=True, hide_index=True)
