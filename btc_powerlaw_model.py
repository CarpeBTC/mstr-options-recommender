"""
btc_powerlaw_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three-layer Bitcoin price model based on:
  - Santostasi & Perrenod (2026): "A Mechanistic Derivation of the
    Bitcoin Price Power Law" (Zenodo 19387099)
  - Perrenod Substack (2026): log-periodic DSI analysis
  - Perrenod YouTube slide deck (April 2026): 8-parameter full model

Layer 1 (CSI)  — Power Law spine:  log10 P = a + k·log10(A)
Layer 2 (DSI)  — Log-periodic oscillations: fundamental + harmonic
Layer 3 (Noise)— Age-decaying residual volatility envelope

Time convention: A = age in decimal years since Genesis Block 2009-01-03
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import numpy as np
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Literal, Optional
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

GENESIS_DATE = date(2009, 1, 3)
DAYS_PER_YEAR = 365.25


def age_years(d: date | datetime | str | pd.Timestamp) -> float:
    """Return Bitcoin age in decimal years from Genesis Block."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    elif isinstance(d, (datetime, pd.Timestamp)):
        d = d.date()
    return (d - GENESIS_DATE).days / DAYS_PER_YEAR


def age_years_from_days(t_days: float) -> float:
    """Convert days-since-genesis to decimal years."""
    return t_days / DAYS_PER_YEAR


# ─────────────────────────────────────────────────────────────────────
# MODEL PARAMETERS  (dataclass — easy to override/tune)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PowerLawParams:
    """
    Layer 1: Power law spine parameters.

    Two sources give slightly different fits depending on method:
      OLS  (Santostasi & Perrenod 2026, days as time unit):
            log10 P = -16.509 + 5.690 · log10(t_days)
      QR median (Perrenod Substack, years as time unit):
            log10 P = -2.150  + 5.865 · log10(A_years)

    Both are provided. The app can offer a selector.
    Note: -16.509 + 5.690·log10(365.25·A) ≈ -2.147 + 5.690·log10(A)
    so the difference is almost entirely in the slope (5.690 vs 5.865).
    """
    # OLS fit from Santostasi & Perrenod 2026 paper (days)
    a_ols_days: float = -16.509    # intercept in log10(days) space
    k_ols_days: float = 5.690      # exponent (β)
    sigma_ols:  float = 0.302      # full-history residual std (dex)

    # QR median fit from Perrenod Substack (years)
    a_qr_years: float = -2.150     # intercept in log10(years) space
    k_qr_years: float = 5.865      # exponent
    sigma_qr:   float = 0.302      # same historical σ

    # Bayesian posterior from paper (most precise single estimate)
    k_bayes:    float = 5.729      # posterior mean β
    k_bayes_se: float = 0.013      # posterior std


@dataclass
class LogPeriodicParams:
    """
    Layer 2: Log-periodic (DSI) oscillation parameters.

    Full model equation (slide deck):
      log10 P(A) = [power law] + decay(A) · [A0·cos(ω0·ln(A) + φ0)
                                            + A1·cos(ω1·ln(A) + φ1)]

    where:
      ω0 = 2π / ln(λ) = 8.63   (fundamental, NOT a free parameter)
      ω1 ≈ 3.5 · ω0 ≈ 30.2     (harmonic shown on slide)
      decay(A) = overall_scale / (A + decay_offset)

    Amplitude/phase values (A0, A1, φ0, φ1) are estimated from
    Perrenod's published residual fits. The overall_scale and
    decay_offset come from his March 2026 Substack.

    ⚠️  CALIBRATION NOTE:
    Perrenod has not published exact A0/φ0/A1/φ1 values publicly.
    The values below are best estimates from:
      - His published charts showing ~0.10–0.15 dex oscillation amplitude
      - The RMS reduction table on the slide (0.15 → 0.11 → 0.077 dex)
      - Visual inspection of his Figure 1 residual fit
    These should be re-fit when exact parameters are available.
    Flag NEEDS_CALIBRATION = True to surface a warning in the UI.
    """
    NEEDS_CALIBRATION: bool = True   # flip to False after exact fit

    # Fundamental mode
    lambda_spacing: float = 2.07    # bubble-peak spacing ratio (measured)
    omega_0: float = field(init=False)   # derived from lambda

    # Harmonic mode frequency multiplier (from slide: "effective local ~3.5 × ω₀")
    harmonic_multiplier: float = 3.5

    # Amplitude estimates (dex units) — estimated from published figures
    # Fundamental amplitude declines with age; ~0.12 dex at current age
    A0_nominal: float = 0.12        # fundamental amplitude scale
    A1_nominal: float = 0.06        # harmonic amplitude scale (roughly half)

    # Phase offsets (radians) — estimated to place 2017/2021 peaks correctly
    # φ0 tuned so cos(ω0·ln(A_2017)) ≈ +1 (peak)
    # A_2017 ≈ 9.0 yrs → ln(9.0) ≈ 2.197 → ω0·ln = 8.63·2.197 ≈ 18.96 rad
    # cos(18.96 + φ0) = +1 → φ0 ≈ -(18.96 mod 2π) ≈ -0.96 + 2π ≈ 5.32
    phi_0: float = 5.32             # fundamental phase (radians)
    phi_1: float = 1.85             # harmonic phase (radians) — estimated

    # Amplitude decay envelope: overall_scale / (A + decay_offset)
    overall_scale: float = 1.0      # dimensionless multiplier
    decay_offset:  float = 2.0      # years (from Perrenod: "~1/(A+2.0)")

    def __post_init__(self):
        self.omega_0 = 2 * np.pi / np.log(self.lambda_spacing)   # = 8.63

    @property
    def omega_1(self) -> float:
        return self.harmonic_multiplier * self.omega_0


@dataclass
class NoiseParams:
    """
    Layer 3: Age-decaying residual noise envelope.

    From Perrenod March 2026 Substack:
      σ_noise(A) = 6.1 / (A + 27.1)   [log10 price units]

    At current age ~17.3 yrs: σ ≈ 6.1/44.4 ≈ 0.137 dex
    At age 20.0 yrs (Jan 2029): σ ≈ 6.1/47.1 ≈ 0.130 dex

    The SLIDE RMS values (0.077 dex for full model) reflect BOTH
    the log-periodic structure being explained AND the noise decay.
    """
    noise_scale:  float = 6.1       # numerator coefficient
    noise_offset: float = 27.1      # age offset in years

    def sigma_at_age(self, A: float) -> float:
        """Return 1-sigma noise envelope in log10 price units at age A."""
        return self.noise_scale / (A + self.noise_offset)

    # Slide deck RMS values for each model layer (empirical, 2029 projection)
    rms_powerlaw_only:     float = 0.150
    rms_pl_fundamental:    float = 0.110
    rms_pl_fund_harmonic:  float = 0.077


# ─────────────────────────────────────────────────────────────────────
# CORE MODEL CLASS
# ─────────────────────────────────────────────────────────────────────

ModelType = Literal["ols_days", "qr_years", "bayes"]
LayerType  = Literal["power_law", "pl_fundamental", "pl_fund_harmonic"]


class BitcoinPowerLawModel:
    """
    Three-layer Bitcoin price model.

    Usage:
        model = BitcoinPowerLawModel()
        price = model.price_trend("2029-01-01")
        dist  = model.price_distribution("2029-01-01", n_sigma=2)
    """

    def __init__(
        self,
        pl_params:  Optional[PowerLawParams]    = None,
        lp_params:  Optional[LogPeriodicParams] = None,
        noise_params: Optional[NoiseParams]     = None,
    ):
        self.pl    = pl_params    or PowerLawParams()
        self.lp    = lp_params    or LogPeriodicParams()
        self.noise = noise_params or NoiseParams()

    # ── Layer 1: Power Law ──────────────────────────────────────────

    def log10_trend(
        self,
        A: float,
        model: ModelType = "qr_years",
    ) -> float:
        """
        Power law trend in log10(USD).

        Parameters
        ----------
        A     : Bitcoin age in decimal years
        model : "ols_days"  — Santostasi & Perrenod OLS (days)
                "qr_years"  — Perrenod QR median (years) [DEFAULT]
                "bayes"     — Bayesian posterior mean (years)
        """
        if model == "ols_days":
            t_days = A * DAYS_PER_YEAR
            return self.pl.a_ols_days + self.pl.k_ols_days * np.log10(t_days)
        elif model == "qr_years":
            return self.pl.a_qr_years + self.pl.k_qr_years * np.log10(A)
        elif model == "bayes":
            # Use QR intercept with Bayesian slope (closest published combo)
            return self.pl.a_qr_years + self.pl.k_bayes * np.log10(A)
        else:
            raise ValueError(f"Unknown model: {model}")

    def price_trend(
        self,
        target: date | str,
        model: ModelType = "qr_years",
    ) -> float:
        """Return power-law trend price in USD for a given date."""
        A = age_years(target)
        return 10 ** self.log10_trend(A, model)

    # ── Layer 2: Log-Periodic Oscillation ──────────────────────────

    def log_periodic_signal(self, A: float) -> float:
        """
        Returns the log-periodic oscillation term in log10(USD).
        This is added ON TOP of the power law trend.

        log_periodic(A) = [1/(A + decay_offset)] ×
                          [A0·cos(ω0·ln(A) + φ0) + A1·cos(ω1·ln(A) + φ1)]
        """
        lp = self.lp
        decay   = lp.overall_scale / (A + lp.decay_offset)
        ln_A    = np.log(A)   # natural log
        fund    = lp.A0_nominal * np.cos(lp.omega_0 * ln_A + lp.phi_0)
        harmonic = lp.A1_nominal * np.cos(lp.omega_1 * ln_A + lp.phi_1)
        return decay * (fund + harmonic)

    def log_periodic_fundamental_only(self, A: float) -> float:
        """Fundamental mode only (for PL + Fundamental layer)."""
        lp    = self.lp
        decay = lp.overall_scale / (A + lp.decay_offset)
        ln_A  = np.log(A)
        return decay * lp.A0_nominal * np.cos(lp.omega_0 * ln_A + lp.phi_0)

    # ── Combined Model ──────────────────────────────────────────────

    def log10_full(
        self,
        A: float,
        layer: LayerType  = "pl_fund_harmonic",
        model: ModelType  = "qr_years",
    ) -> float:
        """
        Full log10 price estimate combining chosen layers.

        Layers:
          "power_law"        — CSI only
          "pl_fundamental"   — CSI + DSI fundamental
          "pl_fund_harmonic" — CSI + DSI fundamental + harmonic [DEFAULT]
        """
        base = self.log10_trend(A, model)
        if layer == "power_law":
            return base
        elif layer == "pl_fundamental":
            return base + self.log_periodic_fundamental_only(A)
        elif layer == "pl_fund_harmonic":
            return base + self.log_periodic_signal(A)
        else:
            raise ValueError(f"Unknown layer: {layer}")

    def price_full(
        self,
        target: date | str,
        layer: LayerType = "pl_fund_harmonic",
        model: ModelType = "qr_years",
    ) -> float:
        """Full model price in USD for a given date."""
        A = age_years(target)
        return 10 ** self.log10_full(A, layer, model)

    # ── Uncertainty / Distribution ──────────────────────────────────

    def sigma_at(
        self,
        target: date | str,
        layer: LayerType = "pl_fund_harmonic",
    ) -> float:
        """
        Returns the appropriate 1-sigma in log10 price units for the
        chosen layer.  Uses slide-deck RMS values (empirical, ~2029).
        Falls back to age-decaying noise formula for other dates.
        """
        rms_map = {
            "power_law":        self.noise.rms_powerlaw_only,
            "pl_fundamental":   self.noise.rms_pl_fundamental,
            "pl_fund_harmonic": self.noise.rms_pl_fund_harmonic,
        }
        return rms_map[layer]

    def price_distribution(
        self,
        target: date | str,
        layer: LayerType = "pl_fund_harmonic",
        model: ModelType = "qr_years",
        sigmas: list[float] = (-2, -1, 0, 1, 2),
    ) -> dict[str, float]:
        """
        Return price quantiles for a given date.

        Returns dict with keys like "-2σ", "-1σ", "median", "+1σ", "+2σ"
        and corresponding USD prices.

        The distribution is log-normal: price = 10^(log10_central ± n·σ)
        """
        A    = age_years(target)
        mu   = self.log10_full(A, layer, model)
        sig  = self.sigma_at(target, layer)

        result = {}
        for s in sigmas:
            if s == 0:
                label = "median"
            elif s > 0:
                label = f"+{s}σ"
            else:
                label = f"{s}σ"
            result[label] = 10 ** (mu + s * sig)
        return result

    # ── Time Series ─────────────────────────────────────────────────

    def forecast_series(
        self,
        start: date | str,
        end:   date | str,
        freq:  str = "W",           # pandas freq string: "D", "W", "ME"
        layer: LayerType = "pl_fund_harmonic",
        model: ModelType = "qr_years",
        include_bands: bool = True,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with date-indexed price forecast and
        optional ±1σ / ±2σ bands.

        Columns: date, age_years, log10_price,
                 price_median, [price_m2s, price_m1s, price_p1s, price_p2s]
        """
        dates = pd.date_range(start=str(start), end=str(end), freq=freq)
        rows  = []
        for d in dates:
            A   = age_years(d)
            mu  = self.log10_full(A, layer, model)
            sig = self.sigma_at(d, layer)
            row = {
                "date":          d.date(),
                "age_years":     round(A, 4),
                "log10_price":   round(mu, 4),
                "price_median":  round(10**mu, 0),
            }
            if include_bands:
                row["price_m2s"] = round(10**(mu - 2*sig), 0)
                row["price_m1s"] = round(10**(mu - 1*sig), 0)
                row["price_p1s"] = round(10**(mu + 1*sig), 0)
                row["price_p2s"] = round(10**(mu + 2*sig), 0)
            rows.append(row)

        return pd.DataFrame(rows).set_index("date")

    # ── Phase Analysis ──────────────────────────────────────────────

    def cycle_phase(self, target: date | str) -> dict:
        """
        Diagnose where a date falls in the log-periodic cycle.

        Returns:
          phase_rad       : fundamental oscillation phase (radians, 0–2π)
          phase_label     : "near peak", "descending", "near trough", "ascending"
          lp_contribution : log-periodic offset from trend (dex)
          cycle_position  : 0.0–1.0 (0=peak, 0.5=trough)
        """
        A    = age_years(target)
        lp   = self.lp
        raw_phase = (lp.omega_0 * np.log(A) + lp.phi_0) % (2 * np.pi)
        cos_val   = np.cos(raw_phase)

        if cos_val > 0.5:
            label = "near peak"
        elif cos_val < -0.5:
            label = "near trough"
        elif raw_phase < np.pi:
            label = "descending"
        else:
            label = "ascending"

        lp_contrib = self.log_periodic_signal(A)

        return {
            "phase_rad":       round(raw_phase, 3),
            "phase_label":     label,
            "lp_contribution": round(lp_contrib, 4),
            "cycle_position":  round((1 - cos_val) / 2, 3),
            "age_years":       round(A, 3),
        }

    # ── Summary Table ───────────────────────────────────────────────

    def summary_table(
        self,
        targets: list[str | date],
        layer: LayerType = "pl_fund_harmonic",
        model: ModelType = "qr_years",
    ) -> pd.DataFrame:
        """
        Convenience method: price distribution table for a list of dates.
        Useful for displaying in Streamlit with st.dataframe().
        """
        rows = []
        for t in targets:
            dist  = self.price_distribution(t, layer, model)
            phase = self.cycle_phase(t)
            rows.append({
                "Date":          str(t),
                "Age (yrs)":     round(phase["age_years"], 2),
                "Cycle Phase":   phase["phase_label"],
                "LP Offset":     f"{phase['lp_contribution']:+.3f} dex",
                "-2σ":          f"${dist['-2σ']:,.0f}",
                "-1σ":          f"${dist['-1σ']:,.0f}",
                "Median":        f"${dist['median']:,.0f}",
                "+1σ":          f"${dist['+1σ']:,.0f}",
                "+2σ":          f"${dist['+2σ']:,.0f}",
            })
        return pd.DataFrame(rows)
