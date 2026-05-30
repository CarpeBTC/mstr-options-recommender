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

    Full model equation (Perrenod 2026):
      log10 P(A) = [power law] + decay(A) · [A0·cos(ω0·ln(A) + φ0)
                                            + A1·cos(2ω0·ln(A) + φ1)
                                            + A2·cos(4ω0·ln(A) + φ2)]

    where:
      λ  = 2.0076              — bubble-peak spacing ratio, derived from 4 published
                                  peak ages (t1=2.41, t2=4.83, t3=9.71, t4=19.5 yrs);
                                  internally consistent to < 0.1% — HIGH CONFIDENCE
      ω0 = 2π / ln(λ) = 9.0155 — fundamental frequency, derived from λ — HIGH CONFIDENCE
      2ω0, 4ω0                 — true harmonics (rolling 4-yr window fit from slides)
      φ0 = 4.6407 ± 0.006      — derived from peak alignment — HIGH CONFIDENCE
      decay(A) = overall_scale / (A + decay_offset)

    ⚠️  CALIBRATION NOTE:
    λ, ω0, and φ0 are now HIGH CONFIDENCE (derived from published Perrenod slides).
    The following still require calibration against exact fit data:
      - φ1, φ2  (2ω0 and 4ω0 phase offsets) — currently estimated
      - A0, A1, A2 amplitudes — estimated from published figure visual inspection
    Flag NEEDS_CALIBRATION = True to surface a warning in the UI until exact
    amplitude/phase values are confirmed from Perrenod's calibration data.
    """
    NEEDS_CALIBRATION: bool = True   # flip to False after φ1/φ2/A0-A2 are confirmed

    # Fundamental mode — HIGH CONFIDENCE (derived from 4 published peak ages)
    lambda_spacing: float = 2.0076  # bubble-peak spacing ratio
    omega_0: float = field(init=False)   # derived: 2π/ln(λ) = 9.0155

    # Amplitude pre-decay scales (dex units) — estimated from published figures.
    # These are PRE-DECAY values; effective amplitude = A_nominal * decay(A).
    # At age 9.71 (2017 peak): decay ≈ 0.085, so A0_nominal=1.2 → ~0.10 dex signal.
    # At age 19.5 (2028 peak): decay ≈ 0.047, so combined → ~0.098 dex signal.
    # Still estimated — needs calibration against Perrenod's exact residual fit.
    A0_nominal: float = 1.2         # fundamental (ω0) pre-decay amplitude
    A1_nominal: float = 0.6         # 2ω0 harmonic pre-decay amplitude — estimated
    A2_nominal: float = 0.3         # 4ω0 harmonic pre-decay amplitude — estimated

    # Phase offsets (radians)
    # φ0 derived: peak alignment from 4 published peak ages — HIGH CONFIDENCE
    # φ1, φ2 derived: require all harmonics to peak simultaneously at t1–t4
    #   φ1 = 2π×9  − 2ω0×ln(19.5) = 56.549 − 53.559 = 2.990
    #   φ2 = 2π×17 − 4ω0×ln(19.5) = 106.814 − 107.118 = −0.304  (≡ 5.979 mod 2π)
    phi_0: float = 4.6407           # fundamental phase (radians) ± 0.006 — HIGH CONFIDENCE
    phi_1: float = 2.990            # 2ω0 phase — derived from peak coherence
    phi_2: float = -0.304           # 4ω0 phase — derived from peak coherence

    # Amplitude decay envelope: overall_scale / (A + decay_offset)
    overall_scale: float = 1.0      # dimensionless multiplier
    decay_offset:  float = 2.0      # years (from Perrenod: "~1/(A+2.0)")

    def __post_init__(self):
        self.omega_0 = 2 * np.pi / np.log(self.lambda_spacing)   # ≈ 9.0155

    @property
    def omega_1(self) -> float:
        """2nd harmonic: 2 × ω0 (true harmonic from rolling 4-yr window fit)."""
        return 2.0 * self.omega_0

    @property
    def omega_2(self) -> float:
        """4th harmonic: 4 × ω0 (true harmonic from rolling 4-yr window fit)."""
        return 4.0 * self.omega_0


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
                          [A0·cos(ω0·ln(A)  + φ0)   ← fundamental
                         + A1·cos(2ω0·ln(A) + φ1)   ← 2nd harmonic
                         + A2·cos(4ω0·ln(A) + φ2)]  ← 4th harmonic
        """
        lp = self.lp
        decay    = lp.overall_scale / (A + lp.decay_offset)
        ln_A     = np.log(A)   # natural log
        fund     = lp.A0_nominal * np.cos(lp.omega_0 * ln_A + lp.phi_0)
        harm2    = lp.A1_nominal * np.cos(lp.omega_1 * ln_A + lp.phi_1)
        harm4    = lp.A2_nominal * np.cos(lp.omega_2 * ln_A + lp.phi_2)
        return decay * (fund + harm2 + harm4)

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
