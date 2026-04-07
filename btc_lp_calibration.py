"""
btc_lp_calibration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calibration script to fit the log-periodic oscillation parameters
(A0, phi_0, A1, phi_1) to real Bitcoin price data.

Run this ONCE with historical price data to derive the exact values
for your LogPeriodicParams dataclass.  Results can then be hard-coded
into btc_powerlaw_model.py with NEEDS_CALIBRATION = False.

Usage:
    python btc_lp_calibration.py --csv btc_price_history.csv

CSV format expected:
    date,close
    2010-07-18,0.09
    ...

Or supply prices via Glassnode / Kraken / Coinbase API fetch.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import argparse
import json
from pathlib import Path

from btc_powerlaw_model import (
    BitcoinPowerLawModel,
    PowerLawParams,
    LogPeriodicParams,
    age_years,
    DAYS_PER_YEAR,
)


# ─────────────────────────────────────────────────────────────────────
# STEP 1: Load and prepare price data
# ─────────────────────────────────────────────────────────────────────

def load_price_data(csv_path: str) -> pd.DataFrame:
    """
    Load BTC price CSV with columns [date, close].
    Returns DataFrame with age_years and log10_price columns,
    filtered to t > 560 days (matching Santostasi & Perrenod 2026).
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=["close"])
    df["age_years"]   = df["date"].apply(lambda d: age_years(d.date()))
    df["log10_price"] = np.log10(df["close"])

    # Filter: exclude pre-exchange era (< 560 days = 1.533 years)
    df = df[df["age_years"] > 1.533].copy()
    print(f"Loaded {len(df)} observations from {df['date'].min().date()} "
          f"to {df['date'].max().date()}")
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 2: Fit power law and extract residuals
# ─────────────────────────────────────────────────────────────────────

def fit_power_law_qr(df: pd.DataFrame) -> tuple[float, float]:
    """
    OLS in log-log space (Quantile Regression approximation via median
    is complex; this uses OLS as a starting point).
    Returns (intercept, slope).
    """
    log_A = np.log10(df["age_years"].values)
    log_P = df["log10_price"].values
    # OLS
    A_mat = np.column_stack([np.ones_like(log_A), log_A])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_P, rcond=None)
    intercept, slope = coeffs
    print(f"Power law fit: log10(P) = {intercept:.4f} + {slope:.4f}·log10(A)")
    return intercept, slope


def get_stage1_residuals(df: pd.DataFrame, intercept: float, slope: float) -> np.ndarray:
    """Compute residuals after removing power law trend."""
    log_A = np.log10(df["age_years"].values)
    fitted = intercept + slope * log_A
    return df["log10_price"].values - fitted


# ─────────────────────────────────────────────────────────────────────
# STEP 3: Fit log-periodic oscillations to residuals
# ─────────────────────────────────────────────────────────────────────

def lp_model(A: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two-mode log-periodic model (fundamental + harmonic).

    params = [overall_scale, phi_0, phi_1, A0_ratio, decay_offset]
    Fixed: omega_0 = 8.63, omega_1 = 3.5 * omega_0
    A0_nominal and A1_nominal derived from overall_scale and A0_ratio.
    """
    overall_scale, phi_0, phi_1, A0_ratio, decay_offset = params
    omega_0 = 8.63
    omega_1 = 3.5 * omega_0

    # Amplitude: overall_scale defines total, A0_ratio splits between modes
    A0 = overall_scale * A0_ratio         # fundamental fraction
    A1 = overall_scale * (1 - A0_ratio)   # harmonic fraction

    decay   = 1.0 / (A + decay_offset)
    ln_A    = np.log(A)
    fund    = A0 * np.cos(omega_0 * ln_A + phi_0)
    harmonic = A1 * np.cos(omega_1 * ln_A + phi_1)
    return decay * (fund + harmonic)


def fit_log_periodic(df: pd.DataFrame, residuals: np.ndarray) -> dict:
    """
    Fit log-periodic parameters to stage-1 residuals.
    Uses differential evolution (global optimizer) to avoid local minima.

    Returns dict with best-fit parameters.
    """
    A = df["age_years"].values

    def objective(params):
        pred = lp_model(A, params)
        return np.sum((residuals - pred)**2)

    # Parameter bounds
    #  overall_scale: amplitude envelope scale (0.01 to 0.5 dex)
    #  phi_0, phi_1: phase offsets (0 to 2π)
    #  A0_ratio: fraction of amplitude in fundamental (0.4 to 0.9)
    #  decay_offset: age offset in decay denominator (0.5 to 5.0 years)
    bounds = [
        (0.05, 0.50),    # overall_scale
        (0.0, 2*np.pi),  # phi_0
        (0.0, 2*np.pi),  # phi_1
        (0.40, 0.90),    # A0_ratio
        (0.5, 5.0),      # decay_offset
    ]

    print("Running differential evolution optimizer (may take ~30s)...")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=1000, tol=1e-8,
        workers=-1,   # use all CPU cores
        popsize=20,
    )

    # Polish with Nelder-Mead
    result2 = minimize(objective, result.x, method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 50000})

    p = result2.x
    overall_scale, phi_0, phi_1, A0_ratio, decay_offset = p

    fitted  = lp_model(A, p)
    ss_res  = np.sum((residuals - fitted)**2)
    ss_tot  = np.sum((residuals - np.mean(residuals))**2)
    r2      = 1 - ss_res / ss_tot
    rms     = np.sqrt(np.mean((residuals - fitted)**2))

    results = {
        "overall_scale": round(float(overall_scale), 5),
        "phi_0":         round(float(phi_0 % (2*np.pi)), 5),
        "phi_1":         round(float(phi_1 % (2*np.pi)), 5),
        "A0_nominal":    round(float(overall_scale * A0_ratio), 5),
        "A1_nominal":    round(float(overall_scale * (1 - A0_ratio)), 5),
        "A0_ratio":      round(float(A0_ratio), 4),
        "decay_offset":  round(float(decay_offset), 4),
        "R2_lp":         round(float(r2), 4),
        "rms_after_lp":  round(float(rms), 5),
        "omega_0":       8.63,
        "omega_1":       round(3.5 * 8.63, 4),
    }
    return results


# ─────────────────────────────────────────────────────────────────────
# STEP 4: Fit noise decay envelope to stage-2 residuals
# ─────────────────────────────────────────────────────────────────────

def fit_noise_decay(
    df: pd.DataFrame,
    stage1_residuals: np.ndarray,
    lp_params: dict,
) -> dict:
    """
    After removing log-periodic signal, fit the noise decay envelope
    σ(A) = noise_scale / (A + noise_offset) to the absolute residuals.
    """
    A = df["age_years"].values
    lp_signal  = lp_model(A, [
        lp_params["overall_scale"],
        lp_params["phi_0"],
        lp_params["phi_1"],
        lp_params["A0_ratio"],
        lp_params["decay_offset"],
    ])
    stage2_res = np.abs(stage1_residuals - lp_signal)

    def noise_model(A, scale, offset):
        return scale / (A + offset)

    from scipy.optimize import curve_fit
    popt, _ = curve_fit(noise_model, A, stage2_res, p0=[6.0, 27.0], maxfev=10000)
    scale, offset = popt

    fitted_noise = noise_model(A, scale, offset)
    rms = np.sqrt(np.mean((stage2_res - fitted_noise)**2))

    return {
        "noise_scale":  round(float(scale), 4),
        "noise_offset": round(float(offset), 4),
        "rms_noise_fit": round(float(rms), 5),
    }


# ─────────────────────────────────────────────────────────────────────
# STEP 5: Print and save calibrated parameters
# ─────────────────────────────────────────────────────────────────────

def print_calibration_results(
    pl_intercept: float,
    pl_slope: float,
    lp_results: dict,
    noise_results: dict,
):
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"\nPower Law (OLS from this dataset):")
    print(f"  intercept (a):  {pl_intercept:.5f}")
    print(f"  slope (k):      {pl_slope:.5f}")

    print(f"\nLog-Periodic Parameters:")
    for k, v in lp_results.items():
        print(f"  {k:20s}: {v}")

    print(f"\nNoise Decay Parameters:")
    for k, v in noise_results.items():
        print(f"  {k:20s}: {v}")

    print("\n--- Copy into LogPeriodicParams / NoiseParams: ---")
    print(f"""
LogPeriodicParams(
    NEEDS_CALIBRATION  = False,
    lambda_spacing     = 2.07,          # fixed
    A0_nominal         = {lp_results['A0_nominal']},
    A1_nominal         = {lp_results['A1_nominal']},
    phi_0              = {lp_results['phi_0']},
    phi_1              = {lp_results['phi_1']},
    overall_scale      = {lp_results['overall_scale']},
    decay_offset       = {lp_results['decay_offset']},
    harmonic_multiplier= 3.5,
)

NoiseParams(
    noise_scale  = {noise_results['noise_scale']},
    noise_offset = {noise_results['noise_offset']},
)
""")


def save_calibration(output_path: str, pl_intercept, pl_slope, lp, noise):
    results = {
        "power_law": {"intercept": pl_intercept, "slope": pl_slope},
        "log_periodic": lp,
        "noise": noise,
    }
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"\nCalibration saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def run_calibration(csv_path: str, output_path: str = "lp_calibration.json"):
    print(f"Loading data from {csv_path}...")
    df = load_price_data(csv_path)

    print("\nFitting power law spine...")
    pl_intercept, pl_slope = fit_power_law_qr(df)
    stage1_res = get_stage1_residuals(df, pl_intercept, pl_slope)
    print(f"Stage-1 residual std: {stage1_res.std():.4f} dex")

    print("\nFitting log-periodic oscillations...")
    lp_results = fit_log_periodic(df, stage1_res)
    print(f"  R² of LP fit on residuals: {lp_results['R2_lp']}")
    print(f"  RMS after LP removal:      {lp_results['rms_after_lp']} dex")

    print("\nFitting noise decay envelope...")
    noise_results = fit_noise_decay(df, stage1_res, lp_results)

    print_calibration_results(pl_intercept, pl_slope, lp_results, noise_results)
    save_calibration(output_path, pl_intercept, pl_slope, lp_results, noise_results)

    return lp_results, noise_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate BTC log-periodic model")
    parser.add_argument("--csv",    required=True,         help="Path to BTC price CSV (date,close)")
    parser.add_argument("--output", default="lp_calibration.json", help="Output JSON path")
    args = parser.parse_args()
    run_calibration(args.csv, args.output)
