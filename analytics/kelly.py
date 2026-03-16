import numpy as np
import pandas as pd


def compute_returns(strikes: list[float], premiums: dict[float, float],
                    scenarios: list[dict]) -> pd.DataFrame:
    """
    Compute option return for each (strike, scenario) pair.
    R = (MSTR_target - strike) / premium - 1  if in the money, else -1
    Returns DataFrame indexed by strike, columns = scenario labels.
    """
    rows = {}
    for strike in strikes:
        premium = premiums.get(strike)
        if premium is None or premium <= 0:
            continue
        row = {}
        for s in scenarios:
            target = s["mstr_price"]
            if target > strike:
                row[s["label"]] = (target - strike) / premium - 1
            else:
                row[s["label"]] = -1.0
        rows[strike] = row
    return pd.DataFrame(rows).T  # index=strike, columns=scenario labels


def single_asset_kelly(returns_df: pd.DataFrame, probabilities: dict[str, float],
                       kelly_fraction: float = 0.5,
                       r_period: float = 0.0) -> pd.Series:
    """
    Compute Kelly fraction for each strike using excess returns over the
    opportunity-cost rate: R_excess = R - r_period.
    Formula: f* = E[R_excess]^2 / E[R_excess^2], then apply kelly_fraction.
    """
    labels = list(probabilities.keys())
    probs = np.array([probabilities[l] for l in labels if l in returns_df.columns])
    cols = [l for l in labels if l in returns_df.columns]

    results = {}
    for strike, row in returns_df.iterrows():
        r = np.array([row[c] for c in cols]) - r_period   # excess returns
        e_r = np.dot(probs, r)
        e_r2 = np.dot(probs, r ** 2)
        if e_r2 > 0 and e_r > 0:
            f_star = (e_r ** 2) / e_r2
        else:
            f_star = 0.0
        results[strike] = min(f_star * kelly_fraction, 1.0)
    return pd.Series(results)


def single_asset_kelly_full(returns_df: pd.DataFrame, probabilities: dict[str, float],
                            r_period: float = 0.0) -> pd.Series:
    """
    Full (unadjusted) Kelly f* = E[R_excess]^2 / E[R_excess^2], before
    kelly_fraction multiplier.  R_excess = R - r_period.
    """
    labels = list(probabilities.keys())
    probs = np.array([probabilities[l] for l in labels if l in returns_df.columns])
    cols = [l for l in labels if l in returns_df.columns]

    results = {}
    for strike, row in returns_df.iterrows():
        r = np.array([row[c] for c in cols]) - r_period   # excess returns
        e_r = np.dot(probs, r)
        e_r2 = np.dot(probs, r ** 2)
        if e_r2 > 0 and e_r > 0:
            results[strike] = min((e_r ** 2) / e_r2, 1.0)
        else:
            results[strike] = 0.0
    return pd.Series(results)


def expected_return(returns_df: pd.DataFrame, probabilities: dict[str, float]) -> pd.Series:
    """Raw expected return E[R] (no opportunity-cost adjustment) — for display."""
    labels = list(probabilities.keys())
    probs = np.array([probabilities[l] for l in labels if l in returns_df.columns])
    cols = [l for l in labels if l in returns_df.columns]

    results = {}
    for strike, row in returns_df.iterrows():
        r = np.array([row[c] for c in cols])
        results[strike] = float(np.dot(probs, r))
    return pd.Series(results)


def prob_profit(returns_df: pd.DataFrame, probabilities: dict[str, float]) -> pd.Series:
    labels = list(probabilities.keys())
    cols = [l for l in labels if l in returns_df.columns]
    probs_arr = np.array([probabilities[l] for l in labels if l in returns_df.columns])

    results = {}
    for strike, row in returns_df.iterrows():
        r = np.array([row[c] for c in cols])
        results[strike] = float(np.dot(probs_arr, (r > 0).astype(float)))
    return pd.Series(results)


def blend_kelly(jacobian_kelly: pd.Series, bhm_kelly: pd.Series) -> pd.Series:
    combined = pd.DataFrame({"j": jacobian_kelly, "b": bhm_kelly})
    return combined.mean(axis=1)


def allocate(kelly_fractions: pd.Series, bankroll: float) -> pd.Series:
    return (kelly_fractions * bankroll).round(0)


def build_portfolio_metrics(strikes: list[float], premiums: dict[float, float],
                             j_scenarios: list[dict], bhm_scenarios: list[dict],
                             kelly_fraction: float, bankroll: float,
                             r_period: float = 0.0) -> pd.DataFrame:
    """
    Build the full Portfolio Growth Metrics table.

    E[R] columns show RAW expected returns (for readability / chart display).
    Kelly f* and $ Allocated use EXCESS returns (R - r_period) so the
    opportunity cost of non-invested capital is priced into the allocation.

    Columns: Premium | Jac E[R] | BHM E[R] | Blended E[R] |
             Jac Kelly f* | BHM Kelly f* | Blended Kelly f* |
             Adj. Kelly | $ Allocated | Contracts
    """
    j_probs = {s["label"]: s["prob"] for s in j_scenarios}
    b_probs = {s["label"]: s["prob"] for s in bhm_scenarios}

    j_returns = compute_returns(strikes, premiums, j_scenarios)
    b_returns = compute_returns(strikes, premiums, bhm_scenarios)

    valid_strikes = sorted(set(j_returns.index) & set(b_returns.index))

    # Raw E[R] — displayed as-is so users can compare against the hurdle line
    j_er = expected_return(j_returns.loc[valid_strikes], j_probs)
    b_er = expected_return(b_returns.loc[valid_strikes], b_probs)

    # Kelly f* uses excess returns (R - r_period) — opportunity cost baked in
    j_kelly_full = single_asset_kelly_full(j_returns.loc[valid_strikes], j_probs, r_period)
    b_kelly_full = single_asset_kelly_full(b_returns.loc[valid_strikes], b_probs, r_period)
    blended_full = blend_kelly(j_kelly_full, b_kelly_full)

    adj_kelly = blended_full * kelly_fraction
    dollar_alloc = allocate(adj_kelly, bankroll)
    contract_cost = pd.Series({s: premiums[s] * 100 for s in valid_strikes})
    contracts = (dollar_alloc / contract_cost).apply(lambda x: max(0, int(x)))

    df = pd.DataFrame({
        "Premium": [premiums[s] for s in valid_strikes],
        "Jac E[R]": j_er.round(2),
        "BHM E[R]": b_er.round(2),
        "Blended E[R]": ((j_er + b_er) / 2).round(2),
        "Jac Kelly f*": j_kelly_full.round(4),
        "BHM Kelly f*": b_kelly_full.round(4),
        "Blended Kelly f*": blended_full.round(4),
        "Adj. Kelly": adj_kelly.round(4),
        "$ Allocated": dollar_alloc,
        "Contracts": contracts,
    }, index=valid_strikes)
    df.index.name = "Strike"

    return df.sort_values("$ Allocated", ascending=False)
