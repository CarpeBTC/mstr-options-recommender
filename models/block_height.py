import math
from datetime import date

# Reference point from spreadsheet
REF_DATE = date(2026, 3, 4)
REF_BLOCK_HEIGHT = 940_267
BLOCKS_PER_HOUR = 6
BLOCKS_PER_DAY = BLOCKS_PER_HOUR * 24

# Regression coefficients from spreadsheet (intercept, slope)
QUANTILES = {
    "q=0.01": (-70.27, 5.902),
    "q=0.25": (-67.96, 5.758),
    "OLS":    (-64.86, 5.561),
    "q=0.75": (-60.90, 5.291),
    "q=0.85": (-57.56, 5.058),
    "q=0.99": (-51.37, 4.644),
}

SCENARIO_PROBS = {
    "q=0.01": 0.24,
    "q=0.25": 0.00,  # OLS used as central scenario instead
    "OLS":    0.50,
    "q=0.75": 0.10,
    "q=0.85": 0.14,
    "q=0.99": 0.01,
    "below_q01": 0.01,
}


def get_block_height(target_date: date) -> int:
    days_elapsed = (target_date - REF_DATE).days
    return REF_BLOCK_HEIGHT + days_elapsed * BLOCKS_PER_DAY


def get_btc_price(target_date: date) -> dict[str, float]:
    height = get_block_height(target_date)
    return {
        label: math.exp(intercept + slope * math.log(height))
        for label, (intercept, slope) in QUANTILES.items()
    }


def get_scenario_prices(target_date: date) -> list[dict]:
    prices = get_btc_price(target_date)
    scenarios = []
    for label, prob in SCENARIO_PROBS.items():
        if label == "below_q01":
            scenarios.append({"label": "below q=0.01", "prob": prob, "btc_price": 0.0})
        elif label == "q=0.25":
            continue  # skip — OLS covers the central scenario
        else:
            scenarios.append({"label": label, "prob": prob, "btc_price": prices[label]})
    return scenarios
