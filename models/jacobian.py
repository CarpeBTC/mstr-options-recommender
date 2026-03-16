import math
from datetime import date

GENESIS_DATE = date(2009, 1, 3)

# Regression coefficients from spreadsheet (intercept, slope)
QUANTILES = {
    "q=0.01": (-35.7239, 5.3596),
    "q=0.05": (-36.1146, 5.4254),
    "q=0.15": (-35.5207, 5.3707),
    "q=0.25": (-35.3487, 5.3631),
    "q=0.50": (-32.5032, 5.0671),
    "OLS":    (-31.2831, 4.9256),
    "q=0.75": (-28.3512, 4.6154),
    "q=0.85": (-25.3631, 4.2777),
    "q=0.95": (-22.9326, 4.0300),
    "q=0.99": (-21.6900, 3.9150),
}

SCENARIO_PROBS = {
    "q=0.01": 0.04,
    "q=0.05": 0.10,
    "q=0.15": 0.10,
    "q=0.25": 0.25,
    "q=0.50": 0.00,  # not used as standalone scenario
    "OLS":    0.25,
    "q=0.75": 0.10,
    "q=0.85": 0.10,
    "q=0.95": 0.04,
    "q=0.99": 0.01,
    "below_q01": 0.01,
}


def get_btc_price(target_date: date) -> dict[str, float]:
    days = (target_date - GENESIS_DATE).days
    return {
        label: math.exp(intercept + slope * math.log(days))
        for label, (intercept, slope) in QUANTILES.items()
    }


def get_scenario_prices(target_date: date) -> list[dict]:
    prices = get_btc_price(target_date)
    scenarios = []
    for label, prob in SCENARIO_PROBS.items():
        if label == "below_q01":
            scenarios.append({"label": "below q=0.01", "prob": prob, "btc_price": 0.0})
        else:
            scenarios.append({"label": label, "prob": prob, "btc_price": prices[label]})
    return scenarios
