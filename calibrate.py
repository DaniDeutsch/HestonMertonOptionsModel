import numpy as np
from scipy.optimize import minimize
from model import heston_jump_price_MC
from data_loader import get_option_data
from risk_free import get_risk_free_rate
import json
import os

def calibrate(ticker, expiration_indices=[0], strike_window=20):
    all_data = []
    r = get_risk_free_rate()

    for expiration_index in expiration_indices:
        data = get_option_data(ticker, expiration_index=expiration_index, strike_window=strike_window)
        calls_df = data["calls_df"]
        S0 = data["S0"]
        T = data["T"]
        expiration = data["expiration"]

        for _, row in calls_df.iterrows():
            K = row["strike"]
            market_price = row["mid_price"]
            all_data.append((S0, K, T, r, market_price, expiration))

    def objective(params):
        v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j = params
        error = 0.0
        for S0, K, T, r, market_price, _ in all_data:
            model_price = heston_jump_price_MC(S0, K, T, r, v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j)
            error += (model_price - market_price) ** 2
        return error / len(all_data)

    # Initial guess
    initial_params = [0.04, 2.0, 0.04, 0.3, -0.7, 0.2, -0.05, 0.2]

    # Bounds for parameters
    bounds = [
        (1e-4, 1.0),   # v0
        (0.1, 5.0),    # kappa
        (1e-4, 1.0),   # theta
        (0.01, 1.0),   # sigma_v
        (-0.99, 0.99), # rho
        (0.0, 1.0),    # lambda_j
        (-0.2, 0.2),   # mu_j
        (0.01, 0.5)    # sigma_j
    ]

    result = minimize(objective, initial_params, method="L-BFGS-B", bounds=bounds)

    if not result.success:
        raise RuntimeError("Calibration failed: " + result.message)

    calibrated_params = {
        "v0": result.x[0],
        "kappa": result.x[1],
        "theta": result.x[2],
        "sigma_v": result.x[3],
        "rho": result.x[4],
        "lambda_j": result.x[5],
        "mu_j": result.x[6],
        "sigma_j": result.x[7]
    }

    expiration_str = "_".join([str(i) for i in expiration_indices])
    cal_file = f"calibrations/{ticker}_{expiration_str}.json"
    os.makedirs("calibrations", exist_ok=True)
    with open(cal_file, "w") as f:
        json.dump(calibrated_params, f, indent=4)

    print(f"Calibration complete and saved to {cal_file}")
    return calibrated_params
