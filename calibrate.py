import numpy as np
from scipy.optimize import minimize
from model import heston_jump_price_MC
from data_loader import get_option_data
import json
import os

def calibrate(ticker, expiration_index=0, strike_window=10, folder="calibrations"):
    data = get_option_data(ticker, expiration_index, strike_window)
    S0 = data["S0"]
    T = data["T"]
    r = 0.05
    calls_df = data["calls_df"]
    expiration = data["expiration"]

    def loss_function(params):
        v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j = params
        loss = 0
        for _, row in calls_df.iterrows():
            K = row["strike"]
            market_price = row["mid_price"]
            model_price = heston_jump_price_MC(
                S0, K, T, r, v0, kappa, theta, sigma_v, rho,
                lambda_j, mu_j, sigma_j, n_sim=1500
            )
            loss += (model_price - market_price) ** 2
        return loss / len(calls_df)

    initial_params = [0.04, 2.0, 0.04, 0.3, -0.7, 0.2, -0.05, 0.2]
    bounds = [
        (0.001, 1.0), (0.1, 5.0), (0.01, 1.0), (0.01, 1.0),
        (-0.99, 0.0), (0.01, 1.0), (-0.3, 0.0), (0.01, 0.5)
    ]

    result = minimize(loss_function, initial_params, bounds=bounds, method='L-BFGS-B', options={'maxiter': 15})
    calibrated_params = dict(zip(
        ["v0", "kappa", "theta", "sigma_v", "rho", "lambda_j", "mu_j", "sigma_j"],
        result.x
    ))

    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{ticker.upper()}_{expiration}.json"
    with open(filename, "w") as f:
        json.dump(calibrated_params, f, indent=2)

    print(f"Calibration complete. Params saved to {filename}")
    return calibrated_params

if __name__ == "__main__":
    ticker = input("Enter ticker (e.g. AAPL): ").upper()
    expiration_index = int(input("Enter expiration index (e.g. 0): "))
    strike_window = float(input("Enter strike window (+/- around spot): "))
    calibrate(ticker, expiration_index=expiration_index, strike_window=strike_window)