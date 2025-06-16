import numpy as np
import pandas as pd
from calibrate import calibrate
from model import heston_jump_price_MC
from data_loader import get_option_data
from risk_free import get_risk_free_rate

# --- User Input ---
ticker = input("Enter ticker (e.g. AAPL): ").upper()
calibration_indices = list(map(int, input("Enter calibration expiration indices (comma-separated): ").split(",")))
test_index = int(input("Enter test expiration index: "))
strike_window = float(input("Enter strike window (+/- around spot): "))

# --- Calibrate on Multiple Expirations ---
params = calibrate(ticker, expiration_indices=calibration_indices, strike_window=strike_window)
r = get_risk_free_rate()

# --- Load Test Expiry Data ---
test_data = get_option_data(ticker, expiration_index=test_index, strike_window=strike_window)
calls_df = test_data["calls_df"]
S0 = test_data["S0"]
T = test_data["T"]
expiration = test_data["expiration"]

# --- Evaluate Model on Test Set ---
y_true = calls_df["mid_price"].values
y_pred = [
    heston_jump_price_MC(S0, K, T, r,
        params["v0"], params["kappa"], params["theta"], params["sigma_v"], params["rho"],
        params["lambda_j"], params["mu_j"], params["sigma_j"])
    for K in calls_df["strike"].values
]

# --- Compute MAPE ---
mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

print(f"\nTest on expiration {expiration}: MAPE = {mape:.2f}%")
