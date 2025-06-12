import json
import os
import matplotlib.pyplot as plt
from data_loader import get_option_data
from model import heston_jump_price_MC
from calibrate import calibrate

# --- User Input ---
ticker = input("Enter ticker (e.g. AAPL): ").upper()
expiration_index = int(input("Enter expiration index (e.g. 0): "))
strike_window = float(input("Enter strike window (+/- around spot): "))

# --- Load Market Data ---
data = get_option_data(ticker, expiration_index=expiration_index, strike_window=strike_window)
S0 = data["S0"]
T = data["T"]
r = 0.05
calls_df = data["calls_df"]
expiration = data["expiration"]

# --- Load or Run Calibration ---
cal_file = f"calibrations/{ticker}_{expiration}.json"

if os.path.exists(cal_file):
    with open(cal_file, "r") as f:
        params = json.load(f)
    print(f"Loaded saved parameters from {cal_file}")
else:
    print(f"No calibration found for {ticker} {expiration}. Running calibration...")
    params = calibrate(ticker, expiration_index=expiration_index, strike_window=strike_window)

# --- Compute Model Prices ---
strikes = calls_df["strike"].values
market_prices = calls_df["mid_price"].values
model_prices = [
    heston_jump_price_MC(S0, K, T, r,
        params["v0"], params["kappa"], params["theta"], params["sigma_v"], params["rho"],
        params["lambda_j"], params["mu_j"], params["sigma_j"])
    for K in strikes
]

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(strikes, market_prices, label="Market Price", marker="o")
plt.plot(strikes, model_prices, label="Model Price", marker="x")
plt.xlabel("Strike Price")
plt.ylabel("Option Price ($)")
plt.title(f"{ticker} Option Pricing Fit\nExpiration: {expiration}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
