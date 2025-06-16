import json
import os
from risk_free import get_risk_free_rate
from data_loader import get_option_data
from model import heston_jump_price_MC
from calibrate import calibrate

# --- User Input ---
ticker = input("Enter ticker (e.g. AAPL): ").upper()
strike_to_check = float(input("Enter strike to analyze: "))
expiration_index = int(input("Enter expiration index (e.g. 0 for nearest): "))
strike_window = float(input("Enter strike window (+/- around spot): "))

# --- Load Market Data ---
data = get_option_data(ticker, expiration_index=expiration_index, strike_window=strike_window)
S0 = data["S0"]
T = data["T"]
r = get_risk_free_rate()
calls_df = data["calls_df"]
expiration = data["expiration"]

# --- Check for existing calibration file ---
cal_file = f"calibrations/{ticker}_{expiration}.json"

if os.path.exists(cal_file):
    with open(cal_file, "r") as f:
        params = json.load(f)
    print(f"Loaded saved parameters from {cal_file}")
else:
    print(f"No calibration found for {ticker} {expiration}. Running calibration...")
    params = calibrate(ticker, expiration_index=expiration_index, strike_window=strike_window)

# --- Select strike ---
selected = calls_df[calls_df["strike"] == strike_to_check]
if selected.empty:
    print(f"⚠️ No call option found at strike {strike_to_check}.")
    exit()

market_price = (selected["bid"].values[0] + selected["ask"].values[0]) / 2
model_price = heston_jump_price_MC(S0, strike_to_check, T, r,
    params["v0"], params["kappa"], params["theta"], params["sigma_v"], params["rho"],
    params["lambda_j"], params["mu_j"], params["sigma_j"]
)

# --- Output ---
print(f"\n--- {ticker} Option @ Strike {strike_to_check} ---")
print(f"Expiration: {expiration}")
print(f"Current Stock Price: ${S0:.2f}")
print(f"Time to Expiration: {T:.3f} years")
print(f"Risk-Free Rate: {r:.2%}")
print(f"Market Option Price: ${market_price:.2f}")
print(f"Model Option Price:  ${model_price:.2f}")
print(f"Difference:          ${model_price - market_price:.2f}")
