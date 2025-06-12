import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize

# --- Pricing Model ---
def heston_jump_price_MC(S0, K, T, r, v0, kappa, theta, sigma_v, rho,
                         lambda_j, mu_j, sigma_j, n_steps=252, n_sim=3000, seed=42):
    dt = T / n_steps
    discount_factor = np.exp(-r * T)
    chol = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))
    S_paths = np.full((n_sim, n_steps + 1), S0)
    v_paths = np.full((n_sim, n_steps + 1), v0)
    np.random.seed(seed)
    for t in range(1, n_steps + 1):
        Z = np.random.normal(size=(n_sim, 2))
        dW = Z @ chol.T * np.sqrt(dt)
        v_prev = v_paths[:, t - 1]
        v_next = np.abs(v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * dW[:, 1])
        v_paths[:, t] = v_next
        jumps = np.random.poisson(lambda_j * dt, n_sim)
        jump_sizes = np.exp(mu_j + sigma_j * np.random.randn(n_sim)) * jumps
        S_prev = S_paths[:, t - 1]
        S_next = S_prev * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev) * dW[:, 0]) * (1 + jump_sizes)
        S_paths[:, t] = S_next
    payoffs = np.maximum(S_paths[:, -1] - K, 0)
    return discount_factor * np.mean(payoffs)

# --- User Inputs ---
ticker = input("Enter ticker (e.g., AAPL): ").upper()
strike_to_model = float(input("Enter strike to model: "))
strike_window = float(input("Calibrate using strikes within +/-: "))
expiration_index = int(input("Expiration index (e.g., 0 for nearest): "))

# --- Market Data Pull ---
stock = yf.Ticker(ticker)
S0 = stock.history(period="1d")["Close"].iloc[-1]
r = 0.05
expirations = stock.options
expiration = expirations[expiration_index]
T = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.today()).days / 365
option_chain = stock.option_chain(expiration)
calls = option_chain.calls
calls["mid_price"] = (calls["bid"] + calls["ask"]) / 2
market_calls = calls[(np.abs(calls["strike"] - S0) < strike_window) & (calls["mid_price"] > 0)].copy()

# --- Loss Function for Calibration ---
def loss_function(params):
    v0, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j = params
    loss = 0
    for _, row in market_calls.iterrows():
        K = row["strike"]
        market_price = row["mid_price"]
        model_price = heston_jump_price_MC(S0, K, T, r, v0, kappa, theta, sigma_v, rho,
                                           lambda_j, mu_j, sigma_j, n_sim=1500)
        loss += (model_price - market_price) ** 2
    return loss / len(market_calls)

# --- Calibrate Parameters ---
initial_params = [0.04, 2.0, 0.04, 0.3, -0.7, 0.2, -0.05, 0.2]
bounds = [
    (0.001, 1.0), (0.1, 5.0), (0.01, 1.0), (0.01, 1.0),
    (-0.99, 0.0), (0.01, 1.0), (-0.3, 0.0), (0.01, 0.5)
]
result = minimize(loss_function, initial_params, bounds=bounds, method='L-BFGS-B', options={'maxiter': 15})
params = result.x

# --- Model Selected Strike ---
selected_call = calls[calls["strike"] == strike_to_model]
if selected_call.empty:
    print(f"\n⚠️ No call option found at strike {strike_to_model}")
    exit()

market_price = (selected_call["bid"].values[0] + selected_call["ask"].values[0]) / 2
model_price = heston_jump_price_MC(S0, strike_to_model, T, r, *params)

# --- Output ---
print(f"\n--- {ticker} Option @ Strike {strike_to_model} ---")
print(f"Current Stock Price: ${S0:.2f}")
print(f"Time to Expiration: {T:.3f} years")
print(f"Risk-Free Rate: {r:.2%}")
print(f"Market Option Price: ${market_price:.2f}")
print(f"Model Option Price:  ${model_price:.2f}")
print(f"Difference:          ${model_price - market_price:.2f}")
