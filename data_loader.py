import yfinance as yf
from datetime import datetime
import pandas as pd

def get_option_data(ticker, expiration_index=0, strike_window=10):
    stock = yf.Ticker(ticker)
    
    # Get current stock price
    S0 = stock.history(period="1d")["Close"].iloc[-1]

    # Get expiration date
    expirations = stock.options
    expiration = expirations[expiration_index]
    T = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.today()).days / 365

    # Get call option chain
    calls = stock.option_chain(expiration).calls
    calls = calls.copy()
    calls["mid_price"] = (calls["bid"] + calls["ask"]) / 2
    calls = calls[calls["mid_price"] > 0]
    
    # Filter by strike window
    calls = calls[abs(calls["strike"] - S0) < strike_window]

    return {
        "S0": S0,
        "T": T,
        "calls_df": calls,
        "expiration": expiration
    }

if __name__ == "__main__":
    ticker = input("Enter ticker (e.g. AAPL): ").upper()
    expiration_index = int(input("Enter expiration index (e.g. 0 for nearest): "))
    strike_window = float(input("Enter strike window (+/- around spot): "))

    data = get_option_data(ticker, expiration_index=expiration_index, strike_window=strike_window)
    print(f"S0: {data['S0']:.2f}, T: {data['T']:.3f} years, Expiration: {data['expiration']}")
    print(data['calls_df'][["strike", "bid", "ask", "mid_price"]].head())
