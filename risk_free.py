import yfinance as yf

def get_risk_free_rate(source="^IRX", fallback=0.05):
    try:
        data = yf.Ticker(source).history(period="1d")
        latest = data["Close"].iloc[-1] / 100  # Convert from % to decimal
        return round(latest, 5)
    except Exception as e:
        print(f"[WARN] Failed to fetch risk-free rate, using fallback ({fallback}): {e}")
        return fallback