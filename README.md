# Heston–Merton Options Model

This project uses a Monte Carlo simulation of the Heston model with Merton jumps to price equity call options. It calibrates to live option chain data via Yahoo Finance and compares model output to market prices.

---

## 🔧 Files

| File               | Purpose |
|--------------------|---------|
| `model.py`         | Heston + Merton Monte Carlo pricing model |
| `calibrate.py`     | Calibrates model to real options and saves parameters |
| `analyze_strike.py`| Main entry point: loads or creates calibration and analyzes one strike |

---

## 📁 Calibration Output

Each calibration is saved to:

```
calibrations/{TICKER}_{EXPIRATION}.json
```

Example:
```
calibrations/AAPL_2024-06-21.json
```

---

## ▶️ How to Use

1. Run:
   ```
   python analyze_strike.py
   ```

2. Follow prompts:
   - Enter ticker (e.g., `AAPL`)
   - Enter expiration index (e.g., `0`)
   - Enter strike window (e.g., `10`)
   - Enter strike to analyze

3. The model will:
   - Pull live options data
   - Load or calibrate parameters
   - Price the selected strike
   - Print model vs market comparison

---

## 📦 Installation

Install all required libraries with:

```
pip install -r requirements.txt
```

---

## 📊 Example Output

```
--- AAPL Option @ Strike 215 ---
Expiration: 2024-06-21
Current Stock Price: $213.52
Time to Expiration: 0.354 years
Risk-Free Rate: 5.00%
Market Option Price: $4.80
Model Option Price:  $5.02
Difference:          $0.22
```
