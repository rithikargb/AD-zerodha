import yfinance as yf
import pandas as pd

stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

period = "5y"
interval = "1d"

stock_data = {}

for stock in stocks:
    print(f"Downloading data for {stock}...")
    stock_data[stock] = yf.Ticker(stock).history(period=period, interval=interval)

for stock, data in stock_data.items():
    filename = f"{stock.replace('.NS', '')}_5Y_Daily.csv"
    data.to_csv(filename)
    print(f"Saved {filename}")

print("Download complete!")
