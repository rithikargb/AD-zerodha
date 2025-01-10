import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA", "BRK-B", "JPM","^NSEI","^BSESN"]

start_date = "2020-01-01"
end_date = "2024-12-31"

all_data = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Ticker'] = ticker
    all_data = pd.concat([all_data, data])


all_data.to_csv("C:/Users/KIIT/Desktop/historical_data_2020-2024.csv", index=True)