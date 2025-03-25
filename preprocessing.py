# fixes for handling multi-stock datasets + voids cross-ticker data leakage
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('./ind_market_dataset.csv')

print("NaN values per column before filling:")
print(data.isna().sum())

# Remove empty columns and duplicate columns
data = data.dropna(axis=1, how='all')
data = data.loc[:, ~data.columns.str.contains(r'\.\d+$')]

print("\nColumns after initial cleaning:")
print(data.columns)

# Convert date column to datetime and sort
data['Date'] = pd.to_datetime(data['Date'])  # Ensure date column exists
data = data.sort_values(['Ticker', 'Date'])

# Handle missing tickers (if any)
data['Ticker'] = data['Ticker'].fillna('Unknown')

# Process numeric columns
num_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')

# Fill NA values with TICKER-SPECIFIC means
data[num_cols] = data.groupby('Ticker')[num_cols].transform(
    lambda x: x.fillna(x.mean())
)

print("\nNaN values after processing:")
print(data.isna().sum())

# Create target variable using TICKER-SPECIFIC shifts
data['Target'] = data.groupby('Ticker')['Close'].shift(-1) > data['Close']
data['Target'] = data['Target'].astype(int)
data = data.dropna(subset=['Target'])  # Remove last row of each ticker

# Scale features PER TICKER
scaler = MinMaxScaler()
scaled_features = data.groupby('Ticker').apply(
    lambda group: scaler.fit_transform(group[num_cols])
)

# Rebuild scaled DataFrame
scaled_df = pd.DataFrame(
    np.vstack(scaled_features),
    columns=[f"{col}_scaled" for col in num_cols]
)
data = pd.concat([data.reset_index(drop=True), scaled_df], axis=1)

print("\nPreprocessing completed successfully. Final columns:")
print(data.columns)

#CHatGPT version was better only for single CSV datasets, but the perplexct code provided is universally applicable to both formats.
