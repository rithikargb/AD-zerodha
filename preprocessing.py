import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('./historical_data_2020-2024.csv')  # Correct path

# Check for NaN values in each column and print to debug
print("NaN values per column before filling:")
print(data.isna().sum())

# Drop columns that are completely NaN (likely the unwanted ones like 'Close.11', 'High.11', etc.)
data = data.dropna(axis=1, how='all')

# Remove columns with a numeric suffix (Close.1, Close.2, etc.)
data = data.loc[:, ~data.columns.str.contains(r'\.\d+$')]

# Check the columns after cleanup
print("Columns after dropping NaN-only and duplicate columns:")
print(data.columns)

# Handle remaining NaNs in 'Ticker' column by filling them with 'Unknown'
data['Ticker'] = data['Ticker'].fillna('Unknown')

# Check for NaN values again to ensure they were filled
print("NaN values per column after filling:")
print(data.isna().sum())

# Define the numeric columns to process
num_col = ['Close', 'Open', 'High', 'Low', 'Volume']

# Convert numeric columns to proper type and handle NaNs by filling with the column mean
data[num_col] = data[num_col].apply(pd.to_numeric, errors='coerce')  # Ensures columns are numeric
data[num_col] = data[num_col].fillna(data[num_col].mean())  # Fill NaNs with the column mean

# Check for NaN values again to ensure they were filled
print("NaN values per column after filling numeric columns:")
print(data.isna().sum())

# Now add the Target column (to predict if the next day's Close is higher)
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop the last row as it has no target (since the last row's target is shifted out)
data = data[:-1]

# Apply MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[num_col])

# Now your dataset is ready for training
print("Preprocessing completed successfully.")
