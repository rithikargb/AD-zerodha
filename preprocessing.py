import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('./historical_data_2020-2024.csv')


print("NaN values per column before filling:")
print(data.isna().sum())


data = data.dropna(axis=1, how='all')


data = data.loc[:, ~data.columns.str.contains(r'\.\d+$')]


print("Columns after dropping NaN-only and duplicate columns:")
print(data.columns)


data['Ticker'] = data['Ticker'].fillna('Unknown')


print("NaN values per column after filling:")
print(data.isna().sum())

num_col = ['Close', 'Open', 'High', 'Low', 'Volume']

data[num_col] = data[num_col].apply(pd.to_numeric, errors='coerce')
data[num_col] = data[num_col].fillna(data[num_col].mean())
print("NaN values per column after filling numeric columns:")
print(data.isna().sum())

data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

data = data[:-1]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[num_col])
print("Preprocessing completed successfully.")