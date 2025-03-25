import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Attention, Dropout, Input
import tensorflow as tf

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tickers = ['HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'RELIANCE.NS', 'TCS.NS']
epsilon = 0.01
time_steps = 60

# Dataset creation with Open price target
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step, 1])
    return np.array(X), np.array(y)

# Model architecture
def build_alstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_output = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    attention = Attention()([lstm_output, lstm_output])
    attention = tf.keras.layers.LayerNormalization()(attention)
    attention = Dropout(0.2)(attention)
    dense_layer = Dense(64, activation='relu')(attention)
    output = Dense(1)(dense_layer)  # Single output for Open price
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Adversarial sample generation
@tf.function(reduce_retracing=True)
def generate_adversarial_samples(model, x_batch, y_batch, epsilon=0.01):
    x_batch_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    y_batch_tensor = tf.convert_to_tensor(y_batch, dtype=tf.float32)
    
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    with tf.GradientTape() as tape:
        tape.watch(x_batch_tensor)
        predictions = model(x_batch_tensor)
        loss = loss_fn(y_batch_tensor, predictions)
    
    gradient = tape.gradient(loss, x_batch_tensor)
    adversarial_samples = x_batch_tensor + epsilon * tf.sign(gradient)
    
    return tf.clip_by_value(adversarial_samples, 0.0, 1.0)

# Main execution
predictions = {}
actual_open_prices = {}
for ticker in tickers:
# Load data for current ticker
    data = pd.read_csv('/workspaces/AD-zerodha/data/ind_market_dataset.csv')
    ticker_data = data[data['Ticker'] == ticker]
    actual_open_prices[ticker] = ticker_data['Open'].iloc[-1]

# Clean and normalize only the relevant ticker data
    num_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    ticker_data.loc[:, num_cols] = ticker_data[num_cols].apply(pd.to_numeric, errors='coerce')
    ticker_data.loc[:, num_cols] = ticker_data[num_cols].fillna(ticker_data[num_cols].mean())
    ticker_data = ticker_data[:-1]  # Remove last row to avoid NaN target

# Use a separate scaler for each ticker
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ticker_data[num_cols])

# Extract the Open price column separately for proper inverse scaling
    open_scaler = MinMaxScaler()
    open_prices = ticker_data[['Open']].values
    open_scaler.fit(open_prices)

# Create dataset
    X, y = create_dataset(scaled_data, time_steps)

# Split dataset
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

# Train model
    model = build_alstm_model((X_train.shape[1], X_train.shape[2]))
    epochs = 60
    batch_size = 64

    for epoch in range(epochs):
        epoch_clean_loss = 0
        epoch_adv_loss = 0
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size].astype(np.float32)
            y_batch = y_train[i:i+batch_size].astype(np.float32)

        # Clean batch training
            loss_clean = model.train_on_batch(x_batch, y_batch)
            epoch_clean_loss += loss_clean

        # Adversarial training
            x_adv = generate_adversarial_samples(model, x_batch, y_batch, epsilon)
            loss_adv = model.train_on_batch(x_adv, y_batch)
            epoch_adv_loss += loss_adv

        print(f"Epoch {epoch+1}/{epochs} | {ticker}")
        print(f"Clean Loss: {epoch_clean_loss/len(X_train)*batch_size:.4f}")
        print(f"Adv Loss: {epoch_adv_loss/len(X_train)*batch_size:.4f}\n")

# Prediction
    y_pred = model.predict(X_test).flatten()

# Inverse transform Open price using the Open-price-specific scaler
    rescaled_preds = open_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Store best prediction (latest available)
    predictions[ticker] = {
        'predicted_open': rescaled_preds[-1],
    }


# Display final results
print("\nNext Day Opening Price Predictions:")
for ticker in tickers:
    print(f"\n{ticker}:")
    last_open_price = actual_open_prices[ticker]  
    print(f" Actual Open price: ${last_open_price:.2f}")
    print(f" Predicted Open: ${predictions[ticker]['predicted_open']:.2f}")