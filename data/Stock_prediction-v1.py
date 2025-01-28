import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Attention, Dropout, Input
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

epsilon = 0.01

data = pd.read_csv('historical_data_2020-2024.csv')

num_col = ['Close', 'Open', 'High', 'Low', 'Volume']
data[num_col] = data[num_col].apply(pd.to_numeric, errors='coerce')
data[num_col] = data[num_col].fillna(data[num_col].mean())
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data[:-1]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[num_col])

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_features)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def build_alstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_output = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    attention = Attention()([lstm_output, lstm_output])
    attention = tf.keras.layers.LayerNormalization()(attention)
    attention = Dropout(0.2)(attention)
    dense_layer = Dense(64, activation='relu')(attention)
    output = Dense(1)(dense_layer)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

input_shape = (X_train.shape[1], X_train.shape[2])
model = build_alstm_model(input_shape)
def generate_adversarial_samples(model, x_batch, y_batch, epsilon=0.01):
    x_batch_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    y_batch_tensor = tf.convert_to_tensor(y_batch, dtype=tf.float32)

    loss_object = tf.keras.losses.MeanSquaredError()

    with tf.GradientTape() as tape:
        tape.watch(x_batch_tensor)
        predictions = model(x_batch_tensor, training=True)
        loss = loss_object(y_batch_tensor, predictions)
    
    gradient = tape.gradient(loss, x_batch_tensor)
    adversarial_samples = x_batch_tensor + epsilon * tf.sign(gradient)
    return tf.clip_by_value(adversarial_samples, 0.0, 1.0)

batch_size = 64
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        loss_clean = model.train_on_batch(x_batch, y_batch)

        x_adversarial = generate_adversarial_samples(model, x_batch, y_batch)

        loss_adversarial = model.train_on_batch(x_adversarial, y_batch)
        
    print(f"Clean Loss: {loss_clean:.4f}, Adversarial Loss: {loss_adversarial:.4f}")

y_pred = model.predict(X_test)

y_pred = y_pred.reshape(-1, 1)
y_pred_2d = np.hstack((y_pred, np.zeros_like(y_pred), np.zeros_like(y_pred), np.zeros_like(y_pred), np.zeros_like(y_pred)))
y_pred_rescaled = scaler.inverse_transform(y_pred_2d)

y_test = y_test.reshape(-1, 1)
y_test_2d = np.hstack((y_test, np.zeros_like(y_test), np.zeros_like(y_test), np.zeros_like(y_test), np.zeros_like(y_test)))
y_test_rescaled = scaler.inverse_transform(y_test_2d)

print("Predicted Stock Prices (rescaled):", y_pred_rescaled[:5, 0])
print("Actual Stock Prices (rescaled):", y_test_rescaled[:5, 0])