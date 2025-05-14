import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_lstm_data(series, window=20):
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(window, len(scaled_series)):
        X.append(scaled_series[i - window:i, 0])
        y.append(scaled_series[i, 0])
    return np.array(X), np.array(y), scaler

def train_lstm_model(X, y):
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def forecast_lstm(model, series, window, forecast_days, scaler):
    input_seq = series[-window:].values.reshape(-1, 1)
    input_seq = scaler.transform(input_seq)
    forecast = []
    for _ in range(forecast_days):
        input_reshaped = np.reshape(input_seq, (1, window, 1))
        pred = model.predict(input_reshaped, verbose=0)[0][0]
        forecast.append(pred)
        input_seq = np.append(input_seq[1:], [[pred]], axis=0)
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast
