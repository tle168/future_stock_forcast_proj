import ta
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def compute_indicators(df):
    df = df.copy()
    df['MA5'] = ta.trend.sma_indicator(df['Close'], window=5)
    df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df["MACD_signal"] = ta.trend.macd_signal(df["Close"])
    # Bollinger Bands
    df["BB_high"] = ta.volatility.bollinger_hband(df["Close"], window=20)
    df["BB_low"] = ta.volatility.bollinger_lband(df["Close"], window=20)

    # Ichimoku Cloud
    ichimoku = ta.trend.ichimoku_base_line(df["High"], df["Low"], window1=9, window2=26)
    df["Ichimoku"] = ichimoku
    return df

def generate_signals(df):

    last = df.iloc[-1]
    signals = []
    if last['MA5'] > last['MA20']:
        signals.append('Giao cắt vàng')
    if last['MA5'] < last['MA20']:
        signals.append('Giao cắt tử thần')
    if last['RSI'] > 70:
        signals.append('Quá mua')
    elif last['RSI'] < 30:
        signals.append('Quá bán')
    return signals
def print_signals(df):
    # Calculate short-term and long-term SMAs
    df = df.copy()
    short_sma = df['Close'].rolling(window=20).mean()
    long_sma = df['Close'].rolling(window=50).mean()

    # Generate buy and sell signals
    buy_signal = short_sma > long_sma
    sell_signal = short_sma < long_sma
    
    print_data=pd.DataFrame()
    # print_data["Date"]=df['Date']
    print_data["Price"]=df['Close']
    print_data['buy_signal']=buy_signal
    print_data["sell_signal"]=sell_signal
    return print_data

def predict_stock_price(df):
    #
    # Xóa các dòng có giá trị NaN do chỉ báo kỹ thuật cần nhiều ngày tính toán
    df.dropna(inplace=True)

    # Xác định biến độc lập (X) và biến phụ thuộc (y)
    features = ["MA5", "MA20", "RSI", "MACD", "MACD_signal", "BB_high", "BB_low", "Ichimoku"]
    X = df[features]
    y = df["Close"].shift(-1)  # Dự đoán giá ngày mai

    # Loại bỏ hàng cuối cùng vì không có nhãn (y)
    X = X[:-1]
    y = y[:-1]

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán trên tập test
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Dự đoán giá cổ phiếu ngày mai
    latest_features = X.iloc[-1].values.reshape(1, -1)
    predicted_price = model.predict(latest_features)[0]

    return predicted_price