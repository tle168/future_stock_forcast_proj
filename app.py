import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vnstock import * # Nạp thư viện để sử dụng
from datetime import datetime, timedelta
from ollama_helper import *
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from model_lstm import prepare_lstm_data, train_lstm_model, forecast_lstm
from ta_signals import compute_indicators, generate_signals, print_signals, predict_stock_price
from charts import plot_price_forecast, plot_heatmap
import io
import requests
import json
import openpyxl # Cần thiết cho to_excel với engine='openpyxl'



# --- Cấu hình Ollama ---
OLLAMA_HOST = "http://localhost:11434" # Mặc định Ollama chạy ở đây
OLLAMA_MODEL = "llama3" # Thay đổi model nếu cần, ví dụ: "mistral"
OLLAMA_API_ENDPOINT = f"{OLLAMA_HOST}/api/generate"


def last_n_days(n):
    """
    Return a date value in YYYY-MM-DD format for last n days. If n = 0, return today's date.
    """
    date_value = (datetime.today() - timedelta(days=n)).strftime('%Y-%m-%d')
    return date_value

LAST_1Y = last_n_days(365)
LAST_30D = last_n_days(30)
LAST_WEEK = last_n_days(7)
YESTERDAY = last_n_days(1) # ngày hôm qua (không phải là ngày cuối cùng giao dịch, đơn giản là ngày liền trước)
Today = datetime.today().strftime('%Y-%m-%d') # ngày hôm nay (không phải là ngày cuối cùng giao dịch, đơn giản là ngày liền trước)
# =========================
# Xử lý dữ liệu và LSTM
# =========================
end_date=Today
start_date = LAST_1Y
# --- Các hàm chức năng ---

def fetch_stock_data(ticker, source='VCI'):
    f"""
    Lấy dữ liệu lịch sử cổ phiếu từ {source}.
    """
    end_date=Today
    start_date = LAST_1Y
    try:
       
        stock = Vnstock().stock(symbol=ticker, source='VCI') # Định nghĩa biến vnstock lưu thông tin mã chứng khoán & nguồn dữ liệu bạn sử dụng
        df = stock.quote.history(start=start_date, end=end_date, interval='1D') # Thiết lập Date tải dữ liệu và khung Date tra cứu là 1 ngày
        
        if df is not None and not df.empty:
            df.columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            # Đảm bảo cột 'Date' là datetime và sắp xếp
            # Kiểm tra xem cột 'Date' có tồn tại không
            if 'Date' not in df.columns:
                # Thử các tên cột phổ biến khác cho ngày tháng
                date_cols = ['time', 'TradingDate', 'Date','datetime']
                for col in date_cols:
                    if col in df.columns:
                        df.rename(columns={col: 'Date'}, inplace=True)
                        break
            if 'Date' not in df.columns:
                st.error(f"Không tìm thấy cột ngày tháng trong dữ liệu từ {source} cho {ticker}.")
                return pd.DataFrame()

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')

            # Đảm bảo các cột cần thiết khác tồn tại
            required_cols = {'Close': ['Close', 'close'], 'Mở cửa': ['Open', 'open'],
                             'High': ['High', 'high'], 'Low': ['Low', 'low'],
                             'Volume': ['Volume', 'volume', 'KLGD khớp lệnh']}
            for vn_col, en_cols in required_cols.items():
                if vn_col not in df.columns:
                    for en_col in en_cols:
                        if en_col in df.columns:
                            df.rename(columns={en_col: vn_col}, inplace=True)
                            break
                if vn_col not in df.columns:
                     st.warning(f"Không tìm thấy cột '{vn_col}' trong dữ liệu. Một số tính năng có thể không hoạt động.")


            df.reset_index(drop=True, inplace=True)
            return df
        else:
            st.warning(f"Không có dữ liệu trả về từ {source} cho mã {ticker} trong khoảng Date đã chọn.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu từ {source} cho mã {ticker}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """
    Tính toán các chỉ báo kỹ thuật.
    """
    if df.empty or 'Close' not in df.columns:
        st.warning("Thiếu cột 'Close' để tính chỉ báo kỹ thuật.")
        return df.copy() # Trả về bản sao để tránh thay đổi df gốc nếu có lỗi

    df_tech = df.copy()

    # Moving Averages
    df_tech['MA5'] = SMAIndicator(close=df_tech['Close'], window=5, fillna=True).sma_indicator()
    df_tech['MA20'] = SMAIndicator(close=df_tech['Close'], window=20, fillna=True).sma_indicator()
    df_tech['MA50'] = SMAIndicator(close=df_tech['Close'], window=50, fillna=True).sma_indicator()
    df_tech['MA200'] = SMAIndicator(close=df_tech['Close'], window=200, fillna=True).sma_indicator()

    # RSI
    df_tech['RSI'] = RSIIndicator(close=df_tech['Close'], window=14, fillna=True).rsi()

    # MACD
    macd = MACD(close=df_tech['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df_tech['MACD_line'] = macd.macd()
    df_tech['MACD_signal'] = macd.macd_signal()
    df_tech['MACD_hist'] = macd.macd_diff() # MACD Histogram

    # Bollinger Bands
    bb = BollingerBands(close=df_tech['Close'], window=20, window_dev=2, fillna=True)
    df_tech['BB_high'] = bb.bollinger_hband()
    df_tech['BB_low'] = bb.bollinger_lband()
    df_tech['BB_mid'] = bb.bollinger_mavg() # Thường là MA20

    return df_tech

def generate_signals(df):
    """
    Tạo tín hiệu mua/bán dựa trên các chỉ báo.
    Đây là một ví dụ đơn giản, cần được tùy chỉnh và kiểm nghiệm kỹ lưỡng.
    """
    if df.empty:
        return df.copy()

    df_sig = df.copy()
    df_sig['Signal'] = 0 # 1 for Buy, -1 for Sell, 0 for Hold

    # MA Crossover (ví dụ: MA5 cắt lên MA20)
    if 'MA5' in df_sig.columns and 'MA20' in df_sig.columns:
        # Mua khi MA5 cắt lên MA20 từ dưới
        buy_condition = (df_sig['MA5'] > df_sig['MA20']) & (df_sig['MA5'].shift(1) <= df_sig['MA20'].shift(1))
        # Bán khi MA5 cắt xuống MA20 từ trên
        sell_condition = (df_sig['MA5'] < df_sig['MA20']) & (df_sig['MA5'].shift(1) >= df_sig['MA20'].shift(1))
        
        df_sig.loc[buy_condition, 'Signal'] = 1
        df_sig.loc[sell_condition, 'Signal'] = -1

    # RSI (ví dụ)
    if 'RSI' in df_sig.columns:
        # Mua khi RSI cắt lên từ vùng quá bán (ví dụ: vượt 30 từ dưới lên)
        buy_rsi_condition = (df_sig['RSI'] > 30) & (df_sig['RSI'].shift(1) <= 30)
        # Bán khi RSI cắt xuống từ vùng quá mua (ví dụ: xuống dưới 70 từ trên xuống)
        sell_rsi_condition = (df_sig['RSI'] < 70) & (df_sig['RSI'].shift(1) >= 70)
        
        # Kết hợp với tín hiệu hiện tại, ưu tiên tín hiệu mới nếu chưa có tín hiệu
        df_sig.loc[buy_rsi_condition & (df_sig['Signal'] == 0), 'Signal'] = 1
        df_sig.loc[sell_rsi_condition & (df_sig['Signal'] == 0), 'Signal'] = -1


    # MACD Crossover (ví dụ)
    if 'MACD_line' in df_sig.columns and 'MACD_signal' in df_sig.columns:
        # Mua khi MACD line cắt lên Signal line
        buy_macd_condition = (df_sig['MACD_line'] > df_sig['MACD_signal']) & (df_sig['MACD_line'].shift(1) <= df_sig['MACD_signal'].shift(1))
        # Bán khi MACD line cắt xuống Signal line
        sell_macd_condition = (df_sig['MACD_line'] < df_sig['MACD_signal']) & (df_sig['MACD_line'].shift(1) >= df_sig['MACD_signal'].shift(1))

        df_sig.loc[buy_macd_condition & (df_sig['Signal'] == 0), 'Signal'] = 1
        df_sig.loc[sell_macd_condition & (df_sig['Signal'] == 0), 'Signal'] = -1

    df_sig['Buy_Signal_Price'] = np.nan
    df_sig['Sell_Signal_Price'] = np.nan
    if 'Close' in df_sig.columns:
        df_sig.loc[df_sig['Signal'] == 1, 'Buy_Signal_Price'] = df_sig['Close']
        df_sig.loc[df_sig['Signal'] == -1, 'Sell_Signal_Price'] = df_sig['Close']

    return df_sig

def plot_market_price(df, ticker):
    """
    Vẽ đồ thị giá Close và Volume.
    Màu của cột Volume sẽ thay đổi: xanh nếu tăng, đỏ nếu giảm so với ngày trước.
    """
    # Đảm bảo các cột cần thiết tồn tại
    if df.empty or 'Date' not in df.columns or 'Close' not in df.columns or 'Volume' not in df.columns:
        st.warning("Thiếu dữ liệu 'Date', 'Close' hoặc 'Volume' để vẽ đồ thị giá thị trường.")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=(f'Giá Close {ticker}', 'Volume giao dịch'),
                        row_heights=[0.7, 0.3])

    # 1. Đồ thị giá Close
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Giá Close',
                             line=dict(color='blue')), row=1, col=1)

    # 2. Đồ thị Volume với màu sắc thay đổi
    # Tính toán thay đổi Volume để xác định màu sắc
    df_plot = df.copy() # Tạo bản sao để tránh SettingWithCopyWarning
    df_plot['Volume_Change'] = df_plot['Volume'].diff()
    
    # Tạo danh sách màu
    colors = []
    for i in range(len(df_plot)):
        if i == 0 or pd.isna(df_plot['Volume_Change'].iloc[i]): # Ngày đầu tiên hoặc không có dữ liệu so sánh
            colors.append('grey')
        elif df_plot['Volume_Change'].iloc[i] > 0:
            colors.append('green')
        elif df_plot['Volume_Change'].iloc[i] < 0:
            colors.append('red')
        else: # Volume_Change == 0
            colors.append('grey')

    fig.add_trace(go.Bar(x=df_plot['Date'], y=df_plot['Volume'], name='Volume',
                         marker_color=colors), row=2, col=1)

    fig.update_layout(
        title_text=f"Phân tích giá thị trường cổ phiếu: {ticker}",
        xaxis_rangeslider_visible=False,
        height=600,
        legend_title_text='Chú giải'
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Giá (VNĐ)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

def plot_recommendations(df_signals, ticker):
    """
    Vẽ đồ thị giá với các tín hiệu mua/bán.
    """
    if df_signals.empty or 'Date' not in df_signals.columns or 'Close' not in df_signals.columns:
        st.warning("Thiếu dữ liệu để vẽ đồ thị khuyến nghị.")
        return

    fig = go.Figure()

    # Giá Close
    fig.add_trace(go.Scatter(x=df_signals['Date'], y=df_signals['Close'],
                             name='Giá Close', line=dict(color='blue')))

    # Đường MA
    if 'MA20' in df_signals.columns:
        fig.add_trace(go.Scatter(x=df_signals['Date'], y=df_signals['MA20'],
                                 name='MA20', line=dict(color='orange', dash='dash')))
    if 'MA50' in df_signals.columns:
        fig.add_trace(go.Scatter(x=df_signals['Date'], y=df_signals['MA50'],
                                 name='MA50', line=dict(color='purple', dash='dash')))

    # Tín hiệu mua
    if 'Buy_Signal_Price' in df_signals.columns:
        fig.add_trace(go.Scatter(x=df_signals['Date'], y=df_signals['Buy_Signal_Price'],
                                name='Tín hiệu Mua', mode='markers',
                                marker=dict(color='green', size=10, symbol='triangle-up')))

    # Tín hiệu bán
    if 'Sell_Signal_Price' in df_signals.columns:
        fig.add_trace(go.Scatter(x=df_signals['Date'], y=df_signals['Sell_Signal_Price'],
                                name='Tín hiệu Bán', mode='markers',
                                marker=dict(color='red', size=10, symbol='triangle-down')))

    fig.update_layout(
        title_text=f"Khuyến nghị Mua/Bán cho {ticker} (Dựa trên chỉ báo cơ bản)",
        xaxis_title="Date",
        yaxis_title="Giá (VNĐ)",
        legend_title="Chú giải",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Thêm đồ thị RSI và MACD
    fig_indicators = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   vertical_spacing=0.1,
                                   subplot_titles=('RSI (14)', 'MACD'))

    if 'RSI' in df_signals.columns:
        fig_indicators.add_trace(go.Scatter(x=df_signals['Date'], y=df_signals['RSI'], name='RSI'), row=1, col=1)
        fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1, annotation_text="Quá mua (70)", annotation_position="bottom right")
        fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1, annotation_text="Quá bán (30)", annotation_position="bottom right")

    if 'MACD_line' in df_signals.columns and 'MACD_signal' in df_signals.columns and 'MACD_hist' in df_signals.columns:
        fig_indicators.add_trace(go.Scatter(x=df_signals['Date'], y=df_signals['MACD_line'], name='MACD Line', line=dict(color='blue')), row=2, col=1)
        fig_indicators.add_trace(go.Scatter(x=df_signals['Date'], y=df_signals['MACD_signal'], name='Signal Line', line=dict(color='orange')), row=2, col=1)
        
        # Màu cho MACD Histogram
        macd_hist_colors = np.where(df_signals['MACD_hist'] > 0, 'green', 'red')
        fig_indicators.add_trace(go.Bar(x=df_signals['Date'], y=df_signals['MACD_hist'], name='MACD Histogram',
                                        marker_color=macd_hist_colors), row=2, col=1)

    fig_indicators.update_layout(height=400, legend_title="Chỉ báo")
    fig_indicators.update_xaxes(title_text="Date", row=2, col=1)
    st.plotly_chart(fig_indicators, use_container_width=True)

def predict_price_regression(df, days_to_predict):
    """
    Dự đoán giá cổ phiếu sử dụng mô hình hồi quy tuyến tính.
    """
    if df.empty or 'Close' not in df.columns or len(df) < 20: # Cần đủ dữ liệu để huấn luyện
        st.warning("Không đủ dữ liệu 'Close' (ít nhất 20 ngày) để thực hiện dự đoán.")
        return pd.DataFrame()

    df_pred = df.copy()
    df_pred['Time_Step'] = np.arange(len(df_pred.index)) # Tạo biến Date

    # Chuẩn bị dữ liệu
    X = df_pred[['Time_Step']] # Đặc trưng
    y = df_pred['Close']    # Mục tiêu

    model = LinearRegression()
    model.fit(X, y) # Huấn luyện trên toàn bộ dữ liệu hiện có

    # Tạo các time step cho tương lai
    last_time_step = df_pred['Time_Step'].iloc[-1]
    future_time_steps = np.arange(last_time_step + 1, last_time_step + 1 + days_to_predict).reshape(-1, 1)

    # Dự đoán
    predicted_prices = model.predict(future_time_steps)

    # Tạo DataFrame cho kết quả dự đoán
    last_date = df_pred['Date'].iloc[-1]
    # Tạo ngày làm việc tiếp theo, bỏ qua cuối tuần nếu muốn (phức tạp hơn)
    # Hiện tại chỉ cộng thêm ngày
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

    predictions_df = pd.DataFrame({
        'Date Dự Đoán': future_dates,
        'Giá Dự Đoán': predicted_prices
    })
    return predictions_df


def predict_price_lstm(df, days_to_predict):
    """
    Dự đoán giá cổ phiếu sử dụng mô hình LSTM.
    """
    if df.empty or 'Đóng cửa' not in df.columns or len(df) < 60:  # Cần ít nhất 60 ngày dữ liệu để huấn luyện
        st.warning("Không đủ dữ liệu 'Đóng cửa' (ít nhất 60 ngày) để thực hiện dự đoán.")
        return pd.DataFrame()

    # Chuẩn bị dữ liệu
    df_pred = df[['Đóng cửa']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_pred)

    # Tạo dữ liệu huấn luyện
    def create_dataset(data, time_steps=60):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    time_steps = 60
    X, y = create_dataset(scaled_data, time_steps)

    # Reshape dữ liệu để phù hợp với LSTM (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Xây dựng mô hình LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile mô hình
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Huấn luyện mô hình
    model.fit(X, y, batch_size=32, epochs=20, verbose=0)

    # Dự đoán cho tương lai
    last_60_days = scaled_data[-time_steps:]
    future_predictions = []
    for _ in range(days_to_predict):
        input_data = last_60_days.reshape((1, time_steps, 1))
        predicted_price = model.predict(input_data, verbose=0)
        future_predictions.append(predicted_price[0, 0])
        last_60_days = np.append(last_60_days[1:], predicted_price, axis=0)

    # Chuyển đổi giá trị dự đoán về thang đo ban đầu
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Tạo DataFrame cho kết quả dự đoán
    last_date = df['Thời Gian'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

    predictions_df = pd.DataFrame({
        'Thời Gian Dự Đoán': future_dates,
        'Giá Dự Đoán': future_predictions.flatten()
    })
    return predictions_df

def call_ollama_api(prompt_text):
    """
    Gửi yêu cầu đến Ollama API và nhận phản hồi.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt_text,
            "stream": False # Nhận toàn bộ phản hồi một lần
        }
        # Thêm headers Content-Type
        headers = {'Content-Type': 'application/json'}
        response = requests.post(OLLAMA_API_ENDPOINT, json=payload, headers=headers, timeout=180) # Tăng timeout
        response.raise_for_status() # Ném lỗi nếu HTTP status code là 4xx/5xx
        
        response_data = response.json()
        return response_data.get("response", "Không nhận được nội dung phản hồi từ Ollama.")

    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối đến Ollama: {e}")
        st.error(f"Đảm bảo Ollama đang chạy tại {OLLAMA_HOST} và model '{OLLAMA_MODEL}' đã được pull.")
        st.error(f"Chi tiết lỗi: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Nội dung phản hồi lỗi từ server (nếu có): {e.response.text}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi giải mã JSON từ phản hồi của Ollama.")
        st.error(f"Nội dung phản hồi thô: {response.text}")
        return None


def generate_ollama_prompt(ticker, df_with_indicators):
    """
    Tạo prompt cho Ollama dựa trên dữ liệu và chỉ báo.
    """
    if df_with_indicators.empty or 'Close' not in df_with_indicators.columns:
        return f"Không có đủ dữ liệu cho cổ phiếu {ticker} để phân tích."

    latest_data = df_with_indicators.iloc[-1]
    prompt = f"""
    Bạn là một chuyên gia phân tích kỹ thuật thị trường chứng khoán Việt Nam.
    Hãy phân tích chuyên sâu về cổ phiếu {ticker} dựa trên các dữ liệu và chỉ báo kỹ thuật gần nhất được cung cấp dưới đây.

    Dữ liệu ngày {latest_data.get('Date', pd.Timestamp('now')).strftime('%Y-%m-%d')}:
    - Giá Close: {latest_data.get('Close', 'N/A'):,.0f} VNĐ
    - MA5: {latest_data.get('MA5', 'N/A'):,.0f}
    - MA20: {latest_data.get('MA20', 'N/A'):,.0f}
    - MA50: {latest_data.get('MA50', 'N/A'):,.0f}
    - MA200: {latest_data.get('MA200', 'N/A'):,.0f}
    - RSI (14): {latest_data.get('RSI', 'N/A'):.2f}
    - MACD Line: {latest_data.get('MACD_line', 'N/A'):.2f}
    - MACD Signal: {latest_data.get('MACD_signal', 'N/A'):.2f}
    - MACD Histogram: {latest_data.get('MACD_hist', 'N/A'):.2f}
    - Bollinger Bands High: {latest_data.get('BB_high', 'N/A'):,.0f}
    - Bollinger Bands Mid (MA20): {latest_data.get('BB_mid', 'N/A'):,.0f}
    - Bollinger Bands Low: {latest_data.get('BB_low', 'N/A'):,.0f}

    Dựa vào các thông tin trên, hãy cung cấp một bài phân tích chi tiết bao gồm:
    1.  **Đánh giá xu hướng hiện tại:** Xác định xu hướng ngắn hạn và trung hạn của cổ phiếu (tăng, giảm, đi ngang). Phân tích vị trí của giá so với các đường MA quan trọng (MA20, MA50, MA200).
    2.  **Phân tích chỉ báo RSI:** Đánh giá mức độ quá mua/quá bán. RSI đang ở vùng nào và có tín hiệu phân kỳ nào không?
    3.  **Phân tích chỉ báo MACD:** Tín hiệu từ MACD line, signal line và histogram. MACD có đang cho tín hiệu mua/bán hay xác nhận xu hướng không?
    4.  **Phân tích Bollinger Bands:** Giá đang ở vị trí nào so với dải Bollinger? Dải Bollinger đang co thắt hay mở rộng, điều này có ý nghĩa gì?
    5.  **Xác định các ngưỡng hỗ trợ và kháng cự:** Dựa trên các đường MA, Bollinger Bands, hoặc các mức giá quan trọng trước đó.
    6.  **Khuyến nghị hành động:** Đưa ra một khuyến nghị cụ thể (Mua, Bán, Nắm giữ, Theo dõi thêm) kèm theo giải thích rõ ràng dựa trên các phân tích ở trên. Nêu rõ các điều kiện để khuyến nghị đó còn hiệu lực hoặc cần xem xét lại.
    7.  **Rủi ro tiềm ẩn (nếu có):** Dựa trên các chỉ báo, có dấu hiệu rủi ro nào cần lưu ý không?

    Hãy trình bày bài phân tích một cách chuyên nghiệp, rõ ràng, và dễ hiểu cho nhà đầu tư.
    Lưu ý: Đây là phân tích dựa trên dữ liệu kỹ thuật và không phải là lời khuyên đầu tư tài chính được cá nhân hóa.
    """
    return prompt

def to_excel(df):
    """Xuất DataFrame ra file Excel dạng bytes."""
    output = io.BytesIO()
    # Sử dụng openpyxl làm engine
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='DuDoanGia')
    # writer.save() không cần thiết khi sử dụng 'with' statement
    processed_data = output.getvalue()
    return processed_data

# --- Giao diện Streamlit ---
st.set_page_config(layout="wide", page_title="Phân Tích Cổ Phiếu VN")

st.title("📈 Công Cụ Phân Tích Cổ Phiếu Việt Nam")
st.markdown("Chào mừng bạn đến với công cụ phân tích cổ phiếu cơ bản, sử dụng dữ liệu từ `vnstock` và phân tích AI từ `Ollama`.")

# --- Sidebar ---
st.sidebar.header("⚙️ Tùy chọn Phân Tích")
stock_ticker = st.sidebar.text_input("Nhập mã cổ phiếu (VD: FPT, HPG):", "FPT").upper()
data_source = st.sidebar.selectbox("Chọn nguồn dữ liệu:", ["VCI","TCBS"], index=0)
window = st.sidebar.slider("Số ngày dữ liệu lịch sử để tải:", 30, 360, 120, step=10) # ~1 năm đến ~5.5 năm
forecast_days = st.sidebar.slider("Dự báo số ngày tới", 5, 30, 10)

run_analysis = st.sidebar.button("🚀 Chạy Phân Tích", type="primary", use_container_width=True)
raw_df=pd.DataFrame()
# --- Xử lý chính ---
if run_analysis and stock_ticker:
    with st.spinner(f"Đang tải và phân tích dữ liệu cho {stock_ticker}..."):
        # 1. Lấy dữ liệu
        # raw_df = fetch_stock_data(stock_ticker, source=data_source)
        stock = Vnstock().stock(symbol=stock_ticker, source='VCI') # Định nghĩa biến vnstock lưu thông tin mã chứng khoán & nguồn dữ liệu bạn sử dụng
        raw_df = stock.quote.history(start=start_date, end=end_date, interval='1D')
        raw_df.columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        raw_df.set_index('Date', inplace=True)

        close_series = raw_df['Close']
        X, y, scaler = prepare_lstm_data(close_series, window)
        model = train_lstm_model(X, y)
        forecast = forecast_lstm(model, close_series, window, forecast_days, scaler)

        forecast_dates = pd.date_range(start=raw_df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({"Forecast": forecast.flatten()}, index=forecast_dates)

        indicators = compute_indicators(raw_df)
        signals = generate_signals(indicators)
        predict_price=predict_stock_price(indicators)
        print_data = print_signals(raw_df)

        if raw_df.empty or 'Close' not in raw_df.columns: # Kiểm tra cột Close
            st.error(f"Không tìm thấy dữ liệu hoặc thiếu cột 'Close' cho mã {stock_ticker} từ nguồn {data_source}.")
            st.caption(f"Đã thử tìm các cột như 'Close', 'close' nhưng không thành công. Vui lòng kiểm tra lại mã cổ phiếu hoặc nguồn dữ liệu.")
        else:
            st.success(f"Đã tải thành công {len(raw_df)} bản ghi cho mã {stock_ticker} từ {data_source} (từ {raw_df['Date'].min().strftime('%Y-%m-%d')} đến {raw_df['Date'].max().strftime('%Y-%m-%d')}).")

            # 2. Tính toán chỉ báo
            df_processed = calculate_technical_indicators(raw_df) # raw_df đã được copy bên trong hàm
            df_with_signals = generate_signals(df_processed) # df_processed đã được copy bên trong hàm

            # 3. Tạo các Tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Giá Thị Trường",
                "💡 Khuyến Nghị (Chỉ báo)",
                "🔮 Dự Đoán Giá",
                "🤖 AI Phân Tích (Ollama)"
            ])

            with tab1:
                st.header(f"Giá Thị Trường và Volume: {stock_ticker}")
                st.markdown("""
                Đồ thị dưới đây hiển thị giá Close lịch sử và Volume giao dịch của cổ phiếu.
                Volume giao dịch được tô màu:
                - **<font color='green'>Xanh</font>**: Volume tăng so với phiên trước.
                - **<font color='red'>Đỏ</font>**: Volume giảm so với phiên trước.
                - **<font color='grey'>Xám</font>**: Volume không đổi hoặc là phiên đầu tiên.
                *Lưu ý: Dữ liệu được cung cấp bởi vnstock và chỉ mang tính chất tham khảo.*
                """, unsafe_allow_html=True)
                if not df_processed.empty:
                    plot_market_price(df_processed, stock_ticker)
                else:
                    st.warning("Không có dữ liệu để vẽ đồ thị giá thị trường.")

            with tab2:
                st.header(f"Khuyến Nghị Dựa Trên Chỉ Báo Kỹ Thuật: {stock_ticker}")
                st.markdown("""
                Đồ thị này hiển thị giá Close cùng với các tín hiệu mua/bán được tạo ra từ các quy tắc đơn giản dựa trên MA, RSI và MACD.
                **LƯU Ý QUAN TRỌNG:** Các tín hiệu này chỉ mang tính chất tham khảo, dựa trên các công thức kỹ thuật cơ bản và **KHÔNG PHẢI LÀ LỜI KHUYÊN ĐẦU TƯ**. Luôn thực hiện nghiên cứu của riêng bạn.
                """)
                if not df_with_signals.empty:
                    plot_recommendations(df_with_signals, stock_ticker)
                    st.subheader("Dữ liệu chỉ báo và tín hiệu 10 ngày gần nhất:")
                    st.dataframe(df_with_signals[['Date', 'Close', 'MA5', 'MA20', 'RSI', 'MACD_line', 'MACD_signal', 'Signal']].sort_values(by='Date', ascending=False).head(10).set_index('Date'))
                else:
                    st.warning("Không có dữ liệu để tạo khuyến nghị.")

            with tab3:
                st.header(f"Dự Đoán Giá Cổ Phiếu (Hồi quy tuyến tính đơn giản): {stock_ticker}")
                st.markdown(f"""
                Phần này sử dụng mô hình Hồi quy Tuyến tính đơn giản để dự đoán giá Close cho **{window}** ngày giao dịch tiếp theo.
                **CẢNH BÁO:** Đây là một mô hình dự đoán rất cơ bản và **KHÔNG NÊN** được coi là dự báo tài chính chính xác.
                Thị trường chứng khoán bị ảnh hưởng bởi nhiều yếu tố phức tạp mà mô hình này không thể nắm bắt.
                """)
                
                if not df_processed.empty and len(df_processed) > 0 :
                    predictions_df = predict_price_lstm(df_processed, window)
                    #df_processed['Date'].index# Dự đoán giá cổ phiếu sử dụng hồi quy tuyến tính


                    if not forecast_df.empty:
                        st.subheader(f"Kết quả dự đoán cho {window} ngày tới:")
                        st.subheader(f"Dự đoán giá ngày mai: {predict_price:.2f} VND")
                        st.dataframe(forecast_df.style.format({"Giá Dự Đoán": "{:,.0f} VNĐ"}))

                        # Vẽ đồ thị dự đoán
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['Close'], name='Giá Lịch Sử', line=dict(color='royalblue')))
                        fig_pred.add_trace(go.Scatter(x=forecast_df['Date Dự Đoán'], y=forecast_df['Giá Dự Đoán'], name='Giá Dự Đoán', line=dict(color='tomato', dash='dash')))
                        fig_pred.update_layout(title=f'Dự đoán giá {stock_ticker} (Hồi quy tuyến tính)', xaxis_title='Date', yaxis_title='Giá (VNĐ)')
                        st.plotly_chart(fig_pred, use_container_width=True)

                        # Nút xuất Excel
                        excel_data = to_excel(forecast_df)
                        st.download_button(
                            label="📥 Tải xuống kết quả dự đoán (Excel)",
                            data=excel_data,
                            file_name=f"du_doan_gia_{stock_ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    else:
                        st.warning("Không thể tạo dự đoán với dữ liệu hiện tại.")
                else:
                    st.warning("Không đủ dữ liệu lịch sử để thực hiện dự đoán (cần ít nhất 20 ngày dữ liệu có cột 'Close').")


            with tab4:
                st.header(f"Phân Tích Chuyên Sâu Bởi AI (Sử dụng Ollama - {OLLAMA_MODEL}): {stock_ticker}")
                st.markdown(f"""
                Chức năng này sử dụng mô hình ngôn ngữ lớn (LLM) chạy trên Ollama (model: `{OLLAMA_MODEL}`) để đưa ra phân tích và nhận định dựa trên các chỉ báo kỹ thuật đã tính toán.
                **LƯU Ý:** Phân tích này được tạo tự động bởi AI và chỉ mang tính chất tham khảo, không phải là lời khuyên đầu tư.
                Hãy đảm bảo Ollama đang chạy trên máy của bạn (`{OLLAMA_HOST}`) và đã pull model `{OLLAMA_MODEL}`.
                """)

                if not df_with_signals.empty:
                    with st.spinner(f"🤖 AI ({OLLAMA_MODEL}) đang phân tích cổ phiếu {stock_ticker}... (Quá trình này có thể mất vài phút)"):
                        prompt = generate_ollama_prompt(stock_ticker, df_with_signals)
                        # st.text_area("Prompt gửi đến Ollama (để debug):", prompt, height=300) # Bỏ comment để xem prompt
                        
                        ai_analysis = ask_ollama(prompt)

                        if ai_analysis:
                            st.subheader("Kết quả phân tích từ AI:")
                            st.markdown(ai_analysis)
                        else:
                            st.error(f"Không thể nhận phản hồi từ Ollama. Vui lòng kiểm tra lại cài đặt Ollama, đảm bảo model '{OLLAMA_MODEL}' đã được pull và Ollama server đang chạy.")
                else:
                    st.warning("Không có đủ dữ liệu chỉ báo để AI phân tích.")
elif run_analysis and not stock_ticker:
    st.warning("⚠️ Vui lòng nhập mã cổ phiếu vào ô bên trái và nhấn 'Chạy Phân Tích'.")

st.sidebar.markdown("---")
st.sidebar.info(f"""
    **Thông tin ứng dụng:**
    - **Phiên bản:** 1.1 (09/05/2025)
    - **Nguồn dữ liệu:** `vnstock` (TCBS, VCI)
    - **Phân tích AI:** `Ollama` (Model: {OLLAMA_MODEL})
    - **Lưu ý:** Thông tin chỉ mang tính tham khảo.
""")
st.sidebar.markdown(f"*{datetime.now().strftime('%A, %d/%m/%Y, %H:%M:%S')}*")
