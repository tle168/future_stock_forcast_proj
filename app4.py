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
# Thư viện cho LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
# Các thư viện khác
import io
import requests
import json
import openpyxl # Cần thiết cho to_excel với engine='openpyxl'
# Cho News Fetching (nếu dùng scraping thực tế)
from bs4 import BeautifulSoup

# --- Cấu hình Ollama ---
OLLAMA_HOST = "http://localhost:11434/v1"
OLLAMA_MODEL = "Gemma3"
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

# --- Các hàm chức năng ---

def fetch_stock_data(ticker, source='VCI', days_to_fetch=1000):
    """
    Lấy dữ liệu lịch sử cổ phiếu từ vnstock.
    """
    end_date=Today
    start_date = last_n_days(days_to_fetch)
    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=days_to_fetch)
    try:
        stock = Vnstock().stock(symbol=ticker, source='VCI') # Định nghĩa biến vnstock lưu thông tin mã chứng khoán & nguồn dữ liệu bạn sử dụng
        df = stock.quote.history(start=start_date, end=end_date, interval='1D') # Thiết lập Date tải dữ liệu và khung Date tra cứu là 1 ngày

        # df = vnstock.stock_historical_data(
        #     symbol=ticker,
        #     start_date=start_date.strftime('%Y-%m-%d'),
        #     end_date=end_date.strftime('%Y-%m-%d'),
        #     resolution='1D',
        #     type='stock',
        #     beautify=True,
        #     source=source
        # )
        if df is not None and not df.empty:
            df.columns=['Thời Gian', 'Open', 'High', 'Low', 'Close', 'Volume']
            if 'Thời Gian' not in df.columns:
                date_cols = ['time', 'TradingDate', 'Date']
                for col in date_cols:
                    if col in df.columns:
                        df.rename(columns={col: 'Thời Gian'}, inplace=True)
                        break
            if 'Thời Gian' not in df.columns:
                st.error(f"Không tìm thấy cột ngày tháng trong dữ liệu từ {source} cho {ticker}.")
                return pd.DataFrame()

            df['Thời Gian'] = pd.to_datetime(df['Thời Gian'])
            df = df.sort_values(by='Thời Gian')

            required_cols = {'Đóng cửa': ['Close', 'close'], 'Mở cửa': ['Open', 'open'],
                             'Cao nhất': ['High', 'high'], 'Thấp nhất': ['Low', 'low'],
                             'Khối lượng': ['Volume', 'volume', 'KLGD khớp lệnh']}
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
            st.warning(f"Không có dữ liệu trả về từ {source} cho mã {ticker} trong khoảng thời gian đã chọn.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu từ {source} cho mã {ticker}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """
    Tính toán các chỉ báo kỹ thuật.
    """
    if df.empty or 'Đóng cửa' not in df.columns:
        st.warning("Thiếu cột 'Đóng cửa' để tính chỉ báo kỹ thuật.")
        return df.copy()

    df_tech = df.copy()
    df_tech['MA5'] = SMAIndicator(close=df_tech['Đóng cửa'], window=5, fillna=True).sma_indicator()
    df_tech['MA20'] = SMAIndicator(close=df_tech['Đóng cửa'], window=20, fillna=True).sma_indicator()
    df_tech['MA50'] = SMAIndicator(close=df_tech['Đóng cửa'], window=50, fillna=True).sma_indicator()
    df_tech['MA200'] = SMAIndicator(close=df_tech['Đóng cửa'], window=200, fillna=True).sma_indicator()
    df_tech['RSI'] = RSIIndicator(close=df_tech['Đóng cửa'], window=14, fillna=True).rsi()
    macd = MACD(close=df_tech['Đóng cửa'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df_tech['MACD_line'] = macd.macd()
    df_tech['MACD_signal'] = macd.macd_signal()
    df_tech['MACD_hist'] = macd.macd_diff()
    bb = BollingerBands(close=df_tech['Đóng cửa'], window=20, window_dev=2, fillna=True)
    df_tech['BB_high'] = bb.bollinger_hband()
    df_tech['BB_low'] = bb.bollinger_lband()
    df_tech['BB_mid'] = bb.bollinger_mavg()
    return df_tech

def generate_signals(df):
    """
    Tạo tín hiệu mua/bán dựa trên các chỉ báo.
    """
    if df.empty:
        return df.copy()
    df_sig = df.copy()
    df_sig['Signal'] = 0
    if 'MA5' in df_sig.columns and 'MA20' in df_sig.columns:
        buy_condition = (df_sig['MA5'] > df_sig['MA20']) & (df_sig['MA5'].shift(1) <= df_sig['MA20'].shift(1))
        sell_condition = (df_sig['MA5'] < df_sig['MA20']) & (df_sig['MA5'].shift(1) >= df_sig['MA20'].shift(1))
        df_sig.loc[buy_condition, 'Signal'] = 1
        df_sig.loc[sell_condition, 'Signal'] = -1
    if 'RSI' in df_sig.columns:
        buy_rsi_condition = (df_sig['RSI'] > 30) & (df_sig['RSI'].shift(1) <= 30)
        sell_rsi_condition = (df_sig['RSI'] < 70) & (df_sig['RSI'].shift(1) >= 70)
        df_sig.loc[buy_rsi_condition & (df_sig['Signal'] == 0), 'Signal'] = 1
        df_sig.loc[sell_rsi_condition & (df_sig['Signal'] == 0), 'Signal'] = -1
    if 'MACD_line' in df_sig.columns and 'MACD_signal' in df_sig.columns:
        buy_macd_condition = (df_sig['MACD_line'] > df_sig['MACD_signal']) & (df_sig['MACD_line'].shift(1) <= df_sig['MACD_signal'].shift(1))
        sell_macd_condition = (df_sig['MACD_line'] < df_sig['MACD_signal']) & (df_sig['MACD_line'].shift(1) >= df_sig['MACD_signal'].shift(1))
        df_sig.loc[buy_macd_condition & (df_sig['Signal'] == 0), 'Signal'] = 1
        df_sig.loc[sell_macd_condition & (df_sig['Signal'] == 0), 'Signal'] = -1
    df_sig['Buy_Signal_Price'] = np.nan
    df_sig['Sell_Signal_Price'] = np.nan
    if 'Đóng cửa' in df_sig.columns:
        df_sig.loc[df_sig['Signal'] == 1, 'Buy_Signal_Price'] = df_sig['Đóng cửa']
        df_sig.loc[df_sig['Signal'] == -1, 'Sell_Signal_Price'] = df_sig['Đóng cửa']
    return df_sig

def plot_market_price(df, ticker):
    """
    Vẽ đồ thị giá đóng cửa và khối lượng.
    """
    if df.empty or 'Thời Gian' not in df.columns or 'Đóng cửa' not in df.columns or 'Khối lượng' not in df.columns:
        st.warning("Thiếu dữ liệu 'Thời Gian', 'Đóng cửa' hoặc 'Khối lượng' để vẽ đồ thị giá thị trường.")
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Giá đóng cửa {ticker}', 'Khối lượng giao dịch'), row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=df['Thời Gian'], y=df['Đóng cửa'], name='Giá đóng cửa', line=dict(color='blue')), row=1, col=1)
    df_plot = df.copy()
    df_plot['Volume_Change'] = df_plot['Khối lượng'].diff()
    colors = []
    for i in range(len(df_plot)):
        if i == 0 or pd.isna(df_plot['Volume_Change'].iloc[i]):
            colors.append('grey')
        elif df_plot['Volume_Change'].iloc[i] > 0:
            colors.append('green')
        elif df_plot['Volume_Change'].iloc[i] < 0:
            colors.append('red')
        else:
            colors.append('grey')
    fig.add_trace(go.Bar(x=df_plot['Thời Gian'], y=df_plot['Khối lượng'], name='Khối lượng', marker_color=colors), row=2, col=1)
    fig.update_layout(title_text=f"Phân tích giá thị trường cổ phiếu: {ticker}", xaxis_rangeslider_visible=False, height=600, legend_title_text='Chú giải')
    fig.update_xaxes(title_text="Thời Gian", row=2, col=1)
    fig.update_yaxes(title_text="Giá (VNĐ)", row=1, col=1)
    fig.update_yaxes(title_text="Khối lượng", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

def plot_recommendations(df_signals, ticker):
    """
    Vẽ đồ thị giá với các tín hiệu mua/bán.
    """
    if df_signals.empty or 'Thời Gian' not in df_signals.columns or 'Đóng cửa' not in df_signals.columns:
        st.warning("Thiếu dữ liệu để vẽ đồ thị khuyến nghị.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_signals['Thời Gian'], y=df_signals['Đóng cửa'], name='Giá Đóng Cửa', line=dict(color='blue')))
    if 'MA20' in df_signals.columns:
        fig.add_trace(go.Scatter(x=df_signals['Thời Gian'], y=df_signals['MA20'], name='MA20', line=dict(color='orange', dash='dash')))
    if 'MA50' in df_signals.columns:
        fig.add_trace(go.Scatter(x=df_signals['Thời Gian'], y=df_signals['MA50'], name='MA50', line=dict(color='purple', dash='dash')))
    if 'Buy_Signal_Price' in df_signals.columns:
        fig.add_trace(go.Scatter(x=df_signals['Thời Gian'], y=df_signals['Buy_Signal_Price'], name='Tín hiệu Mua', mode='markers', marker=dict(color='green', size=10, symbol='triangle-up')))
    if 'Sell_Signal_Price' in df_signals.columns:
        fig.add_trace(go.Scatter(x=df_signals['Thời Gian'], y=df_signals['Sell_Signal_Price'], name='Tín hiệu Bán', mode='markers', marker=dict(color='red', size=10, symbol='triangle-down')))
    fig.update_layout(title_text=f"Khuyến nghị Mua/Bán cho {ticker} (Dựa trên chỉ báo cơ bản)", xaxis_title="Thời Gian", yaxis_title="Giá (VNĐ)", legend_title="Chú giải", height=600)
    st.plotly_chart(fig, use_container_width=True)

    fig_indicators = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('RSI (14)', 'MACD'))
    if 'RSI' in df_signals.columns:
        fig_indicators.add_trace(go.Scatter(x=df_signals['Thời Gian'], y=df_signals['RSI'], name='RSI'), row=1, col=1)
        fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1, annotation_text="Quá mua (70)", annotation_position="bottom right")
        fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1, annotation_text="Quá bán (30)", annotation_position="bottom right")
    if 'MACD_line' in df_signals.columns and 'MACD_signal' in df_signals.columns and 'MACD_hist' in df_signals.columns:
        fig_indicators.add_trace(go.Scatter(x=df_signals['Thời Gian'], y=df_signals['MACD_line'], name='MACD Line', line=dict(color='blue')), row=2, col=1)
        fig_indicators.add_trace(go.Scatter(x=df_signals['Thời Gian'], y=df_signals['MACD_signal'], name='Signal Line', line=dict(color='orange')), row=2, col=1)
        macd_hist_colors = np.where(df_signals['MACD_hist'] > 0, 'green', 'red')
        fig_indicators.add_trace(go.Bar(x=df_signals['Thời Gian'], y=df_signals['MACD_hist'], name='MACD Histogram', marker_color=macd_hist_colors), row=2, col=1)
    fig_indicators.update_layout(height=400, legend_title="Chỉ báo")
    fig_indicators.update_xaxes(title_text="Thời Gian", row=2, col=1)
    st.plotly_chart(fig_indicators, use_container_width=True)

def create_sequences(data, n_steps_in, n_steps_out):
    """
    Helper function to create input/output sequences for LSTM.
    """
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(data):
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def predict_price_lstm(df_full, column_to_predict='Đóng cửa', n_steps_in=60, n_steps_out=1, epochs=50, batch_size=32, days_to_predict_future=7):
    """
    Dự đoán giá cổ phiếu sử dụng mô hình LSTM.
    """
    if df_full.empty or column_to_predict not in df_full.columns or len(df_full) < n_steps_in + n_steps_out:
        st.warning(f"Không đủ dữ liệu '{column_to_predict}' (cần ít nhất {n_steps_in + n_steps_out} ngày) để huấn luyện mô hình LSTM.")
        return pd.DataFrame()

    # 0. Set random seed for reproducibility (optional, but good for consistent results during development)
    tf.random.set_seed(42)
    np.random.seed(42)

    # 1. Select and Prepare Data
    data_to_predict = df_full[column_to_predict].values.reshape(-1, 1)

    # 2. Scale Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_predict)

    # 3. Create Sequences
    X, y = create_sequences(scaled_data, n_steps_in, n_steps_out)

    if X.shape[0] == 0:
        st.warning(f"Không thể tạo đủ sequences với n_steps_in={n_steps_in}. Cần thêm dữ liệu lịch sử.")
        return pd.DataFrame()

    # 4. Reshape X for LSTM: [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1)) # 1 feature (giá đóng cửa)

    # 5. Build LSTM Model
    model = Sequential()
    model.add(LSTM(units=100, activation='relu', return_sequences=True, input_shape=(n_steps_in, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=n_steps_out)) # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 6. Train Model
    st.write(f"Bắt đầu huấn luyện mô hình LSTM với {epochs} epochs...")
    progress_bar = st.progress(0)
    # For simplicity, no explicit train/test split here, training on all available sequences
    # In a real scenario, a validation split or cross-validation would be important.
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0,
                        callbacks=[TrainingProgressCallback(progress_bar, epochs)])
    st.write("Huấn luyện mô hình LSTM hoàn tất.")

    # 7. Make Future Predictions
    last_sequence = scaled_data[-n_steps_in:] # Lấy sequence cuối cùng từ dữ liệu đã scale
    current_batch = last_sequence.reshape((1, n_steps_in, 1))
    future_predictions_scaled = []

    for _ in range(days_to_predict_future):
        current_pred = model.predict(current_batch, verbose=0)[0] # Dự đoán 1 bước tiếp theo
        future_predictions_scaled.append(current_pred)
        # Cập nhật current_batch: bỏ giá trị cũ nhất, thêm giá trị dự đoán mới nhất
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # 8. Inverse Transform Predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1,1))

    # 9. Create Prediction DataFrame
    last_date = df_full['Thời Gian'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict_future + 1)]

    predictions_df = pd.DataFrame({
        'Thời Gian Dự Đoán': future_dates,
        'Giá Dự Đoán (LSTM)': future_predictions.flatten() # flatten để chuyển thành 1D array
    })

    return predictions_df

# Custom Callback for Streamlit Progress Bar
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress((epoch + 1) / self.total_epochs)


def call_ollama_api(prompt_text):
    """
    Gửi yêu cầu đến Ollama API và nhận phản hồi.
    """
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(OLLAMA_API_ENDPOINT, json=payload, headers=headers, timeout=180)
        response.raise_for_status()
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
    if df_with_indicators.empty or 'Đóng cửa' not in df_with_indicators.columns:
        return f"Không có đủ dữ liệu cho cổ phiếu {ticker} để phân tích."
    latest_data = df_with_indicators.iloc[-1]
    prompt = f"""
    Bạn là một chuyên gia phân tích kỹ thuật thị trường chứng khoán Việt Nam.
    Hãy phân tích chuyên sâu về cổ phiếu {ticker} dựa trên các dữ liệu và chỉ báo kỹ thuật gần nhất được cung cấp dưới đây.

    Dữ liệu ngày {latest_data.get('Thời Gian', pd.Timestamp('now')).strftime('%Y-%m-%d')}:
    - Giá đóng cửa: {latest_data.get('Đóng cửa', 'N/A'):,.0f} VNĐ
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

# News Fetching Function (if needed)
def generate_ollama_summary_prompt(news_title, news_content):
    """
    Tạo prompt cho Ollama để tóm tắt tin tức.
    """
    return f"""
    Bạn là một trợ lý AI có khả năng tóm tắt tin tức một cách chính xác và ngắn gọn.
    Hãy tóm tắt những điểm chính của bài báo sau đây bằng tiếng Việt. Tập trung vào thông tin quan trọng nhất liên quan đến cổ phiếu hoặc thị trường.
    Độ dài tóm tắt khoảng 3-5 câu.

    Tiêu đề: {news_title}

    Nội dung:
    {news_content}

    Tóm tắt:
    """

def fetch_simulated_news(ticker):
    """
    Mô phỏng việc lấy tin tức. Thay thế bằng logic lấy tin thật.
    Trong môi trường thực tế có thể dùng tool_code như <tool_code>g.search_news(query=f"tin tức cổ phiếu {ticker} Việt Nam", limit=5)</tool_code>
    sau đó xử lý kết quả trả về.
    """
    st.caption(f"Đang mô phỏng việc lấy tin tức cho {ticker}... Trong thực tế, bạn sẽ dùng công cụ tìm kiếm hoặc API tin tức.")
    current_date = datetime.now()
    news_items = [
        {
            "title": f"Triển vọng tích cực cho {ticker} trong quý tới sau báo cáo lợi nhuận",
            "link": f"https://cafef.vn/{ticker.lower()}-trien-vong-tich-cuc-quy-toi-{current_date.strftime('%Y%m%d')}.chn",
            "snippet": f"Công ty Cổ phần {ticker} vừa công bố báo cáo tài chính quý vừa qua với những con số ấn tượng, vượt kỳ vọng của giới phân tích. Doanh thu tăng trưởng 25% so với cùng kỳ, lợi nhuận sau thuế đạt mức cao kỷ lục...",
            "published_date": (current_date - timedelta(days=1)).strftime('%d/%m/%Y'),
            "source": "CafeF"
        },
        {
            "title": f"{ticker} dự kiến mở rộng nhà máy, nâng công suất thêm 30%",
            "link": f"https://vietstock.vn/{ticker.lower()}-du-kien-mo-rong-nha-may-{current_date.strftime('%Y%m%d')}.htm",
            "snippet": f"Theo thông tin từ Đại hội đồng cổ đông thường niên, {ticker} đã thông qua kế hoạch đầu tư mở rộng nhà máy hiện tại và xây dựng thêm một phân xưởng mới. Dự kiến sau khi hoàn thành, tổng công suất sẽ tăng thêm 30%, đáp ứng nhu cầu thị trường đang tăng cao.",
            "published_date": (current_date - timedelta(days=3)).strftime('%d/%m/%Y'),
            "source": "Vietstock"
        },
        {
            "title": f"Phân tích kỹ thuật cổ phiếu {ticker}: Tín hiệu nào cho nhà đầu tư?",
            "link": f"https://vneconomy.vn/phan-tich-ky-thuat-{ticker.lower()}-tin-hieu-nao-{current_date.strftime('%Y%m%d')}.htm",
            "snippet": f"Sau một giai đoạn tích lũy, cổ phiếu {ticker} đang cho thấy những dấu hiệu bứt phá khỏi vùng kháng cự quan trọng. Các chỉ báo như RSI, MACD đều ủng hộ xu hướng tăng giá ngắn hạn. Tuy nhiên, nhà đầu tư cần chú ý đến ngưỡng cản tiếp theo...",
            "published_date": (current_date - timedelta(days=5)).strftime('%d/%m/%Y'),
            "source": "VnEconomy"
        },
         {
            "title": f"Cảnh báo rủi ro từ biến động tỷ giá có thể ảnh hưởng đến {ticker}",
            "link": f"https://baodautu.vn/canh-bao-rui-ro-ty-gia-{ticker.lower()}-{current_date.strftime('%Y%m%d')}.html",
            "snippet": f"Một số chuyên gia kinh tế nhận định rằng những biến động gần đây trên thị trường ngoại hối có thể tạo ra áp lực không nhỏ lên các doanh nghiệp có hoạt động xuất nhập khẩu lớn như {ticker}. Chi phí đầu vào có thể tăng nếu không có biện pháp phòng ngừa rủi ro tỷ giá hiệu quả.",
            "published_date": (current_date - timedelta(days=2)).strftime('%d/%m/%Y'),
            "source": "Báo Đầu Tư"
        },
        {
            "title": f"Bài viết về {ticker} từ Fireant.vn",
            "link": f"https://fireant.vn/bai-viet/{ticker.lower()}-*{current_date.strftime('%Y%m%d')}.html",
            "snippet": f"Khối ngoại quay lại mua ròng cổ phiếu {ticker}.",
            "published_date": (current_date - timedelta(days=2)).strftime('%d/%m/%Y'),
            "source": "Fireant"
        }
    ]
    
    return news_items
def to_excel(df):
    """Xuất DataFrame ra file Excel dạng bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Đổi tên cột nếu là dự đoán LSTM
        sheet_name = 'DuDoanGia_LSTM' if 'Giá Dự Đoán (LSTM)' in df.columns else 'DuDoanGia'
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    processed_data = output.getvalue()
    return processed_data

# --- Giao diện Streamlit ---
st.set_page_config(layout="wide", page_title="Phân Tích Cổ Phiếu", page_icon="📈")

st.title("📈 Công Cụ Phân Tích Cổ Phiếu")
st.markdown("Chào mừng bạn đến với công cụ phân tích cổ phiếu, sử dụng dữ liệu từ `vnstock`, dự đoán giá bằng LSTM và phân tích AI từ `Ollama`.")

# --- Sidebar ---
st.sidebar.header("⚙️ Tùy chọn Phân Tích")
stock_ticker = st.sidebar.text_input("Nhập mã cổ phiếu (VD: FPT, HPG):", "FPT").upper()
data_source = st.sidebar.selectbox("Chọn nguồn dữ liệu:", ["TCBS", "VCI", "SSI", "VND", "DNSE", "VPS"], index=0)
days_to_fetch_sb = st.sidebar.slider("Số ngày dữ liệu lịch sử để tải (cho LSTM và chỉ báo):", 60, 2000, 730, step=10)
days_for_prediction = st.sidebar.slider("Số ngày dự đoán giá tiếp theo (LSTM):", 1, 30, 7) # Giảm max cho LSTM
lstm_epochs = st.sidebar.slider("Số epochs huấn luyện LSTM:", 10, 100, 50, step=10)
lstm_n_steps_in = st.sidebar.slider("Số ngày lịch sử làm đầu vào cho LSTM (lookback window):", 30, 120, 60, step=5)


run_analysis = st.sidebar.button("🚀 Chạy Phân Tích", type="primary", use_container_width=True)

# --- Xử lý chính ---
if run_analysis and stock_ticker:
    with st.spinner(f"Đang tải và phân tích dữ liệu cho {stock_ticker}..."):
        raw_df = fetch_stock_data(stock_ticker, source=data_source, days_to_fetch=days_to_fetch_sb)

        if raw_df.empty or 'Đóng cửa' not in raw_df.columns:
            st.error(f"Không tìm thấy dữ liệu hoặc thiếu cột 'Đóng cửa' cho mã {stock_ticker} từ nguồn {data_source}.")
        else:
            st.success(f"Đã tải thành công {len(raw_df)} bản ghi cho {stock_ticker} (từ {raw_df['Thời Gian'].min().strftime('%Y-%m-%d')} đến {raw_df['Thời Gian'].max().strftime('%Y-%m-%d')}).")
            df_processed = calculate_technical_indicators(raw_df)
            df_with_signals = generate_signals(df_processed)

            # Lấy tin tức (mô phỏng)
            news_data = fetch_simulated_news(stock_ticker)

            tab1, tab2, tab3, tab4,tab_news = st.tabs([
                "📊 Giá Thị Trường",
                "💡 Khuyến Nghị (Chỉ báo)",
                "🔮 Dự Đoán Giá (LSTM)",
                "🤖 AI Phân Tích (Ollama)",
                "📰 Tin Tức & Tóm Tắt AI"
            ])

            with tab1:
                st.header(f"Giá Thị Trường và Khối Lượng: {stock_ticker}")
                st.markdown("""...""", unsafe_allow_html=True) # Giữ nguyên mô tả cũ
                if not df_processed.empty:
                    plot_market_price(df_processed, stock_ticker)
                else:
                    st.warning("Không có dữ liệu để vẽ đồ thị giá thị trường.")

            with tab2:
                st.header(f"Khuyến Nghị Dựa Trên Chỉ Báo Kỹ Thuật: {stock_ticker}")
                st.markdown("""...""") # Giữ nguyên mô tả cũ
                if not df_with_signals.empty:
                    plot_recommendations(df_with_signals, stock_ticker)
                    st.subheader("Dữ liệu chỉ báo và tín hiệu 10 ngày gần nhất:")
                    st.dataframe(df_with_signals[['Thời Gian', 'Đóng cửa', 'MA5', 'MA20', 'RSI', 'MACD_line', 'MACD_signal', 'Signal']].sort_values(by='Thời Gian', ascending=False).head(10).set_index('Thời Gian'))
                else:
                    st.warning("Không có dữ liệu để tạo khuyến nghị.")

            with tab3:
                st.header(f"Dự Đoán Giá Cổ Phiếu (Sử dụng LSTM): {stock_ticker}")
                st.markdown(f"""
                Phần này sử dụng mô hình **LSTM (Long Short-Term Memory)**, một loại mạng nơ-ron hồi quy (RNN) trong Deep Learning,
                để dự đoán giá đóng cửa cho **{days_for_prediction}** ngày giao dịch tiếp theo.
                - **Cửa sổ nhìn lại (Lookback window):** Mô hình sử dụng dữ liệu của **{lstm_n_steps_in}** ngày trước đó để dự đoán.
                - **Số epochs huấn luyện:** {lstm_epochs}.
                
                **CẢNH BÁO:** Dự đoán bằng LSTM phức tạp hơn và có thể cho kết quả tốt hơn hồi quy tuyến tính trong một số trường hợp,
                nhưng vẫn **KHÔNG NÊN** được coi là dự báo tài chính tuyệt đối chính xác. Kết quả phụ thuộc nhiều vào dữ liệu,
                cấu hình mô hình và quá trình huấn luyện. Quá trình huấn luyện có thể mất một chút thời gian.
                """)

                if not df_processed.empty and len(df_processed) >= lstm_n_steps_in + 1: # Cần đủ dữ liệu cho lookback window + ít nhất 1 điểm để dự đoán
                    with st.spinner(f"Đang chuẩn bị dữ liệu và huấn luyện mô hình LSTM cho {stock_ticker}... (Có thể mất vài phút)"):
                        predictions_df = predict_price_lstm(
                            df_full=df_processed,
                            column_to_predict='Đóng cửa',
                            n_steps_in=lstm_n_steps_in,
                            epochs=lstm_epochs,
                            days_to_predict_future=days_for_prediction
                        )

                    if not predictions_df.empty:
                        st.subheader(f"Kết quả dự đoán LSTM cho {days_for_prediction} ngày tới:")
                        # Đổi tên cột để phù hợp với logic hiển thị và xuất excel
                        predictions_df.rename(columns={'Giá Dự Đoán (LSTM)': 'Giá Dự Đoán'}, inplace=True)
                        st.dataframe(predictions_df.style.format({"Giá Dự Đoán": "{:,.0f} VNĐ"}))

                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=df_processed['Thời Gian'], y=df_processed['Đóng cửa'], name='Giá Lịch Sử', line=dict(color='royalblue')))
                        fig_pred.add_trace(go.Scatter(x=predictions_df['Thời Gian Dự Đoán'], y=predictions_df['Giá Dự Đoán'], name='Giá Dự Đoán (LSTM)', line=dict(color='tomato', dash='dash')))
                        fig_pred.update_layout(title=f'Dự đoán giá {stock_ticker} (LSTM)', xaxis_title='Thời Gian', yaxis_title='Giá (VNĐ)')
                        st.plotly_chart(fig_pred, use_container_width=True)

                        excel_data = to_excel(predictions_df)
                        st.download_button(
                            label="📥 Tải xuống kết quả dự đoán LSTM (Excel)",
                            data=excel_data,
                            file_name=f"du_doan_gia_lstm_{stock_ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    else:
                        st.warning("Không thể tạo dự đoán LSTM với dữ liệu hoặc cấu hình hiện tại.")
                else:
                    st.warning(f"Không đủ dữ liệu lịch sử để thực hiện dự đoán LSTM (cần ít nhất {lstm_n_steps_in + 1} ngày dữ liệu có cột 'Đóng cửa').")

            with tab4:
                st.header(f"Phân Tích Chuyên Sâu Bởi AI (Sử dụng Ollama - {OLLAMA_MODEL}): {stock_ticker}")
                st.markdown(f"""...""") # Giữ nguyên mô tả cũ
                if not df_with_signals.empty:
                    with st.spinner(f"🤖 AI ({OLLAMA_MODEL}) đang phân tích cổ phiếu {stock_ticker}..."):
                        prompt = generate_ollama_prompt(stock_ticker, df_with_signals)
                        ai_analysis = ask_ollama(prompt)
                        if ai_analysis:
                            st.subheader("Kết quả phân tích từ AI:")
                            st.markdown(ai_analysis)
                        else:
                            st.error(f"Không thể nhận phản hồi từ Ollama.")
                else:
                    st.warning("Không có đủ dữ liệu chỉ báo để AI phân tích.")
            with tab_news:
                st.header(f"Tin Tức Liên Quan Đến {stock_ticker} & Tóm Tắt AI")
                if news_data:
                    # Giới hạn số lượng tin tức để tránh quá nhiều cuộc gọi API tới Ollama cùng lúc
                    max_news_to_summarize = st.slider("Số lượng tin tức hiển thị và tóm tắt tối đa:", 1, len(news_data), min(5, len(news_data)))
                    
                    for i, news_item in enumerate(news_data[:max_news_to_summarize]):
                        st.subheader(f"📰 [{news_item['title']}]({news_item['link']})")
                        st.caption(f"Nguồn: {news_item['source']} - Ngày đăng: {news_item['published_date']}")
                        
                        # Sử dụng st.expander để người dùng có thể xem nội dung đầy đủ nếu muốn
                        with st.expander("Xem nội dung gốc (đoạn trích)"):
                            st.markdown(news_item['snippet'][:1000] + "..." if len(news_item['snippet']) > 1000 else news_item['snippet']) # Giới hạn độ dài hiển thị

                        # Nút để yêu cầu tóm tắt (tránh gọi API hàng loạt ngay lúc đầu)
                        # Hoặc có thể tóm tắt ngay nếu số lượng tin ít
                        summary_placeholder = st.empty() # Placeholder để hiển thị tóm tắt
                        
                        # Tóm tắt ngay
                        with st.spinner(f"AI ({OLLAMA_MODEL}) đang tóm tắt tin '{news_item['title'][:30]}...'"):
                            summary_prompt = generate_ollama_summary_prompt(news_item['title'], news_item['snippet'])
                            # st.text_area("Prompt tóm tắt:", summary_prompt, height=100, key=f"prompt_sum_{i}") # Debug
                            summary = ask_ollama(summary_prompt) # Có thể dùng model khác nhẹ hơn nếu cần
                            if summary:
                                summary_placeholder.markdown(f"**📝 Tóm tắt AI:**\n{summary}")
                            else:
                                summary_placeholder.warning("Không thể tạo tóm tắt cho tin này.")
                        st.divider()
                else:
                    st.info(f"Không tìm thấy tin tức nào cho {stock_ticker} (mô phỏng).")
elif run_analysis and not stock_ticker:
    st.warning("⚠️ Vui lòng nhập mã cổ phiếu vào ô bên trái và nhấn 'Chạy Phân Tích'.")

st.sidebar.markdown("---")
st.sidebar.info(f"""
    **Thông tin ứng dụng:**
    - **Phiên bản:** 1.2 (14/05/2025) - LSTM Prediction
    - **Nguồn dữ liệu:** `vnstock`
    - **Phân tích AI:** `Ollama` (Model: {OLLAMA_MODEL})
    - **Lưu ý:** Thông tin chỉ mang tính tham khảo.
""")
st.sidebar.markdown(f"*{datetime.now().strftime('%A, %d/%m/%Y, %H:%M:%S')}*")
