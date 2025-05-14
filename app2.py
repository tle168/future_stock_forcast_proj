import streamlit as st
import pandas as pd
from model_lstm import prepare_lstm_data, train_lstm_model, forecast_lstm
from ta_signals import compute_indicators, generate_signals, print_signals, predict_stock_price
from charts import *
from ollama_helper import ask_ollama
from data_fetcher import fetch_stock_data, load_data
from datetime import datetime, timedelta, time
from vnstock import * # Nạp thư viện để sử dụng
from email_alert import send_email
import plotly.tools as tls
import matplotlib.pyplot as plt
from config import *
# ==========================
# Thiết lập cấu hình ngày tháng
# ==========================

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

st.set_page_config(layout="wide")
st.title("📈 AI Stock Forecast Dashboard")

# Sidebar
symbol = st.sidebar.text_input("🔍 Mã cổ phiếu", value="FPT")
window = st.sidebar.slider("🕒 Số ngày quá khứ (window)", 10, 60, 20)
forecast_days = st.sidebar.slider("Dự báo số ngày tới", 5, 30, 10)
st.sidebar.markdown("## Tải dữ liệu")
# Load data from CSV as default
data = pd.DataFrame()


if st.sidebar.button("Fetch Data & Analyze"):
    stock = Vnstock().stock(symbol=symbol, source='VCI') # Định nghĩa biến vnstock lưu thông tin mã chứng khoán & nguồn dữ liệu bạn sử dụng
    data = stock.quote.history(start=start_date, end=end_date, interval='1D') # Thiết lập thời gian tải dữ liệu và khung thời gian tra cứu là 1 ngày
    data.columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # data = pd.read_csv(f"data/{symbol}.csv")
    # #data = load_data(symbol, start_date, end_date)
    # data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    st.sidebar.info("Data loaded successfully!")

if len(data) > 0:
    try:
        close_series = data['Close']
        X, y, scaler = prepare_lstm_data(close_series, window)
        model = train_lstm_model(X, y)
        forecast = forecast_lstm(model, close_series, window, forecast_days, scaler)

        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({"Forecast": forecast.flatten()}, index=forecast_dates)

        indicators = compute_indicators(data)
        signals = generate_signals(indicators)
        predict_price=predict_stock_price(indicators)
        print_data = print_signals(data)
        

        tab1, tab2, tab3 = st.tabs(["📊 Biểu đồ", "📈 Dữ liệu", "🤖 AI Phân tích"])
        with tab1:
            st.subheader("Giá thực tế & dự báo")
            st.plotly_chart(plot_price_forecast(data, forecast_df), use_container_width=True)

            st.subheader(f"Biểu đồ khối lượng giao dịch của {symbol}")
            # st.plotly_chart(plot_heatmap(data), use_container_width=True)
            st.bar_chart(data['Volume'], use_container_width=True)
        with tab2:
            st.subheader("Giá dự báo")
            st.subheader(f"Dự đoán giá ngày mai: {predict_price:.2f} VND")

            st.write(f"Giá dự báo cho {symbol} trong {forecast_days} ngày tới:")
            
            st.table(forecast_df["Forecast"].reset_index().rename(columns={"index": "Ngày", "Forecast": "Giá dự báo"}))
            # @st.dialog("📧 Gửi dự báo qua email")
            # def send_email_dialog():
            st.write("📧 Gửi dự báo qua email")
            st.write("Nhập địa chỉ email để nhận dự báo:")
            email = st.text_input("Email: ", value="thach.le168@gmail.com")
            # email = st.text_input("Email: ", value="thach.le168@gmail.com")
            if st.button("Gửi email"):
                send_email(forecast_df, email)
                st.success("Email đã được gửi thành công!")
            # st.info("Email đã được gửi thành công!")
            
            
        with tab3:
            st.subheader("Phân tích kỹ thuật")
            for sig in signals:
                st.info(f"🔸 {sig}")
                st.write(print_data.tail(10))

            prompt = f"Dựa trên giá đóng cửa gần đây của cổ phiếu {symbol}: {close_series.tail(20).tolist()}, các chỉ báo kỹ thuật là: {signals}. Bạn có thể đưa ra phân tích kỹ thuật chuyên sâu và gợi ý cẩn trọng không?"
            if st.button("🧠 Phân tích bằng Ollama"):
                result = ask_ollama(prompt)
                with st.spinner("Asking Ollama ({OLLAMA_MODEL}) for analysis..."):
                    time.sleep(5)
                st.subheader("Khuyến nghị từ AI")
                st.markdown(result)
    except FileNotFoundError:
        st.warning("Stock data not found. Please load data again!!!")
        

    

    
   
