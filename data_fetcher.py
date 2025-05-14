# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from config import DEFAULT_START_DATE_YEARS_AGO
from vnstock import * # Nạp thư viện để sử dụng

@st.cache_data(ttl=3600) # Cache 1 giờ
def fetch_stock_data(ticker: str, start_date: datetime = None, end_date: datetime = datetime.now()):
    """Fetches historical stock data from TCBS."""
    if start_date is None:
        start_date = end_date - timedelta(days=365 * DEFAULT_START_DATE_YEARS_AGO)

    try:
        st.write(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}...")
        #data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        stock = Vnstock().stock(symbol=ticker, source='TCBS') # Định nghĩa biến vnstock lưu thông tin mã chứng khoán & nguồn dữ liệu bạn sử dụng
        data = stock.quote.history(start=start_date, end=end_date, interval='1D') # Thiết lập thời gian tải dữ liệu và khung thời gian tra cứu là 1 ngày
        
        if data.empty:
            st.error(f"No data found for {ticker}. Check symbol or date range.")
            return None
        # Đảm bảo cột chuẩn hóa (thường yfinance đã làm)
        #data.columns = [col.capitalize() for col in data.columns]
        data.columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        st.write("Data fetched successfully.")
        return data["Close"].values.reshape(-1, 1), data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Thêm hàm fetch_fundamental_data nếu cần (dùng Alpha Vantage,...)
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
end_date=YESTERDAY
start_date = LAST_1Y
def load_data(ticker, start_date, end_date):
    stock = Vnstock().stock(symbol=ticker, source='VCI') # Định nghĩa biến vnstock lưu thông tin mã chứng khoán & nguồn dữ liệu bạn sử dụng
    data = stock.quote.history(start=start_date, end=end_date, interval='1D') # Thiết lập thời gian tải dữ liệu và khung thời gian tra cứu là 1 ngày
    data.columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    data.set_index('Date', inplace=True)
    return data["Close"].values.reshape(-1, 1), data