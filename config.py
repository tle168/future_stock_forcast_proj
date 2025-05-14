import os

# --- Ollama Configuration ---
# Lấy từ biến môi trường khi chạy trong Docker, hoặc đặt giá trị mặc định
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "Gemma3" # Hoặc mô hình bạn muốn dùng
OLLAMA_API_ENDPOINT_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_ENDPOINT_CHAT = f"{OLLAMA_BASE_URL}/api/chat" # Dùng endpoint này nếu model hỗ trợ chat

# --- Data Configuration ---
DEFAULT_TICKER = "AAPL" # Mã chứng khoán mặc định
DEFAULT_START_DATE_YEARS_AGO = 3
PREDICTION_DAYS = 30 # Số ngày muốn dự đoán

# --- Technical Indicators ---
INDICATOR_LIST = [
    {"kind": "sma", "length": 20},
    {"kind": "sma", "length": 50},
    {"kind": "sma", "length": 200},
    {"kind": "ema", "length": 20},
    {"kind": "rsi", "length": 14},
    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
    {"kind": "bbands", "length": 20, "std": 2},
    {"kind": "obv"}, # On Balance Volume
    {"kind": "atr", "length": 14}, # Average True Range
    # Thêm các chỉ báo khác nếu muốn
]

# --- API Keys (nếu cần cho dữ liệu cơ bản) ---
# ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", None)