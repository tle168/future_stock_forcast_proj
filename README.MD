# 📁 Cấu trúc thư mục dự án

future_stock_forecast/
├── app.py
├── model_lstm.py
├── charts.py
├── ta_signals.py
├── ollama_helper.py
├── requirements.txt
├── README.md
└── data/
    └── FPT.csv  # Dữ liệu mẫu cho mã cổ phiếu FPT

# Dự Báo Giá Cổ Phiếu Bằng LSTM

## Tính năng chính
- Dự báo giá cổ phiếu tương lai bằng LSTM
- Tính chỉ số MA5, MA20, RSI, MACD
- Vẽ biểu đồ kỹ thuật (line chart, forecast, heatmap)
- Giao diện web đơn giản qua Streamlit
- Phân tích AI bằng Ollama (Llama3)

## Hướng dẫn sử dụng
1. Cài các thư viện từ `requirements.txt`
2. Khởi chạy Ollama bằng Docker:
   ```bash
   docker run -d -p 11434:11434 ollama/ollama
   ollama run llama3

# Chạy app:
streamlit run app.py

# Cấu trúc dự án
app.py: giao diện chính

model_lstm.py: mô hình dự báo

charts.py, ta_signals.py: biểu đồ & chỉ báo

ollama_helper.py: gọi AI phân tích




Viết chương trình python chạy web streamlit để phân tích cổ phiếu tại thị trường việt nam, các tính năng chính như sau:
1. sidebar có các ô nhập mã cổ phiếu,
2. chương trình sẽ lấy giá cổ phiếu qua thư viện vnstock từ VCI hoặc TCBS, 
Nếu dữ liệu >0, thì tạo các streamlit tab
Tab1 - name: Giá thị trường và vẽ đồ thì bao gồm giá đóng cửa, volume trong tab giá thị trường, nếu volume giảm thì đổi màu thành đỏ, tăng là màu xanh
4. khi có đầy đủ dữ liệu bắt đầu tính toán: MA5, MA20, MA50, MA200, RSI, MACD để đưa ra khuyên nghị mua vào, bán ra (vẽ đồ thị có tín hiệu mua vào bán ra) vẽ đồ thị trong streamlit tab "Khuyến nghị"
5. sử dụng máy học LLM để dự đoán giá cổ phiếu trong số lượng ngày lấy từ slidebar trong sidebar
6. dựa vào kết quar tính toàn tại bước 4, sử dụng Ollama AI để đưa ra khuyến nghị và phân tích cổ phiếu chuyên nghiêp. Kết quả hiển thị tại Tab AI Phân tích
7. Ollama chạy trên Docker tại máy localhost
8. bổ sung thêm tính năng xuất ra excel kết quả dự đoán.





# Git updates
## …or create a new repository on the command line

echo "# future_stock_forcast_proj" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/tle168/future_stock_forcast_proj.git
git push -u origin main

## …or push an existing repository from the command line
git remote add origin https://github.com/tle168/future_stock_forcast_proj.git
git branch -M main
git push -u origin main
