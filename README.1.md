# 📊 Customer Value Segmentation — RFM + Random Forest

Ứng dụng Streamlit phân loại khách hàng thành 3 nhóm giá trị **Low / Medium / High Value** dựa trên phân tích RFM (Recency, Frequency, Monetary) và mô hình Random Forest, sử dụng bộ dữ liệu **Online Retail** (UCI Machine Learning Repository).

## 🚀 Demo

Quy trình gồm 5 bước, mỗi bước là một tab trong ứng dụng:

1. **Khám phá dữ liệu (EDA)** — preview, kích thước, giá trị thiếu, dữ liệu trùng lặp, thống kê mô tả, phân phối Quantity/UnitPrice
2. **Tiền xử lý** — loại trùng lặp, tách đơn hủy, loại giá trị không hợp lệ, xử lý missing values
3. **RFM & Phân khúc** — tính Recency/Frequency/Monetary cho từng khách hàng, gán nhãn phân khúc giá trị
4. **Huấn luyện & Đánh giá** — huấn luyện Random Forest, xem accuracy/F1/confusion matrix/feature importance
5. **Dự đoán thử** — nhập thông tin một khách hàng mới và dự đoán phân khúc

## 📊 Khám phá dữ liệu (EDA)

Một số biểu đồ từ bước khám phá dữ liệu trên bộ Online Retail gốc (541,909 dòng) — các biểu đồ này cũng được tạo động ngay trong tab **"Khám phá dữ liệu"** của ứng dụng Streamlit.

**Giá trị thiếu:** `Description` thiếu 0.27%, `CustomerID` thiếu ~25% — đây là lý do bước tiền xử lý phải tách riêng tập dữ liệu có/không có CustomerID.

<img src="images/missing_values.png" width="500"/>

**Phân bố giao dịch theo quốc gia:** phần lớn giao dịch đến từ Anh (United Kingdom), phần còn lại rải rác ở châu Âu.

<img src="images/top_countries.png" width="600"/>

**Phân phối Quantity & UnitPrice:** đa số giao dịch có số lượng và đơn giá nhỏ, lệch phải mạnh (right-skewed).

<p float="left">
  <img src="images/quantity_distribution.png" width="420"/>
  <img src="images/unitprice_distribution.png" width="420"/>
</p>

**Boxplot (thang gốc):** cho thấy rõ các outlier cực đoan — Quantity dao động từ khoảng -80,000 đến +80,000, UnitPrice có giá trị âm và một điểm gần 40,000 — chính là các đơn hủy, phí điều chỉnh và lỗi nhập liệu cần được xử lý ở bước tiền xử lý.

<img src="images/boxplots.png" width="700"/>

## 🗂️ Cấu trúc dự án

```
.
├── app.py                          # Ứng dụng Streamlit chính
├── requirements.txt                # Thư viện cần thiết
├── images/                         # Biểu đồ EDA dùng trong README
├── scripts/
│   ├── preprocess_online_retail.py # Script tiền xử lý độc lập (chạy ngoài Streamlit)
│   └── train_rf_customer_value.py  # Script huấn luyện/đánh giá độc lập (chạy ngoài Streamlit)
├── data/
│   └── README.md                   # Hướng dẫn tải bộ dữ liệu
└── .gitignore
```

## 🔧 Cài đặt & chạy

```bash
# 1. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Chạy ứng dụng
streamlit run app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`. Tải lên file `online_retail.csv` ở thanh bên trái (xem `data/README.md` để biết cách lấy dữ liệu).

## 📈 Phương pháp

- **RFM**: mỗi khách hàng được tính điểm Recency/Frequency/Monetary theo tứ phân vị (1-4), cộng lại thành `RFM_Score` (3-12).
- **Nhãn phân khúc**: `RFM_Score` được chia theo ngưỡng cố định — Low Value (3-6), Medium Value (7-9), High Value (10-12).
- **Mô hình**: `RandomForestClassifier` (scikit-learn) với các đặc trưng Recency, Frequency, Monetary, Tenure, NumProducts, AvgOrderValue, Country (one-hot top 10 quốc gia).
- **Đánh giá**: train/test split có stratify, 5-fold cross-validation, accuracy, F1-macro, confusion matrix, feature importance.

## 📚 Nguồn dữ liệu

Bộ dữ liệu **Online Retail** — giao dịch của một công ty bán lẻ trực tuyến tại Anh từ 01/12/2010 đến 09/12/2011.

- Nguồn: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
- Tác giả: Chen, D., Sain, S.L., Guo, K. (2015)
- Giấy phép: CC BY 4.0

## 📄 License

MIT License — xem file `LICENSE`.
