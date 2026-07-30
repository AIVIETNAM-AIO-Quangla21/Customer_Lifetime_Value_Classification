# Dữ liệu

File `online_retail.csv` **không** được đưa vào repo này (xem `.gitignore`) vì dung lượng khá lớn (~47MB) — không nên commit file dữ liệu lớn lên Git.

## Cách lấy dữ liệu

Đây là bộ dữ liệu **Online Retail** công khai từ UCI Machine Learning Repository:

- Trang chính thức: https://archive.ics.uci.edu/dataset/352/online+retail
- Tác giả: Daqing Chen, Sai Laing Sain, Kun Guo (2015)
- Giấy phép: CC BY 4.0

Sau khi tải về, đặt file vào đây với tên:

```
data/online_retail.csv
```

Hoặc đơn giản hơn: khi chạy ứng dụng Streamlit (`streamlit run app.py`), dùng luôn nút **"Tải lên online_retail.csv"** ở thanh bên trái — không cần đặt file vào thư mục `data/`.
