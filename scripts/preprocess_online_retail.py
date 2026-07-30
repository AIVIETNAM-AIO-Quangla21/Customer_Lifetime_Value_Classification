"""
Script tiền xử lý dữ liệu Online Retail
Input : online_retail.csv
Output: online_retail_cleaned.csv (+ online_retail_returns.csv chứa các đơn hủy)
"""

import pandas as pd
import numpy as np

# ============================================================
# 0. ĐỌC DỮ LIỆU
# ============================================================
df = pd.read_csv("online_retail.csv", encoding="latin1")
print(f"Số dòng ban đầu: {len(df):,}")

# ============================================================
# 1. LOẠI BỎ DÒNG TRÙNG LẶP HOÀN TOÀN
# ============================================================
n_before = len(df)
df = df.drop_duplicates(keep="first")
print(f"Đã loại {n_before - len(df):,} dòng trùng lặp -> còn {len(df):,} dòng")

# ============================================================
# 2. CHUẨN HÓA KIỂU DỮ LIỆU
# ============================================================
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df["StockCode"] = df["StockCode"].astype(str)
# CustomerID giữ ở dạng float (vì còn NaN); sẽ ép về Int64 sau khi xử lý missing

# ============================================================
# 3. TÁCH ĐƠN HÀNG HỦY / TRẢ HÀNG (InvoiceNo bắt đầu bằng 'C')
# ============================================================
is_cancelled = df["InvoiceNo"].str.startswith("C")
df_returns = df[is_cancelled].copy()
df = df[~is_cancelled].copy()
print(f"Tách {len(df_returns):,} dòng là đơn hủy (InvoiceNo có 'C') -> lưu riêng, còn {len(df):,} dòng")

# Kiểm tra Quantity âm còn sót lại (không thuộc đơn hủy) -> đây là dữ liệu bất thường, loại bỏ
odd_negative = df["Quantity"] < 0
print(f"Phát hiện {odd_negative.sum():,} dòng Quantity âm nhưng không phải đơn hủy -> loại bỏ (nghi lỗi nhập liệu)")
df = df[~odd_negative].copy()

# ============================================================
# 4. LOẠI BỎ GIAO DỊCH KHÔNG HỢP LỆ VỀ GIÁ / SỐ LƯỢNG
# ============================================================
n_before = len(df)
df = df[(df["UnitPrice"] > 0) & (df["Quantity"] > 0)].copy()
print(f"Loại {n_before - len(df):,} dòng có UnitPrice<=0 hoặc Quantity<=0 -> còn {len(df):,} dòng")

# ============================================================
# 5. TÁCH CÁC MÃ HÀNG KHÔNG PHẢI SẢN PHẨM THỰC (phí ship, phí ngân hàng...)
# ============================================================
non_product_codes = ["POST", "D", "M", "BANK CHARGES", "DOT", "C2", "PADS", "CRUK"]
is_non_product = df["StockCode"].isin(non_product_codes)
df_non_product = df[is_non_product].copy()
df = df[~is_non_product].copy()
print(f"Tách {len(df_non_product):,} dòng mã phi sản phẩm (POST, D, M...) -> còn {len(df):,} dòng")

# ============================================================
# 6. XỬ LÝ MISSING VALUES
# ============================================================
# 6a. Description: điền theo Description phổ biến nhất của cùng StockCode
desc_map = (
    df.dropna(subset=["Description"])
    .groupby("StockCode")["Description"]
    .agg(lambda x: x.value_counts().idxmax())
)
df["Description"] = df["Description"].fillna(df["StockCode"].map(desc_map))
n_desc_missing = df["Description"].isna().sum()
df = df.dropna(subset=["Description"])
print(f"Điền Description theo StockCode, loại nốt {n_desc_missing:,} dòng không tra được -> còn {len(df):,} dòng")

# 6b. CustomerID: tách riêng để tùy mục đích phân tích mà dùng
#     - Phân tích khách hàng (RFM, phân khúc...) -> dùng df_known_customer
#     - Phân tích doanh thu/sản phẩm tổng thể -> có thể dùng cả df
df_known_customer = df.dropna(subset=["CustomerID"]).copy()
df_known_customer["CustomerID"] = df_known_customer["CustomerID"].astype("Int64")
print(f"Tập con có CustomerID hợp lệ: {len(df_known_customer):,}/{len(df):,} dòng "
      f"({len(df_known_customer)/len(df)*100:.1f}%) -> dùng cho phân tích theo khách hàng")

# ============================================================
# 7. TẠO FEATURE PHỤC VỤ PHÂN TÍCH
# ============================================================
for d in (df, df_known_customer):
    d["TotalPrice"] = d["Quantity"] * d["UnitPrice"]
    d["Year"] = d["InvoiceDate"].dt.year
    d["Month"] = d["InvoiceDate"].dt.month
    d["Day"] = d["InvoiceDate"].dt.day
    d["Hour"] = d["InvoiceDate"].dt.hour
    d["DayOfWeek"] = d["InvoiceDate"].dt.day_name()

# ============================================================
# 8. LƯU KẾT QUẢ
# ============================================================
df.to_csv("online_retail_cleaned.csv", index=False)
df_known_customer.to_csv("online_retail_cleaned_with_customerid.csv", index=False)
df_returns.to_csv("online_retail_returns.csv", index=False)

print("\n=== TÓM TẮT ===")
print(f"Dữ liệu sạch (dùng cho phân tích doanh thu/sản phẩm): {len(df):,} dòng")
print(f"Dữ liệu sạch có CustomerID (dùng cho phân tích khách hàng): {len(df_known_customer):,} dòng")
print(f"Đơn hàng bị hủy/trả (lưu riêng để phân tích returns): {len(df_returns):,} dòng")
