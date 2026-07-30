# =====================================================================
# APP.PY - Ứng dụng Streamlit
# Phân loại khách hàng theo giá trị (Low / Medium / High Value)
# Pipeline: Khám phá dữ liệu -> Tiền xử lý -> Tính RFM -> Random Forest
# =====================================================================

import os
import io
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

plt.style.use("ggplot")

NON_PRODUCT_CODES = ["POST", "D", "M", "BANK CHARGES", "DOT", "C2", "PADS", "CRUK"]
FEATURE_COLS = ["Recency", "Frequency", "Monetary", "Tenure", "NumProducts", "AvgOrderValue"]

st.set_page_config(
    page_title="Phân khúc khách hàng - Random Forest",
    page_icon="📊",
    layout="wide",
)

# =====================================================================
# HÀM TIỆN ÍCH (CÓ CACHE ĐỂ TRÁNH TÍNH LẠI KHÔNG CẦN THIẾT)
# =====================================================================

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    """Đọc file CSV gốc."""
    return pd.read_csv(file, encoding="latin1")


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame):
    """Thực hiện toàn bộ bước tiền xử lý, trả về dữ liệu sạch + log từng bước."""
    logs = []
    df = df.copy()

    n0 = len(df)
    df = df.drop_duplicates(keep="first")
    logs.append(f"Loại bỏ dòng trùng lặp: {n0:,} → {len(df):,} dòng (đã bỏ {n0 - len(df):,})")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)

    n1 = len(df)
    is_cancelled = df["InvoiceNo"].str.startswith("C")
    df_returns = df[is_cancelled].copy()
    df = df[~is_cancelled].copy()
    logs.append(f"Tách đơn hủy (InvoiceNo bắt đầu bằng 'C'): {n1:,} → {len(df):,} dòng (đã tách {len(df_returns):,})")

    n2 = len(df)
    odd_negative = df["Quantity"] < 0
    df = df[~odd_negative].copy()
    logs.append(f"Loại Quantity âm bất thường (không phải đơn hủy): {n2:,} → {len(df):,} dòng (đã bỏ {n2 - len(df):,})")

    n3 = len(df)
    df = df[(df["UnitPrice"] > 0) & (df["Quantity"] > 0)].copy()
    logs.append(f"Loại UnitPrice<=0 hoặc Quantity<=0: {n3:,} → {len(df):,} dòng (đã bỏ {n3 - len(df):,})")

    n4 = len(df)
    is_non_product = df["StockCode"].isin(NON_PRODUCT_CODES)
    df = df[~is_non_product].copy()
    logs.append(f"Loại mã phi sản phẩm (POST, D, M...): {n4:,} → {len(df):,} dòng (đã bỏ {n4 - len(df):,})")

    desc_map = (
        df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda x: x.value_counts().idxmax())
    )
    df["Description"] = df["Description"].fillna(df["StockCode"].map(desc_map))
    n5 = len(df)
    df = df.dropna(subset=["Description"])
    logs.append(f"Điền Description theo StockCode, loại dòng không tra được: {n5:,} → {len(df):,} dòng")

    n6 = len(df)
    df_known_customer = df.dropna(subset=["CustomerID"]).copy()
    df_known_customer["CustomerID"] = df_known_customer["CustomerID"].astype(int)
    logs.append(
        f"Tập con có CustomerID hợp lệ: {len(df_known_customer):,}/{n6:,} dòng "
        f"({len(df_known_customer) / n6 * 100:.1f}%) — dùng để tính RFM"
    )

    df_known_customer["TotalPrice"] = df_known_customer["Quantity"] * df_known_customer["UnitPrice"]
    df_known_customer["Year"] = df_known_customer["InvoiceDate"].dt.year
    df_known_customer["Month"] = df_known_customer["InvoiceDate"].dt.month
    df_known_customer["DayOfWeek"] = df_known_customer["InvoiceDate"].dt.day_name()

    return df_known_customer, df_returns, logs


def score_quartile(series: pd.Series, ascending_score: bool = True) -> pd.Series:
    ranks = series.rank(method="first")
    labels = [1, 2, 3, 4] if ascending_score else [4, 3, 2, 1]
    return pd.qcut(ranks, 4, labels=labels).astype(int)


@st.cache_data(show_spinner=False)
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Tính RFM cho từng khách hàng và gán nhãn phân khúc giá trị."""
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        Tenure=("InvoiceDate", lambda x: (x.max() - x.min()).days),
        NumProducts=("StockCode", "nunique"),
    ).reset_index()
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]

    top_country = df.groupby("CustomerID")["Country"].agg(lambda x: x.value_counts().idxmax())
    rfm = rfm.merge(top_country.rename("Country"), on="CustomerID")

    rfm["R_score"] = score_quartile(rfm["Recency"], ascending_score=False)
    rfm["F_score"] = score_quartile(rfm["Frequency"], ascending_score=True)
    rfm["M_score"] = score_quartile(rfm["Monetary"], ascending_score=True)
    rfm["RFM_Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    # Ngưỡng cố định theo tổng điểm RFM (3-12): Low=3-6, Medium=7-9, High=10-12
    rfm["ValueSegment"] = pd.cut(
        rfm["RFM_Score"],
        bins=[2, 6, 9, 12],
        labels=["Low Value", "Medium Value", "High Value"],
        include_lowest=True,
    )
    return rfm


def build_features(rfm: pd.DataFrame):
    top_countries = rfm["Country"].value_counts().nlargest(10).index
    rfm = rfm.copy()
    rfm["Country_grouped"] = np.where(rfm["Country"].isin(top_countries), rfm["Country"], "Other")
    country_dummies = pd.get_dummies(rfm["Country_grouped"], prefix="Country")
    X = pd.concat([rfm[FEATURE_COLS], country_dummies], axis=1)
    y = rfm["ValueSegment"].astype(str)
    return X, y, list(top_countries)


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title("Ma trận nhầm lẫn")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    return fig


def plot_feature_importance(importances: pd.Series, top_n: int = 15):
    fig, ax = plt.subplots(figsize=(6, 5))
    importances.head(top_n).sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Mức độ quan trọng")
    ax.set_title(f"Top {top_n} đặc trưng quan trọng nhất")
    plt.tight_layout()
    return fig


# =====================================================================
# SIDEBAR
# =====================================================================

st.sidebar.header("⚙️ Cấu hình")
uploaded_file = st.sidebar.file_uploader("Tải lên online_retail.csv", type=["csv"])

default_path = "online_retail.csv"
data_source = uploaded_file
if data_source is None and os.path.exists(default_path):
    data_source = default_path
    st.sidebar.caption(f"Đang dùng file mặc định: `{default_path}`")

st.sidebar.subheader("Tham số Random Forest")
n_estimators = st.sidebar.slider("n_estimators (số cây)", 50, 500, 300, step=50)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 10, 2)
test_size = st.sidebar.slider("Tỷ lệ tập test", 0.1, 0.4, 0.2, step=0.05)
random_state = st.sidebar.number_input("random_state", value=42, step=1)
balanced = st.sidebar.checkbox("class_weight = 'balanced'", value=True)

train_button = st.sidebar.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "Quy trình: Khám phá dữ liệu → Tiền xử lý → Tính RFM & gán nhãn phân khúc "
    "→ Huấn luyện & đánh giá Random Forest → Dự đoán thử."
)

# =====================================================================
# MAIN
# =====================================================================

st.title("📊 Phân loại khách hàng theo giá trị (RFM + Random Forest)")
st.caption("Bộ dữ liệu Online Retail — phân khúc khách hàng thành Low / Medium / High Value")

if data_source is None:
    st.info("👈 Vui lòng tải lên file `online_retail.csv` ở thanh bên trái để bắt đầu.")
    st.stop()

raw_df = load_data(data_source)
df_clean, df_returns, logs = clean_data(raw_df)
rfm = compute_rfm(df_clean)
X, y, top_countries = build_features(rfm)

tab_eda, tab_prep, tab_rfm, tab_model, tab_predict = st.tabs(
    ["🔍 Khám phá dữ liệu", "🧹 Tiền xử lý", "🧮 RFM & Phân khúc", "🌲 Huấn luyện & Đánh giá", "🔮 Dự đoán thử"]
)

# ------------------------------- TAB EDA -------------------------------
with tab_eda:
    st.subheader("Xem trước dữ liệu")
    st.dataframe(raw_df.head(), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số dòng", f"{raw_df.shape[0]:,}")
    c2.metric("Khách hàng (unique)", f"{raw_df['CustomerID'].nunique():,}")
    c3.metric("Hóa đơn (unique)", f"{raw_df['InvoiceNo'].nunique():,}")
    c4.metric("Sản phẩm (unique)", f"{raw_df['StockCode'].nunique():,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Giá trị thiếu")
        missing = raw_df.isnull().sum()
        missing_pct = (missing / len(raw_df) * 100).round(2)
        st.dataframe(
            pd.DataFrame({"Số lượng thiếu": missing, "Tỷ lệ (%)": missing_pct}),
            use_container_width=True,
        )
    with col_b:
        st.subheader("Dữ liệu trùng lặp")
        st.metric("Số dòng trùng lặp hoàn toàn", f"{raw_df.duplicated().sum():,}")
        st.subheader("Top quốc gia")
        st.dataframe(raw_df["Country"].value_counts().head(10), use_container_width=True)

    st.subheader("Thống kê mô tả")
    st.dataframe(raw_df.describe(), use_container_width=True)

    st.subheader("Phân phối Quantity & UnitPrice")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(raw_df["Quantity"].clip(-50, 50), bins=50, ax=ax)
        ax.set_title("Phân phối Quantity (đã giới hạn [-50, 50] để dễ nhìn)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(raw_df["UnitPrice"].clip(0, 50), bins=50, ax=ax)
        ax.set_title("Phân phối UnitPrice (đã giới hạn [0, 50] để dễ nhìn)")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=raw_df["Quantity"], ax=ax)
        ax.set_title("Boxplot Quantity (thang gốc, có outlier)")
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=raw_df["UnitPrice"], ax=ax)
        ax.set_title("Boxplot UnitPrice (thang gốc, có outlier)")
        st.pyplot(fig)

# ----------------------------- TAB TIỀN XỬ LÝ -----------------------------
with tab_prep:
    st.subheader("Các bước tiền xử lý đã thực hiện")
    for i, log in enumerate(logs, 1):
        st.write(f"**Bước {i}.** {log}")

    st.success(f"✅ Dữ liệu sạch cuối cùng: **{len(df_clean):,} dòng**, "
               f"**{df_clean['CustomerID'].nunique():,} khách hàng** duy nhất.")

    st.subheader("Kiểm tra lại giá trị thiếu sau khi làm sạch")
    st.dataframe(df_clean.isnull().sum().rename("Số lượng thiếu"), use_container_width=True)

    st.subheader("Xem trước dữ liệu sạch")
    st.dataframe(df_clean.head(20), use_container_width=True)

    csv_buffer = io.StringIO()
    df_clean.to_csv(csv_buffer, index=False)
    st.download_button("⬇️ Tải dữ liệu đã làm sạch (CSV)", csv_buffer.getvalue(),
                        file_name="online_retail_cleaned.csv", mime="text/csv")

# ------------------------------- TAB RFM -------------------------------
with tab_rfm:
    st.subheader("Bảng chỉ số RFM theo khách hàng")
    st.dataframe(rfm.head(20), use_container_width=True)

    st.subheader("Thống kê RFM")
    st.dataframe(rfm[FEATURE_COLS].describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Phân bố nhãn phân khúc")
        seg_counts = rfm["ValueSegment"].value_counts().reindex(
            ["Low Value", "Medium Value", "High Value"]
        )
        st.bar_chart(seg_counts)
        st.dataframe(seg_counts.rename("Số khách hàng"), use_container_width=True)
    with col2:
        st.subheader("Phân bố điểm RFM_Score")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(rfm["RFM_Score"], bins=10, discrete=True, ax=ax)
        ax.set_title("Phân bố tổng điểm RFM (3-12)")
        st.pyplot(fig)

    st.caption(
        "Nhãn phân khúc được gán theo ngưỡng cố định trên tổng điểm RFM: "
        "**Low Value** = 3–6 điểm, **Medium Value** = 7–9 điểm, **High Value** = 10–12 điểm."
    )

    csv_buffer2 = io.StringIO()
    rfm.to_csv(csv_buffer2, index=False)
    st.download_button("⬇️ Tải bảng RFM đã gán nhãn (CSV)", csv_buffer2.getvalue(),
                        file_name="customer_rfm_labeled.csv", mime="text/csv")

# --------------------------- TAB HUẤN LUYỆN --------------------------
with tab_model:
    st.subheader("Huấn luyện mô hình Random Forest")

    if train_button:
        with st.spinner("Đang huấn luyện mô hình..."):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced" if balanced else None,
                random_state=random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)

            cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1_macro")
            y_pred = rf.predict(X_test)

            report_dict = classification_report(
                y_test, y_pred, target_names=le.classes_, output_dict=True
            )
            cm = confusion_matrix(y_test, y_pred)
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

            st.session_state.trained = True
            st.session_state.model = rf
            st.session_state.label_encoder = le
            st.session_state.feature_columns = list(X.columns)
            st.session_state.top_countries = top_countries
            st.session_state.metrics = {
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
            }
            st.session_state.report_df = pd.DataFrame(report_dict).transpose()
            st.session_state.cm = cm
            st.session_state.class_names = list(le.classes_)
            st.session_state.importances = importances

    if not st.session_state.get("trained", False):
        st.info("👈 Chọn tham số ở thanh bên trái rồi nhấn **'Huấn luyện mô hình'** để bắt đầu.")
    else:
        m = st.session_state.metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy (test)", f"{m['accuracy']:.3f}")
        c2.metric("F1-macro (test)", f"{m['f1_macro']:.3f}")
        c3.metric("F1-macro (5-fold CV)", f"{m['cv_f1_mean']:.3f} ± {m['cv_f1_std']:.3f}")

        st.subheader("Báo cáo phân loại chi tiết")
        st.dataframe(st.session_state.report_df.round(3), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ma trận nhầm lẫn")
            fig_cm = plot_confusion_matrix(st.session_state.cm, st.session_state.class_names)
            st.pyplot(fig_cm)
        with col2:
            st.subheader("Mức độ quan trọng của đặc trưng")
            fig_fi = plot_feature_importance(st.session_state.importances)
            st.pyplot(fig_fi)

        model_buffer = io.BytesIO()
        joblib.dump(
            {
                "model": st.session_state.model,
                "label_encoder": st.session_state.label_encoder,
                "feature_columns": st.session_state.feature_columns,
            },
            model_buffer,
        )
        st.download_button(
            "⬇️ Tải mô hình đã huấn luyện (.pkl)",
            model_buffer.getvalue(),
            file_name="rf_customer_value_model.pkl",
        )

# --------------------------- TAB DỰ ĐOÁN THỬ --------------------------
with tab_predict:
    st.subheader("Dự đoán phân khúc cho một khách hàng mới")

    if not st.session_state.get("trained", False):
        st.info("Vui lòng huấn luyện mô hình ở tab **'Huấn luyện & Đánh giá'** trước.")
    else:
        col1, col2, col3 = st.columns(3)
        recency = col1.number_input("Recency (ngày kể từ lần mua gần nhất)", min_value=0, value=30)
        frequency = col2.number_input("Frequency (số hóa đơn)", min_value=1, value=5)
        monetary = col3.number_input("Monetary (tổng chi tiêu)", min_value=0.0, value=500.0)

        col4, col5, col6 = st.columns(3)
        tenure = col4.number_input("Tenure (số ngày gắn bó)", min_value=0, value=100)
        num_products = col5.number_input("NumProducts (số sản phẩm khác nhau)", min_value=1, value=10)
        country = col6.selectbox("Country", st.session_state.top_countries + ["Other"])

        avg_order_value = monetary / frequency if frequency > 0 else 0.0
        st.caption(f"AvgOrderValue tự tính = Monetary / Frequency = {avg_order_value:,.2f}")

        if st.button("🔮 Dự đoán phân khúc", type="primary"):
            row = {col: 0 for col in st.session_state.feature_columns}
            row["Recency"] = recency
            row["Frequency"] = frequency
            row["Monetary"] = monetary
            row["Tenure"] = tenure
            row["NumProducts"] = num_products
            row["AvgOrderValue"] = avg_order_value
            country_col = f"Country_{country}"
            if country_col in row:
                row[country_col] = 1
            else:
                row["Country_Other"] = 1

            X_new = pd.DataFrame([row])[st.session_state.feature_columns]
            model = st.session_state.model
            le = st.session_state.label_encoder
            pred = model.predict(X_new)[0]
            proba = model.predict_proba(X_new)[0]

            pred_label = le.inverse_transform([pred])[0]
            st.success(f"### Phân khúc dự đoán: **{pred_label}**")

            proba_df = pd.Series(proba, index=le.classes_).sort_values(ascending=False)
            st.bar_chart(proba_df)
