"""
Xay dung mo hinh Random Forest phan loai khach hang theo gia tri
(Low Value / Medium Value / High Value) dua tren chi so RFM
(Recency, Frequency, Monetary)

Input : online_retail.csv (du lieu goc)
Output:
    - customer_rfm_labeled.csv      : bang RFM + nhan phan khuc tung khach hang
    - confusion_matrix.png          : ma tran nham lan tren tap test
    - feature_importance.png        : muc do quan trong cua cac dac trung
    - rf_customer_value_model.pkl   : mo hinh da huan luyen (joblib)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

RANDOM_STATE = 42

# ============================================================
# 1. DOC & LAM SACH DU LIEU (rut gon tu buoc tien xu ly truoc)
# ============================================================
df = pd.read_csv("online_retail.csv", encoding="latin1")
df = df.drop_duplicates()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["InvoiceNo"] = df["InvoiceNo"].astype(str)

df = df[~df["InvoiceNo"].str.startswith("C")]                # bo don huy
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]         # bo gia tri khong hop le
df = df.dropna(subset=["CustomerID"])                          # chi giu khach hang xac dinh duoc
non_product_codes = ["POST", "D", "M", "BANK CHARGES", "DOT", "C2", "PADS", "CRUK"]
df = df[~df["StockCode"].astype(str).isin(non_product_codes)]

df["CustomerID"] = df["CustomerID"].astype(int)
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print(f"So dong giao dich hop le dung de tinh RFM: {len(df):,}")
print(f"So khach hang duy nhat: {df['CustomerID'].nunique():,}")

# ============================================================
# 2. TINH CHI SO RFM CHO TUNG KHACH HANG
# ============================================================
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

print("\nThong ke RFM:")
print(rfm[["Recency", "Frequency", "Monetary", "Tenure", "NumProducts", "AvgOrderValue"]].describe())

# ============================================================
# 3. TAO NHAN PHAN KHUC GIA TRI (Low / Medium / High) TU DIEM RFM
# ============================================================
def score_quartile(series, ascending_score=True):
    ranks = series.rank(method="first")
    labels = [1, 2, 3, 4] if ascending_score else [4, 3, 2, 1]
    return pd.qcut(ranks, 4, labels=labels).astype(int)

rfm["R_score"] = score_quartile(rfm["Recency"], ascending_score=False)   # Recency thap -> diem cao
rfm["F_score"] = score_quartile(rfm["Frequency"], ascending_score=True)
rfm["M_score"] = score_quartile(rfm["Monetary"], ascending_score=True)
rfm["RFM_Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

rfm["ValueSegment"] = pd.cut(
    rfm["RFM_Score"],
    bins=[2, 6, 9, 12],
    labels=[
        "Low Value",
        "Medium Value",
        "High Value"
    ],
    include_lowest=True
)


print("\nPhan bo nhan:")
print(rfm["ValueSegment"].value_counts())

# ============================================================
# 4. CHUAN BI DU LIEU HUAN LUYEN
# ============================================================
feature_cols = ["Recency", "Frequency", "Monetary", "Tenure", "NumProducts", "AvgOrderValue"]

top_countries = rfm["Country"].value_counts().nlargest(10).index
rfm["Country_grouped"] = np.where(rfm["Country"].isin(top_countries), rfm["Country"], "Other")
country_dummies = pd.get_dummies(rfm["Country_grouped"], prefix="Country")

X = pd.concat([rfm[feature_cols], country_dummies], axis=1)
y = rfm["ValueSegment"].astype(str)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# ============================================================
# 5. HUAN LUYEN MO HINH RANDOM FOREST
# ============================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1_macro")
print(f"\nF1-macro trung binh qua 5-fold CV (tren tap train): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# 6. DANH GIA MO HINH TREN TAP TEST
# ============================================================
y_pred = rf.predict(X_test)

print(f"\nAccuracy tren tap test: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-macro tren tap test: {f1_score(y_test, y_pred, average='macro'):.4f}")
print("\nBao cao phan loai chi tiet:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Ma tran nham lan
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(le.classes_)))
ax.set_yticks(range(len(le.classes_)))
ax.set_xticklabels(le.classes_, rotation=20)
ax.set_yticklabels(le.classes_)
ax.set_xlabel("Du doan")
ax.set_ylabel("Thuc te")
ax.set_title("Ma tran nham lan - Random Forest")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 6))
importances.head(15).plot(kind="barh", ax=ax, color="#4C72B0")
ax.invert_yaxis()
ax.set_xlabel("Muc do quan trong")
ax.set_title("Top 15 dac trung quan trong nhat")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nTop 10 dac trung quan trong nhat:")
print(importances.head(10))

# ============================================================
# 7. LUU KET QUA
# ============================================================
rfm.to_csv("customer_rfm_labeled.csv", index=False)
joblib.dump({"model": rf, "label_encoder": le, "feature_columns": list(X.columns)},
            "rf_customer_value_model.pkl")

print("\nDa luu: customer_rfm_labeled.csv, confusion_matrix.png, feature_importance.png, rf_customer_value_model.pkl")
