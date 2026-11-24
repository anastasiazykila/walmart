"""
WALMART CUSTOMER ANALYSIS - Portfolio Project
Author: Anastasia (anastasiazykila)
Dataset: Walmart_customer_purchases.csv

Pertanyaan Bisnis:
1. Produk apa yang paling disukai customer (berdasarkan rating)?
2. Customer bisa dikelompokkan jadi berapa segmen berdasarkan umur & jumlah belanja?
3. Bisakah kita memprediksi apakah seorang customer akan jadi Repeat Customer?

Tools: Python, Pandas, Seaborn, Matplotlib, Plotly, Streamlit, Scikit-Learn
Cara menjalankan:
    pip install -r requirements.txt
    streamlit run walmart.py
"""

import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ============================================================
# SETUP STREAMLIT
# ============================================================
st.set_page_config(page_title="Walmart Customer Analysis", layout="wide")
st.title("🛒 Walmart Customer Analysis")
st.write("""
Project ini menganalisa data pembelian customer Walmart untuk menjawab 3 pertanyaan bisnis:
1. Produk apa yang paling disukai customer?
2. Customer bisa dibagi jadi berapa segmen?
3. Bisakah kita memprediksi Repeat Customer?
""")

# ============================================================
# 1. LOAD DATASET (FIX: pakai path relatif, bukan C:/Users/Ani/...)
# ============================================================
# File CSV harus ada di folder yang sama dengan file ini
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_customer_purchases.csv")
    return df

df = load_data()
st.subheader("1. Data Mentah")
st.write(f"Jumlah baris: **{df.shape[0]}** | Jumlah kolom: **{df.shape[1]}**")
st.dataframe(df.head())

# ============================================================
# 2. CLEANING / PREPARE DATA
# ============================================================
st.subheader("2. Data Cleaning")

# Hapus duplikat dan nilai kosong
rows_before = df.shape[0]
df_cleaned = df.drop_duplicates().dropna().copy()
rows_after = df_cleaned.shape[0]
st.write(f"Baris sebelum cleaning: {rows_before} → Sesudah cleaning: {rows_after} (dihapus {rows_before - rows_after} baris duplikat / kosong)")

# Konversi kolom tanggal jika ada
if 'Purchase_Date' in df_cleaned.columns:
    df_cleaned['Purchase_Date'] = pd.to_datetime(df_cleaned['Purchase_Date'])

# Fungsi untuk menyederhanakan kategori (Top 10, sisanya jadi 'Other')
# Tujuannya: biar one-hot encoding tidak meledak jadi ratusan kolom
def reduce_categories(data, column, top_n=10):
    top_categories = data[column].value_counts().nlargest(top_n).index
    return data[column].apply(lambda x: x if x in top_categories else 'Other')

columns_to_reduce = ['City', 'Product_Name', 'Category']
for col in columns_to_reduce:
    if col in df_cleaned.columns:
        df_cleaned[col] = reduce_categories(df_cleaned, col, top_n=10)

st.write("✅ Cleaning selesai: duplikat dihapus, NaN dihapus, tanggal dikonversi, kategori disederhanakan jadi Top 10 + Other.")
st.dataframe(df_cleaned.head())

# ============================================================
# 3. EDA - Produk dengan Rating Tertinggi
# ============================================================
st.subheader("3. EDA - Produk Apa yang Paling Disukai Customer?")

# Ambil top 10 produk berdasarkan jumlah pembelian
top_products = df_cleaned['Product_Name'].value_counts().nlargest(10).index
df_top_products = df_cleaned[df_cleaned['Product_Name'].isin(top_products)]

# Hitung rata-rata rating
avg_rating = df_top_products.groupby('Product_Name')['Rating'].mean().sort_values()

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=avg_rating.values, y=avg_rating.index, palette='coolwarm', ax=ax)
ax.set_title('Rata-rata Rating Produk (Top 10 Produk)')
ax.set_xlabel('Rata-rata Rating')
ax.set_ylabel('Nama Produk')
st.pyplot(fig)

st.write("""
**Insight:** Produk dengan rating tertinggi bukan selalu produk yang paling banyak dibeli.
Artinya, ada produk yang dibeli banyak orang tapi ratingnya biasa aja — itu peluang untuk diperbaiki kualitasnya.
""")

# ============================================================
# 4. SEGMENTASI CUSTOMER (KMeans)
# ============================================================
st.subheader("4. Segmentasi Customer dengan KMeans")

st.write("""
Tujuan: Mengelompokkan customer berdasarkan **Umur** dan **Jumlah Belanja**,
agar marketing bisa menargetkan promo yang berbeda untuk tiap segmen.
""")

# FIX: Hanya pakai StandardScaler untuk KMeans (tidak double scaling dengan MinMax sebelumnya)
# Ambil data yang sudah dibersihkan, bukan data mentah
X_cluster = df_cleaned[['Age', 'Purchase_Amount']].copy()

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# Elbow Method (opsional, untuk menentukan jumlah cluster)
# Di sini kita pakai 4 cluster berdasarkan percobaan sebelumnya
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cleaned['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Tampilkan rata-rata tiap cluster
cluster_summary = df_cleaned.groupby('Cluster')[['Age', 'Purchase_Amount']].mean().round(2)
st.write("Rata-rata Umur & Belanja per Cluster:")
st.dataframe(cluster_summary)

# Visualisasi Segmentasi
# Untuk visualisasi, pakai nilai asli (bukan yang sudah di-scale) biar mudah dibaca
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x=df_cleaned['Age'],
    y=df_cleaned['Purchase_Amount'],
    hue=df_cleaned['Cluster'],
    palette='Set2',
    s=60,
    ax=ax2
)
ax2.set_title('Segmentasi Customer Berdasarkan Umur dan Jumlah Belanja')
ax2.set_xlabel('Umur')
ax2.set_ylabel('Jumlah Belanja (Rp / $)')
st.pyplot(fig2)

st.write("""
**Insight Segmentasi (contoh interpretasi):**
- Cluster dengan umur muda + belanja tinggi = target utama untuk promo premium
- Cluster dengan umur tua + belanja rendah = butuh promo yang berbeda (misal: diskon, bukan produk premium)
- Interpretasi lengkap ada di README dan di laporan akhir.
""")

# ============================================================
# 5. PREDIKSI REPEAT CUSTOMER
# ============================================================
st.subheader("5. Prediksi Repeat Customer - Logistic Regression vs Random Forest")

st.write("""
Pertanyaan: Bisakah kita memprediksi apakah seorang customer akan belanja lagi (Repeat Customer)?
Model yang dibandingkan: Logistic Regression vs Random Forest.
""")

target_column = "Repeat_Customer"

# FIX: Pakai df_cleaned (yang sudah dibersihkan), bukan df mentah
# FIX: Pisahkan target dulu, baru encoding fitur. Target tidak boleh ikut di-one-hot.
# Pastikan target dalam bentuk 0/1
if df_cleaned[target_column].dtype == object:
    # Jika target masih berupa 'Yes'/'No' atau 'True'/'False', ubah ke 0/1
    df_cleaned[target_column] = df_cleaned[target_column].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, True: 1, False: 0})

X = df_cleaned.drop([target_column], axis=1).copy()
y = df_cleaned[target_column].copy()

# Hapus kolom tanggal untuk modeling (tidak bisa dipakai langsung oleh model)
if 'Purchase_Date' in X.columns:
    X = X.drop(['Purchase_Date'], axis=1)

# Label Encoding untuk kolom dengan kategori > 100 (biar tidak meledak)
for col in X.select_dtypes(include=['object', 'category']).columns:
    if X[col].nunique() > 100:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# One-hot encoding untuk sisa fitur kategorikal
X = pd.get_dummies(X, drop_first=True)

# Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Logistic Regression (butuh standardisasi) ---
scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

# --- Random Forest (tidak butuh standardisasi) ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --- Evaluasi & Tampilkan di Streamlit (FIX: sebelumnya hanya print di terminal) ---
acc_logreg = accuracy_score(y_test, y_pred_logreg)
acc_rf = accuracy_score(y_test, y_pred_rf)

col1, col2 = st.columns(2)
with col1:
    st.metric("Akurasi Logistic Regression", f"{acc_logreg:.2%}")
    st.text("Classification Report - Logistic Regression:")
    st.text(classification_report(y_test, y_pred_logreg))
with col2:
    st.metric("Akurasi Random Forest", f"{acc_rf:.2%}")
    st.text("Classification Report - Random Forest:")
    st.text(classification_report(y_test, y_pred_rf))

# Kesimpulan otomatis
if acc_rf > acc_logreg:
    st.success(f"🏆 Random Forest lebih baik ({acc_rf:.2%} vs {acc_logreg:.2%}). Model ini lebih cocok untuk memprediksi Repeat Customer di dataset ini.")
else:
    st.success(f"🏆 Logistic Regression lebih baik ({acc_logreg:.2%} vs {acc_rf:.2%}). Model yang lebih sederhana justru lebih akurat di dataset ini.")

st.write("""
**Insight Bisnis:**
Jika kita bisa memprediksi siapa yang akan jadi Repeat Customer, kita bisa:
- Kasih promo khusus ke customer yang diprediksi TIDAK akan balik, biar mereka balik lagi
- Kasih reward ke customer yang diprediksi akan balik, biar mereka makin loyal
- Hemat budget marketing karena tidak buang promo ke semua orang
""")

st.markdown("---")
st.write("Dibuat oleh Anastasia | Tools: Python, Pandas, Seaborn, Matplotlib, Scikit-Learn, Streamlit | Dataset: Walmart Customer Purchases")
