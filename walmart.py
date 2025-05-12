import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# 1. Load dataset
df = pd.read_csv("C:/Users/Ani/Documents/A DATA SCIENCE/Walmart_customer_purchases.csv")

# 2. Hapus duplikat dan nilai kosong
df_cleaned = df.drop_duplicates().dropna()

# 3. Konversi kolom tanggal jika ada
if 'Purchase_Date' in df_cleaned.columns:
    df_cleaned['Purchase_Date'] = pd.to_datetime(df_cleaned['Purchase_Date'])

# 4. Fungsi untuk menyederhanakan kategori
def reduce_categories(data, column, top_n=10):
    top_categories = data[column].value_counts().nlargest(top_n).index
    return data[column].apply(lambda x: x if x in top_categories else 'Other')


# 5. Terapkan penyederhanaan pada kolom tertentu
columns_to_reduce = ['City', 'Product_Name', 'Category']
for col in columns_to_reduce:
    if col in df_cleaned.columns:
        df_cleaned[col] = reduce_categories(df_cleaned, col, top_n=10)

# 6. One-hot encoding untuk kolom kategorikal
categorical_cols = ['Gender', 'City', 'Category', 'Payment_Method', 'Discount_Applied', 'Repeat_Customer', 'Product_Name']
categorical_cols = [col for col in categorical_cols if col in df_cleaned.columns]  # Pastikan kolom ada
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols)

# 7. Normalisasi kolom numerik
numerical_cols = ['Age', 'Purchase_Amount', 'Rating']
numerical_cols = [col for col in numerical_cols if col in df_encoded.columns]  # Pastikan kolom ada
scaler = MinMaxScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# 8. Simpan hasil preprocessing ke file baru
df_encoded.to_csv("C:/Users/Ani/Documents/A DATA SCIENCE/Walmart_customer_purchases_cleaned.csv", index=False)

print(f"Shape after encoding: {df_encoded.shape}")

# ================= EDA (Exploratory Data Analysis) =================

# Cek struktur awal data
print("\nInfo Data:")
print(df.info())
print("\n5 Data Teratas:")
print(df.head())

# Statistik deskriptif
print("\nStatistik Deskriptif:")
print(df.describe())
# Ambil top 10 produk berdasarkan jumlah pembelian
top_products = df['Product_Name'].value_counts().nlargest(10).index
df_top_products = df[df['Product_Name'].isin(top_products)]

# Hitung rata-rata rating
avg_rating = df_top_products.groupby('Product_Name')['Rating'].mean().sort_values()

# Visualisasi Rata-rata Rating Produk (Top 10 Produk)

fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=avg_rating.values, y=avg_rating.index, palette='coolwarm', ax=ax) 
ax.set_title('Rata-rata Rating Produk (Top 10 Produk)')
ax.set_xlabel('Rata-rata Rating')
ax.set_ylabel('Nama Produk')
st.pyplot(fig)  # Menampilkan grafik di Streamlit

# ================= Segmentasi Customer =================

# Normalize the data for better clustering performance
scaler = StandardScaler()
df_encoded[['Age', 'Purchase_Amount']] = scaler.fit_transform(df_encoded[['Age', 'Purchase_Amount']])

# Inisialisasi dan fit KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
X = df_encoded[['Age', 'Purchase_Amount']]
kmeans.fit(X)

# Tambahkan hasil clustering ke dataframe
df_encoded['Cluster'] = kmeans.labels_

# Visualisasi Segmentasi Customer
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(
    x=df_encoded['Age'],
    y=df_encoded['Purchase_Amount'],
    hue=df_encoded['Cluster'],
    palette='Set2',
    s=60,
    ax=ax
)
ax.set_title('Segmentasi Customer Berdasarkan Umur dan Jumlah Belanja')
ax.set_xlabel('Umur (Normalized)')
ax.set_ylabel('Jumlah Belanja (Normalized)')
st.pyplot(fig)  # Menampilkan grafik di Streamlit


# ======================= Prediksi Repeat Customer =======================

st.subheader('Prediksi Pelanggan Repeat Customer')


# 2. Cek nama semua kolom
print("Kolom-kolom dalam dataset:", df.columns.tolist())

# 3. Asumsikan kolom target bernama 'Repeat_Customer'
target_column = "Repeat_Customer"  # Ganti jika nama kolom berbeda

# 4. Pisahkan fitur dan target
X = df.drop([target_column], axis=1)
y = df[target_column]

# 5. ANALISIS: Jumlah kategori unik per kolom kategorikal
print("\n[INFO] Jumlah kategori unik per kolom kategorikal:")
for col in X.select_dtypes(include=['object', 'category']).columns:
    n_unique = X[col].nunique()
    print(f"{col}: {n_unique} kategori unik")

# 6. Label Encoding kolom dengan kategori > 100
for col in X.select_dtypes(include=['object', 'category']).columns:
    if X[col].nunique() > 100:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        print(f"[ENCODE] Kolom '{col}' di-label encode karena punya >100 kategori.")
        
# 7. One-hot encoding untuk sisa fitur kategorikal
X = pd.get_dummies(X, drop_first=True)

# 8. Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Standarisasi fitur untuk Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

# 11. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 12. Evaluasi
print("\n[HASIL] Logistic Regression:\n", classification_report(y_test, y_pred_logreg))
print("\n[HASIL] Random Forest:\n", classification_report(y_test, y_pred_rf))