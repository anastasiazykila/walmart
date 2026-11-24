# 🛒 Walmart Customer Analysis

**Portfolio Project — Data Analysis + Machine Learning**

Analisa data pembelian customer Walmart untuk menjawab 3 pertanyaan bisnis:
1. Produk apa yang paling disukai customer?
2. Customer bisa dibagi jadi berapa segmen?
3. Bisakah kita memprediksi siapa yang akan jadi Repeat Customer?

---

## 📊 Dataset

- **File:** `Walmart_customer_purchases.csv`
- **Sumber:** Dataset Walmart Customer Purchases (Kaggle / public dataset)
- **Kolom utama:** `Age`, `Gender`, `City`, `Product_Name`, `Category`, `Purchase_Amount`, `Purchase_Date`, `Payment_Method`, `Discount_Applied`, `Rating`, `Repeat_Customer`

---

## 🧹 Data Cleaning (Prepare Data)

Yang dilakukan sebelum analisa:
- Hapus duplikat dan nilai kosong (`drop_duplicates()` + `dropna()`)
- Konversi `Purchase_Date` ke datetime (`pd.to_datetime()`)
- Sederhanakan kategori: hanya Top 10 kategori yang dipertahankan, sisanya jadi `Other` — biar one-hot encoding tidak meledak jadi ratusan kolom
- Pisahkan target (`Repeat_Customer`) dari fitur sebelum encoding, agar tidak terjadi data leakage

---

## 🔍 Langkah Analisa (Data Analysis Lifecycle)

### 1. Ask Question
- Produk apa yang paling disukai?
- Bagaimana segmentasi customer-nya?
- Siapa yang akan jadi Repeat Customer?

### 2. Get Data
- Load CSV dengan Pandas

### 3. Investigate Data
- Cek `df.info()`, `df.describe()`, `df.isnull().sum()`, `df.duplicated().sum()`

### 4. Prepare Data
- Cleaning seperti di atas + StandardScaler untuk clustering & modeling

### 5. Analyze Data
- **EDA:** Rata-rata rating per produk (Top 10) — visualisasi dengan Seaborn
- **Segmentasi:** KMeans (4 cluster) berdasarkan `Age` + `Purchase_Amount`
- **Prediksi:** Logistic Regression vs Random Forest untuk memprediksi `Repeat_Customer`

### 6. Present Results
- Dashboard Streamlit dengan grafik + metrik akurasi + insight bisnis

---

## 📈 Hasil & Insight

### EDA — Rating Produk
- Produk dengan rating tertinggi bukan selalu produk yang paling banyak dibeli
- Artinya ada produk yang laku tapi kualitasnya masih bisa diperbaiki

### Segmentasi — KMeans (4 Cluster)
- Cluster dibedakan berdasarkan umur dan jumlah belanja
- Contoh penggunaan: 
  - Cluster umur muda + belanja tinggi → target promo premium
  - Cluster umur tua + belanja rendah → butuh promo diskon, bukan premium
- Lihat grafik segmentasi di dashboard untuk detail tiap cluster

### Prediksi — Repeat Customer
- Model dibandingkan dengan `classification_report` dan akurasi
- Model yang lebih baik dipilih berdasarkan akurasi di data uji (lihat di dashboard)
- **Manfaat bisnis:** 
  - Prediksi siapa yang tidak akan balik → kasih promo khusus biar balik
  - Prediksi siapa yang akan balik → kasih reward biar makin loyal
  - Hemat budget marketing karena promo lebih tertarget

---

## 🛠️ Tools

- Python
- Pandas (cleaning + analysis)
- Seaborn + Matplotlib (visualisasi)
- Scikit-Learn (KMeans, Logistic Regression, Random Forest, StandardScaler)
- Streamlit (dashboard)

---

## ▶️ Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run walmart.py
```

Buka di browser: `http://localhost:8501`

---

## 📁 Struktur File

```
walmart/
├── walmart.py                              # Code utama (Streamlit app)
├── Walmart_customer_purchases.csv          # Dataset (taruh di folder yang sama)
├── requirements.txt                        # Daftar library
└── README.md                               # File ini
```

### requirements.txt
```
streamlit
pandas
seaborn
matplotlib
scikit-learn
```

---

## 🔧 Perbaikan dari Versi Sebelumnya

Versi ini memperbaiki 4 bug dari versi awal:
1. **Path hardcoded** `C:/Users/Ani/...` → diganti path relatif, jadi bisa di-run orang lain
2. **Data mentah dipakai untuk modeling** (`df` bukan `df_cleaned`) → sekarang pakai `df_cleaned` yang sudah dibersihkan
3. **Double scaling** (MinMax + StandardScaler untuk kolom yang sama) → sekarang hanya StandardScaler untuk KMeans
4. **Target ikut di-one-hot** → sekarang target dipisahkan dulu sebelum encoding, agar tidak leakage
5. **Hasil model hanya di-print di terminal** → sekarang ditampilkan di Streamlit dengan `st.metric` + `st.text`

---

Dibuat oleh Anastasia — Portfolio Data Analysis
