# 🎬 MOVREC — Rencana Implementasi

## Deskripsi Proyek

MovRec adalah sistem rekomendasi film Netflix berbasis Machine Learning menggunakan metode Content-Based Filtering.

Sistem akan merekomendasikan film berdasarkan:

* Genre
* Deskripsi film
* Mood pengguna
* Kemiripan konten film

Aplikasi dibuat dalam bentuk website interaktif menggunakan Streamlit.

---

# 📌 Tujuan Proyek

* Membangun sistem rekomendasi film berbasis Machine Learning
* Mengimplementasikan metode TF-IDF dan Cosine Similarity
* Membantu pengguna menemukan film sesuai preferensi
* Membuat antarmuka modern dan interaktif
* Melakukan deployment aplikasi secara online

---

# 🛠️ Teknologi yang Digunakan

| Teknologi         | Fungsi                    |
| ----------------- | ------------------------- |
| Python            | Bahasa pemrograman utama  |
| Pandas            | Pengolahan data           |
| Scikit-learn      | Machine Learning          |
| TF-IDF Vectorizer | Ekstraksi fitur teks      |
| Cosine Similarity | Menghitung kemiripan film |
| Streamlit         | Web aplikasi              |
| Joblib            | Menyimpan model           |
| Matplotlib        | Visualisasi data          |
| Seaborn           | Analisis visual           |
| TMDB API          | Mengambil poster film     |
| GitHub            | Version control           |
| Streamlit Cloud   | Deployment aplikasi       |

---

# 📂 Struktur Project

```bash id="wubc83"
movrec-app/
│
├── app.py
├── requirements.txt
├── implementation_plan.md
├── README.md
│
├── movies.pkl
├── tfidf.pkl
├── tfidf_matrix.pkl
│
└── notebook.ipynb
```

---

# 🚀 Tahapan Implementasi

## 1. Pengumpulan Dataset

### Kegiatan

* Mengambil dataset Netflix
* Memahami struktur dataset
* Menentukan fitur penting

### Hasil

* Dataset siap diproses

---

## 2. Preprocessing Data

### Kegiatan

* Membersihkan missing value
* Menggabungkan fitur penting
* Membersihkan teks
* Feature engineering

### Hasil

* Data teks siap digunakan

---

## 3. Pembuatan Sistem Rekomendasi

### Kegiatan

* Menggunakan TF-IDF Vectorizer
* Membuat TF-IDF Matrix
* Menghitung Cosine Similarity
* Membuat fungsi rekomendasi film

### Hasil

* Sistem rekomendasi berbasis konten

---

## 4. Penyimpanan Model

### Kegiatan

* Menyimpan dataframe
* Menyimpan model TF-IDF
* Menyimpan matrix similarity

### Hasil

* File model `.pkl`

---

## 5. Pengembangan Website Streamlit

### Kegiatan

* Membuat tampilan UI modern
* Membuat fitur pencarian film
* Membuat fitur rekomendasi berdasarkan mood
* Membuat dashboard analytics
* Menambahkan poster film

### Hasil

* Website rekomendasi film interaktif

---

## 6. Pengujian Sistem

### Kegiatan

* Menguji hasil rekomendasi
* Menguji tampilan website
* Menguji deployment aplikasi

### Hasil

* Sistem berjalan dengan baik

---

## 7. Deployment

### Kegiatan

* Upload project ke GitHub
* Membuat requirements.txt
* Deploy menggunakan Streamlit Cloud

### Hasil

* Aplikasi dapat diakses online

---

# 📊 Metode Rekomendasi

## Content-Based Filtering

Metode ini bekerja dengan:

1. Menggabungkan informasi film
2. Mengubah teks menjadi vector menggunakan TF-IDF
3. Menghitung kemiripan menggunakan Cosine Similarity
4. Menampilkan film yang paling mirip

---

# ✨ Fitur Aplikasi

* 🔍 Cari film sejenis
* 😊 Rekomendasi berdasarkan mood
* 🎲 Surprise Me
* 📊 Dashboard analytics
* 🎬 Poster film otomatis
* 🌙 Tampilan cinematic dark mode

---

# ✅ Hasil yang Diharapkan

Aplikasi mampu:

* Memberikan rekomendasi film yang relevan
* Menampilkan UI yang menarik
* Berjalan secara online
* Membantu pengguna menemukan film dengan mudah

---

# 🔮 Pengembangan Selanjutnya

* Sistem login pengguna
* Hybrid recommendation system
* Personalisasi berdasarkan histori tontonan
* Integrasi trending movie realtime
* Deep learning recommendation system
