# BearingCare - Diagnosis Kerusakan Bearing

Proyek Machine Learning MKB UPI · Kasus 04

**Aplikasi:** https://bearingcare-kasus04.streamlit.app/  
**Repository:** https://github.com/zaldi15/bearingcare-ml-kasus04  
**Video demo:** 

## Anggota kelompok

| Nama | NIM | Kontribusi |
|---|---|---|
| [Zaldi Arifa] | [2406326] | [] |
| [Nama anggota 2] | [NIM] | [Bagian yang dikerjakan] |


## 1. Masalah dan tujuan

Analis getaran di pabrik semen harus memeriksa banyak rekaman dari bearing motor dan pompa. Pemeriksaan manual memerlukan waktu dan keahlian membaca karakteristik getaran. Proyek ini mengembangkan model klasifikasi untuk membantu teknisi memperkirakan kondisi bearing dari ringkasan satu rekaman getaran.

Empat kelas yang diprediksi adalah:

- `normal` — tidak terindikasi kerusakan berdasarkan pola data model;
- `outer_race` — indikasi kerusakan cincin luar;
- `inner_race` — indikasi kerusakan cincin dalam;
- `bola` — indikasi kerusakan elemen bola.

Aplikasi menampilkan hasil klasifikasi, probabilitas masing-masing kelas, dan saran persiapan suku cadang. Prediksi digunakan sebagai **pendukung inspeksi**, bukan pengganti pemeriksaan teknisi.

## 2. Dataset dan pembersihan

Dataset berasal dari berkas `raw.csv` yang disediakan untuk Kasus 04 pada [repositori tugas MKB](https://github.com/nurhikam/mkb-ml-project). Data bersifat **sintetis**. Setiap baris merepresentasikan ringkasan rekaman getaran selama satu detik.

- Data awal: **15.220 baris, 14 kolom**.
- Data setelah penghapusan duplikat identik: **15.000 rekaman**.
- Masukan model: **12 fitur asli** (11 numerik dan satu kategori `posisi_sensor`).
- Target: `kondisi`. Kolom `id_rekaman` tidak digunakan untuk prediksi.

Fitur mencakup posisi sensor, RPM, beban, suhu housing, RMS dan puncak getaran, kurtosis, crest factor, energi BPFO/BPFI/BSF, serta frekuensi dominan.

| Permasalahan pada data mentah | Jumlah | Penanganan |
|---|---:|---|
| Baris duplikat identik | 220 | Dihapus sebelum pembagian data |
| Variasi penulisan label target | 16 variasi | Diseragamkan menjadi empat kelas |
| `suhu_housing_C` kosong | 153 | Imputasi median dari data latih |
| `kurtosis` kosong | 456 | Imputasi median dari data latih |
| `kecepatan_rpm` bernilai nol | 61 | Ditandai sebagai missing, lalu diimputasi |
| `rms_g` bernilai negatif | 45 | Ditandai sebagai missing, lalu diimputasi |

Imputer dan transformasi yang mempelajari parameter dari data hanya di-*fit* pada data latih untuk menghindari *data leakage*.

Setelah pembersihan, distribusi kelas adalah **8.219 normal**, **3.039 outer_race**, **2.178 inner_race**, dan **1.564 bola**. Karena proporsi kelas tidak sama, metrik utama yang digunakan adalah **macro F1-score**.

## 3. Metode pemodelan

Data dibagi secara **stratified** menjadi 70% train (10.500), 15% validation (2.250), dan 15% test (2.250), dengan `random_state=42`.

Model yang dibandingkan:

1. **Logistic Regression** sebagai baseline wajib, menggunakan fitur asli dengan imputasi, encoding kategori, dan standardisasi.
2. **Random Forest**, menggunakan fitur asli dan fitur turunan.
3. **Extra Trees**, menggunakan fitur asli dan fitur turunan.

Fitur turunan yang dicoba mencakup total energi BPFO/BPFI/BSF, rasio masing-masing energi terhadap total energi, frekuensi dominan relatif terhadap RPM, dan rasio puncak terhadap RMS.

Model dipilih berdasarkan **macro F1 pada validation**. Model terpilih kemudian dilatih ulang pada gabungan train dan validation, sebelum dievaluasi pada test. Data test tidak digunakan untuk memilih model.

## 4. Hasil eksperimen

| Model | Macro F1 validation | Macro F1 test |
|---|---:|---:|
| Logistic Regression (baseline) | 0,6894 | 0,6898 |
| **Random Forest (final)** | **0,7984** | **0,7972** |
| Extra Trees | 0,7855 | Tidak dievaluasi pada test |

Akurasi Random Forest pada test adalah **0,8369**. Kenaikan macro F1 terhadap baseline pada test sekitar **0,1073 poin**.

F1 tiap kelas pada test:

| Kelas | F1-score |
|---|---:|
| `normal` | 0,8712 |
| `outer_race` | 0,8279 |
| `inner_race` | 0,7879 |
| `bola` | 0,7018 |

Kelas `bola` masih menjadi tantangan utama. Recall-nya sekitar **0,5983**, dan **83 dari 234** rekaman `bola` pada test diprediksi sebagai `normal`. Hasil ini menunjukkan bahwa model belum cukup andal untuk digunakan sebagai satu-satunya dasar keputusan perawatan.

Prediksi multikelas menggunakan kelas dengan probabilitas terbesar (`argmax`), sehingga tidak menerapkan threshold biner.

## 5. Aplikasi web

**[Buka BearingCare](https://bearingcare-kasus04.streamlit.app/)**

Aplikasi dibuat dengan **Streamlit** dan memuat model tersimpan secara langsung; **tidak ada API FastAPI terpisah pada deployment saat ini**.

Alur aplikasi:

```text
Input teknisi / CSV
        ↓
Validasi dan persiapan fitur (src/features.py)
        ↓
Model Random Forest (artifacts/model.joblib)
        ↓
Kelas prediksi + probabilitas + saran suku cadang
```

Fitur aplikasi:

- **Satu rekaman:** mengisi parameter getaran dan melihat hasil diagnosis.
- **Prediksi CSV:** mengunggah beberapa rekaman sekaligus (maksimum 2.000 baris per unggahan), meninjau hasil, dan mengunduh CSV hasil prediksi.
- **Validasi masukan:** nilai tertentu dibatasi sesuai domainnya; contoh, getaran puncak lebih kecil daripada RMS ditolak dalam formulir.

## 6. Struktur repositori

Struktur aplikasi yang sudah digunakan:

```text
bearingcare-ml-kasus04/
├── artifacts/
│   ├── model.joblib
│   └── metadata.json
├── src/
│   └── features.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## 7. Menjalankan secara lokal

Gunakan Python 3.12 dengan struktur folder seperti di atas. Dari direktori utama proyek:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser, biasanya pada `http://localhost:8501`. Model dan metadata harus sudah berada di folder `artifacts/`.

Untuk mengulangi eksperimen, gunakan notebook praktikum bersama dataset `raw.csv`. Prosesnya mencakup **Load Data → EDA → Data Cleaning → Split Data → Baseline → Model Comparison → Evaluation → Save Model**.

## 8. Keterbatasan dan pengembangan

- Dataset masih sintetis dan belum diuji pada data getaran aktual di pabrik semen.
- Model belum mempertimbangkan perubahan karakteristik antar-mesin, antar-sensor, atau kerusakan gabungan.
- Probabilitas model belum dikalibrasi sebagai ukuran risiko yang dapat diandalkan.
- Kesalahan terbesar perlu ditindaklanjuti pada kelas `bola`, terutama kasus yang diprediksi `normal`.
- Saran suku cadang bersifat umum; ukuran dan spesifikasi komponen tetap harus diperiksa oleh teknisi.

Pengembangan berikutnya dapat menggunakan data lapangan, validasi antar-mesin, evaluasi kalibrasi, dan pemantauan penurunan kinerja model dari waktu ke waktu.

## Referensi dan alat bantu

- [Instruksi proyek Machine Learning MKB UPI](https://github.com/nurhikam/mkb-ml-project)
- [Aplikasi BearingCare](https://bearingcare-kasus04.streamlit.app/)

Asisten AI digunakan untuk membantu penyusunan kode awal, analisis, dan dokumentasi. Pengujian, interpretasi hasil, serta penjelasan setiap bagian proyek tetap menjadi tanggung jawab anggota kelompok.
