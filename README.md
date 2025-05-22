# Laporan Proyek Machine Learning - Satriatama Putra

## Daftar Isi

- [Domain Proyek](#domain-proyek)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Referensi](#referensi)


## Domain Proyek

Pada proyek ini akan dibahas mengenai permasalahan dalam bidang peternakan dan kesehatan hewan, khususnya untuk memprediksi penyakit pada hewan ternak berdasarkan gejala-gejala yang ditunjukkan. Dengan kemajuan teknologi dan penerapan machine learning, diagnosis dini terhadap penyakit hewan ternak dapat dilakukan dengan lebih cepat dan akurat, sehingga penanganan dapat dilakukan lebih efektif.

![lumpy disease](https://github.com/user-attachments/assets/9e7ad02f-a368-42c8-af24-7bddb90e3b5d)  
**Gambar 1. Lumpy Skin Disease**

Kesehatan hewan ternak merupakan aspek krusial dalam industri peternakan. Penyakit pada hewan ternak dapat menyebabkan kerugian ekonomi yang signifikan, mulai dari penurunan produktivitas hingga kematian ternak yang berdampak langsung pada pendapatan peternak. Selain itu, beberapa penyakit pada hewan ternak berpotensi menjadi zoonosis (penyakit yang dapat ditularkan dari hewan ke manusia), yang dapat mengancam kesehatan masyarakat secara luas [1] [2].

Deteksi dini dan diagnosis yang akurat sangat penting untuk mengatasi penyakit pada hewan ternak. Veteriner dan peternak harus mampu mengidentifikasi gejala-gejala spesifik dan mengaitkannya dengan penyakit tertentu untuk memberikan penanganan yang tepat [3] [4]. Namun, proses diagnosis manual membutuhkan keahlian khusus dan pengalaman yang mungkin tidak dimiliki oleh semua peternak, terutama di daerah dengan keterbatasan akses ke layanan kesehatan hewan.

Penerapan machine learning dalam prediksi penyakit hewan ternak menawarkan solusi yang efisien dan dapat diakses secara luas. Dengan menganalisis data historis tentang gejala dan diagnosa, model machine learning dapat membantu mengidentifikasi pola yang mungkin tidak terlihat oleh manusia, serta memberikan prediksi yang akurat tentang kemungkinan penyakit berdasarkan gejala yang diamati.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, maka diperoleh rumusan masalah pada proyek ini, yaitu:
1. Bagaimana cara mengidentifikasi penyakit pada hewan ternak berdasarkan gejala yang diamati dengan akurasi tinggi?
2. Apa saja faktor atau gejala yang paling berpengaruh dalam prediksi penyakit hewan ternak?
3. Bagaimana membangun model machine learning yang dapat membedakan antara penyakit dengan gejala yang mirip seperti lumpy virus dan pneumonia?

### Goals

Berdasarkan rumusan masalah di atas, maka diperoleh tujuan dari proyek ini, yaitu:
1. Mengembangkan model machine learning yang dapat memprediksi jenis penyakit pada hewan ternak berdasarkan gejala yang ditunjukkan dengan akurasi tinggi.
2. Mengidentifikasi faktor-faktor kunci atau gejala yang memiliki korelasi tinggi dengan diagnosis penyakit tertentu.
3. Meningkatkan kemampuan diferensiasi model dalam membedakan penyakit dengan gejala yang mirip, khususnya lumpy virus dan pneumonia.

### Solution Statements

Berdasarkan rumusan masalah dan tujuan di atas, maka disimpulkan beberapa solusi yang dapat dilakukan untuk mencapai tujuan dari proyek ini, yaitu:
1. Melakukan analisis eksploratif data untuk memahami distribusi gejala pada setiap penyakit, serta korelasi antara gejala dan diagnosis.
2. Mengembangkan model prediksi penyakit menggunakan dua algoritma machine learning yang berbeda:
   - **XGBoost (eXtreme Gradient Boosting)**: Algoritma berbasis gradient boosting yang dapat menangani data kategorikal dengan baik dan mencegah overfitting melalui regularisasi.
   - **Random Forest**: Metode ensemble learning berbasis pohon keputusan yang robust terhadap outlier dan dapat memberikan informasi tentang kepentingan fitur.
3. Menerapkan teknik persiapan data (feature engineering dan feature encoding) untuk meningkatkan performa model, termasuk:
   - Pengelompokan gejala menjadi sindrom untuk menangkap pola gejala yang lebih komprehensif
   - Encoding fitur kategorikal menggunakan teknik one-hot encoding
   - Seleksi fitur berdasarkan mutual information dan chi-square test
   - Standarisasi fitur numerik untuk memastikan semua fitur berkontribusi secara setara

## Data Understanding
![dataset](https://github.com/user-attachments/assets/5e655ea9-a0ab-4ea6-b190-35f0ae68dfa8)
**Gambar 2. Preview Dataset Penyakit Hewan Ternak**

Dataset yang digunakan dalam proyek ini adalah ["Livestock Symptoms and Diseases"](https://www.kaggle.com/datasets/researcher1548/livestock-symptoms-and-diseases) yang diambil dari platform Kaggle. Dataset ini berisi informasi tentang hewan ternak, gejala-gejala yang ditunjukkan, dan diagnosis penyakit.

### Informasi Dataset
- Jumlah sampel: 43,778 entri
- Jumlah fitur: 7 kolom
- Format: CSV (animal_disease_dataset.csv)

### Variabel-variabel pada dataset:
1. **Animal**: Jenis hewan ternak (buffalo, cow, goat, sheep)
2. **Age**: Usia hewan ternak dalam tahun
3. **Temperature**: Suhu tubuh hewan ternak dalam derajat Fahrenheit
4. **Symptom_1, Symptom_2, Symptom_3**: Tiga gejala utama yang ditunjukkan oleh hewan
5. **Disease**: Diagnosis penyakit (target variabel) dengan 5 kelas: anthrax, blackleg, foot and mouth disease, lumpy virus, dan pneumonia

### Exploratory Data Analysis (EDA)

#### 1. Deskripsi Statistik Variabel

**Tabel 1. Deskripsi Statistik Variabel Numerik**
| Statistik | Age            | Temperature    |
|-----------|----------------|----------------|
| count     | 43,778         | 43,778         |
| mean      | 7.01           | 102.30         |
| std       | 3.74           | 1.31           |
| min       | 1.00           | 100.0          |
| 25%       | 4.00           | 101.2          |
| 50%       | 7.00           | 102.3          |
| 75%       | 10.00          | 103.4          |
| max       | 15.00          | 105.0          |

Berdasarkan statistik deskriptif, usia hewan berkisar antara 1-15 tahun dengan rata-rata 7 tahun, dan suhu tubuh berkisar antara 100-105°F dengan rata-rata 102.3°F.

#### 2. Distribusi Variabel Kategorikal
![distribusi_kategorikal](https://github.com/user-attachments/assets/3fdac933-bbe3-40e4-a70f-f37fc817df0a)
**Gambar 3. Distribusi Variabel Kategorikal**

- **Animal**: Distribusi cukup seimbang antara empat jenis hewan (buffalo, cow, goat, sheep) dengan proporsi masing-masing sekitar 25%.
- **Disease**: Distribusi juga relatif seimbang dengan anthrax, blackleg, dan foot and mouth disease masing-masing sekitar 22%, pneumonia 17%, dan lumpy virus 16%.
- **Symptoms**: Gejala paling umum adalah "loss of appetite" (23%), "depression" (15%), dan "painless lumps" (11%).

#### 3. Analisis Hubungan antar Variabel
![distribusi_gejala_penyakit](https://github.com/user-attachments/assets/8fbfafb8-ac56-4c12-ba10-80f4542af818)  
**Gambar 4. Distribusi Gejala berdasarkan Penyakit**

- **Anthrax**: Ditandai dengan gejala pernapasan (shortness of breath, chest discomfort) dan sistemik (fatigue, sweats, chills).
- **Blackleg**: Memiliki ciri khas "crackling sound" yang tidak muncul di penyakit lainnya.
- **Foot and mouth disease**: Ditandai dengan gangguan mobilitas (difficulty walking, lameness) dan lesi pada mulut dan kaki.
- **Lumpy virus dan Pneumonia**: Memiliki profil gejala yang hampir identik (painless lumps, depression, loss of appetite), sehingga sulit dibedakan hanya dari gejala klinis.
  
![distribusi_umur_temp_penyakit](https://github.com/user-attachments/assets/80de816d-6494-4258-af44-cda07b16a903)
**Gambar 5. Distribusi Umur dan Temperatur berdasarkan Penyakit**

- Foot and mouth disease dan anthrax cenderung menyerang hewan dengan usia lebih muda (sekitar 6 tahun).
- Pneumonia dan lumpy virus menunjukkan suhu tubuh median yang sedikit lebih tinggi dibandingkan penyakit lain.
- Buffalo dan sapi secara konsisten lebih tua (7.8-8.1 tahun) dibandingkan kambing dan domba (5.4-5.6 tahun) terlepas dari jenis penyakitnya.

#### 4. Analisis 

![analisis_korelasi](https://github.com/user-attachments/assets/d54795e1-29aa-46ab-8813-c4352db21f57)
**Gambar 6. Korelasi Fitur dengan Penyakit**

- Gejala "painless lumps" memiliki korelasi tertinggi (+0.8) dengan diagnosis penyakit tertentu (kemungkinan lumpy virus).
- "Loss of appetite" (+0.7) merupakan indikator kuat kedua untuk diagnosis penyakit.
- Gejala pernapasan dan sistemik menunjukkan korelasi negatif (-0.3), yang berarti keberadaannya cenderung mengindikasikan penyakit spesifik (kemungkinan anthrax).
- Umur dan temperatur menunjukkan korelasi minimal dengan diagnosis, mengindikasikan bahwa faktor ini kurang signifikan dibanding gejala klinis.

## Data Preparation

Pada tahap persiapan data atau data preparation dilakukan beberapa proses untuk memastikan data dalam kondisi optimal untuk digunakan dalam pemodelan machine learning.

### 1. Feature Encoding

#### Label Encoding untuk Target Variable

Label encoding diterapkan pada variabel target (Disease) untuk mengubah kategori penyakit menjadi nilai numerik.

```
anthrax: 0
blackleg: 1
foot and mouth: 2
lumpy virus: 3
pneumonia: 4
```

#### One-Hot Encoding untuk Variabel Animal

Variabel kategorikal "Animal" diubah menjadi representasi biner menggunakan one-hot encoding, menghasilkan empat kolom baru: Animal_buffalo, Animal_cow, Animal_goat, dan Animal_sheep.

#### Encoding untuk Fitur Symptoms

Gejala-gejala diubah menjadi format biner (has_symptom) untuk menandai keberadaan gejala tertentu pada setiap hewan.

```python
for symptom in all_symptoms:
    feature_name = f"has_{symptom.replace(' ', '_')}"
    symptom_features[feature_name] = df[symptom_cols].apply(
        lambda row: int(symptom in row.values), axis=1
    )
```

### 2. Feature Engineering

Pengelompokan gejala menjadi sindrom untuk menangkap pola gejala yang lebih komprehensif:

1. **Respiratory Syndrome**: Shortness of breath, chest discomfort, coughing, rapid breathing
2. **Systemic Syndrome**: Fatigue, sweats, chills, fever, depression, loss of appetite
3. **Foot Mouth Syndrome**: Difficulty walking, lameness, blisters, sores, drooling
4. **Lumps Syndrome**: Painless lumps, painful lumps, swelling in neck/extremities/abdomen

![distribusi_sindrom](https://github.com/user-attachments/assets/2f71aa1d-45e3-4936-8d90-c2c67c5b024e)
**Gambar 7. Distribusi Sindrom berdasarkan Penyakit**

### 3. Feature Selection

Seleksi fitur dilakukan untuk mengidentifikasi variabel yang paling relevan dalam memprediksi penyakit:

1. **Mutual Information Score**: Mengukur dependensi antara fitur dan target
2. **Chi-Square Test**: Mengukur independensi antara fitur kategorikal dan target

![MI](https://github.com/user-attachments/assets/f2fd48e6-e6eb-4fa9-87b4-e1edd2aacef2)
**Gambar 8. Top 20 Fitur berdasarkan Mutual Information Score**

Berdasarkan hasil analisis, 31 fitur terpilih sebagai input untuk model, termasuk:
- Fitur syndrome yang dibuat (respiratory, systemic, foot_mouth, lumps)
- Fitur demografis (Age, Temperature, jenis hewan)
- Gejala spesifik dengan skor MI tertinggi (has_painless_lumps, has_depression, dll)

### 4. Train-Test Split

Dataset dibagi menjadi data latih (75%) dan data uji (25%) dengan stratifikasi berdasarkan target untuk memastikan distribusi kelas yang seimbang.

```
X_train shape: (32833, 31)
X_test shape: (10945, 31)
```

### 5. Feature Scaling

Standarisasi diterapkan pada fitur numerik (Age dan Temperature) agar semua fitur berkontribusi secara setara pada model.

```python
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
```

### 6. Class Balancing

Analisis distribusi kelas menunjukkan dataset yang relatif seimbang dengan rasio ketidakseimbangan (max/min) sebesar 1.35, sehingga tidak diperlukan teknik resampling.

## Modeling

Dua model machine learning dikembangkan untuk memprediksi penyakit pada hewan ternak berdasarkan gejala-gejala yang ditunjukkan:

### 1. XGBoost (eXtreme Gradient Boosting)

XGBoost adalah implementasi dari gradient boosted decision trees yang dirancang untuk kecepatan dan performa tinggi.

**Konfigurasi Model:**
```python
xgb_model = XGBClassifier(
    objective='multi:softprob',
    n_estimators=100,          
    learning_rate=0.1,         
    max_depth=5,               
    min_child_weight=1,        
    gamma=0,                   
    subsample=0.8,             
    colsample_bytree=0.8,      
    random_state=42            
)
```

**Kelebihan XGBoost:**
- Dapat menangani fitur kategorikal dengan baik
- Bekerja efektif dengan data yang tidak seimbang
- Menangkap pola kompleks dalam data
- Mencegah overfitting melalui regularisasi

### 2. Random Forest

Random Forest adalah metode ensemble learning yang membangun beberapa decision tree dan menggabungkan prediksinya.

**Konfigurasi Model:**
```python
rf_model = RandomForestClassifier(
    n_estimators=100,      
    max_depth=None,        
    min_samples_split=2,   
    min_samples_leaf=1,    
    max_features='sqrt',   
    bootstrap=True,        
    random_state=42,       
    n_jobs=-1              
)
```

**Kelebihan Random Forest:**
- Menangani fitur kategorikal dengan baik
- Tahan terhadap overfitting melalui bagging
- Menyediakan ukuran kepentingan fitur
- Berkinerja baik dengan campuran jenis fitur

## Evaluation

Evaluasi model dilakukan menggunakan beberapa metrik untuk mendapatkan gambaran komprehensif tentang performa model:

### Metrik Evaluasi

1. **Accuracy**: Proporsi prediksi yang benar dari keseluruhan data
2. **Precision**: Proporsi prediksi positif yang benar 
3. **Recall**: Proporsi kasus positif yang berhasil diprediksi
4. **F1 Score**: Rata-rata harmonik dari precision dan recall
5. **ROC AUC**: Area di bawah kurva ROC, mengukur kemampuan model membedakan kelas

### Hasil Evaluasi Model

**Tabel 2. Perbandingan Performa Model**
|             | XGBoost | Random Forest |
|-------------|---------|---------------|
| Accuracy    | 0.829   | 0.815         |
| Precision   | 0.828   | 0.815         |
| Recall      | 0.829   | 0.815         |
| F1 Score    | 0.827   | 0.815         |
| ROC AUC     | 0.957   | 0.953         |

![comp_model](https://github.com/user-attachments/assets/2d36f94d-2bc4-4435-b751-e5264c21d1fd)

**Gambar 9. Perbandingan Performa Model**

**Tabel 3. Performa Per-Kelas**
| Disease          | XGBoost Precision | RF Precision | XGBoost Recall | RF Recall | XGBoost F1 | RF F1   |
|------------------|-------------------|--------------|----------------|-----------|------------|---------|
| anthrax          | 1.000             | 1.000        | 1.000          | 1.000     | 1.000      | 1.000   |
| blackleg         | 1.000             | 1.000        | 1.000          | 1.000     | 1.000      | 1.000   |
| foot and mouth   | 1.000             | 1.000        | 1.000          | 1.000     | 1.000      | 1.000   |
| lumpy virus      | 0.474             | 0.433        | 0.404          | 0.413     | 0.436      | 0.423   |
| pneumonia        | 0.490             | 0.449        | 0.561          | 0.469     | 0.523      | 0.459   |

![f1_disease](https://github.com/user-attachments/assets/8fa486cd-840f-4c43-820d-b7eeaabea110)

**Gambar 10. Perbandingan F1 Score berdasarkan Penyakit**

### Confusion Matrix

![conf_mx_xg](https://github.com/user-attachments/assets/f2916f3f-69c9-40cd-9c52-c4a52de7f22b)

**Gambar 11. Confusion Matrix Model XGBoost**

![conf_mx_rf](https://github.com/user-attachments/assets/275fd230-30ad-483c-8f80-897eaaa61821)

**Gambar 12. Confusion Matrix Model Random Forest**

### Analisis Hasil

Berdasarkan hasil evaluasi, dapat disimpulkan bahwa:

1. **XGBoost unggul dengan akurasi 82.9%** dibandingkan Random Forest (81.5%), menunjukkan performa yang lebih baik secara keseluruhan.

2. **Kedua model berhasil memprediksi dengan sempurna (F1=1.0)** untuk tiga penyakit:
   - Anthrax
   - Blackleg
   - Foot and mouth disease

3. **Tantangan utama ada pada prediksi** dua penyakit:
   - Lumpy virus (XGBoost F1: 0.436, RF F1: 0.423)
   - Pneumonia (XGBoost F1: 0.523, RF F1: 0.459)

4. **XGBoost secara signifikan lebih baik dalam mendeteksi pneumonia** (F1: 0.523 vs 0.459) dengan recall yang lebih tinggi (56.1% vs 46.9%).

5. **Kedua model mengalami kesulitan membedakan lumpy virus dan pneumonia** karena keduanya memiliki profil gejala yang hampir identik.

### Kesimpulan

XGBoost adalah model superior untuk prediksi penyakit pada ternak, menunjukkan performa yang lebih baik pada semua metrik. Model ini khususnya unggul dalam:

1. Akurasi keseluruhan yang lebih tinggi dan performa yang seimbang (presisi/recall)
2. Kemampuan deteksi pneumonia yang jauh lebih baik
3. Performa yang sedikit lebih baik dalam mendeteksi lumpy virus
4. Performa sempurna pada tiga penyakit yang mudah dibedakan

Faktor pembeda utama adalah kemampuan superior XGBoost dalam membedakan kasus-kasus yang menantang antara lumpy virus dan pneumonia, yang memiliki gejala serupa.

## Referensi

[1] H. M. Prosser *et al.*, "[Application of artificial intelligence and machine learning in bovine respiratory disease prevention, diagnosis, and classification](https://avmajournals.avma.org/view/journals/ajvr/86/S1/ajvr.24.10.0327.xml)," *American Journal of Veterinary Research*, vol. 86, no. S1, pp. S22–S26, Feb. 2025. doi: 10.2460/ajvr.24.10.0327.

[2]  Food and Agriculture Organization of the United Nations, World Health Organization, and World Organisation for Animal Health, "([https://iris.who.int/bitstream/handle/10665/325620/9789241514934-eng.pdf](https://iris.who.int/bitstream/handle/10665/325620/9789241514934-eng.pdf))". Geneva: WHO, 2019.

[3]  Food and Agriculture Organization of the United Nations, "[Zoonoses](https://www.fao.org/one-health/areas-of-work/zoonoses/en)," *FAO*. [Online]. Available: [https://www.fao.org/one-health/areas-of-work/zoonoses/en](https://www.fao.org/one-health/areas-of-work/zoonoses/en), Accessed: May 22, 2025.

[4]  M. Imani, A. Beikmohammadi, and H. R. Arabnia, "([https://www.mdpi.com/2227-7080/13/3/88](https://www.mdpi.com/2227-7080/13/3/88))," *Technologies*, vol. 13, no. 3, p. 88, Feb. 2025. doi: 10.3390/technologies13030088.

[5]  World Organisation for Animal Health, "[Controlling the cross-border spread of livestock diseases](https://www.woah.org/en/article/controlling-the-cross-border-spread-of-livestock-diseases/)," *WOAH*, Mar. 2025. [Online]. Available: [https://www.woah.org/en/article/controlling-the-cross-border-spread-of-livestock-diseases/](https://www.woah.org/en/article/controlling-the-cross-border-spread-of-livestock-diseases/), Accessed: May 22, 2025.
