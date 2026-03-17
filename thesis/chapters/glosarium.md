# Panduan Tata Tulis dan Glosarium Skripsi

Dokumen ini berfungsi sebagai acuan standar penulisan istilah asing (bahasa Inggris) dan tata bahasa Indonesia baku untuk menjaga konsistensi naskah skripsi dari Bab I hingga Bab V.

---

## 📌 Aturan Emas Penulisan (EYD & Istilah Asing)

1. **Cetak Miring (*Italic*):** SEMUA istilah bahasa Inggris yang belum diserap secara resmi ke dalam bahasa Indonesia baku **wajib** dicetak miring.
2. **Aturan Imbuhan + Bahasa Asing:** Jika kata asing mendapat imbuhan bahasa Indonesia (awalan/akhiran/partikel), **wajib** dipisah dengan tanda hubung (`-`), dan HANYA kata asingnya yang dicetak miring.
   - *Benar:* di-*fine-tuning*, *input*-nya, men-*generate*, me-*resampling*
   - *Salah:* di*fine-tuning*, *inputnya*, di-*finetuning*
3. **Penamaan Arsitektur/Model:** Nama model, *dataset*, atau algoritma yang merupakan *proper noun* (kata benda khas/merek/entitas) **TIDAK PERLU** dicetak miring. Cukup gunakan huruf kapital di awal kata.
   - *Benar:* UrbanSound8K, PANNs, ResNet38, AudioSet, AdamW, PyTorch.
   - *Salah:* *UrbanSound8k*, panns, resnet38.

---

## 📖 Glosarium Istilah Skripsi

Gunakan ejaan dan format di bawah ini secara konsisten di seluruh naskah:

### 1. Ranah Data & Audio (Pemrosesan Sinyal)
- **Audio Mentah:** *raw waveform* (bukan *raw-waveform* atau *Raw Waveform* di tengah kalimat).
- **Frekuensi Pencuplikan:** *sampling rate*
- **Penggabungan Saluran:** *down-mixing* (pakai tanda hubung).
- **Fitur Spektrogram:** *log-mel spectrogram* (huruf kecil semua jika di tengah kalimat).
- **Pemotongan/Penambahan Durasi:** *random cropping*, *center cropping*, *zero-padding*, *truncating*.
- **Ekstraksi Fitur Buatan:** *hand-crafted features*.
- **MFCC:** *mel-frequency cepstral coefficients*.

### 2. Ranah Pembelajaran Mesin (*Machine Learning*)
- **Pembelajaran Mesin:** *machine learning*.
- **Pembelajaran Secara Transfer:** *transfer learning*.
- **Klasifikasi Suara Lingkungan:** *environmental sound classification* (ESC).
- **Algoritma Khusus:** *support vector machine* (SVM), *random forest* (RF).
- **Kumpulan Data:** *dataset* (kata ini sangat sering lupa dicetak miring, mohon diperhatikan).
- **Daftar Nama Dataset:** UrbanSound8K, AudioSet.
- **Pembagian Data:** *fold*, *k-fold cross validation*.
- **Pelatihan Ulang:** *transfer learning*, *fine-tuning*, *freeze base*.
- **Pelatihan dari Awal:** *training from scratch*.
- **Kondisi Pelatihan:** *overfitting*, *underfitting*, kebocoran data (*data leakage*).
- **Pengaturan Lanjut:** *hyperparameter*, *learning rate*, *batch size*, *epoch*.
- **Masukan:** *input*.
- **Kuat, Tangguh, Tahan Uji:** *robust*.

### 3. Ranah Arsitektur Jaringan (*Deep Learning*)
- **Pembelajaran Mendalam:** *deep learning*.
- **Jaringan Saraf:** *convolutional neural network* (CNN).
- **Lapisan:** lapisan ekstraksi fitur (*feature extraction layer*), lapisan klasifikasi (*classification layer*), *fully connected layer*.
- **Komponen Konvolusi:** *kernel*, *stride*, *node*, *neuron*, *filter*, bobot (*weight*).
- **Fungsi Aktivasi:** ReLU, *sigmoid*, *softmax*.
- **Fungsi Kerugian:** *loss function*, *cross-entropy loss*, *cost-sensitive learning*.
- **Sudah Pernah Dilatih:** *pre-trained*.
- **Arsitektur Pre-trained PANNs:** *Pre-trained Audio Neural Networks* (PANNs).

### 4. Ranah Evaluasi
- **Metrik Utama:** *F1-score* (huruf 'F' besar, 'score' kecil, pakai tanda hubung), *Accuracy*, *Precision*, *Recall*.
- **Matriks Uji:** *confusion matrix*.
- **Prediksi:** *True Positive* (TP), *True Negative* (TN), *False Positive* (FP), *False Negative* (FN).