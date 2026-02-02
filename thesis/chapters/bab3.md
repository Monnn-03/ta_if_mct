# BAB III : Metode Penelitian

## Alur Penelitian
[Gambar Alur Penelitian]
1. Identifikasi Domain Penelitian
- Tahap awal penelitian dilakukan dengan mencari referensi domain penelitian untuk klasifikasi suara. Pencarian ini dilakukan berdasarkan beberapa faktor prioritas yang menjadi pertimbangan pemilihan domain, salah satunya adalah ketersediaan dataset dan model klasifikasi. Domain yang dipilih adalah domain keselamatan publik untuk penyandang Tuna Rungu.
2. Melakukan Studi Literatur
- Dalam menemukan urgensi penelitian, dilakukan studi literatur mengenai kebutuhan alat klasifikasi suara yang unggul dalam mengurangi risiko kecelakaan penyandang Tuna Rungu. 
3. Mencari Dataset
- Ketersediaan dataset menjadi kunci dalam pelatihan model klasifikasi. Maka dari itu, pengambilan dataset dilakukan secara sekunder (dataset publik) dan ditemukan dataset UrbanSound8K yang menyimpan suara-suara di lingkungan perkotaan. Dari dataset tersebut, akan dipilih kelas data suara yang mewakili suara yang menjadi sinyal ancaman bahaya di lingkungan perkotaan. Walaupun begitu, terdapat kendala keterbatasan jumlah dataset yang dapat menjadi faktor kegagalan model dalam berlatih mengklasifikasikan suara tersebut.
4. Mencari Pre-trained Model
- Dikarenakan keterbatasan dataset menjadi faktor kegagalan model untuk belajar, maka melatih model dari awal menjadi langkah yang kurang efektif. Karena itu, model yang sudah pernah dilatih sebelumnya dengan dataset yang cukup (Pre-trained model) menjadi solusi pelatihan model dengan dataset yang terbatas. Akhirnya, PANNs dipilih menjadi Pre-trained model dalam penelitian ini.
5. Membaca Dokumentasi Dataset dan Pre-trained Model
- Setelah pemilihan dataset dan model, diberikan beberapa dokumentasi sebagai panduan dan catatan dalam menjalankan penelitian. Dataset UrbanSound8K telah membagi data ke dalam 10 fold. Maka dari itu, pembuat dataset memperingatkan untuk tidak mencampur kumpulan fold yang telah dirancang, karena mencampur data tersebut berpotensi menyebabkan kebocoran data yang mempengaruhi model saat belajar. Fenomena ini disebut overfitting, dimana model pandai dalam menebak suara pada dataset, tetapi bodoh dalam menebak suara diluar dataset.
- PANNs dalam dokumentasinya menyediakan berbagai macam arsitektur model dengan representasi input yang berbeda. Pemilihan arsitektur model yang tepat menjadi kunci dalam menciptakan model klasifikasi suara yang aman bagi penyandang Tuna Rungu. Dari semua jenis model dalam PANNs, dipilih ketiga arsitektur model yang unggul dalam performanya mewakili representasi inputnya, yaitu Res1dNet31 (Raw Waveform), ResNet38 (Log-mel Spectrogram) dan Hybrid yang menggunakan kedua representasi input tersebut. Permasalahan dalam membandingkan ketiga model ini akhirnya menjadi rumusan masalah dalam penelitian ini.
6. Melakukan Preprocessing Dataset
- Sebelum menggunakan dataset, dilakukan beberapa proses dalam mempersiapkan dataset mentah menjadi siap pakai untuk penelitian ini.
7. Menyesuaikan Konfigurasi Model
- Setelah mempersiapkan data, dilakukan penyesuaian pada konfigurasi model sebelum digunakan pada penelitian ini.
8. Melakukan Uji Coba Transfer Learning pada Pre-trained Model
- Tahap ini menjadi tahapan utama dalam mendapatkan model klasifikasi yang unggul dalam mengklasifikan suara yang lebih spesifik dan sesuai dengan tujuan penelitian. Proses ini dapat memakan waktu yang cukup lama dikarenakan akan terdapat proses Trial and Error pada proses pelatihannya beserta konfigurasi kode yang belum sesuai.
9. Mendapatkan Hasil & Evaluasi Fine-tuning Model
- Setelah menjalani proses pelatihan model dengan konfigurasi yang sudah sesuai, maka akan didapatkan hasil yang menggambarkan performa model saat melakukan pembelajaran. Evaluasi pembelajaran model akan digambarkan pada Confusion Matrix, F1-Score, Grafik Lost-Accuracy dan waktu training model. Keluaran model yang sudah dilakukan Transfer Learning juga menjadi keluaran model nyata yang dapat diuji coba dengan suara audio dari berbagai sumber yang sesuai dengan 4 kelas tersebut.
10. Analisis Hasil dan Kesimpulan
- Setelah mendapatkan hasil evaluasi model, dilakukan perbandingan performa model beserta menyimpulkan model manakah yang paling unggul dibandingkan kedua model lainnya.

## Akuisisi Data
- Dataset yang digunakan adalah dataset publik UrbanSound8K. Dataset didownload secara manual dari Kaggle berbentuk .zip. Dataset tersebut kemudian diekstrak dan berisikan 10 folder yang masing-masing mewakili 1 fold beserta file .csv yang menampilkan tabel nama file audio beserta metadatanya.

## Pre-processing Data
- Dikarenakan dataset UrbanSound8K telah menyusun 10 fold dalam 10 folder masing-masing,Maka akan digabungkan 10 fold menjadi 5 fold (tanpa mencampur secara acak). Lalu, dilakukan seleksi kelas pada setiap foldnya dan melakukan teknik distribusi yang cocok pada keseimbangan jumlah data antar kelasnya. Setelah itu, akan dilakukan pembagian data latih (Train) dan data uji (Test) agar data yang sudah dipakai untuk belajar tidak digunakan untuk menguji model lagi. Hal tersebut berupaya menghindari model untuk menghafal, bukan mempelajari karakteristik suara.
- Setelah menggabungkan 10 fold tadi, 5 fold tersebut masih terdiri dari kelas-kelas yang tidak semuanya digunakan. Maka dari itu, akan dipisahkan dan disisakan 4 kelas yang mewakili suara ancaman bahaya, yaitu gun_shot, siren, dog_bark, dan car_horn pada 5 fold tersebut.

## Konfigurasi Model
- Penyesuaian komponen meliputi jumlah keluaran kelas dan pembekuan otak lama pada model (Freeze Base). Hal tersebut dilakukan agar model mengeluarkan hasil klasifikasi pada jumlah keluaran kelas yang ditentukan dan model tidak menghapus ingatan yang telah dipelajari oleh dataset yang berjumlah besar sebelumnya. Model diharapkan memiliki bekal awal yang cukup dan hanya menambah ingatan pola yang baru pada karakteristik suara yang lebih spesifik (Transfer Learning).
- Dalam menjaga keseimbangan jumlah distribusi data antar kelas, diterapkan Sistem Weight Penalty. Weight Penalty adalah metode untuk menyeimbangkan performa akurasi pada kelas tertentu yang memiliki keterbatasan jumlah sampel. Model yang salah menebak kelas dengan jumlah data sampel yang lebih sedikit akan diberikan poin penalti lebih banyak dibandingkan model yang salah menebak kelas dengan jumlah data cukup.

## Parameter Pelatihan
Dalam mencapai keseimbangan hasil pelatihan antar jenis model dan tranparansi konfigurasi parameter yang dapat mempengaruhi pelatihan model, maka akan dijabarkan parameter pelatihan pada tabel x.x [diisi sesuai struktur].
- Th : Parameter, Nilai / Keterangan
- Td :
Model Architecture,PANNs (ResNet38 / Res1dNet31 / Wavegram-Logmel)
Pre-trained Dataset,AudioSet (Google)
Metode Transfer Learning,Fine-tuning pada Fully Connected Layer akhir
Input Sampling Rate,32.000 Hz
Input Duration,5 Detik (160.000 samples)
Preprocessing,Padding (Zero-pad) & Random Cropping
Jumlah Epoch,15
Batch Size,8
Learning Rate,0.001
Optimizer,Adam
Loss Function,Cross Entropy Loss dengan Class Weights
Validasi,5-Fold Cross Validation (Group Split)
Random Seed,42
Hardware,GPU CUDA RTX 1050

## Analisis dan Evaluasi
Setelah melatih model, hasil pelatihan model akan digambarkan dengan beberapa tolak ukur :
1. Confusion Matrix (Untuk melihat jumlah tebakan benar pada suara tertentu)
2. F1-Score (Untuk melihat apakah semua suara yang sesuai kelasnya terdeteksi, melihat seberapa presisi tebakan dia)
3. Grafik loss & accuracy (Untuk melihat proses model belajar)
4. Waktu training (Untuk menjadi acuan performa komputasi model)