# Research Logbook

## February

### 2026-02-09
- Membuat paragraf pembuka subbab 3.3. Konfigurasi Model
- Menambahkan penjelasan Sigmoid menjadi linear pada bagian 3.3.1.
- Menambahkan ilustrasi transfer learning pada 3.3.1.
- Menambahkan penjelasan detail dan rumus pada bagian 3.3.2.

### 2026-02-08
- Merevisi kata Tuna Rungu menjadi tunarungu pada Bab III
- Memberikan ilustrasi pada fold mapping
- Memberikan urgensi dilakukan fold mapping dan teknik validasinya.
- Menambahkan penjelasan seleksi kelas suara
- Menambah sub subbab tentang Augmentasi Temporal

### 2026-02-07
- Menggabungkan 3.2. Akuisisi Data dengan 3.3. Pra-pemrosesan data menjadi 3.2. Pengumpulan dan Pra-pemrosesan Data

### 2026-02-06
- Train ketiga Spectrogram Fold 5
- Revisi Bab III : 3.1. Alur Penelitian

### 2026-02-05
- Memperbaiki augmentasi temporal di `datareader.py` menjadi 1 detik.
- Tidak memakai Sigmoid pada PANNs models.
- Setup WandB Dashboard
- Train ketiga Spectrogram Fold 1-4

### 2026-02-04
- Belajar tentang Learning Rate, CPU dan GPU.
- Belajar Baca Grafik Loss-Accuracy.

### 2026-02-03
- Bimbingan bersama Pak MCT (Detail lihat di TAsys)
- Mencari tahu tentang deterministic, mematikan nya saja
- Mempelajari dasar config training.

### 2026-02-02
- Menambahkan output berupa Confusion Matrix dan Total Time
- Melakukan train ulang pada konfigurasi hasil evaluasi
- Membuat Markdown dan Konversi ke Tex untuk Bab III

### 2026-02-01
- Mendapatkan hasil train ResNet38 pertama

## January

### 2026-01-31
- Train ulang ResNet38, mengganti `num_workers` menjadi 0

### 2026-01-30
- Train ResNet38

### 2026-01-29
- Mengkonfigurasi ulang `datareader.py`
- Mengkonfigurasi ulang `model.py`
- Mengkonfigurasi ulang `train.py`
- Menuliskan Alur Penelitian

### 2026-01-28
- Menggabungkan 10 fold menjadi 5 fold saja
- Membuat ulang `create_5fold_split.py`
- Menuliskan Langkah Preprocessing Data

### 2026-01-27
- Membuat Outline Bab III
- Menentukan inti poin isi dari setiap bagiannya

### 2026-01-26
- Bimbingan dengan Pak MCT (Detail lihat di TAsys) [Dl 2 Minggu Max]
- Buat ulang draft Bab III dengan Markdown

### 2026-01-25
- Revisi Bab I Final

### 2026-01-24
- Revisi Rancangan Pengujian sebelum bimbingan

### 2026-01-23
- Bab III tinggal Rancangan Pengujian


### 2026-01-22
- Revisi Metode Pengembangan

### 2026-01-20
- Review Bab III
- Revisi Alur Penelitian
- Revisi Alat dan Bahan
- Buat ulang Draft Metode Pengembangan - Rancangan Pengujian

### 2026-01-13
- Masih mencari alur penelitian
- Menjabarkan alur penelitian
- Buat draft dulu untuk bagian Bab III lainnya

### 2026-01-12
- Mencari alur penelitian

### 2026-01-10
- Bab I Selesai Revisi
- Susun Bab III (Kalau masih kuat wkwkwk)

### 2026-01-09
- Revisi Latar Belakang dari Revisi Detaiil Pak MCT

### 2026-01-08
- Revisi Latar Belakang Paragraf 1 dan 2

### 2026-01-07
- Bimbingan Pertama di Awal Tahun bersama Pak MCT (Detail ditulis pada TASYS)

### 2026-01-06
- Fiksasi Bab I sebelum bimbingan

### 2026-01-05
- Meninjau kembali Bab I

### 2026-01-03
- Input penelitian terdahulu ke LaTeX

### 2026-01-02
- Mencari penelitian terdahulu (State-of-the-art)

## December

### 2025-12-30
- Poles Seluruh Bab I (Rumusan Masalah - Manfaat Penelitian)

### 2025-12-29
- Poles seluruh Bab I (Latar Belakang)

### 2025-12-27
- Buat seluruh Draft untuk Bab I

### 2025-12-26
- Selesaikan Draft Paragraf 4 dan 5 latar belakang

### 2025-12-22
- Selesaikan Draft Paragraf 3 latar belakang

### 2025-12-21
- Selesaikan Draft Paragraf 1 dan 2 Latar Belakang

### 2025-12-17
- Menyusun Bab I : Latar Belakang
1. Paragraf 1
2. Paragraf 2

### 2025-12-16
- Catatan dari bimbingan Pak Martin :
1. Pelajari kenapa pada arsitektur ketiga jenis modelnya menggunakan angka 128000.
2. Sudah boleh menyusun Bab I
3. Pelajari asal usul untuk orang yang membuat arsitektur model tersebut (Kenapa menggunakan model ini? dll)
- Mempelajari bahwasannya 128000 Hz adalah hasil daripada samplerate audio (32000 Hz) dikalikan dengan durasi audio (yaitu 4 detik).

### 2025-12-13
- Melakukan train pada ketiga jenis model dan membandingkan ketiganya.

### 2025-12-09
- Memasukkan model ResNet38 (Input Spectro), Res1dNet31 (Input Waveform), dan Wavegram-Logmel-CNN (Input Hybrid) ke dalam `model.py`.

### 2025-12-02
- Untuk seluruh notulensi, harap ditulis juga pada website sistem TA.
- Catatan Bimbingan TA dari Pak Martin:
1. Nama audio dari UrbanSound8K masih terkontaminasi dengan kata "Fold". Harus diperhatikan dan jangan terkecoh dengan kontaminasi nama file audio tersebut.
2. Berdasarkan observasi, beberapa model PANNs Qiuqiangkong inputnya langsung audio, sedangkan piczakCNN itu hanya melihat spectrogram.
3. Fokus explore model dengan tiga jenis input, yaitu waveform, hybrid dan spectrogram.
4. Kemungkinan Judul Penelitian : Perbandingan Hasil Ketiga Model dengan Input Spectrogram, Hybrid, dan Audio Langsung (Waveform).
5. Model CNN, Resnet, Mobile, Itu Spectrogram
6. Model Dainet - Wavegram CNN itu audio mentah
7. Model Wavegram Logmel itu Hybrid
8. Ambil 1 perwakilan dari tiap jenis model, kita ambil yang terbaik.
9. Post kode di github TA (untuk dataset cukup di `.gitignore`).
- Memindahkan semua code dari komputer lokal ke folder yang sudah remote dengan repo TA.

## November

### 2025-11-28
- Menyelesaikan PANNs Transfer pada `model.py` dan menjalankan.
- Mendapatkan hasil dengan torchinfo dari kedua model.

### 2025-11-25
- Menyelesaikan PiczakCNN pada `model.py` dan menjalankannya.

### 2025-11-24
- Belajar library dan sintaks dasar `model.py`
1. `torch`
2. `torch.nn`
3. `torch.nn.Functional`
4. Class dan `nn.Module`
5. `Conv2D`, `ReLU`, `Dropout`, `MaxPooling2D`.

### 2025-11-23
- Menonton tutorial membuat struktur dasar `model.py`

### 2025-11-21
- Menjalankan `datareader.py` dan berhasil membaca 2000+ data `.wav` dan menampilkan mel-spectrogram.

### 2025-11-20
- Membuat dan menyesuaikan 4 kelas yang ada ke `datareader.py`

### 2025-11-16
- Mempelajari Data Reader yang dibuat di tutorial Pak Martin.

### 2025-11-14
- Memanipulasi Dataset pada GoogleDrive Package dan menjalankan di Google Colab.

### 2025-11-11
- Menonton tutorial Audio Classification dari Pak Martin yang diberikan oleh Bang Julio.
- Mencoba model sederhana dari AI.

### 2025-11-10
- Membuat model sendiri pada Google Colab. Tetapi masih terdapat kendala teknis pada import dataset.
- Catatan Bimbingan TA dari Pak Martin:
1. Susun ulang fold jadi 5 fold (buat aja dalam folder):
- Dilebur, jadiin satu, baru dipisah jadi 5 fold lagi
- Distribusi kelas di tiap fold itu sebisa mungkin sama.
2. Solusi dataset:
- upload ke googledrive, pakai package gdrive
- opsi lain pakai huggingface
3. buat data reader (pelajari dari julio). Kalau julio belum ngirim sampai minggu ini, kabarin.
4. buat model (model.py), jalankan dengan torchinfo summary (ngeprint layer per layer model).

### 2025-11-09
- Menghitung setiap class pada dataset UrbanSound8K dan diproyeksikan pada histogram.

### 2025-11-05
- Catatan dari Pak Martin (Bimbingan TA):
1. Transfer learning untuk Piczak gunakan model pretrained dari Bang julio, tambahkan class, layer terakhir ganti nanti untuk train suara dari saya.
2. Compare dengan model pretrained github quiqiangkong
3. Pastikan strategi pemilihan dataset (5 fold) 1,2 = 1; 3,4 = 2 ....

### 2025-11-04
- Membuat simulasi kode di Colab untuk memproyeksikan Spectrogram dari suara sirene.

## October

### 2025-10-22
1. Mencari aspek yang berbeda untuk kebaruan penelitian.
2. Didapatkan dua pilihan, ganti model atau ganti domain?
3. Apakah kita bandingkan menggunakan Transfer Learning (PANNs) dari pre-model? Mari kita saksikan pada bimbingan berikutnya.

### 2025-10-13
1. Lihat topik Bang Julio dan cari aspek yang sekiranya tidak ada pada topiknya.
2. Aspek Beliau :
- Domain tentang Keselamatan Tuna Rungu
- Best Model, One DB (Mencari yang terbaik)
- Data yang masuk digeser geser (time shift)
- Noise -> Spectro -> Spec Augment, atur awal time shift random (Ada kemungkinan 50:50 data di skip ke proses berikutnya)
- Direkam HP, waktunya berapa, jaraknya berapa, situasi lingkungan
- Metode Piczak CNN
- Checkpointing Auto Dari Model
- Suara -> Klakson Mobil, Ambulans, Knalpot Motor, Mesin Mobil.
3. Aspek pengembangan. Cari Model, Augmentasi, Domain, Param yang berbeda dengan punya Bang Jul, lalu Bandingkan.
4. Pytorch is a must

### 2025-10-12
- Menentukan domain untuk klasifikasi apakah musik atau berganti ke suara??
- Menentukan untuk melakukan klasifikasi suara perkotaan (Urban) dengan domain Public Safety (Keamanan, seperti suara tembakan, sirene, dll). Tetapi untuk topik alternatif yang sesuai dengan topik awal, mengganti genre menjadi pop daerah dengan adanya transfer learning.

### 2025-10-04
- Mencari genre baru dan saran jenis Klasifikasi Musik, berkaitan dengan musik tradisional di Indonesia (Untuk Sementara, Ansambel Gamolan Lampung secara SingleClass).

## September

### 2025-09-27
- Preview MFCC dari https://github.com/bissessk/Musical-Instrument-Classification-Using-Deep-Learning di Python.

### 2025-09-19
- Brainstorming tentang genre yang dibawakan apakah tetap atau harus diganti

### 2025-09-18
- Meeting TA ke-2 dengan Pak Martin
- Diberikan tugas sebagai berikut :
1. Download dataset https://github.com/bissessk/Musical-Instrument-Classification-Using-Deep-Learning, dan preview MFCC nya di Python.
2. Pertimbangkan dataset korean ballad yang terbatas
3. Opsi dataset untuk build model awal (non korean ballad):
- https://magenta.withgoogle.com/datasets/nsynth#files
- https://github.com/bissessk/Musical-Instrument-Classification-Using-Deep-Learning
- https://research.google.com/audioset/dataset/index.html
4. Cari yang multi class. Ada atau tidak.





### 2025-09-16
- Mempelajari dan membandingkan metode CNN, RNN, Spectrogram biasa, dan Mel-Spectro.
- Mencari dataset musik, sejauh ini baru MTG-Jamendo yang ditemukan.
- Mencari lebih banyak tentang urgensi genre Korean Ballad berdasarkan kelangkaan penelitian dan popularitas Korean Wave di Indonesia.

### 2025-09-15
- Mempelajari mengenai Singleclass, Multiclass, Singlelabel dan Multilabel.

## August

### 2025-08-29
- Meeting TA pertama dengan Pak Martin 
- Diberikan Tugas Sebagai Berikut :
1. Tentukan Single-class atau Multi-class
2. Pelajari dan Bandingkan CNN, RNN, Spectrogram dan Melspectro
3. Cari dataset musik dan etika menggunakan dataset musik
4. Cari urgensi kenapa harus genre Korean Ballad