# Research Logbook

## Oktober

### 2025-11-05
- Catatan dari Pak Martin (Bimbingan TA):
1. Transfer learning untuk Piczak gunakan model pretrained dari Bang julio, tambahkan class, layer terakhir ganti nanti untuk train suara dari saya.
2. Compare dengan model pretrained github quiqiangkong
3. Pastikan strategi pemilihan dataset (5 fold) 1,2 = 1; 3,4 = 2 ....

### 2025-11-04
- Membuat simulasi kode di Colab untuk memproyeksikan Spectrogram dari suara sirene.

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

## Agustus

### 2025-08-29
- Meeting TA pertama dengan Pak Martin 
- Diberikan Tugas Sebagai Berikut :
1. Tentukan Single-class atau Multi-class
2. Pelajari dan Bandingkan CNN, RNN, Spectrogram dan Melspectro
3. Cari dataset musik dan etika menggunakan dataset musik
4. Cari urgensi kenapa harus genre Korean Ballad