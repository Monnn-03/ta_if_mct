# Environmental Sound Classification (ESC)
Environmental Sound Classification (ESC) merupakan jenis klasifikasi suara yang lebih merujuk kepada suara-suara akustik pada lingkungan sekitar kita.
Suara-suara tersebut dibedakan menjadi dua jenis, yaitu suara alami yang berasal dari aktivitas alam, dan suara lingkungan buatan yang berasal dari aktivitas manusia atau perkotaan. ESC memiliki berbagai manfaat, mulai dari lingkup pengetahuan hingga digunakan sebagai alat bantu bagi penyandang tunarungu.

# Konsep Suara Digital
Konsep suara digital diawali dengan bentuk suara pada mulanya, di mana suara yang kita dengar berupa suara dalam bentuk fisik. Agar suara dapat didengar oleh mesin, suara perlu diterjemahkan ke dalam format yang bisa dimengerti bahasa mesin. Konversi tersebut mengalami dua tahap, yaitu mengubah sinyal fisik menjadi sinyal analog, kemudian sinyal analog tersebut dikonversikan lagi menjadi sinyal digital. Sinyal digital inilah yang dapat dimengerti oleh mesin, di mana bentuknya berubah dari gelombang fisik menjadi bentuk biner (terdiri dari 0 dan 1) yang dapat dimengerti oleh bahasa mesin. Konversi tersebut perlu dilakukan karena mesin tidak dapat membaca suara secara langsung, melainkan mesin hanya dapat membaca nilai biner (diskrit).

Sinyal digital nantinya akan berisi nilai amplitudo dari gelombang suara yang didapatkan oleh mesin. 
Nilai amplitudo ini yang menjadi komponen dasar pemrosesan sinyal suara pada mesin nantinya.

## Sampling Rate & Teorema Nyquist
Pengambilan seluruh nilai amplitudo tersebut bergantung pada kekuatan mesin dalam mengambil banyak sampel dalam satu waktu yang dilambangkan dengan istilah Sampling Rate, yaitu banyak nilai titik sampel yang diambil per detik durasi audio (Hz). Sebagai contoh jika audio asli memiliki sampling rate 32.000 Hz, maka audio tersebut mengambil detail 32.000 titik sampel dalam 1 detik. Semakin besar nilai sampling rate suatu audio, maka semakin jelas rekaman suara yang diambil.

Harry Nyquist mengemukakan teoremanya (Teorema Nyquist) bahwa nilai sampling rate pada sebuah audio harus 2 kali frekuensi pendengaran manusia pada umumnya (2 x f). Ini didasarkan pada konsep frekuensi suara yang merupakan banyaknya gelombang suara dalam 1 detik.  Gelombang suara harus terdiri dari dua bagian, yaitu 1 bagian puncak dan 1 bagian lembah. Teorema ini memastikan bahwa nilai sampling rate menangkap 2 titik sampel bagian dalam gelombang agar sesuai dengan nilai frekuensi pada audio tersebut. 

## Bit Depth
Jika sampling rate memastikan pengambilan titik sampel pada frekuensi, Bit Depth mengatur detail penempatan nilai amplitudo pada sebuah audio. Nilai bit depth yang tinggi memperjelas transisi kekuatan atau volume suara pada audio. Proyeksi nilai amplitudo tersebut dikalkulasikan dalam rumus berikut yang menentukan jumlah, batas bawah dan batas atas nilai amplitudo.

- Banyak nilai n bit depth = 2^n
- Nilai batas bawah n bit depth = -2^(n-1)
- Nilai batas atas n bit depth = 2^(n-1) - 1

Nilai batas bawah mendapatkan lebih banyak angka daripada nilai batas atas disebabkan oleh angka 0 ikut ke dalam nilai tersebut, dan ini diatur dalam hukum Two's Complement.

## Format Saluran Suara
Detail pengambilan rekaman suara bergantung pada jumlah dan kualitas perangkat perekam. Banyak jumlah perangkat yang merekam dari berbeda sudut pandang menciptakan pengalaman mendengar suara rekaman yang realistik. Maka dari itu, teknologi pengalaman suara tersebut diatur dalam format saluran suara. Seiring perkembangan teknologi, format saluran suara juga semakin berkembang dan canggih. 

Umumnya terdapat dua format saluran suara yang digunakan, yaitu satu saluran (mono) dan dua saluran (stereo). Keunggulan mendengar suara dengan format stereo ialah dapat mendengarkan suara kejadian pada rekaman dari sisi kiri dan kanan dibandingkan dengan format mono yang hanya merekam pada satu sisi saja. Mesin membaca format suara stereo yang menghasilkan dua kelompok nilai amplitudo yang dibungkus dalam dua array.

# Dataset UrbanSound8K
Dataset UrbanSound8K merupakan kumpulan data yang mengandung 8732 suara lingkungan perkotaan (format `.wav`) dengan kategori 10 kelas, di mana kategori tersebut mencakup :
a. air_conditioner (pendingin ruangan)
b. car_horn (klakson mobil)
c. children playing (anak-anak sedang bermain)
d. dog_bark (gonggongan anjing)
e. drilling (aktivitas mengebor)
f. engine_drilling (mesin pengebor)
g. gun_shot (tembakan senjata)
h. jackhammer (penghancur permukaan keras)
i. siren (sirene)
j. street_music (musik jalanan)

Selain file audio, dataset ini juga menyediakan file `.csv` sebagai informasi metadata untuk setiap file audionya. Terdapat format penamaan file audio didalam file metadata tersebut dengan format : `[fsID]-[classID]-[occurenceID]-[sliceID].wav`. Penomoran classID dimulai dari angka 0 sampai 9, sesuai dengan urutan yang tertera di atas.

Semua file audio telah disusun ke dalam 10 fold, dengan format penamaan folder : `fold1` - `fold10`. Penyusunan ini dibuat sedemikian rupa karena ada beberapa file audio yang direkam di tempat dan kejadian yang sama, sehingga beberapa rekaman tersebut disatukan pada fold yang sama demi menghindari kebocoran data. Kebocoran data merupakan fenomena yang terjadi saat data yang seharusnya menjadi data uji, tercampur dengan data latih sehingga model mendapatkan karakteristik suara yang sama saat validasi pembelajaran (dan berlaku sebaliknya).

Maka dari itu, pembuat dataset sangat tidak menganjurkan pengacakan data ulang karena berpotensi mengakibatkan model mengalami overfitting, yaitu model memiliki nilai akurasi tinggi hanya saat menebak data lokal. Ketika diberikan data baru yang belum pernah dipelajari sebelumnya, maka model akan menjadi bodoh.

# Pra-pemrosesan Data
Salah satu keberhasilan model dalam klasifikasi adalah kualitas data yang cukup baik dan sesuai konteksnya. Sebelum data digunakan untuk pelatihan model, pra-pemrosesan (pre-processing) data perlu dilakukan agar kualitas data sesuai dengan kebutuhan input model untuk mendapatkan hasil performa maksimal. 

## Down-mixing
Down-mixing merupakan proses penggabungan beberapa saluran suara menjadi lebih sedikit. Penggunaan down-mixing lebih efektif jika format saluran suara yang sedikit sudah cukup untuk merepresentasikan karakteristik suara yang ingin dipelajari. Proses tersebut juga membuat komputasi menjadi lebih ringan karena hanya sedikit kelompok nilai amplitudo yang diproses.

## Resampling
Resampling adalah proses mengubah jumlah titik sampel yang berada dalam 1 detik. Perubahan tersebut dapat mempengaruhi kejelasan suara, sehingga sample rate tujuan harus memiliki nilai yang dapat menangkap seluruh frekuensi gelombang pada suara. 

## Augmentasi

# Representasi Input Suara
## Raw Waveform
Raw Waveform merupakan representasi angka-angka yang mewakili nilai amplitudo pada audio, yang jika divisualisasikan akan membentuk sebuah gelombang audio tersebut. Format audionya biasanya berupa `.wav` atau `.aiff`, menyimpulkan kualitas audio yang masih murni belum terkena proses kompresi.

[GAMBAR_RAW_WAVEFORM]

Pada gelombang sinyal suara yang sudah mengalami konversi digital, direpresentasikan ke dalam bentuk array yang berisikan nilai-nilai amplitudo berdasarkan durasi (detik) dikali dengan nilai sampling rate nya. Amplitudo merupakan titik simpangan terjauh dalam gelombang suara. Nilai tersebut dapat berupa nilai positif bahkan negatif yang umumnya berada pada rentang -1 sampai 1. Semakin jauh nilai tersebut dari titik nol, maka getaran yang dihasilkan suara tersebut semakin tinggi (kencang).

Semua data pada dataset UrbanSound8K memiliki format `.wav`, sehingga membuktikan keaslian kualitas dataset pada penelitian ini.

## Log-mel Spectrogram
Jika Raw Waveform merupakan representasi suara dalam bentuk domain amplitudo dan waktu, maka Spectrogram merupakan representasi grafis fitur suara dalam bentuk domain frekuensi dan waktu.
Proses pembentukan spectrogram diawali dengan bentuk Raw Waveform yang diproses dengan metode Fast-Fourier Transform.
Metode tersebut dilatarbelakangi dengan masalah mengenai cara dekomposisi setiap frekuensi yang ada pada sebuah gelombang suara.
Cara kerja metode FFT ini adalah dengan mengambil bentuk awal potongan 1 detik gelombang suara dan menghitung kemunculan gelombang suara dengan panjang gelombang yang diberi variasi setiap saat.
Proses tersebut menghasilkan sebuah grafik baru yang menandakan setiap frekuensi yang didapatkan dari gelombang suara tersebut.

Setelah mendapatkan grafik frekuensi dari gelombang suaranya, penggambaran frekuensinya masih berupa satuan Hertz (Hz) dan merepresentasikan seluruh nilai frekuensi dengan jelas.
Pada kenyataannya, manusia memiliki rentang pendengaran pada suara dengan rentang frekuensi tertentu saja.
Maka dari itu, Skala Mel (Mel-Scale) hadir dalam memberikan rentang pembacaan frekuensi suara pada mesin agar menyerupai rentang frekuensi suara pada pendengaran manusia, sehingga membuat representasi fitur suara menjadi lebih realistis.
Spectrogram yang telah diproses dengan Mel-Scale dinamai dengan Mel-Spectrogram.

Namun, proyeksi pada Mel-Spectrogram masih memberikan rentang suara yang cukup jauh.
Ini mengakibatkan nilai warna pada detail minor suara yang ingin dipelajari tidak terproyeksi dengan maksimal, sehingga sulit untuk mengklasifikasi dengan detail yang tidak sesuai dengan pendengaran manusia.
Maka dari itu, proses logaritma menjadi solusi yang menyamaratakan nilai amplitudo pada proyeksi Mel-Spectrogram.
Nilai amplitudo yang kecil pada rentang amplitudo diubah melalui rumus logaritma yang menghasilkan nilai baru dengan satuan Desibel (DB).
Proyeksi Mel-Spectrogram yang telah diberikan proses logaritma (dapat disebut Log-mel Spectrogram) sangat menyerupai pendengaran manusia sehingga mesin dapat mempelajari suara tersebut layaknya manusia mempelajarinya.

# CNN

# Pre-trained Audio Neural Networks (PANNs)
Dalam dunia pre-trained model, PANNs menjadi salah satu pilihan yang menyediakan model klasifikasi suara dengan bobot terlatih dari dataset suara berskala besar (AudioSet).
Arsitektur ini dibuat dan dilatih oleh seorang peneliti bernama Qiuqiang Kong.
Pertimbangan penggunaan PANNs disebabkan oleh keterbatasan jumlah data pada dataset sekunder.
Walaupun begitu, arsitektur mengeluarkan output klasifikasi multi-label (Multi-label Classification) ke dalam 527 kelas pada dataset AudioSet, sehingga diperlukan penyesuaian pada lapisan akhir model agar dapat menyesuaikan output pada kelas suara yang ditentukan pada penelitian ini.

Penelitian Qiuqiang Kong mengenai PANNs juga telah menguji performa setiap arsitektur yang dipetakan ke dalam 3 metrik, yaitu mAP, AUC, dan d-prime.
Semua arsitektur yang ada dalam PANNs memiliki representasi input yang berbeda-beda.
Arsitektur PANNs dengan representasi input 1 dimensi ada DaiNet, LeeNet, Res1dNet, dan Wavegram-CNN.
Dilanjutkan dengan arsitektur representasi input 2 dimensi yaitu CNN, ResNet, dan MobileNet.
Yang terakhir, satu-satunya arsitektur dengan pendekatan input Hybrid yaitu Wavegram-Logmel-CNN.
Dalam penelitian ini, akan diambil 3 model unggul yang mewakili representasi inputnya masing-masing, yaitu Res1dNet31, ResNet38, dan Wavegram-Logmel-CNN14.

## Res1dNet31
Res1dNet31 merupakan arsitektur ResNet (Residual Network) 1 dimensi milik PANNs yang paling unggul dalam menangani representasi input domain waktu dibandingkan model lainnya dengan jenis input yang sama.

Arsitektur ini menerima input berupa sinyal digital 1 dimensi (Raw Waveform) dengan sample rate 32 KHz.

Proses komputasi dimulai dengan mengekstrak fitur Raw Waveform menggunakan lapisan konvolusi awal  dengan Kernel berukuran 3 dan stride bernilai 5. 
Informasi diproses lebih dalam melalui backbone ResNet dengan 14 blok residu, di mana setiap blok terdiri dari 2 lapisan konvolusi residu yang masing-masing mengandung dilatasi 1 dan 2. 
Akhirnya, hasil daripada backbone dilanjutkan ke 2 lapisan terakhir untuk mendapatkan kesimpulan klasifikasi final.
Seluruh lapisan pada arsitektur ini berjumlah 31 lapisan.

Keunggulan arsitektur ini dirancang untuk mempelajari fitur data suara mentah yang diperoleh secara alami dibandingkan fitur Logmel-Spectrogram yang sudah diproses dengan metode buatan manusia (FFT dan Logmel).

## ResNet38
ResNet38 merupakan salah satu arsitektur ResNet 2 dimensi milik PANNs dengan nilai mAP tertinggi dalam menangani jenis input dengan dimensi tersebut.

Arsitektur ini menerima input berupa Raw Waveform yang nantinya akan diekstrak dan diproses menjadi Log-mel Spectrogram (2 dimensi) dengan 1000 frames dan 64 mel bins.

Proses komputasi dimulai dengan blok konvolusi 512 filter dengan kernel berukuran 3x3 dan mengandung Batch Normalization (BN) serta ReLU.
Proses blok tersebut dilakukan sebanyak 2 kali dan dilakukan Max Pooling 2x2.
Setelah itu, fitur dimasukkan ke dalam backbone yang berisikan 16 Basic Blocks dan yang diikuti dengan Max Pooling 2x2 setiap iterasinya.
Lapisan akhir pada arsitektur ini berupa lapisan Fully Connected dengan 527 jawaban kelas serta fungsi aktivasi sigmoid.

Keunggulan arsitektur ini terletak pada kemampuannya dalam menangani input domain frekuensi yang sudah tergambar dalam grafis (2 dimensi).
Hal ini dapat memudahkan arsitektur mengenali pola yang sudah diproses sedemikian rupa menyerupai pendengaran manusia.

## Wavegram_Logmel_Cnn14
Wavegram_Logmel_Cnn14 merupakan arsitektur dengan pendekatan Hybrid pada keluarga PANNs yang dicetuskan oleh Qiuqiang Kong sendiri dalam penelitiannya.

Arsitektur ini menerima input Raw Waveform yang diproses dalam dua pendekatan, yaitu Wavegram dan Log-mel Spectrogram.
Untuk membentuk Wavegram, salah satu lapisan diawali dengan menempatkan lapisan konvolusi CNN 1 dimensi dengan panjang filter 11 dan stride 5.
Lalu diikuti dengan 3 blok konvolusi CNN 1 dimensi, di mana setiap blok mengandung 2 lapisan konvolusi dengan dilatasi 1 dan 2.
Fitur nantinya akan di reshape menjadi sebuah Wavegram, bersamaan dengan ekstraksi fitur Log-mel Spectrogram.
Kedua fitur ini nantinya akan digabungkan dan diproses melalui arsitektur CNN14.

Penggabungan kedua pendekatan ini berfungsi untuk menutupi kekurangan satu sama lain.
Raw Waveform murni yang diekstrak dan sulit dipelajari, dilengkapi dengan Log-mel Spectrogram dengan proyeksi pola yang mudah untuk dipelajari walaupun mengalami kehilangan informasi secara utuh.

# Transfer Learning
Setelah memilih model pre-trained, tahap selanjutnya adalah Transfer Learning.
Metode ini bekerja dengan cara mengadaptasi model pre-trained dengan data yang memiliki domain lebih spesifik.
Dalam kasus penelitian ini, PANNs yang sudah paham mengenali suara yang umum diarahkan untuk lebih spesifik mengenali suara kedaruratan di perkotaan.

Untuk melaksanakan transfer learning, terdapat dua jenis lingkup pelatihan model pre-trained, yaitu Freeze Base dan Fine Tuning.

## Freeze Base
Freeze Base merupakan jenis transfer learning yang bekerja dengan membekukan lapisan backbone pada model.
Pembekuan lapisan tersebut menyebabkan parameter bobot tidak akan disesuaikan oleh data yang baru.
Penyesuaian hanya dilakukan pada lapisan belakang (head) yang memutuskan label pada data tersebut, dalam penelitian ini menyesuaikan output klasifikasi 527 kelas multi-label menjadi 4 kelas dengan klasifikasi single-label.
Mekanisme ini menjadi sesuai jika dihadapkan dengan kondisi data yang sangat terbatas, sehingga menghindari kerusakan bobot akibat kekurangan data untuk dipelajari.

## Fine Tuning
Berbeda dengan Freeze Base, metode Fine Tuning menyesuaikan bobot lapisan backbone pada konteks data baru yang ingin dipelajari.
Ini mengizinkan model untuk menyesuaikan ulang bobot pada lapisan backbone, sehingga model mempunyai bobot yang telah diperbarui dan bisa menangani kasus klasifikasi dengan konteks khusus yang sudah dipelajari. 
Untuk mendapatkan hasil yang efektif, diperlukan data dengan jumlah yang cukup agar model memiliki bekal cukup dalam memperbarui bobotnya.

# Strategi Optimasi Pelatihan

# Metrik Evaluasi






