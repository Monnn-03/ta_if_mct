# TINJAUAN PUSTAKA
Lakukan pembahasan secara sistematis dengan menjelaskan masalah apa yang diangkat di penelitian terdahulu, metode yang digunakan, kontribusi yang diberikan, serta analisis penulis terkait dengan keunggulan atau keterbatasannya. Tuangkan perbandingan penelitian terdahulu dengan penelitian yang akan dikerjakan, minimal 5 jurnal pembanding (3 -4 tahun terakhir).

1. A Study on AI-Driven Acoustic Feature Design for Urban Noise Classification (2025) [Shen Yu] 
a. Masalah : Sumber daya akustik perkotaan mengalami kekurangan skenario, keragaman akustik, metode augmentasi umum yang rentan merusak karakteristik suara, serta jaringan saraf tiruan yang tidak interpretabilitas dan robust.
b. Metode : Melatih model ResNet-TF dari nol, Dataset : UN15 
c. Kontribusi : Menciptakan standar pra-pemrosesan data, pemodelan dalam domain waktu-frekuensi (grafis), serta augmentasi screening (upaya menangani ketidakseimbangan pada kelas minoritas)
d. Keunggulan : Berhasil mengintegrasikan pemodelan domain waktu-frekuensi secara grafis yang mampu menangkap detail suara dengan lebih baik pada domain dataset suara perkotaan. 
e. Keterbatasan : Membutuhkan waktu lebih untuk melakukan augmentasi dan screening data buatan untuk kelas minoritas.

2. Multi-Class Urban Sound Classification with Deep
Learning Architectures (2024) [Avadhani et al.] 
a. Masalah : Tantangan dalam mengkategorikan suara di perkotaan dengan tepat.
b. Metode : Melakukan ekstraksi fitur manual menggunakan MFCC dan Melatih Model ANN, CNN, RNN, dan LSTM dari Nol.
c. Kontribusi : Memberikan kelebihan dan pertimbangan dari setiap model yang digunakan untuk aplikasi di dunia perkotaan.
d. Keunggulan : Mendapatkan hasil akurasi ANN sebesar 94.79%, CNN sebesar 93,64%, dan LSTM sebesar 86,09% pada dataset UrbanSound8K. 
e. Keterbatasan : Tidak ada penanganan ketidakseimbangan antar kelas dataset (akurasi dipertanyakan), serta masih sangat bergantung pada MFCC yang melewati proses kompresi lagi sehingga informasi penting audio dapat hilang.

3. Robust Forest Sound Classification Using  Pareto-Mordukhovich Optimized MFCC in Environmental Monitoring (2025) [Qurthobi et al.] 
a. Masalah : Membuat sistem deteksi ancaman dini berbasis klasifikasi audio pada lingkungan dengan ketersediaan dataset yang terbatas.
b. Metode : Menggunakan ekstraksi fitur manual berupa MFCC dan melatih model hybrid gabungan secara manual (CNN-BiLSTM) dari nol.
c. Kontribusi : Membuktikan bahwa performa model hibrida dibandingkan dengan pre-trained model mandiri.
d. Keunggulan : Berhasil menciptakan arsitektur hibrida dari gabungan kedua arsitektur yang berbeda jenis.
e. Keterbatasan : Memiliki nilai loss yang cukup tinggi (0.7209) dan akurasi rendah (78.52%) akibat penggunaan MFCC dan arsitektur gabungan yang berat. Metode hibrida arsitektur dan augmentasi yang digunakan tidak cukup efisien dibandingkan pendekatan hibrida pada level input.

4. Graph-Based Audio Classification Using Pre-Trained Models and Graph Neural Networks (2024) [Castro-Ospina et al.] 
a. Masalah : Tantangan mencari representasi data klasifikasi suara lingkungan yang optimal menggunakan pendekatan non-konvensional. 
b. Metode : Menggunakan model pre-trained sebagai pengekstraksi fitur awal, lalu distrukturisasi ulang menjadi simpul-simpul graf untuk dilatih ke dalam Graph Neural Networks.
c. Kontribusi : Mengusulkan metode baru yang memetakan fitur audio spasial-temporal ke dalam bentuk graf dalam tugas klasifikasi multi-kelas.
d. Keunggulan : Membuktikan bahwa arsitektur GAT (\textit{Graph Attention Networks}) mampu memberikan performa paling kompetitif dibandingkan varian GNN lainnya dalam mengenali struktur graf audio.
e. Keterbatasan : Transformasi audio menjadi struktur graf menambah lapisan kompleksitas komputasi yang tinggi, namun hanya menghasilkan akurasi yang moderat (83%) untuk suara lingkungan. Selain itu, model ini belum menguji secara komparatif format input mentah (\textit{raw waveform} vs spektral) serta tidak memiliki mitigasi terhadap ketidakseimbangan data kelas.

5. A Comparative Study of Deep Audio Models for Spectrogram- and Waveform-Based SingFake Detection (2025) [Nguyen-Duc et al.]
a. Masalah : Tantangan dalam mendeteksi suara sintesis (nyanyian deepfake) akibat tingginya kompleksitas variasi nada, warna suara, dan iringan latar. 
b. Metode : Melakukan analisis komparatif performa antara model deep learning yang menggunakan input Spektrogram Log-Mel (AST, Whisper) dengan model yang memproses Raw Waveform (UniSpeech-SAT, HuBERT).
c. Kontribusi : Memberikan tolok ukur komparatif yang komprehensif mengenai efektivitas berbagai format representasi input audio terhadap model AI modern untuk tujuan autentikasi media.
d. Keunggulan : Membuktikan secara empiris bahwa format representasi visual (spektrogram Log-Mel) memiliki kemampuan generalisasi yang lebih superior dibandingkan bentuk gelombang mentah, khususnya pada data pengujian yang belum pernah dilihat sebelumnya.
e. Keterbatasan : Penelitian ini hanya membandingkan kedua format input secara terpisah tanpa mengeksplorasi potensi penggabungan (Hybrid Input). Selain itu, pengujian dilakukan pada domain suara vokal yang teratur, sehingga belum teruji ketahanannya pada suara kebisingan perkotaan yang acak dan memiliki distribusi kelas yang sangat timpang (highly imbalanced).




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
Resampling adalah proses mengubah jumlah titik sampel yang berada dalam 1 detik. Perubahan tersebut dapat mempengaruhi kejelasan suara, sehingga sample rate tujuan harus memiliki nilai yang cukup untuk menangkap seluruh frekuensi gelombang pada suara. Banyaknya titik sampel dalam satu detik juga mempengaruhi proses komputasi.

## Augmentasi
Augmentasi merupakan salah satu teknik yang ampuh dalam memperbanyak variasi data dalam dunia pelatihan model. Variasi tersebut dilakukan dengan cara memanipulasi bagian data, seperti menambahkan gangguan (noise) atau melakukan pemotongan durasi secara acak. Dengan variasi data yang selalu acak pada setiap iterasi pelatihan model (epoch), keyakinan model untuk menjawab secara benar meningkat. Model juga tidak hanya memiliki bobot yang bervariasi, tetapi juga tahan uji ketika menghadapi data yang mengalami sedikit kerusakan.

# Representasi Input Suara
## Raw Waveform
Raw Waveform merupakan representasi angka-angka yang mewakili nilai amplitudo pada audio, yang jika divisualisasikan akan membentuk sebuah gelombang audio tersebut. Format audionya biasanya berupa `.wav` atau `.aiff`, menyimpulkan kualitas audio yang masih murni belum terkena proses kompresi.

[GAMBAR_RAW_WAVEFORM]

Pada gelombang sinyal suara yang sudah mengalami konversi digital, direpresentasikan ke dalam bentuk array yang berisikan nilai-nilai amplitudo berdasarkan durasi (detik) dikali dengan nilai sampling rate nya. Amplitudo merupakan titik simpangan terjauh dalam gelombang suara. Nilai tersebut dapat berupa nilai positif bahkan negatif yang umumnya berada pada rentang -1 sampai 1. Semakin jauh nilai tersebut dari titik nol, maka getaran yang dihasilkan suara tersebut semakin tinggi (kencang).

Semua data pada dataset UrbanSound8K memiliki format `.wav`, sehingga membuktikan keaslian kualitas dataset pada penelitian ini.

## Log-mel Spectrogram
Spectrogram merupakan representasi grafis fitur suara dalam bentuk domain frekuensi dan waktu.
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

# Convolutional Neural Network (CNN)
Convolutional Neural Network (CNN) adalah arsitektur model Deep Learning berupa jaringan saraf yang bekerja dengan cara mengenali pola dan meringkas informasi penting yang didapatkan pada setiap lapisannya. CNN dikenal karena keandalannya dalam mempelajari dan mengenali gambar. Terdapat beberapa bagian dalam arsitektur CNN, yaitu Input Layer (kaki), Backbone (punggung), Neck (leher), dan Head (kepala / klasifikator).

Setiap bagian memiliki fungsinya masing-masing. Input Layer merupakan lapisan awal sebagai penerima input data untuk dijadikan bahan belajar model. Backbone bertugas mengekstraksi fitur input yang telah diberikan dan menghasilkan informasi ringkas karakteristik input. Keberadaan Neck sebenarnya opsional, tetapi bagian ini bertugas untuk meringkas informasi dari backbone yang banyak menjadi deretan 1 baris angka yang sederhana. Head akan memproses ringkasan informasi dari neck dan mengambil keputusan terhadap hasil karakteristik input ke dalam kelas yang telah dipetakan. 

## Kernel
Kernel merupakan komponen pengamat fitur yang biasanya berbentuk persegi berukuran tertentu. Tugasnya adalah menemukan pola (filter) pada fitur yang sedang disorot dan meneruskannya ke lapisan berikutnya.

## Stride
Stride merupakan mekanisme lompatan kernel untuk melewati beberapa urutan fitur yang seharusnya. Fungsinya mempercepat proses pengambilan informasi pada fitur, sehingga meminimalkan komputasi.

## Batch Normalization (BN)
Batch Normalization merupakan mekanisme normalisasi rentang nilai yang diperoleh dari lapisan sebelumnya. Mekanisme ini membuat komputasi nilai menjadi lebih mudah untuk diproses ke lapisan berikutnya.

## ReLU
ReLU adalah komponen yang bertugas untuk menyaring dan membuang nilai negatif pada informasi ekstraksi fitur. Komponen ini hanya akan mengembalikan informasi pola dengan nilai yang kuat didalamnya.

## Max Pooling
Pooling merupakan komponen yang bertugas mengecilkan ukuran data. Hal ini bertujuan untuk mempermudah proses komputasi saat pelatihan. Akan tetapi, fungsi ini mulai sering digantikan dengan stride yang cukup mengambil fungsinya dalam mempercepat pengambilan informasi fitur.

## Fully Connected Layer (Linear)
Fully Connected (FC) Layer merupakan lapisan yang berada di bagian Head sebagai pengambil keputusan. Lapisan ini menerima informasi yang sudah diekstraksi oleh backbone dan diringkas oleh neck, kemudian menghubungkan semua informasi yang didapatkan. Dari kesimpulan tersebut, lapisan ini akan menebak kelas data ini sesuai dengan ciri-ciri fitur tersebut dengan representasi angka mentah (logits).

## Sigmoid (Fungsi Aktivasi Keluaran)
Sigmoid merupakan fungsi yang mengubah hasil angka mentah pada FC Layer menjadi probabilitas independen berupa persentase. Angka yang dihasilkan oleh FC Layer tidak representatif untuk dijadikan hasil akhir untuk dilihat pengguna karena penentuan nilainya sangat abstrak. Maka dari itu, perubahan angka tersebut menjadi nilai persentase lebih informatif untuk disajikan kepada pengguna.



# Pre-trained Audio Neural Networks (PANNs)
Dalam dunia pre-trained model, PANNs menjadi salah satu pilihan yang menyediakan model klasifikasi suara dengan bobot terlatih dari dataset suara berskala besar (AudioSet). Arsitektur ini dibuat dan dilatih oleh seorang peneliti bernama Qiuqiang Kong. Keluaran klasifikasi nya berupa multi-label (Multi-label Classification) ke dalam 527 kelas pada dataset AudioSet, sehingga diperlukan penyesuaian pada lapisan akhir model agar dapat menyesuaikan output pada kelas suara yang ditentukan pada penelitian ini.

[GAMBAR_TABEL_PENGUJIAN_MODEL_PANNs]

Penelitian Qiuqiang Kong mengenai PANNs juga telah menguji performa setiap arsitektur yang dipetakan ke dalam 3 metrik, yaitu mAP, AUC, dan d-prime.
Semua arsitektur yang ada dalam PANNs memiliki representasi input yang berbeda-beda. Arsitektur PANNs dengan representasi input 1 dimensi ada DaiNet, LeeNet, Res1dNet, dan Wavegram-CNN. Dilanjutkan dengan arsitektur representasi input 2 dimensi yaitu CNN, ResNet, dan MobileNet. Terakhir, satu-satunya arsitektur dengan pendekatan input Hybrid yaitu Wavegram-Logmel-CNN. Dalam penelitian ini, akan diambil 3 model unggul yang mewakili representasi inputnya masing-masing, yaitu Res1dNet31, ResNet38, dan Wavegram-Logmel-CNN14.

## Res1dNet31
Res1dNet31 merupakan arsitektur ResNet (Residual Network) 1 dimensi milik PANNs yang paling unggul dalam menangani representasi input domain waktu dibandingkan model lainnya dengan jenis input yang sama. Arsitektur ini menerima input berupa sinyal digital 1 dimensi (Raw Waveform) dengan sample rate 32 KHz. Keunggulan arsitektur ini dirancang untuk mempelajari fitur data suara mentah yang diperoleh secara alami dibandingkan fitur Logmel-Spectrogram yang sudah diproses dengan metode buatan manusia (FFT dan Logmel).

Proses komputasinya diawali dengan mengekstrak fitur Raw Waveform menggunakan lapisan konvolusi awal  dengan Kernel berukuran 3 dan stride bernilai 5. Informasi diproses lebih dalam melalui backbone ResNet dengan 14 blok residu, di mana setiap blok terdiri dari 2 lapisan konvolusi residu yang masing-masing mengandung dilatasi 1 dan 2. Akhirnya, hasil daripada backbone dilanjutkan ke 2 lapisan terakhir untuk mendapatkan kesimpulan klasifikasi final. Seluruh lapisan pada arsitektur ini berjumlah 31 lapisan.

## ResNet38
ResNet38 merupakan salah satu arsitektur ResNet 2 dimensi milik PANNs dengan nilai mAP tertinggi dalam menangani jenis input dengan dimensi tersebut.

Arsitektur ini menerima input berupa Raw Waveform yang nantinya akan diekstrak dan diproses menjadi Log-mel Spectrogram (2 dimensi) dengan 1000 frames dan 64 mel bins.

Proses komputasi dimulai dengan blok konvolusi 512 filter dengan kernel berukuran 3x3 dan mengandung Batch Normalization (BN) serta ReLU.
Proses blok tersebut dilakukan sebanyak 2 kali dan dilakukan Max Pooling 2x2.
Setelah itu, fitur dimasukkan ke dalam backbone yang berisikan 16 Basic Blocks, di mana setiap blok berisikan kernel, BN dan ReLU. Setiap Basic Blocks diakhiri dengan Max Pooling berukuran 2x2.
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

# Strategi Optimasi dan Regulasi Pelatihan
Dalam pelatihan dari nol atau Fine-tuning model, perlu adanya beberapa regulasi untuk memastikan model memiliki cara belajar yang tepat.
Regulasi tersebut mengatur mekanisme jalannya pelatihan. Untuk mendapatkan hasil yang lebih maksimal, strategi optimasi juga diterapkan dalam proses pelatihan model.

## Hyperparameter Pelatihan
Hyperparameter merupakan parameter regulasi pelatihan pada eksperimen pelatihan (pelatihan berulang). Perubahan tersebut bertujuan mendapatkan parameter yang paling optimal untuk menghasilkan performa model terbaik.

## Epoch
Epoch merupakan istilah pengukuran iterasi pelatihan model. 

## Step
Step adalah hitungan langkah dalam setiap epoch ketika pekerja CPU mengambil data dan diantarakan ke lapisan input model.

## Batch Size
Batch Size mengukur banyaknya data (file) yang diambil setiap step.

## Num Workers
Num Workers adalah jumlah pekerja CPU yang mengambil data setiap stepnya.

## Learning Rate
Learning Rate adalah parameter untuk mengatur jarak langkah belajar model.

## Cross-Entropy Loss
Loss merupakan nilai yang cukup penting dalam pelatihan model. Jika nilai akurasi mengukur jumlah tebakan benar model, maka nilai Loss mengukur seberapa yakin model ketika salah menebak data. Kalkulasi nilai ini membutuhkan nilai probabilitas hasil tebakan yang didapatkan dari fungsi Sigmoid.

[Rumus]

Perhitungan ini berlaku sama rata untuk semua kelas yang ditentukan.

## Cost-Sensitive Learning
Cost-Sensitive Learning merupakan adaptasi perhitungan Cross Entropy Loss sebelumnya yang memberikan nilai loss lebih pada kelas yang lebih penting. Adaptasi tersebut hanya menambahkan bobot khusus pada setiap kelasnya, sehingga penambahan nilai Loss digambarkan pada rumus berikut :

[Rumus_Cost-Sensitive-Learn]

## Optimizer
Optimizer merupakan mekanisme penting yang berfungsi sebagai pengatur strategi cara belajar model setelah menebak. Komponen yang diatur dalam optimizer ini adalah bobot dan bias pada arsitektur model.

Bobot dan bias baru pada setiap lapisan akan dikalkulasikan dengan rumus berikut:
- w/b baru = w/b lama - (lr x gradient)

Gradient didapatkan dari hasil kalkulasi nilai loss dengan bobot pada setiap lapisan.

## Adam
AdamW merupakan salah satu jenis optimizer yang terkenal mengandalkan momentum. 

## AdamW
Mengatasi kekurangan pada versi optimizer sebelumnya, AdamW merupakan perbaikan dari Adam dalam menangani Weight Decay.

## ReduceLROnPlateau (LR Scheduler)
Learning Rate (LR) Scheduler merupakan mekanisme untuk mengatur Learning Rate pada pelatihan. Ini melengkapi mekanisme optimizer yang hanya menyesuaikan nilai Gradient saja.

ReduceLROnPlateau merupakan LR Scheduler bawaan Pytorch yang berfungsi mengurangi Learning Rate ketika suatu metrik tidak melebihi nilai target dengan jumlah batas toleransi (patience) yang ditentukan. Pengurangan Learning Rate menggunakan faktor pengali yang akan mengurangi nilai Learning Rate sebelumnya.

## Early Stopping
Early Stopping merupakan mekanisme berupa pemberhentian pelatihan model jika suatu metrik tidak melebihi nilai tertingginya. Penyesuaian batas toleransi (patience) ketika metrik yang ditentukan tidak melewati rekor dilakukan berulang sesuai dengan analisa performa model. 
Mekanisme ini diciptakan untuk menghindari model mengalami overfitting. Jika mekanisme ini tidak diterapkan, model yang sudah pernah mencapai performa tertingginya pada suatu epoch akan terus berlatih dan diuji dengan data yang sama pada sisa epochnya, sehingga model malah menghafal data uji dan data latih. 

## Checkpointing
Saat model mencapai nilai performa terbaiknya, mekanisme Checkpointing hadir untuk melakukan otomatisasi penyimpanan bobot model. Bobot model yang disimpan berekstensi `.pth` dan disimpan pada direktori yang telah ditentukan.

# Metrik Evaluasi

## Confusion Matrix
Confusion Matrix adalah metrik yang memetakan jumlah jawaban benar dan salah model menebak data uji. Nilai metrik tersebut dikategorikan ke dalam 4 istilah:
1. True Positive (TP)
Pada data dengan kelas X, model menebak itu kelas X (Benar).
2. True Negative (TN)
Pada data dengan kelas Y, model menebak itu kelas Y, bukan kelas X (Benar). 
3. False Positive (FP)
Pada data dengan kelas Y, model menebak itu kelas X (Salah).
4. False Negative (FN)
Pada data dengan kelas X, model menebak itu kelas Y (Salah).

## Precision
Precision mengukur seberapa tepat model menebak benar suatu kelas dari total tebakan kelas tersebut. Pengukuran dirumuskan ke dalam rumus berikut:
- Precision = TP / (TP + FP)

## Recall
Recall mengukur seberapa tepat model menebak suatu kelas saat data dengan kelas tersebut dipanggil. Pengukuran dirumuskan ke dalam rumus berikut:
- Recall = TP / (TP + FN)

## F1-Score
F1-Score adalah penggabungan nilai rata-rata dari kedua metrik sebelumnya, yaitu Precision dan Recall.

## Kurva Loss
Kurva loss merupakan metrik pengukur keyakinan model dalam menebak data latih (train loss) maupun data uji (validation loss). Nilai loss akan bernilai tinggi jika model terlalu yakin ketika salah menebak kelas pada data, begitupun sebaliknya. Pembacaan kerdua kurva (train dan validation) loss tersebut penting dalam menganalisis apakah model tersebut mempelajari atau sekedar menghafal data.

Overfitting merupakan fenomena dimana model pintar menebak data latih, tetapi bodoh ketika menebak data uji. Fenomena ini digambarkan dengan kurva train loss yang menurun, tetapi kurva validation loss yang meningkat.

Underfitting merupakan fenomena dimana model payah dalam menebak data latih dan data uji. Ini digambarkan dengan kurva loss keduanya yang cukup tinggi dan sulit untuk turun.


# Struktur Dasar Teori
1. Environmental Sound Classification (ESC)
2. Konsep Suara Digital 
a. Sampling Rate
b. Bit Depth
c. Format Saluran Suara
3. UrbanSound8K (Dataset)
4. Preprocessing Data
a. Down-mixing
b. Resampling
5. Representasi Input Suara
5.1. Raw Waveform
5.2. Log-mel Spectrogram
6. Augmentasi Data
7. Convolutional Neural Network (CNN)
a. Kernel
b. Stride
c. Batch Normalization (BN)
d. Retrified Linear Unit (ReLU)
e. Max Pooling
f. Fully Connected Layer
g. Sigmoid
8. Pre-trained Audio Neural Networks (PANNs)
8.1. Res1dNet31
8.2. ResNet38
8.3. Wavegram_Logmel_Cnn14
9. Transfer Learning
a. Freeze Base
b. Fine Tuning
10. Strategi Optimasi dan Skenario Pelatihan
10.1. Hyperparameter Pelatihan
a. Epoch
b. Batch Size
10.2. Fungsi Kerugian dan Cost Sensitive Learning
a. Cross Entropy Loss
b. Cost Sensitive Learning
10.3. Optimizer (Adam & AdamW) dan Scheduler
a. Optimizer
- Learning Rate
. Adam
. AdamW
b. Scheduler
. ReduceLROnPlateau
10.4. Mitigasi Overfitting (Early Stopping dan Checkpointing)
a. Early Stopping
b. Checkpointing
11. Metrik Evaluasi
a. Confusion Matrix
b. Precision
c. Recall
d. F1-Score
e. Kurva Loss (Loss Curve)





