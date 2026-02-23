# Konsep Suara Digital
Suara yang biasa kita dengar merupakan suara dalam bentuk fisik.
Agar suara dapat didengar oleh mesin, suara perlu diterjemahkan ke dalam format yang bisa dimengerti bahasa mesin.
Konversi tersebut mengalami dua tahap, yaitu mengubah sinyal fisik menjadi sinyal analog, kemudian sinyal analog tersebut dikonversikan lagi menjadi sinyal digital.
Sinyal digital inilah yang dapat dimengerti oleh mesin, di mana bentuknya berubah dari gelombang fisik menjadi bentuk biner (terdiri dari 0 dan 1) yang dapat dimengerti oleh bahasa mesin.
Konversi tersebut perlu dilakukan karena mesin tidak dapat membaca suara secara langsung, melainkan mesin hanya dapat membaca nilai biner (diskrit).

## Sampling Rate
Pengambilan seluruh nilai amplitudo tersebut sangat bergantung pada kekuatan mesin dalam mengambil banyak sampel dalam satu waktu yang dilambangkan dengan istilah Sampling Rate, yaitu banyak nilai titik sampel yang diambil per detik durasi audio (Hz).
Sebagai contoh jika audio asli memiliki sampling rate 32.000 Hz, maka audio tersebut mengambil detail 32.000 titik sampel dalam 1 detik.
Semakin besar nilai sampling rate suatu audio, maka semakin jelas rekaman suara yang diambil.

## Format Saluran Suara
Detail pengambilan rekaman suara bergantung pada jumlah dan kualitas perangkat perekam. 
Banyak jumlah perangkat yang merekam dari berbeda sudut pandang menciptakan pengalaman mendengar suara rekaman yang realistik.
Maka dari itu, salah satu istilah penting dalam domain suara adalah saluran suara.

Dalam penelitian ini, terdapat dua format saluran suara yang dibahas, yaitu satu saluran (mono) dan dua saluran (stereo).
Keunggulan mendengar suara dengan format stereo ialah dapat mendengarkan suara kejadian pada rekaman dari sisi kiri dan kanan dibandingkan dengan format mono yang hanya merekam pada satu sisi saja. Mesin membaca format suara stereo yang menghasilkan dua kelompok nilai amplitudo yang dibungkus dalam dua array.

Merujuk pada konteks klasifikasi suara, pemilihan dan penyamarataan format mono pada dataset suara didasarkan oleh beberapa alasan. 
Klasifikasi suara bertujuan ingin mempelajari karakteristik suara, di mana format suara mono sudah cukup untuk merepresentasikan karakteristik suara yang ingin dipelajari dibandingkan dengan klasifikasi suara berformat stereo yang terkesan ikut mempelajari sisi pengambilan rekaman suara.
Selain menyesuaikan konteks klasifikasi, format suara mono mempermudah proses komputasi karena nilai amplitudo yang dibaca hanya berupa satu array, bukan dua array.

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

## Hybrid
Jenis input Hybrid adalah penggabungan kedua fitur suara yang telah dijelaskan sebelumnya (Raw Waveform dan Log-mel Spectrogram). 
Ini merupakan ide dari penemu PANNs itu sendiri, yaitu Qiuqiangkong, yang bertujuan untuk menciptakan model dengan pembelajaran variasi input yang lebih detil.
Proses ini melibatkan dua kejadian yang berjalan secara paralel, di mana model dengan input Hybrid akan memiliki dua cabang input yang masing-masing meminta input Raw Waveform.
Pada cabang pertama, Raw Waveform akan diubah menjadi Log-mel Spectrogram dengan proses FFT.
Berbeda dengan cabang lainnya yang membiarkan model tersebut memproses Raw Waveform menjadi sebuah bentuk 2 dimensi, yang dinamakan Wavegram.

# Pre-trained Audio Neural Networks (PANNs)

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








