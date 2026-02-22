# Konsep Suara Digital
Suara yang biasa kita dengar merupakan suara dalam bentuk fisik.
Agar suara dapat didengar oleh mesin, suara perlu diterjemahkan ke dalam format yang bisa dimengerti bahasa mesin.
Konversi tersebut mengalami dua tahap, yaitu mengubah sinyal fisik menjadi sinyal analog, kemudian sinyal analog tersebut dikonversikan lagi menjadi sinyal digital.
Sinyal digital inilah yang dapat dimengerti oleh mesin, di mana bentuknya berubah dari gelombang fisik menjadi bentuk biner (terdiri dari 0 dan 1) yang dapat dimengerti oleh bahasa mesin.
Konversi tersebut perlu dilakukan karena mesin tidak dapat membaca suara secara langsung, melainkan mesin hanya dapat membaca nilai biner (diskrit).

Dalam penelitian ini, sinyal suara direpresentasikan ke dalam bentuk array yang berisikan nilai-nilai amplitudo berdasarkan durasinya. Amplitudo merupakan titik simpangan terjauh dalam gelombang suara. Nilai tersebut dapat berupa nilai positif bahkan negatif yang umumnya berada pada rentang -1 sampai 1. Semakin jauh nilai tersebut dari titik nol, maka getaran yang dihasilkan suara tersebut semakin tinggi (kencang).

# Sampling Rate
Pengambilan seluruh nilai amplitudo tersebut sangat bergantung pada kekuatan mesin dalam mengambil banyak sampel dalam satu waktu yang dilambangkan dengan istilah Sampling Rate, yaitu banyak nilai titik sampel yang diambil per detik durasi audio (Hz).
Sebagai contoh jika audio asli memiliki sampling rate 32.000 Hz, maka audio tersebut mengambil detail 32.000 titik sampel dalam 1 detik.
Semakin besar nilai sampling rate suatu audio, maka semakin jelas rekaman suara yang diambil.

# Format Saluran Suara
Detail pengambilan rekaman suara bergantung pada jumlah dan kualitas perangkat perekam. 
Banyak jumlah perangkat yang merekam dari berbeda sudut pandang menciptakan pengalaman mendengar suara rekaman yang realistik.
Maka dari itu, salah satu istilah penting dalam domain suara adalah saluran suara.

Dalam penelitian ini, terdapat dua format saluran suara yang dibahas, yaitu satu saluran (mono) dan dua saluran (stereo).
Keunggulan mendengar suara dengan format stereo ialah dapat mendengarkan suara kejadian pada rekaman dari sisi kiri dan kanan dibandingkan dengan format mono yang hanya merekam pada satu sisi saja. Mesin membaca format suara stereo yang menghasilkan dua kelompok nilai amplitudo yang dibungkus dalam dua array.

Merujuk pada konteks klasifikasi suara, pemilihan dan penyamarataan format mono pada dataset suara didasarkan oleh beberapa alasan. 
Klasifikasi suara bertujuan ingin mempelajari karakteristik suara, di mana format suara mono sudah cukup untuk merepresentasikan karakteristik suara yang ingin dipelajari dibandingkan dengan klasifikasi suara berformat stereo yang terkesan ikut mempelajari sisi pengambilan rekaman suara.
Selain menyesuaikan konteks klasifikasi, format suara mono mempermudah proses komputasi karena nilai amplitudo yang dibaca hanya berupa satu array, bukan dua array.

# Raw Waveform



