# BAB III : Metode Penelitian

## Langkah Penelitian

## Dataset
- Dataset yang digunakan adalah dataset publik UrbanSound8k. Dataset didownload secara manual dari Kaggle berbentuk .zip. Dataset tersebut kemudian diekstrak dan berisikan 10 folder yang masing-masing mewakili 1 fold beserta file .csv yang menampilkan tabel nama file audio beserta metadatanya.

## Pre-processing Data
- Dikarenakan Dataset UrbanSound8K telah menyusun 10 fold dalam 10 folder masing-masing,Maka akan digabungkan 10 fold menjadi 5 fold. Langkah ini akan menghemat waktu train serta membagi porsi data yang lebih adil untuk train dan test.
- Setelah menggabungkan 10 fold tadi, 5 fold tersebut masih terdiri dari kelas-kelas yang tidak semuanya digunakan. Maka dari itu, akan dipisahkan dan disisakan 4 kelas yang mewakili suara ancaman bahaya, yaitu gun_shot, siren, dog_bark, dan car_horn pada 5 fold tersebut. 
- Menyeimbangkan jumlah distribusi data antar kelas, dengan Sistem Weight Penalty. Weight Penalty adalah metode untuk menyeimbangkan performa akurasi pada kelas tertentu yang memiliki keterbatasan jumlah sampel. Model yang salah menebak kelas dengan jumlah data sampel yang lebih sedikit akan diberikan poin penalti lebih banyak dibandingkan model yang salah menebak kelas dengan jumlah data cukup.

## Konfigurasi 3 Model
- Waveform = Res1dNet31
- Spectrogram = ResNet38
- Hybrid = Wavegram-Logmel-CNN14

## Train
- Devices
- Platform

## Evaluasi
- Confusion Matrix (Untuk melihat jumlah tebakan benar pada suara tertentu)
- F1-Score (Untuk melihat apakah semua suara yang sesuai kelasnya terdeteksi, melihat seberapa presisi tebakan dia)
- Grafik loss & accuracy (Untuk melihat proses model belajar)
- Waktu training (Untuk menjadi acuan performa komputasi model)

## Analisis
- Buat tabel perbandingannya