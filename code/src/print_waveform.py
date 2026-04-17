import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 1. Tentukan lokasi file audio kamu 
# (Misalnya suara sirene, ledakan, bor, atau klakson)
file_audio = os.path.join(os.getcwd(), 'audio_test', 'car_horn.wav')

# 2. Memuat / Membaca file audio menggunakan Librosa
# Librosa akan mengembalikan 2 hal:
# - y  : Array berisi angka-angka getaran suara (Amplitudo)
# - sr : Sample Rate (Berapa banyak titik getaran yang direkam per detik)
y, sr = librosa.load(file_audio, sr=None) 
# Catatan: sr=None agar Python membaca kecepatan asli dari file audionya.

# 3. Menyiapkan kanvas gambar (Panjang 14, Lebar 5)
plt.figure(figsize=(14, 5))

# 4. Menggambar gelombang "Detak Jantung" suaranya
librosa.display.waveshow(y, sr=sr, color='blue')

# 5. Memberikan judul dan label sumbu biar rapi dan akademis
plt.title('Raw Waveform - Suara Klakson Mobil', fontsize=16, fontweight='bold')
plt.xlabel('Waktu (Detik)', fontsize=12)
plt.ylabel('Amplitudo', fontsize=12)

# Menambahkan grid (garis bantu) biar grafiknya lebih mudah dibaca
plt.grid(True, linestyle='--', alpha=0.7)

# 6. Tampilkan gambar ke layar!
plt.show()