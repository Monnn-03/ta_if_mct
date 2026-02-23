import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Load file contoh suara terompet
path = librosa.ex('trumpet')
y, sr = librosa.load(path)

# --- PROSES MATEMATIKANYA ---

# A. Spectrogram Standar (STFT) - Masih murni Hertz & Amplitudo
D = np.abs(librosa.stft(y))

# B. Mel-Spectrogram (Pakai Skala Mel, tapi belum Log)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# C. Log-Mel Spectrogram (Sudah Mel + Sudah Desibel/Log)
S_db = librosa.power_to_db(S, ref=np.max)

# --- VISUALISASI ---
plt.figure(figsize=(12, 12))

# Plot 1: Spectrogram Standar (Hertz)
plt.subplot(3, 1, 1)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f')
plt.title('1. Spectrogram Standar (Sumbu Y: Hertz - Masih Kaku)')

# Plot 2: Mel-Spectrogram (Skala Mel)
plt.subplot(3, 1, 2)
librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f')
plt.title('2. Mel-Spectrogram (Sumbu Y: Mel - Sudah Mirip Telinga)')

# Plot 3: Log-Mel Spectrogram (Mel + Desibel)
plt.subplot(3, 1, 3)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('3. Log-Mel Spectrogram (Mel + Log - Detail Muncul Semua!)')

plt.tight_layout()
plt.show()