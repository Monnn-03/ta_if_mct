import torch

# --- TAHAP 1: SI MONO YANG "POLOS" ---
# Anggap ini hasil librosa.load() yang cuma 3 titik sampel suara
data_awal = torch.tensor([0.2, 0.4, 0.6]) 
print(f"1. Data Awal (Mono): {data_awal}")
print(f"   Bentuk (Shape): {data_awal.shape}") # Hasilnya: torch.Size([3])
print(f"   Dimensi (ndim): {data_awal.ndim}D")

print("-" * 30)

# --- TAHAP 2: PROSES UNSQUEEZE (Bungkus Kado) ---
# Kita tambah dimensi di posisi 0 (paling depan)
data_unsqueeze = data_awal.unsqueeze(0)
print(f"2. Setelah Unsqueeze(0): {data_unsqueeze}")
print(f"   Bentuk (Shape): {data_unsqueeze.shape}") # Hasilnya: torch.Size([1, 3])
print(f"   Dimensi (ndim): {data_unsqueeze.ndim}D")

print("-" * 30)

# --- TAHAP 3: SI STEREO (Dua Jalur) ---
# Anggap baris 1 suara kiri, baris 2 suara kanan
data_stereo = torch.tensor([
    [0.2, 0.4, 0.6], # Kiri
    [0.8, 0.2, 0.4]  # Kanan
])
print(f"3. Data Stereo:")
print(data_stereo)
print(f"   Bentuk (Shape): {data_stereo.shape}") # Hasilnya: torch.Size([2, 3])

print("-" * 30)

# --- TAHAP 4: MEAN (Campur Jadi Mono) ---
# Kita ambil rata-rata dari jalur atas dan bawah
data_mixed = torch.mean(data_stereo, dim=0, keepdim=True)
print(f"4. Hasil Mix (Rata-rata): {data_mixed}")
# (0.2 + 0.8)/2 = 0.5, dst.
print(f"   Bentuk (Shape): {data_mixed.shape}") # Hasilnya: torch.Size([1, 3])