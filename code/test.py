import torch
from torch.utils.data import Dataset, DataLoader
import random

# --- 1. Kita buat Dataset Dummy ---
class SimpleDataset(Dataset):
    def __init__(self):
        # Data aslinya cuma angka 1 sampai 100
        self.data = list(range(1, 101))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Simulasi Augmentasi Temporal:
        # Kita ambil angka asli, lalu tambah angka random (noise)
        # Kalau di audio: ini seperti mengambil potongan detik yang beda
        original_val = self.data[idx]
        noise = random.randint(100, 999) 
        
        augmented_val = original_val + noise
        print(f"Noise: {noise}")
        return original_val, noise, augmented_val

# --- 2. Setup DataLoader (DI LUAR LOOP - Cara Benar) ---
dataset = SimpleDataset()
loader = DataLoader(dataset, batch_size=5, shuffle=False)

print("🔍 BUKTI BAHWA AUGMENTASI BERUBAH TIAP EPOCH")
print("="*50)

# --- 3. Loop Epoch ---
for epoch in range(1, 3): # Kita coba 2 epoch saja
    print(f"\n🚀 EPOCH {epoch}")
    
    # Ambil batch pertama saja sebagai contoh
    # Syntax 'iter(loader).next()' atau loop biasa akan memicu __getitem__
    first_batch = next(iter(loader)) 
    
    originals, noise, augmenteds = first_batch
    
    print(f"   Data Asli (Urutan Shuffle): {originals.tolist()}")
    print(f"   Noise (Random):            {noise.tolist()}")
    print(f"   Data Augmentasi (Random):   {augmenteds.tolist()}")

print("\n" + "="*50)
print("KESIMPULAN:")
print("Lihat 'Data Augmentasi' di Epoch 1 vs Epoch 2.")
print("Meskipun DataLoader dibuat di luar loop, nilainya BERBEDA kan?")