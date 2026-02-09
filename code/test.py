import torch
from torch.utils.data import DataLoader, TensorDataset

# 1. Bikin Data Dummy (Angka 0 sampai 9)
data = torch.arange(10)
dataset = TensorDataset(data)

# 2. Bikin DataLoader DI LUAR LOOP (Seperti kodemu)

# Perhatikan shuffle=True

print("--- PEMBUKTIAN SHUFFLE ---")

# 3. Jalankan Loop Epoch
for epoch in range(3):
    print(f"\nEpoch {epoch+1}:")
    # Ambil batch pertama saja sebagai sampel
    for batch in loader:
        print(f"  Isi Batch: {batch[0].tolist()}")

# 