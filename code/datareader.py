import os
import torch
import json
import random
import librosa
import numpy as np
import warnings
from torch.utils.data import Dataset
import torchaudio.transforms as T

warnings.filterwarnings("ignore")

class AudioDataset(Dataset):
    def __init__(self, split_json="split_data.json", fold="fold1", split_type="train", target_sr=32000, fixed_length=160000):
        """
        Args:
            split_json: Path ke file JSON.
            fold: String "fold1", "fold2", dst (sesuai key di JSON).
            split_type: "train" atau "test".
            target_sr: 32000 Hz (Standar PANNs).
            fixed_length: 160000 sample (5 detik). 
                          (Catatan: US8K rata-rata <4 detik. 5 detik lebih efisien daripada 10 detik).
        """
        self.target_sr = target_sr
        self.fixed_length = fixed_length
        self.split_type = split_type
        
        # --- 1. LOAD JSON ---
        if not os.path.exists(split_json):
            raise FileNotFoundError(f"[ERROR] File {split_json} tidak ditemukan! Jalankan create_5fold_split.py dulu.")
            
        with open(split_json, 'r') as f:
            full_data = json.load(f)
            
        # --- 2. AMBIL DATA SESUAI FOLD ---
        # Struktur JSON kita: full_data['fold1']['train']['files']
        try:
            target_data = full_data[fold][split_type]
            self.file_paths = target_data['files']
            self.labels = target_data['labels']
        except KeyError:
            raise KeyError(f"[ERROR] Fold '{fold}' atau tipe '{split_type}' tidak ada di JSON.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Ambil path dan label langsung dari list (tidak perlu parsing nama file lagi)
        file_path = self.file_paths[idx]
        label_id = self.labels[idx] # Sudah berupa integer 0, 1, 2, 3
        
        # --- 3. LOAD AUDIO ---
        try:
            # Load pakai librosa (aman untuk berbagai format)
            # sr=None agar kita tahu SR aslinya dulu
            audio_array, sr = librosa.load(file_path, sr=None)
            
            # Ubah ke Tensor
            waveform = torch.tensor(audio_array, dtype=torch.float32)
            
            # Pastikan dimensi (Channel, Time) -> (1, N)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0) # Tambah channel dim

            # --- 4. RESAMPLING ---
            if sr != self.target_sr:
                # Gunakan torchaudio transforms karena lebih cepat di GPU nantinya
                resampler = T.Resample(sr, self.target_sr)
                waveform = resampler(waveform)

            # --- 5. MIX TO MONO ---
            # Jika audio stereo (2 channel), rata-rata kan jadi mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # --- 6. PADDING / CROPPING (FIX LENGTH) ---
            current_len = waveform.shape[1]
            
            if current_len > self.fixed_length:
                # KEPANJANGAN: Potong
                if self.fixed_length < current_len:
                    if self.split_type == "train":
                        start = random.randint(0, current_len - self.fixed_length)
                    else:
                        start = (current_len - self.fixed_length) // 2
                    waveform = waveform[:, start:start+self.fixed_length]
                    
            elif current_len < self.fixed_length:
                # KEPENDEKAN: Tambah nol (Zero Padding)
                pad_amount = self.fixed_length - current_len
                # Pad format: (padding_kiri, padding_kanan)
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

            # RETURN FINAL
            # Waveform: Tensor [1, 160000]
            # Label: Tensor Scalar (Long)
            return waveform, torch.tensor(label_id).long()

        except Exception as e:
            print(f"❌ GAGAL LOAD: {file_path}")
            print(f"   Errornya: {e}")
            # ---------------------
            print(f"[WARNING] Gagal memuat file...")
            # Return dummy nol biar training tidak berhenti total
            return torch.zeros(1, self.fixed_length), torch.tensor(0).long()

# --- BLOK TEST (PENTING) ---
if __name__ == "__main__":
    import config # Pastikan config.py ada
    
    print("🧪 Testing AudioDataset...")
    
    # Coba load dataset (Pastikan split_data.json sudah ada)
    # Sesuaikan argumen dengan file config kamu
    try:
        ds = AudioDataset(
            split_json="split_data.json", 
            fold="fold1", 
            split_type="train",
            target_sr=32000,
            fixed_length=32000 * 5 # 5 Detik
        )
        
        print(f"✅ Dataset berhasil dimuat! Jumlah sample: {len(ds)}")
        
        # Coba ambil 1 data
        wave, label = ds[0]
        print(f"   Output Shape : {wave.shape}") # Harusnya [1, 160000]
        print(f"   Label ID     : {label} (Tipe: {type(label)})")
        
        # Cek apakah Label ID valid
        mapping = config.IDX_TO_CLASS # Dari config.py
        print(f"   Arti Label   : {mapping[label.item()]}")

    except Exception as e:
        print(f"❌ Terjadi Error saat testing: {e}")