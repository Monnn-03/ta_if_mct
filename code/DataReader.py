import os
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import json
from create_5fold_split import make_5fold_split
import random
import librosa
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class AudioDataset(Dataset):
    def __init__(self, root_dir, fold=0, split_json="split.json", split_type="train", target_sr=32000, fixed_length=320000):
        """
        Args:
            target_sr: Sample rate tujuan (Default PANNs = 32000 Hz)
            fixed_length: Panjang audio dalam sample. 
                          32000 Hz * 10 detik = 320000 sample.
                          Ini penting agar bisa dibatch.
        """
        self.target_sr = target_sr
        self.fixed_length = fixed_length
        
        # Mapping label UrbanSound8K
        self.labels_map = {
            'car_horn': 0, 'dog_bark': 1, 'gun_shot': 2, 'siren': 3
        }
        
        ### -- SPLIT HANDLING --
        split_path = os.path.join(os.getcwd(), split_json)
        
        # Cek apakah file split JSON ada
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                self.splits = json.load(f)
        else:
            # Plan B: Scan manual (Safety Net)
            print("[INFO] File split.json tidak ditemukan, mencoba scan manual...")
            all_samples = []
            for label in self.labels_map:
                label_dir = os.path.join(root_dir, label)
                if os.path.exists(label_dir):
                    for fname in os.listdir(label_dir):
                        fpath = os.path.join(label_dir, fname)
                        if fname.endswith('.wav') and os.path.isfile(fpath):
                            all_samples.append((fpath, label))
            
            # Buat split baru
            if all_samples:
                try:
                    folds = make_5fold_split(all_samples, n_folds=5)
                    with open(split_path, 'w') as f:
                        json.dump(folds, f, indent=2)
                    self.splits = folds
                except Exception as e:
                    print(f"[ERROR] Gagal membuat split: {e}")
                    self.splits = []
            else:
                self.splits = []

        # Memuat daftar sampel berdasarkan fold dan tipe (train/val)
        if self.splits and len(self.splits) > fold:
            self.samples = [(item["file_path"], item["label"]) for item in self.splits[fold][split_type]]
        else:
            self.samples = []
            print(f"[WARN] Split kosong untuk fold {fold} tipe {split_type}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # 1. Parsing Label yang Aman (Sesuai Arahan Pak Martin)
        # Nama file: 100032-3-0-0.wav -> Angka '3' adalah class ID asli
        try:
            filename = os.path.basename(file_path)
            parts = filename.split('-')
            # Ambil bagian index 1 (Class ID)
            original_class_id = int(parts[1])
            
            # Kita punya mapping sendiri (0-3), pastikan labelnya valid
            # Jika label di JSON string ('siren'), convert ke int
            if isinstance(label, str):
                label_id = self.labels_map.get(label, 0)
            else:
                label_id = label
        except:
            # Fallback jika nama file tidak standar US8K
            label_id = label if isinstance(label, int) else 0

        # 2. Load Audio (Waveform) pakai Librosa
        try:
            # sr=None agar membaca sample rate asli dulu
            audio_array, sr = librosa.load(file_path, sr=None)
            
            # Konversi ke Tensor PyTorch
            waveform = torch.tensor(audio_array, dtype=torch.float32)
            
            # Tambah dimensi Channel: (Time) -> (1, Time)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            # 3. Resampling ke 32k (Standar PANNs)
            if sr != self.target_sr:
                resampler = T.Resample(sr, self.target_sr)
                waveform = resampler(waveform)

            # 4. Mix Down to Mono (Jaga-jaga kalau stereo)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 5. Padding / Cutting (Agar panjangnya SAMA semua)
            # PANNs butuh input batch yang seragam panjangnya
            current_len = waveform.shape[1]
            
            if current_len > self.fixed_length:
                # KEPANJANGAN: Potong Random (Random Crop)
                # Biar model belajar bagian yang beda-beda tiap epoch
                start = random.randint(0, current_len - self.fixed_length)
                waveform = waveform[:, start:start+self.fixed_length]
            elif current_len < self.fixed_length:
                # KEPENDEKAN: Padding dengan nol di belakang
                padding = self.fixed_length - current_len
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # RETURN: 
            # waveform: [1, 320000] (Audio Mentah)
            # label_id: int (0-3)
            return waveform, label_id

        except Exception as e:
            print(f"[ERROR] Corrupt file {file_path}: {e}")
            # Return dummy nol biar training gak crash
            return torch.zeros(1, self.fixed_length), 0

if __name__ == "__main__":
    # Test DataReader
    print("--- Test DataReader (Waveform Output) ---")
    
    # Asumsi folder data ada di sini
    base_dir = os.path.join(os.getcwd(), 'data') 

    dataset = AudioDataset(root_dir=base_dir, fold=0, split_type="train")
    
    print(f"Total Samples: {len(dataset)}")
    
    if len(dataset) > 0:
        wave, lbl = dataset[0]
        print(f"\nSample 0:")
        print(f"Shape Waveform: {wave.shape}") # Harusnya [1, 320000]
        print(f"Label ID: {lbl}")
        
        print("\n✅ DataReader Siap! Input sekarang adalah Audio Mentah.")