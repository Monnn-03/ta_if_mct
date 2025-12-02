import os
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import json
from create_5fold_split import make_5fold_split
import random
import torch.nn.functional as F
from utils import preview_mel_spectrogram
import warnings
import librosa  # <--- KITA PAKAI INI SEKARANG
import numpy as np

warnings.filterwarnings("ignore")

class AudioDataset(Dataset):
    def __init__(self, root_dir, fold=0, split_json="split.json", split_type="train", segment_length=41):
        self.labels_map = {
            'car_horn': 0,
            'dog_bark': 1,
            'gun_shot': 2,
            'siren': 3
        }
        
        self.segment_length = segment_length
        
        # --- KONFIGURASI AUDIO (Sesuaikan dengan PANNs: 32000Hz) ---
        self.target_sample_rate = 32000  
        
        ### -- SPLIT HANDLING --
        split_path = os.path.join(os.getcwd(), split_json)
        
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                self.splits = json.load(f)
        else:
            # Plan B: Scan manual jika JSON tidak ada
            all_samples = []
            for label in self.labels_map:
                label_dir = os.path.join(root_dir, label)
                if os.path.exists(label_dir):
                    for fname in os.listdir(label_dir):
                        fpath = os.path.join(label_dir, fname)
                        if fname.endswith('.wav') and os.path.isfile(fpath):
                            all_samples.append((fpath, label))
            
            try:
                folds = make_5fold_split(all_samples, n_folds=5)
                with open(split_path, 'w') as f:
                    json.dump(folds, f, indent=2)
                self.splits = folds
            except Exception as e:
                print(f"Gagal membuat split: {e}")
                self.splits = []
        
        # Transformasi Mel Spectrogram
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=1024,
            hop_length=320, # 10ms hop untuk 32k
            n_mels=64
        )
        
        self.delta_transform = T.ComputeDeltas()
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
        
        # Memuat daftar sampel
        if self.splits:
            self.samples = [(item["file_path"], item["label"]) for item in self.splits[fold][split_type]]
        else:
            self.samples = []
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Cek file
        if not os.path.isfile(file_path):
            print(f"File tidak ditemukan: {file_path}")
            return file_path, label, None
        
        try:
            # === [PERUBAHAN UTAMA: GANTI KE LIBROSA] ===
            # Kita GANTI torchaudio.load dengan librosa.load
            # sr=None agar sample rate asli terbaca
            audio_array, sr = librosa.load(file_path, sr=None) 
            
            # Ubah Numpy ke Tensor
            waveform = torch.tensor(audio_array, dtype=torch.float32)
            
            # Tambah dimensi Channel (Mono -> [1, time])
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            # ===========================================
            
            # 1. Resampling
            if sr != self.target_sample_rate:
                resampler = T.Resample(sr, self.target_sample_rate)
                waveform = resampler(waveform)
            
            # Mixdown Stereo ke Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 2. Ekstraksi Fitur Mel Spectrogram
            mel_spec = self.mel_spectrogram(waveform)
            log_mel_spec = self.amplitude_to_db(mel_spec)
            
            # 3. Segmentasi (Padding)
            current_len = log_mel_spec.shape[2]
            if current_len < self.segment_length:
                padding = self.segment_length - current_len
                log_mel_spec = F.pad(log_mel_spec, (0, padding))
                current_len = self.segment_length # Update panjang setelah padding
            
            # 4. Random Crop
            if current_len > self.segment_length:
                start_frame = random.randint(0, current_len - self.segment_length)
                segment = log_mel_spec[:, :, start_frame:start_frame + self.segment_length]
            else:
                segment = log_mel_spec # Jika pas, ambil semua
            
            # 5. Delta Features
            deltas = self.delta_transform(segment)
            
            # 6. Stack
            stacked_features = torch.cat((segment, deltas), dim=0)
            
            # Handle label jika masih string
            if isinstance(label, str):
                label_id = self.labels_map[label]
            else:
                label_id = label
            
            return file_path, label_id, stacked_features

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Return dummy biar gak crash total
            return file_path, label, None
        
if __name__ == "__main__":
    # Sesuaikan path data Anda
    data_dir = os.path.join(os.getcwd(), 'data') # Asumsi folder utama UrbanSound8K
    
    # Pastikan split.json sudah ada (dari create_splits.py)
    dataset = AudioDataset(root_dir=data_dir, fold=0, split_json="split.json", split_type="train")
    
    print(f"Jumlah sampel dalam dataset: {len(dataset)}")
    
    if len(dataset) > 0:
        # Ambil sampel random
        random_idx = random.randint(0, len(dataset) - 1)
        sample_data = dataset[random_idx]
        file_path, label, features = sample_data
        
        print(f"File path: {file_path}") 
        print(f"Label ID: {label}")
        
        if features is not None:
            print(f"Stacked features shape: {features.shape}")
            # Preview Channel 0 (Spectrogram Asli)
            # Pastikan utils.py Anda sudah benar atau matikan baris ini jika masih error
            try:
                preview_mel_spectrogram(file_path, label, features[0].unsqueeze(0))
            except Exception as e:
                print(f"Gagal preview gambar (tapi data aman): {e}")
        else:
            print("Fitur kosong (Error loading).")