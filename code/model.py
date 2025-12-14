import torch
import torch.nn as nn
import sys
import os

# =========================================================================
# 1. SETUP LIBRARY PANNS
# =========================================================================
# Cek folder audioset_tagging_cnn
panns_path = "audioset_tagging_cnn/pytorch"
if os.path.exists("audioset_tagging_cnn"):
    if panns_path not in sys.path:
        sys.path.insert(0, panns_path)
else:
    # Cek di folder root (siapa tahu struktur folder beda)
    if os.path.exists("../audioset_tagging_cnn"):
        sys.path.insert(0, "../audioset_tagging_cnn/pytorch")

try:
    from models import ResNet38, Res1dNet31, Wavegram_Logmel_Cnn14
except ImportError:
    print("❌ [ERROR] Tidak bisa import PANNs. Pastikan folder 'audioset_tagging_cnn' ada.")

# =========================================================================
# 2. HELPER: DOWNLOAD WEIGHTS (Fitur Penting Anda)
# =========================================================================
def download_weights(filename, url):
    if not os.path.exists(filename):
        print(f"⬇️ Downloading Pre-trained Weights: {filename}...")
        # Gunakan curl sistem (atau bisa pakai requests kalau mau)
        os.system(f"curl -L \"{url}\" -o {filename}")
        print("✅ Download selesai.")

# =========================================================================
# 3. KELAS UTAMA: AudioClassifier
# (Gabungan Logic Anda + Logic Predict)
# =========================================================================
class AudioClassifier(nn.Module):
    def __init__(self, model_type, num_classes=4):
        super(AudioClassifier, self).__init__()
        self.model_type = model_type
        
        # --- A. MODEL SPECTROGRAM (ResNet38) ---
        if model_type == "spectrogram":
            # 1. Setup Arsitektur
            self.base_model = ResNet38(
                sample_rate=32000, window_size=1024, hop_size=320, 
                mel_bins=64, fmin=50, fmax=14000, classes_num=527
            )
            
            # 2. Auto-Download & Load Weights (Ini logic asli Anda)
            path = "ResNet38_mAP=0.434.pth"
            url = "https://zenodo.org/record/3987831/files/ResNet38_mAP%3D0.434.pth?download=1"
            download_weights(path, url)
            
            try:
                # Load weight PANNs (527 kelas) ke dalam otak model
                # map_location='cpu' biar aman di laptop non-GPU
                ckpt = torch.load(path, map_location='cpu')
                self.base_model.load_state_dict(ckpt['model'])
                print(f"✅ Loaded pre-trained weights: {path}")
            except Exception as e:
                print(f"⚠️ Warning: Gagal load weights {path}. Model kosongan. Error: {e}")

            # 3. Ganti Kepala (Head) untuk 4 Kelas
            # Ini dilakukan SETELAH load weights, supaya bagian ini fresh (random)
            self.base_model.fc_audioset = nn.Linear(2048, num_classes)

        # --- B. MODEL WAVEFORM (Res1dNet31) ---
        elif model_type == "waveform":
            self.base_model = Res1dNet31(
                sample_rate=32000, window_size=1024, hop_size=320, 
                mel_bins=64, fmin=50, fmax=14000, classes_num=527
            )
            
            path = "Res1dNet31_mAP=0.365.pth"
            url = "https://zenodo.org/record/3987831/files/Res1dNet31_mAP%3D0.365.pth?download=1"
            download_weights(path, url)

            try:
                ckpt = torch.load(path, map_location='cpu')
                self.base_model.load_state_dict(ckpt['model'])
                print(f"✅ Loaded pre-trained weights: {path}")
            except:
                print(f"⚠️ Warning: Gagal load weights {path}")

            self.base_model.fc_audioset = nn.Linear(2048, num_classes)

        # --- C. MODEL HYBRID (Wavegram-Logmel-Cnn14) ---
        elif model_type == "hybrid":
            self.base_model = Wavegram_Logmel_Cnn14(
                sample_rate=32000, window_size=1024, hop_size=320, 
                mel_bins=64, fmin=50, fmax=14000, classes_num=527
            )
            
            path = "Wavegram_Logmel_Cnn14_mAP=0.439.pth"
            url = "https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1"
            download_weights(path, url)
            
            try:
                ckpt = torch.load(path, map_location='cpu')
                self.base_model.load_state_dict(ckpt['model'])
                print(f"✅ Loaded pre-trained weights: {path}")
            except:
                print(f"⚠️ Warning: Gagal load weights {path}")

            self.base_model.fc_audioset = nn.Linear(2048, num_classes)
            
        else:
            raise ValueError(f"Model type '{model_type}' tidak dikenali!")

    def forward(self, x, mixup_lambda=None):
        # Input Handling: Ubah [Batch, 1, Time] jadi [Batch, Time] agar PANNs senang
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        output_dict = self.base_model(x, mixup_lambda=mixup_lambda)
        return output_dict['clipwise_output']

# --- Test Block ---
if __name__ == "__main__":
    print("Test Membangun Model Spectrogram dengan Auto-Download...")
    model = AudioClassifier("spectrogram", num_classes=4)
    print("✅ Berhasil dibangun.")