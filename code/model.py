import torch
import torch.nn as nn
import sys
import os

# =========================================================================
# 1. SETUP LIBRARY PANNS
# =========================================================================
panns_path = "audioset_tagging_cnn/pytorch"
if os.path.exists("audioset_tagging_cnn"):
    if panns_path not in sys.path:
        sys.path.insert(0, panns_path)
else:
    if os.path.exists("../audioset_tagging_cnn"):
        sys.path.insert(0, "../audioset_tagging_cnn/pytorch")

try:
    from models import ResNet38, Res1dNet31, Wavegram_Logmel_Cnn14
except ImportError:
    print("❌ [ERROR] Tidak bisa import PANNs. Pastikan folder 'audioset_tagging_cnn' ada.")

# =========================================================================
# 2. HELPER: DOWNLOAD WEIGHTS
# =========================================================================
def download_weights(filename, url):
    if not os.path.exists(filename):
        print(f"⬇️ Downloading Pre-trained Weights: {filename}...")
        os.system(f"curl -L \"{url}\" -o {filename}")
        print("✅ Download selesai.")

# =========================================================================
# 3. KELAS UTAMA: AudioClassifier (REVISI: base_model -> base)
# =========================================================================
class AudioClassifier(nn.Module):
    def __init__(self, model_type, num_classes=4):
        super(AudioClassifier, self).__init__()
        self.model_type = model_type
        
        # --- A. MODEL SPECTROGRAM (ResNet38) ---
        if model_type == "spectrogram":
            # Perhatikan: Di sini kita pakai self.base (BUKAN self.base_model)
            self.base = ResNet38(
                sample_rate=32000, window_size=1024, hop_size=320, 
                mel_bins=64, fmin=50, fmax=14000, classes_num=527
            )
            
            path = "ResNet38_mAP=0.434.pth"
            url = "https://zenodo.org/record/3987831/files/ResNet38_mAP%3D0.434.pth?download=1"
            download_weights(path, url)
            
            try:
                ckpt = torch.load(path, map_location='cpu')
                self.base.load_state_dict(ckpt['model'])
                print(f"✅ Loaded pre-trained weights: {path}")
            except Exception as e:
                print(f"⚠️ Warning: Gagal load weights {path}. Error: {e}")

            self.base.fc_audioset = nn.Linear(2048, num_classes)

        # --- B. MODEL WAVEFORM (Res1dNet31) ---
        elif model_type == "waveform":
            self.base = Res1dNet31(
                sample_rate=32000, window_size=1024, hop_size=320, 
                mel_bins=64, fmin=50, fmax=14000, classes_num=527
            )
            
            path = "Res1dNet31_mAP=0.365.pth"
            url = "https://zenodo.org/record/3987831/files/Res1dNet31_mAP%3D0.365.pth?download=1"
            download_weights(path, url)

            try:
                ckpt = torch.load(path, map_location='cpu')
                self.base.load_state_dict(ckpt['model'])
                print(f"✅ Loaded pre-trained weights: {path}")
            except:
                print(f"⚠️ Warning: Gagal load weights {path}")

            self.base.fc_audioset = nn.Linear(2048, num_classes)

        # --- C. MODEL HYBRID (Wavegram-Logmel-Cnn14) ---
        elif model_type == "hybrid":
            self.base = Wavegram_Logmel_Cnn14(
                sample_rate=32000, window_size=1024, hop_size=320, 
                mel_bins=64, fmin=50, fmax=14000, classes_num=527
            )
            
            path = "Wavegram_Logmel_Cnn14_mAP=0.439.pth"
            url = "https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1"
            download_weights(path, url)
            
            try:
                ckpt = torch.load(path, map_location='cpu')
                self.base.load_state_dict(ckpt['model'])
                print(f"✅ Loaded pre-trained weights: {path}")
            except:
                print(f"⚠️ Warning: Gagal load weights {path}")

            self.base.fc_audioset = nn.Linear(2048, num_classes)
            
        else:
            raise ValueError(f"Model type '{model_type}' tidak dikenali!")

    def forward(self, x, mixup_lambda=None):
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        # Panggil self.base (bukan self.base_model)
        output_dict = self.base(x, mixup_lambda=mixup_lambda)
        return output_dict['clipwise_output']

if __name__ == "__main__":
    print("Test Model Building...")
    model = AudioClassifier("spectrogram", num_classes=4)
    print("✅ Struktur Aman.")