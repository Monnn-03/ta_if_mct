import torch
import torch.nn as nn
import sys
import os

# --- SETUP IMPORT PANNs ---
# Pastikan folder ini ada. Kalau belum, git clone dulu.
panns_path = "audioset_tagging_cnn/pytorch"
if os.path.exists("audioset_tagging_cnn"):
    if panns_path not in sys.path:
        sys.path.insert(0, panns_path)
else:
    print("⚠️ WARNING: Folder 'audioset_tagging_cnn' tidak ditemukan.")
    print("   Solusi: Jalankan 'git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git'")

try:
    from models import ResNet38, Res1dNet31, Wavegram_Logmel_Cnn14
except ImportError:
    print("❌ ERROR: Gagal import model PANNs. Cek struktur folder.")

def download_weights(filename, url):
    if not os.path.exists(filename):
        print(f"⬇️ Downloading weights: {filename}...")
        # Gunakan wget atau curl (biar aman di windows/linux pakai try-except atau library requests)
        # Tapi os.system curl sudah cukup oke untuk dev
        os.system(f"curl -L \"{url}\" -o {filename}")
        print("✅ Download selesai.")

class AudioClassifier(nn.Module):
    def __init__(self, model_type, num_classes=4, freeze_base=False):
        """
        Args:
            freeze_base (bool): Jika True, lapisan awal tidak akan di-training (bobot beku).
                                Hanya layer akhir yang belajar. Bagus untuk epoch awal.
        """
        super(AudioClassifier, self).__init__()
        self.model_type = model_type
        
        # Konfigurasi Umum PANNs
        configs = {
            "sample_rate": 32000, "window_size": 1024, "hop_size": 320, 
            "mel_bins": 64, "fmin": 50, "fmax": 14000, "classes_num": 527
        }

        # --- PILIH ARSITEKTUR ---
        if model_type == "spectrogram":
            self.base = ResNet38(**configs)
            url = "https://zenodo.org/record/3987831/files/ResNet38_mAP%3D0.434.pth?download=1"
            path = "ResNet38_mAP=0.434.pth"
            
        elif model_type == "waveform":
            self.base = Res1dNet31(**configs)
            url = "https://zenodo.org/record/3987831/files/Res1dNet31_mAP%3D0.365.pth?download=1"
            path = "Res1dNet31_mAP=0.365.pth"
            
        elif model_type == "hybrid":
            self.base = Wavegram_Logmel_Cnn14(**configs)
            url = "https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1"
            path = "Wavegram_Logmel_Cnn14_mAP=0.439.pth"
            
        else:
            raise ValueError(f"Model '{model_type}' tidak dikenal (pilih: spectrogram, waveform, hybrid)")

        # --- LOAD WEIGHTS ---
        download_weights(path, url)
        try:
            # Load ke CPU biar aman memorinya pas inisialisasi
            ckpt = torch.load(path, map_location='cpu')
            self.base.load_state_dict(ckpt['model'], strict=False) 
            print(f"✅ Pre-trained weights loaded: {model_type}")
        except Exception as e:
            print(f"⚠️ Gagal load weights: {e}")

        # --- MODIFY OUTPUT LAYER ---
        # Layer fc_audioset outputnya 527, kita ganti jadi num_classes (4)
        # Input features tiap model beda? Tidak, ResNet38/Wavegram/Res1dNet rata-rata 2048
        self.base.fc_audioset = nn.Linear(2048, num_classes)

        # --- FREEZE OPTION ---
        if freeze_base:
            print("❄️  Membekukan lapisan awal (Freezing Feature Extractor)...")
            for name, param in self.base.named_parameters():
                # Kecualikan layer terakhir biar tetap bisa belajar
                if "fc_audioset" not in name:
                    param.requires_grad = False

    def forward(self, x, mixup_lambda=None):
        # Handle dimensi input dari DataLoader (Batch, 1, Length) -> (Batch, Length)
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            
        # PANNs return dictionary
        output = self.base(x, mixup_lambda=mixup_lambda)
        return output['clipwise_output']

if __name__ == "__main__":
    # Test instansiasi
    try:
        model = AudioClassifier("hybrid", num_classes=4)
        print("Test Hybrid Model:")
        # Buat dummy input 5 detik (Batch=2, Length=160000)
        dummy_input = torch.randn(2, 160000)
        output = model(dummy_input)
        print(f"Output Shape: {output.shape}") # Harusnya [2, 4]
        print("✅ Model siap digunakan!")
    except Exception as e:
        print(f"❌ Error: {e}")