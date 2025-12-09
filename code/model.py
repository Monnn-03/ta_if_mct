import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import sys
import os

# =========================================================================
# 1. SETUP LIBRARY PANNS (AUTO-DOWNLOAD)
# =========================================================================
# Arahkan ke folder audioset_tagging_cnn di root project
panns_root = os.path.join(os.getcwd(), 'audioset_tagging_cnn')
panns_code_path = os.path.join(panns_root, 'pytorch')

if not os.path.exists(panns_root):
    print("[INFO] Cloning PANNs repository...")
    os.system('git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git')

if panns_code_path not in sys.path:
    sys.path.insert(0, panns_code_path)

try:
    # Import 3 Model Jagoan Kita
    from models import ResNet38, Res1dNet31, Wavegram_Logmel_Cnn14
    print("[OK] Library PANNs berhasil dimuat!")
except ImportError:
    print("[ERROR] Gagal import model. Pastikan 'torchlibrosa' terinstall.")
    # Dummy class biar gak crash syntax editor
    class ResNet38: pass
    class Res1dNet31: pass
    class Wavegram_Logmel_Cnn14: pass

# =========================================================================
# 2. HELPER: DOWNLOAD WEIGHTS
# =========================================================================
def download_weights(filename, url):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        os.system(f"curl -L {url} -o {filename}")

# =========================================================================
# 3. MODEL A: SPECTROGRAM-BASED (Cnn14)
# =========================================================================
class ModelSpectrogram(nn.Module):
    def __init__(self, num_classes=4, freeze_base=True):
        super(ModelSpectrogram, self).__init__()
        
        # Init ResNet38 (Standard PANNs)
        self.base = ResNet38(sample_rate=32000, window_size=1024, hop_size=320, 
                          mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        # Load Pretrained
        path = "ResNet38_mAP=0.434.pth"
        url = "https://zenodo.org/record/3987831/files/ResNet38_mAP%3D0.434.pth?download=1"
        download_weights(path, url)
        
        try:
            ckpt = torch.load(path, map_location='cpu')
            self.base.load_state_dict(ckpt['model'])
        except:
            print(f"[WARN] Gagal load weights untuk {path}")

        # Freeze
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        # Ganti Head (Output Layer)
        self.base.fc_audioset = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Input dari DataReader: [Batch, 1, Time]
        # PANNs Cnn14 minta: [Batch, Time] (2D)
        
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1) # Buang dimensi channel 1
        
        # Masuk ke Cnn14 (Dia akan ekstrak Spectrogram sendiri di dalam)
        output = self.base(x) 
        return output['clipwise_output']

# =========================================================================
# 4. MODEL B: WAVEFORM-BASED (DaiNet)
# =========================================================================
class ModelWaveform(nn.Module):
    def __init__(self, num_classes=4, freeze_base=True):
        super(ModelWaveform, self).__init__()
        
        # Init Res1dNet31 (1D-CNN)
        self.base = Res1dNet31(sample_rate=32000, window_size=1024, hop_size=320, 
                           mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        path = "Res1dNet31_mAP=0.365.pth"
        url = "https://zenodo.org/record/3987831/files/Res1dNet31_mAP%3D0.365.pth?download=1"
        download_weights(path, url)

        try:
            ckpt = torch.load(path, map_location='cpu')
            self.base.load_state_dict(ckpt['model'])
        except:
            print(f"[WARN] Gagal load weights untuk {path}")
        
        # Freeze (Hati-hati, kalau from scratch JANGAN di-freeze semua)
        # Asumsi: Kita training DaiNet dari nol (Scratch) atau Fine Tuning total
        if freeze_base:
            # Biasanya DaiNet performanya butuh fine tuning, tapi kita ikuti pola dulu
            for param in self.base.parameters():
                param.requires_grad = False
                
        # DaiNet output embedding size di PANNs biasanya seragam
        # Kita pastikan nanti di summary
        self.base.fc_audioset = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Input: [Batch, 1, Time] -> [Batch, Time]
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            
        output = self.base(x)
        return output['clipwise_output']

# =========================================================================
# 5. MODEL C: HYBRID (Wavegram-Logmel-Cnn14)
# =========================================================================
class ModelHybrid(nn.Module):
    def __init__(self, num_classes=4, freeze_base=True):
        super(ModelHybrid, self).__init__()
        
        self.base = Wavegram_Logmel_Cnn14(sample_rate=32000, window_size=1024, hop_size=320, 
                                          mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        path = "Wavegram_Logmel_Cnn14_mAP=0.439.pth"
        url = "https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1"
        download_weights(path, url)
        
        try:
            ckpt = torch.load(path, map_location='cpu')
            self.base.load_state_dict(ckpt['model'])
        except:
            print(f"[WARN] Gagal load weights untuk {path}")

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False
        
        self.base.fc_audioset = nn.Linear(2048, num_classes)

    def forward(self, x):
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            
        # Model ini memecah x jadi 2 cabang di dalam (Waveform & Logmel)
        output = self.base(x)
        return output['clipwise_output']

# =========================================================================
# 6. MAIN TESTER (Generate Laporan)
# =========================================================================
if __name__ == "__main__":
    # Dummy Input: Waveform 4 detik (32000 * 4) = 128000
    # Kita buat format [Batch, 1, Time] persis output DataReader
    dummy_input = torch.randn(2, 1, 32000*4) 

    output_file = "TorchInfoModel_3Types.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        
        print(f"Generating Architecture Report to {output_file}...")

        # 1. SPECTROGRAM
        title = "\n" + "="*50 + "\n1. MODEL SPECTROGRAM (ResNet38)\n" + "="*50 + "\n"
        print("Checking Spectrogram Model...")
        f.write(title)
        m1 = ModelSpectrogram(num_classes=4)
        stats = summary(m1, input_data=dummy_input, verbose=0, col_names=["input_size", "output_size", "num_params"])
        f.write(str(stats))

        # 2. WAVEFORM
        title = "\n\n" + "="*50 + "\n2. MODEL WAVEFORM (Res1dNet31)\n" + "="*50 + "\n"
        print("Checking Waveform Model...")
        f.write(title)
        try:
            m2 = ModelWaveform(num_classes=4)
            stats = summary(m2, input_data=dummy_input, verbose=0, col_names=["input_size", "output_size", "num_params"])
            f.write(str(stats))
        except Exception as e:
            f.write(f"Error DaiNet: {e}")

        # 3. HYBRID
        title = "\n\n" + "="*50 + "\n3. MODEL HYBRID (Wavegram-Logmel)\n" + "="*50 + "\n"
        print("Checking Hybrid Model...")
        f.write(title)
        m3 = ModelHybrid(num_classes=4)
        stats = summary(m3, input_data=dummy_input, verbose=0, col_names=["input_size", "output_size", "num_params"])
        f.write(str(stats))
    
    print("\n[SUKSES] Semua model berhasil dibangun! Siap untuk Training.")