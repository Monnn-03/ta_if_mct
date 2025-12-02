import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import sys
import os

# =========================================================================
# 1. SETUP LIBRARY PANNS (Otomatis Import)
# =========================================================================
# Kita arahkan Python untuk mencari file 'models.py' milik PANNs
panns_root = os.path.join(os.getcwd(), 'audioset_tagging_cnn')
panns_code_path = os.path.join(panns_root, 'pytorch')

# Jika belum ada, download dulu
if not os.path.exists(panns_root):
    print("[INFO] Sedang meng-clone repository PANNs...")
    os.system('git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git')

# Masukkan ke path system agar bisa di-import
if panns_code_path not in sys.path:
    sys.path.insert(0, panns_code_path)

# C. Coba Import Cnn14 (Tanpa Try-Except biar ketahuan error aslinya)
try:
    from models import Cnn14
    print("[OK] SUKSES: Library PANNs (Cnn14) berhasil dimuat!")
except ImportError as e:
    print("\n" + "!"*50)
    print("[ERROR] ERROR FATAL: Gagal import Cnn14.")
    print(f"Pesan Error Python: {e}")
    print("!"*50)
    print("SOLUSI MUNGKIN:")
    print("1. Pastikan sudah install: 'pip install torchlibrosa'")
    print("2. Cek apakah folder 'audioset_tagging_cnn/pytorch' ada di folder proyekmu.")
    print("!"*50 + "\n")
    # Kita buat dummy class Cnn14 biar kodingan di bawah tidak crash syntax-nya
    class Cnn14: pass

# =========================================================================
# 2. MODEL A: PICZAK CNN (Training From Scratch)
# =========================================================================
# Ini kode tulisan tangan Anda tadi (sudah benar)
class PiczakCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(PiczakCNN, self).__init__()
        
        # Conv Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=80, kernel_size=(57, 6), stride=(1, 1), padding=(28, 2)),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3))
        self.dropout1 = nn.Dropout(0.5)

        # Conv Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        
        # Classifier
        self.flatten = nn.Flatten()
        
        # Trik Hitung Otomatis (Cerdas!)
        self.fc1_input_dim = None # Nanti diisi otomatis
        
        # Kita definisikan layer linear nanti di forward atau pakai LazyLinear
        # Biar aman dan sama persis kode Anda, kita pakai LazyLinear
        self.fc1 = nn.LazyLinear(5000) 
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(5000, 5000)
        self.relu2 = nn.ReLU()
        
        self.output_layer = nn.Linear(5000, num_classes)

    def forward(self, x):
        # x shape: [Batch, 2, Freq, Time]
        x = self.conv_block1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv_block2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        output = self.output_layer(x)
        return output

# =========================================================================
# 3. MODEL B: PANNS TRANSFER (Transfer Learning Wrapper)
# =========================================================================
class PANNsTransfer(nn.Module):
    def __init__(self, num_classes=4, freeze_base=True):
        super(PANNsTransfer, self).__init__()
        
        # 1. Panggil Model Asli (Cnn14)
        # Kita biarkan dia apa adanya (input 1 channel)
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, 
                                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                                classes_num=527)

        # 2. Load Bobot Pretrained
        pretrained_path = "Cnn14_mAP=0.431.pth"
        if not os.path.exists(pretrained_path):
            print("Downloading weights...")
            os.system(f"curl -L https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1 -o {pretrained_path}")

        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            self.base_model.load_state_dict(checkpoint['model'])

        # 3. MODIFIKASI PENTING (Bypass Preprocessing PANNs)
        # Karena data kita sudah jadi Spectrogram, kita matikan mesin masak PANNs
        self.base_model.spectrogram_extractor = nn.Identity()
        self.base_model.logmel_extractor = nn.Identity()

        # 4. Freeze (Bekukan Ilmu Lama)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # 5. Ganti Kepala (Output Layer)
        self.base_model.fc_audioset = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Input dari DataReader: [Batch, 2, Freq, Time]
        
        # PANNs cuma terima 1 Channel (Spectrogram). Channel 2 (Delta) kita buang.
        x = x[:, 0:1, :, :] 
        
        # PANNs minta input ditukar posisinya: [Batch, 1, Time, Freq]
        x = x.permute(0, 1, 3, 2) 
        
        output = self.base_model(x)
        return output['clipwise_output']

# =========================================================================
# 4. MAIN CHECK (Generate Laporan ke File)
# =========================================================================
if __name__ == "__main__":
    # Dummy Input (2 Channel)
    dummy_input = torch.randn(2, 2, 64, 41)
    
    # Nama file output
    output_file = "TorchInfoModel.txt"

    # Buka file dengan mode 'w' (write) dan encoding 'utf-8' (Wajib biar gak error)
    with open(output_file, "w", encoding="utf-8") as f:
        
        # --- 1. MODEL PICZAK ---
        title1 = "\n" + "="*50 + "\nMODEL 1: PICZAK CNN (From Scratch)\n" + "="*50 + "\n"
        print(title1) # Tampil di layar
        f.write(title1) # Tulis ke file
        
        piczak = PiczakCNN(num_classes=4)
        # verbose=0 artinya: Jangan print ke layar dulu, simpan dulu di variabel
        piczak_stats = summary(piczak, input_data=dummy_input, verbose=0,
                               col_names=["input_size", "output_size", "num_params"])
        
        print(piczak_stats) # Tampil di layar
        f.write(str(piczak_stats)) # Tulis ke file

        # --- 2. MODEL PANNS ---
        title2 = "\n\n" + "="*50 + "\nMODEL 2: PANNS TRANSFER (Transfer Learning)\n" + "="*50 + "\n"
        print(title2)
        f.write(title2)
        
        try:
            panns = PANNsTransfer(num_classes=4, freeze_base=True)
            panns_stats = summary(panns, input_data=dummy_input, verbose=0,
                                  col_names=["input_size", "output_size", "num_params"])
            
            print(panns_stats)
            f.write(str(panns_stats))
            
        except Exception as e:
            error_msg = f"Error loading PANNs: {e}"
            print(error_msg)
            f.write(error_msg)

    print(f"\n\n[SUKSES] Laporan arsitektur telah disimpan ke file: {output_file}")