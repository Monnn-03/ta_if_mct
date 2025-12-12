import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import time

# --- IMPORT DARI KODE KITA SENDIRI ---
from code.datareaderr import AudioDataset
from model import ModelSpectrogram, ModelWaveform, ModelHybrid

# ==============================================================================
# 1. KONFIGURASI (Pusat Kontrol)
# ==============================================================================
CONFIG = {
    "project_name": "TA_SoundClassification",
    "model_type": "waveform",  # PILIHAN: 'spectrogram', 'waveform', 'hybrid'
    "num_classes": 4,
    "batch_size": 16,        # Turunkan jadi 8 jika VRAM GPU habis/Error Out of Memory
    "learning_rate": 0.001,  # Standar PANNs 1e-3
    "epochs": 15,            # Jumlah putaran belajar per fold
    "folds": 5,              # Total Fold (0-4)
    "num_workers": 2,        # Jumlah asisten CPU untuk load data (Windows max 2-4 aman)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "models_saved"
}

# Buat folder penyimpanan jika belum ada
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==============================================================================
# 2. FUNGSI PELATIHAN (Loop Satu Epoch)
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # Mode Latihan (Aktifkan Dropout)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Pakai tqdm untuk progress bar yang keren
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. Reset Gradients (Hapus sisa hitungan sebelumnya)
        optimizer.zero_grad()
        
        # 2. Forward Pass (Menebak)
        outputs = model(inputs) # Output shape: [Batch, 4]
        
        # 3. Hitung Error (Loss)
        loss = criterion(outputs, labels)
        
        # 4. Backward Pass (Belajar/Koreksi Bobot)
        loss.backward()
        optimizer.step()
        
        # Hitung Statistik
        running_loss += loss.item()
        _, predicted = outputs.max(1) # Ambil kelas dengan skor tertinggi
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ==============================================================================
# 3. FUNGSI VALIDASI (Ujian)
# ==============================================================================
def validate(model, loader, criterion, device):
    model.eval() # Mode Ujian (Matikan Dropout, Bekukan BatchNorm)
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # Jangan hitung gradien (Hemat memori)
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ==============================================================================
# 4. FUNGSI UTAMA (MAIN LOOP)
# ==============================================================================
def run_training():
    print(f"🚀 Memulai Training Project: {CONFIG['project_name']}")
    print(f"🎯 Model Target: {CONFIG['model_type'].upper()}")
    print(f"⚙️  Device: {CONFIG['device']}")
    print("="*60)

    # Dataset Folder (Sesuaikan dengan komputer Anda)
    # Cek folder 'UrbanSound8K' atau 'data'
    base_dir = os.path.join(os.getcwd(), 'UrbanSound8K')
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.getcwd(), 'data')
    
    # Loop untuk 5-Fold Cross Validation
    for fold in range(CONFIG["folds"]):
        print(f"\n📦 --- FOLD {fold + 1}/{CONFIG['folds']} ---")
        
        # A. Siapkan Data Loader
        # DataReader kita sudah pintar, tinggal panggil
        train_ds = AudioDataset(root_dir=base_dir, fold=fold, split_type="train")
        val_ds = AudioDataset(root_dir=base_dir, fold=fold, split_type="val")
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], 
                                  shuffle=True, num_workers=CONFIG["num_workers"])
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], 
                                shuffle=False, num_workers=CONFIG["num_workers"])
        
        # B. Pilih Model (Sesuai Config)
        if CONFIG["model_type"] == "spectrogram":
            model = ModelSpectrogram(num_classes=CONFIG["num_classes"])
        elif CONFIG["model_type"] == "waveform":
            model = ModelWaveform(num_classes=CONFIG["num_classes"])
        elif CONFIG["model_type"] == "hybrid":
            model = ModelHybrid(num_classes=CONFIG["num_classes"])
        else:
            raise ValueError("Tipe model tidak dikenali! Pilih: spectrogram, waveform, hybrid")
            
        model = model.to(CONFIG["device"])
        
        # C. Optimizer & Loss
        # Adam biasanya pilihan terbaik untuk Audio
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        criterion = nn.CrossEntropyLoss() # Standar klasifikasi
        
        # D. Loop Epoch
        best_acc = 0.0
        
        for epoch in range(CONFIG["epochs"]):
            start_time = time.time()
            
            # Latih & Uji
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
            val_loss, val_acc = validate(model, val_loader, criterion, CONFIG["device"])
            
            # Cek Juara
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                # Simpan Model Juara
                save_name = f"{CONFIG['model_type']}_fold{fold}_best.pth"
                save_path = os.path.join(CONFIG["save_dir"], save_name)
                torch.save(model.state_dict(), save_path)
            
            end_time = time.time()
            epoch_mins = (end_time - start_time) / 60
            
            # Laporan per Epoch
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
                  f"Train: Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
                  f"Val: Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
                  f"Time: {epoch_mins:.1f}m | "
                  f"{'🏆 BEST' if is_best else ''}")
        
        print(f"🏁 Selesai Fold {fold}. Akurasi Terbaik: {best_acc:.2f}%")
        print("-" * 60)

if __name__ == "__main__":
    run_training()