import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as pd_plt
import matplotlib.pyplot as plt

# --- IMPORT DARI KODE KITA SENDIRI ---
from datareader import AudioDataset
from model import ModelSpectrogram, ModelWaveform, ModelHybrid

# ==============================================================================
# 1. KONFIGURASI (Pusat Kontrol)
# ==============================================================================
CONFIG = {
    "project_name": "TA_SoundClassification",
    # GANTI INI SESUAI MODEL YANG MAU DILATIH (spectrogram / waveform / hybrid)
    "model_type": "spectrogram",  
    "num_classes": 4,
    "batch_size": 16,        
    "learning_rate": 0.001,  
    "epochs": 15,            
    "folds": 5,              
    "num_workers": 2,        
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "models_saved",
    "report_dir": "reports" # Folder baru untuk simpan grafik & excel
}

# Buat folder penyimpanan
os.makedirs(CONFIG["save_dir"], exist_ok=True)
os.makedirs(CONFIG["report_dir"], exist_ok=True)

# ==============================================================================
# 2. HELPER: VISUALISASI GRAFIK
# ==============================================================================
def plot_history(history, model_name, fold):
    """
    Fungsi ini menggambar grafik Loss dan Akurasi secara otomatis
    dan menyimpannya sebagai file PNG.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # GRAFIK 1: LOSS (Error) - Harusnya Turun
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title(f'Loss Curve - {model_name} Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # GRAFIK 2: ACCURACY (Kepintaran) - Harusnya Naik
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Acc')
    plt.title(f'Accuracy Curve - {model_name} Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Simpan Gambar
    filename = f"{CONFIG['report_dir']}/{model_name}_fold{fold}_chart.png"
    plt.savefig(filename)
    plt.close()
    print(f"📊 Grafik disimpan: {filename}")

# ==============================================================================
# 3. FUNGSI PELATIHAN
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
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
# 4. MAIN LOOP
# ==============================================================================
def run_training():
    print(f"🚀 Memulai Training Project: {CONFIG['project_name']}")
    print(f"🎯 Model Target: {CONFIG['model_type'].upper()}")
    print(f"⚙️  Device: {CONFIG['device']}")
    print("="*60)

    # Auto-detect folder data
    base_dir = os.path.join(os.getcwd(), 'UrbanSound8K')
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.getcwd(), 'data')
    
    # List untuk menampung hasil semua fold (buat laporan akhir)
    final_results = []

    for fold in range(CONFIG["folds"]):
        print(f"\n📦 --- FOLD {fold + 1}/{CONFIG['folds']} ---")
        
        # Reset History per Fold
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        train_ds = AudioDataset(root_dir=base_dir, fold=fold, split_type="train")
        val_ds = AudioDataset(root_dir=base_dir, fold=fold, split_type="val")
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], 
                                  shuffle=True, num_workers=CONFIG["num_workers"])
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], 
                                shuffle=False, num_workers=CONFIG["num_workers"])
        
        # Pilih Model
        if CONFIG["model_type"] == "spectrogram":
            model = ModelSpectrogram(num_classes=CONFIG["num_classes"])
        elif CONFIG["model_type"] == "waveform":
            model = ModelWaveform(num_classes=CONFIG["num_classes"])
        elif CONFIG["model_type"] == "hybrid":
            model = ModelHybrid(num_classes=CONFIG["num_classes"])
        else:
            raise ValueError("Model tidak dikenal")
            
        model = model.to(CONFIG["device"])
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        
        for epoch in range(CONFIG["epochs"]):
            start_time = time.time()
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
            val_loss, val_acc = validate(model, val_loader, criterion, CONFIG["device"])
            
            # SIMPAN DATA KE HISTORY
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                save_name = f"{CONFIG['model_type']}_fold{fold}_best.pth"
                torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], save_name))
            
            end_time = time.time()
            epoch_mins = (end_time - start_time) / 60
            
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                  f"{'🏆 BEST' if is_best else ''}")

        # --- SETELAH FOLD SELESAI: GENERATE LAPORAN ---
        print(f"🏁 Menyimpan laporan Fold {fold}...")
        
        # 1. Simpan Excel (CSV)
        df = pd.DataFrame(history)
        df.index.name = 'Epoch'
        csv_name = f"{CONFIG['report_dir']}/{CONFIG['model_type']}_fold{fold}_log.csv"
        df.to_csv(csv_name)
        print(f"📄 Excel disimpan: {csv_name}")
        
        # 2. Simpan Grafik (PNG)
        plot_history(history, CONFIG['model_type'], fold)
        
        # Catat skor akhir
        final_results.append(best_acc)
        print("-" * 60)

    # --- LAPORAN FINAL ---
    avg_acc = sum(final_results) / len(final_results)
    print("\n" + "="*60)
    print(f"🎉 TRAINING SELESAI UNTUK MODEL: {CONFIG['model_type'].upper()}")
    print(f"📊 Rata-rata Akurasi 5-Fold: {avg_acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_training()