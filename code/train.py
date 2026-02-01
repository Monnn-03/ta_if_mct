import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score # WAJIB untuk data tidak seimbang

# --- IMPORT DARI KODE KITA SENDIRI ---
# Pastikan nama file .py nya sesuai
from datareader import AudioDataset
from model import AudioClassifier # Kita pakai satu class ini saja
import config # Mengambil settingan global

# ==============================================================================
# 1. FUNGSI PENGUNCI KEBERUNTUNGAN (SEED) - WAJIB ADA
# ==============================================================================
def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Random Seed dikunci di angka: {seed}")

# ==============================================================================
# 2. KONFIGURASI
# ==============================================================================
CONFIG = {
    "project_name": "TA_SoundClassification",
    # PILIHAN: 'spectrogram', 'waveform', 'hybrid'
    "model_type": "spectrogram",  
    "num_classes": 4,
    "batch_size": 16,        
    "learning_rate": 0.001,  
    "epochs": 15,            
    "folds": 5,              
    "num_workers": 0,        
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "models_saved",
    "report_dir": "reports",
    "seed": 42
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
os.makedirs(CONFIG["report_dir"], exist_ok=True)

# ==============================================================================
# 3. HELPER: VISUALISASI
# ==============================================================================
def plot_history(history, model_name, fold):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    plt.title(f'Loss - {model_name} Fold {fold}')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    # F1-Score (Lebih penting daripada Akurasi untuk kasusmu)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], 'b-o', label='Train F1')
    plt.plot(epochs, history['val_f1'], 'r-o', label='Val F1')
    plt.title(f'F1 Score - {model_name} Fold {fold}')
    plt.xlabel('Epochs'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['report_dir']}/{model_name}_fold{fold}_chart.png")
    plt.close()

# ==============================================================================
# 4. ENGINE TRAINING
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs) # Output shape: [Batch, 4]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Simpan prediksi untuk hitung F1 Score nanti
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_loss = running_loss / len(loader)
    # Hitung F1 Score (Macro Average untuk data imbalance)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    return avg_loss, f1, acc

# ==============================================================================
# 5. MAIN PROGRAM
# ==============================================================================
def run_training():
    # 1. Kunci Random Seed (Paling Awal!)
    seed_everything(CONFIG["seed"])
    
    print(f"🚀 Mulai Training: {CONFIG['model_type'].upper()}")
    print(f"⚙️  Device: {CONFIG['device']}")
    
    # Load JSON Split untuk mengambil Class Weights
    with open("split_data.json", 'r') as f:
        split_data = json.load(f)

    final_results = []

    for fold_idx in range(1, CONFIG["folds"] + 1):
        fold_name = f"fold{fold_idx}" # fold1, fold2...
        print(f"\n📦 --- {fold_name.upper()} ---")
        
        # --- A. DATASET & DATALOADER ---
        # Perhatikan: split_type="test" sesuai kunci di JSON (bukan val)
        train_ds = AudioDataset(split_json="split_data.json", fold=fold_name, split_type="train")
        val_ds = AudioDataset(split_json="split_data.json", fold=fold_name, split_type="test")
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
        
        # --- B. CLASS WEIGHTS (SOLUSI IMBALANCE) ---
        # Ambil bobot dari JSON untuk fold ini
        weights_dict = split_data[fold_name]["weights"]
        # Urutkan bobot berdasarkan index 0, 1, 2, 3
        weights_list = [weights_dict[str(i)] for i in range(CONFIG["num_classes"])]
        weights_tensor = torch.tensor(weights_list).float().to(CONFIG["device"])
        print(f"⚖️  Menggunakan Class Weights: {weights_list}")
        
        # Masukkan bobot ke Loss Function
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        
        # --- C. INIT MODEL ---
        # Panggil class AudioClassifier dari model.py
        model = AudioClassifier(model_type=CONFIG["model_type"], num_classes=CONFIG["num_classes"])
        model = model.to(CONFIG["device"])
        
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        
        # --- D. LOOP EPOCH ---
        history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
        best_f1 = 0.0
        
        for epoch in range(CONFIG["epochs"]):
            train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
            val_loss, val_f1, val_acc = validate(model, val_loader, criterion, CONFIG["device"])
            
            history['train_loss'].append(train_loss)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            
            # Simpan Model Terbaik berdasarkan F1-Score
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), f"{CONFIG['save_dir']}/{CONFIG['model_type']}_{fold_name}_best.pth")
                print(f"   Epoch {epoch+1}: F1 {val_f1:.3f} (New Best!) | Acc: {val_acc:.2f}%")
            else:
                print(f"   Epoch {epoch+1}: F1 {val_f1:.3f} | Loss: {train_loss:.4f}")

        # --- E. LAPORAN PER FOLD ---
        # Simpan CSV
        df = pd.DataFrame(history)
        df.to_csv(f"{CONFIG['report_dir']}/{CONFIG['model_type']}_{fold_name}.csv", index=False)
        
        # Simpan Grafik
        plot_history(history, CONFIG['model_type'], fold_idx)
        
        final_results.append(best_f1)
        print(f"✅ Selesai {fold_name}. Best F1: {best_f1:.3f}")

    # --- LAPORAN AKHIR ---
    print("\n" + "="*50)
    print(f"Rata-rata F1-Score 5-Fold: {sum(final_results)/len(final_results):.3f}")
    print("="*50)

if __name__ == "__main__":
    run_training()