from datetime import datetime
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
import seaborn as sns # Library untuk gambar Confusion Matrix cantik
from sklearn.metrics import f1_score, confusion_matrix
import wandb

# --- IMPORT KODE SENDIRI ---
from datareader import AudioDataset
from model import AudioClassifier

# ==============================================================================
# 1. SETUP & UTILS
# ==============================================================================
def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False # Untuk operasi train GPU, biar lebih cepat tetapi hasilnya relatif sama
    torch.backends.cudnn.benchmark = False

def save_config_log(config, path):
    """Mencatat bukti parameter ke file teks"""
    with open(path, 'w') as f:
        f.write("=== LOG KONFIGURASI TRAINING ===\n")
        f.write(f"Waktu Run : {time.ctime()}\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("================================\n")
    print(f"📄 Bukti konfigurasi disimpan di: {path}")

# ==============================================================================
# 2. KONFIGURASI
# ==============================================================================
CONFIG = {
    "project_name": "TA_SoundClassification",
    # GANTI INI SESUAI GILIRAN (spectrogram / waveform / hybrid)
    "model_type": "spectrogram",  
    "num_classes": 4,
    "batch_size": 8,       # Ukuran batch untuk satu kali epoch training
    "epochs": 50,            
    "folds": 5,              
    "num_workers": 2,      # CPU workers untuk Preprocessing data
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "checkpoints",
    "report_dir": "reports",
    "seed": 42,
    "target_sr": 32000,    # Standar PANNs
    "fixed_length": 32000, # 1 detik pada 32kHz, untuk Augmentasi Temporal
    "learning_rate": 5e-4,

    # OPTIMIZER LOCK
    "optimizer": {
        "name": "AdamW",
        "weight_decay": 1e-4,
    },

    # SCHEDULER LOCK
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "patience": 3,
        "factor": 0.5,
        "target_metric": "val_loss", # Bisa 'train_loss' atau 'val_loss' tergantung kebutuhan
    },
}

# Mapping label untuk Confusion Matrix
LABELS = ['Siren', 'Car Horn', 'Gun Shot', 'Dog Bark'] # Pastikan urutan 0,1,2,3

os.makedirs(CONFIG["save_dir"], exist_ok=True)
os.makedirs(CONFIG["report_dir"], exist_ok=True)

# ==============================================================================
# 3. VISUALISASI
# ==============================================================================
def plot_history(history, model_name, fold):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    plt.title(f'Loss - {model_name} Fold {fold}')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    # Plot F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], 'b-o', label='Train F1')
    plt.plot(epochs, history['val_f1'], 'r-o', label='Val F1')
    plt.title(f'F1 Score - {model_name} Fold {fold}')
    plt.xlabel('Epochs'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['report_dir']}/{model_name}_fold{fold}_chart.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, fold):
    """Menggambar Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Gambar Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    
    plt.title(f'Confusion Matrix - {model_name} Fold {fold}')
    plt.ylabel('Label Asli')
    plt.xlabel('Prediksi Model')
    
    save_path = f"{CONFIG['report_dir']}/{model_name}_fold{fold}_confmat.png"
    plt.savefig(save_path)
    plt.close()
    print(f"🖼️  Confusion Matrix disimpan: {save_path}")

# ==============================================================================
# 4. ENGINE TRAINING
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(loader, desc="Train", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_loss = running_loss / len(loader)
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
    
    # Return prediksi lengkap untuk Confusion Matrix nanti
    return avg_loss, f1, all_labels, all_preds

# ==============================================================================
# 5. MAIN PROGRAM
# ==============================================================================
def run_training():
    seed_everything(CONFIG["seed"])

    activity = input("Masukkan aktivitas Anda saat ini: ")
    current_time = datetime.now().strftime("%Y-%m-%d %H%M")

    run_name = f"{current_time}_{activity}"

    wandb.init(
        project="TA_SoundClassification", # Nama Proyek di Dashboard
        config=CONFIG,                     # Kamus settingan (LR, Batch Size, dll)
        name=run_name                    # Nama Run di Dashboard
    )
    
    print(f"🚀 Mulai Training: {CONFIG['model_type'].upper()}")
    
    # --- BUKTI 1: SIMPAN LOG KONFIGURASI ---
    # Ini bukti otentik buat dosen bahwa settingan gak berubah
    log_path = f"{CONFIG['report_dir']}/{CONFIG['model_type']}_parameters.txt"
    save_config_log(CONFIG, log_path)

    with open("split_data.json", 'r') as f:
        split_data = json.load(f)

    # Variabel untuk menampung waktu total seluruh fold
    total_training_start = time.time()
    final_scores = []
    
    # Loop Fold
    for fold_idx in [1]:
        fold_name = f"fold{fold_idx}"
        print(f"\n📦 --- {fold_name.upper()} ---")
        
        # Mulai Stopwatch per Fold
        fold_start_time = time.time()

        train_ds = AudioDataset(split_json="split_data.json", fold=fold_name, split_type="train", target_sr=CONFIG["target_sr"], fixed_length=CONFIG["fixed_length"])
        val_ds = AudioDataset(split_json="split_data.json", fold=fold_name, split_type="test", target_sr=CONFIG["target_sr"], fixed_length=CONFIG["fixed_length"])
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
        
        # Class Weights
        weights_dict = split_data[fold_name]["weights"]
        weights_list = [weights_dict[str(i)] for i in range(CONFIG["num_classes"])]
        weights_tensor = torch.tensor(weights_list).float().to(CONFIG["device"])
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        
        model = AudioClassifier(model_type=CONFIG["model_type"], num_classes=CONFIG["num_classes"])
        model = model.to(CONFIG["device"])

        # LOGIKA PEMILIHAN OPTIMIZER BERDASARKAN CONFIG
        opt_name = CONFIG["optimizer"]["name"]
        lr = CONFIG["learning_rate"]
        wd = CONFIG["optimizer"]["weight_decay"]

        print(f"Menggunakan Optimizer: {opt_name}") # Log biar jelas

        if opt_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Optimizer belum didaftarkan: {opt_name}")

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=CONFIG["scheduler"]["patience"], factor=CONFIG["scheduler"]["factor"])
        
        history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
        best_f1 = 0.0
        last_checkpoint_path = None
        
        # Variabel untuk menyimpan prediksi terbaik (untuk Confusion Matrix)
        best_labels_true = []
        best_labels_pred = []

        # Early Stopping Variables
        patience_counter = 5
        early_stop_counter = 0

        # Loop Epoch
        for epoch in range(CONFIG["epochs"]):
            train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
            val_loss, val_f1, labels_true, labels_pred = validate(model, val_loader, criterion, CONFIG["device"])
            
            history['train_loss'].append(train_loss)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)

            # Step Scheduler
            if CONFIG["scheduler"]["target_metric"] == "val_loss":
                scheduler.step(val_loss)
            elif CONFIG["scheduler"]["target_metric"] == "train_loss":
                scheduler.step(train_loss)
            else:
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']

            #  KIRIM LAPORAN KE WANDB (TARUH DISINI)
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "learning_rate": current_lr,
            })

            if val_f1 > best_f1:
                best_f1 = val_f1
                # Simpan prediksi saat momen terbaik ini untuk Confusion Matrix
                best_labels_true = labels_true
                best_labels_pred = labels_pred

                root_checkpoint_dir = f"{CONFIG['save_dir']}/{CONFIG['model_type']}"
                os.makedirs(root_checkpoint_dir, exist_ok=True)

                checkpoint_filename = f"fold{fold_idx}_epoch{epoch+1}_{best_f1:.3f}_best.pth"

                new_checkpoint_path = f"{root_checkpoint_dir}/{checkpoint_filename}"
                
                if last_checkpoint_path is not None and os.path.exists(last_checkpoint_path):
                    try:
                        os.remove(last_checkpoint_path)
                        print(f"   🗑️ Menghapus checkpoint lama: {os.path.basename(last_checkpoint_path)}")
                    except Exception as e:
                        print(f"⚠️  Gagal menghapus checkpoint lama: {e}")

                # Simpan Model (Checkpoint)
                torch.save(model.state_dict(), new_checkpoint_path)
                last_checkpoint_path = new_checkpoint_path
                print(f"   Epoch {epoch+1}: F1 {val_f1:.3f} (New Best!) [Checkpoint disimpan!]")

                early_stop_counter = 0 # Reset counter jika ada rekor baru
            else:
                print(f"   Epoch {epoch+1}: F1 {val_f1:.3f}")

                early_stop_counter += 1
                if early_stop_counter >= patience_counter:
                    print(f"Early stopping di epoch {epoch+1} setelah {patience_counter} epoch tanpa adanya rekor baru.")
                    break

        # Stop Stopwatch per Fold
        fold_duration = time.time() - fold_start_time
        print(f"⏱️  Waktu Training Fold {fold_idx}: {fold_duration/60:.2f} menit")

        # --- GENERATE LAPORAN PER FOLD ---
        # 1. Simpan Grafik Loss & F1
        plot_history(history, CONFIG['model_type'], fold_idx)
        
        # 2. Simpan Confusion Matrix (BUKTI 2)
        plot_confusion_matrix(best_labels_true, best_labels_pred, CONFIG['model_type'], fold_idx)
        
        # 3. Simpan CSV (History Lengkap)
        df = pd.DataFrame(history)
        df['fold_duration_sec'] = fold_duration # Catat waktu juga di excel
        df.to_csv(f"{CONFIG['report_dir']}/{CONFIG['model_type']}_{fold_name}.csv", index=False)
        
        final_scores.append(best_f1)
        
        # BERSIH-BERSIH MEMORI (PENTING!)
        import gc
        del model, train_loader, val_loader, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    # --- LAPORAN FINAL ---
    total_duration = time.time() - total_training_start
    print("\n" + "="*50)
    print(f"🎉 SELESAI: {CONFIG['model_type'].upper()}")
    print(f"📊 Rata-rata F1-Score: {sum(final_scores)/len(final_scores):.3f}")
    print(f"⏱️  Total Waktu Komputasi: {total_duration/60:.2f} menit")
    print("="*50)

if __name__ == "__main__":
    run_training()