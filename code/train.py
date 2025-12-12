import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Import kode kita
from datareader import AudioDataset
from model import ModelSpectrogram, ModelWaveform, ModelHybrid

# ==============================================================================
# KONFIGURASI
# ==============================================================================
# GANTI INI SESUAI FILE YANG MAU DICEK
MODEL_TYPE = "waveform"       # spectrogram / waveform / hybrid
FOLD_TO_TEST = 1              # Fold mana yang mau dites (misal Fold 1 yang akurasinya 85%)
MODEL_PATH = "models_saved/waveform_fold1_best.pth" # Path file .pth hasil training tadi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_CLASSES = 4
LABELS = ['Car Horn', 'Dog Bark', 'Gun Shot', 'Siren'] # Urutan 0, 1, 2, 3

# ==============================================================================
# FUNGSI EVALUASI
# ==============================================================================
def run_evaluation():
    print(f"🔍 Memulai Evaluasi untuk Model: {MODEL_TYPE.upper()} (Fold {FOLD_TO_TEST})")
    
    # 1. Siapkan Data (Pakai Split Validation dari Fold tersebut)
    # Pastikan path dataset benar
    base_dir = os.path.join(os.getcwd(), 'UrbanSound8K')
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.getcwd(), 'data')
        
    val_ds = AudioDataset(root_dir=base_dir, fold=FOLD_TO_TEST, split_type="val")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Struktur Model
    if MODEL_TYPE == "spectrogram":
        model = ModelSpectrogram(num_classes=NUM_CLASSES)
    elif MODEL_TYPE == "waveform":
        model = ModelWaveform(num_classes=NUM_CLASSES)
    elif MODEL_TYPE == "hybrid":
        model = ModelHybrid(num_classes=NUM_CLASSES)
    
    # 3. Load Bobot (Weights) yang sudah dilatih
    if os.path.exists(MODEL_PATH):
        print(f"📥 Loading weights dari: {MODEL_PATH}")
        # Map location cpu jaga-jaga kalau gpu beda
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(f"❌ Error: File {MODEL_PATH} tidak ditemukan!")
        return

    model.to(DEVICE)
    model.eval()
    
    # 4. Prediksi Masal
    all_preds = []
    all_labels = []
    
    print("⏳ Sedang memprediksi...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Buat Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 6. Gambar Confusion Matrix yang Cantik
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Asli')
    plt.title(f'Confusion Matrix - {MODEL_TYPE.upper()} (Fold {FOLD_TO_TEST})')
    
    # Simpan Gambar
    save_file = f"reports/conf_matrix_{MODEL_TYPE}_fold{FOLD_TO_TEST}.png"
    plt.savefig(save_file)
    print(f"✅ Gambar Confusion Matrix disimpan di: {save_file}")
    plt.show() # Tampilkan di layar juga
    
    # 7. Print Rapor Lengkap
    print("\n" + "="*50)
    print(f"📊 CLASSIFICATION REPORT: {MODEL_TYPE.upper()}")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=LABELS))

if __name__ == "__main__":
    run_evaluation()