import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# --- UPDATE IMPORT: Gunakan AudioClassifier ---
from datareader import AudioDataset
from model import AudioClassifier 

# ==============================================================================
# KONFIGURASI (UBAH DISINI SEBELUM JALAN)
# ==============================================================================
# 1. Pilih Model yang mau dicek (spectrogram / waveform / hybrid)
MODEL_TYPE = "hybrid"

# 2. Pilih Fold mana yang mau dites (Harus cocok dengan nama file modelnya)
# Misal: Kalau mau cek fold 2, pastikan path model mengarah ke fold 2
FOLD_TO_TEST = 4 

# 3. Path ke file .pth yang sudah disimpan
MODEL_PATH = "models_saved/hybrid_fold4_best.pth" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_CLASSES = 4
LABELS = ['Car Horn', 'Dog Bark', 'Gun Shot', 'Siren'] # Urutan label 0-3

# ==============================================================================
# FUNGSI EVALUASI
# ==============================================================================
def run_evaluation():
    print(f"🔍 Memulai Evaluasi untuk Model: {MODEL_TYPE.upper()} (Fold {FOLD_TO_TEST})")
    
    # --- 1. Siapkan Data (Pakai Split Validation dari Fold tersebut) ---
    base_dir = os.path.join(os.getcwd(), 'UrbanSound8K')
    if not os.path.exists(base_dir):
        # Fallback kalau folder namanya 'data'
        base_dir = os.path.join(os.getcwd(), 'data')
    
    # Kita ambil data validasi saja untuk pengujian
    val_ds = AudioDataset(root_dir=base_dir, fold=FOLD_TO_TEST, split_type="val")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"📂 Jumlah Data Validasi: {len(val_ds)} sampel")
    
    # --- 2. Load Struktur Model (UPDATE BARU) ---
    print(f"🏗️ Membangun arsitektur {MODEL_TYPE}...")
    try:
        # Cukup panggil satu class ini saja
        model = AudioClassifier(model_type=MODEL_TYPE, num_classes=NUM_CLASSES)
    except Exception as e:
        print(f"❌ Error saat membangun model: {e}")
        return

    # --- 3. Load Bobot (Weights) Hasil Training ---
    if os.path.exists(MODEL_PATH):
        print(f"📥 Loading weights dari: {MODEL_PATH}")
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print("✅ Weights berhasil dimuat!")
        except Exception as e:
            print(f"❌ Gagal load state_dict. Pastikan arsitektur model cocok. Error: {e}")
            return
    else:
        print(f"❌ Error: File model '{MODEL_PATH}' tidak ditemukan!")
        print("   Pastikan path-nya benar dan file .pth sudah didownload.")
        return

    model.to(DEVICE)
    model.eval()
    
    # --- 4. Loop Prediksi ---
    all_preds = []
    all_labels = []
    
    print("⏳ Sedang melakukan prediksi masal...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            
            # Ambil kelas dengan probabilitas tertinggi
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- 5. Visualisasi Hasil ---
    # Buat Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plotting Grafik
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Asli (Kenyataan)')
    plt.title(f'Confusion Matrix - {MODEL_TYPE.upper()} (Fold {FOLD_TO_TEST})')
    
    # Simpan Gambar Laporan
    if not os.path.exists("reports"):
        os.makedirs("reports")
        
    save_file = f"reports/eval_matrix_{MODEL_TYPE}_fold{FOLD_TO_TEST}.png"
    plt.savefig(save_file)
    print(f"\n✅ Gambar Confusion Matrix disimpan di: {save_file}")
    plt.show() 
    
    # --- 6. Rapor Nilai (Precision, Recall, F1-Score) ---
    print("\n" + "="*60)
    print(f"📊 LAPORAN EVALUASI: {MODEL_TYPE.upper()}")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=LABELS))
    print("="*60)

if __name__ == "__main__":
    run_evaluation()