import os
import config
import glob
import json 
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# --- BAGIAN 1: KONFIGURASI ---
# Folder tempat audio kamu berada (hasil dari arrange_data.py)
DATA_DIR = "data"
# Nama file output
OUTPUT_FILE = "split_data.json"
# Daftar kelas target (Urutan ini PENTING, 0=siren, 1=car_horn, dst)
CLASSES = ['siren', 'car_horn', 'gun_shot', 'dog_bark']

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}

# --- BAGIAN 2: FUNGSI PENCARI FILE ---
def get_files_and_labels(fold_indices=5):
    """
    Tugas: Masuk ke folder Fold X, ambil semua file .wav, catat labelnya.
    Input: Daftar angka fold, misal [2, 3, 4, 5]
    Output: Dua list panjang (daftar path file, daftar label angka)
    """
    X_files = []
    y_labels = []

    for fold in fold_indices:
        for class_name in CLASSES:
            # Pola pencarian: data/Fold1/siren/*.wav
            search_pattern = os.path.join(DATA_DIR, f"Fold{fold}", class_name, "*.wav")
            
            # GLOB beraksi: Temukan semua file yang cocok dengan pola
            found_files = glob.glob(search_pattern)
            
            # Masukkan ke keranjang
            X_files.extend(found_files)
            
            # Buat label angka (misal: 0) sebanyak file yang ditemukan
            label_id = CLASS_TO_IDX[class_name]
            y_labels.extend([label_id] * len(found_files))
            
    return X_files, y_labels

# --- BAGIAN 3: PROGRAM UTAMA ---
def main():
    print(f"🚀 Memulai pemetaan data dari folder '{DATA_DIR}'...")
    
    final_split = {}
    all_folds = [1, 2, 3, 4, 5] # Kita punya 5 fold hasil gabungan
    
    # Loop 5 kali untuk 5 Skenario Cross Validation
    for i in all_folds:
        test_fold = [i]                       # Satu fold untuk Ujian
        train_folds = [x for x in all_folds if x != i] # Sisanya untuk Belajar
        
        print(f"\n📂 Skenario Fold {i}:")
        print(f"   - Test  : Fold {test_fold}")
        print(f"   - Train : Fold {train_folds}")
        
        # Panggil fungsi pencari file tadi
        X_train, y_train = get_files_and_labels(train_folds)
        X_test, y_test = get_files_and_labels(test_fold)
        
        # Cek keamanan (takutnya path salah dan tidak nemu file)
        if len(X_train) == 0:
            print("❌ ERROR: Tidak ada file ditemukan! Cek nama folder 'data' kamu.")
            return

        # --- BAGIAN 4: HITUNG BOBOT (Solusi Data Tidak Seimbang) ---
        # "Hakim" sklearn menghitung keadilan bobot
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        
        # Bungkus bobot jadi kamus: {"0": 1.2, "1": 0.5}
        # Kita ubah key jadi string supaya bisa disimpan di JSON
        weight_dict = {str(k): float(v) for k, v in zip(np.unique(y_train), weights)}
        print(f"   ⚖️  Class Weights: {weight_dict}")

        # --- BAGIAN 5: SUSUN DATA ---
        final_split[f"fold{i}"] = {
            "train": {
                "files": X_train,
                "labels": y_train
            },
            "test": {
                "files": X_test,
                "labels": y_test
            },
            "weights": weight_dict
        }

    # --- BAGIAN 6: SIMPAN JSON ---
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_split, f, indent=4)
        
    print(f"\n✅ SUKSES! File '{OUTPUT_FILE}' siap digunakan.")

if __name__ == "__main__":
    main()