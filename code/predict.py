import torch
import librosa
import numpy as np
import cv2
from model import AudioClassifier  # Pastikan file model.py ada di sebelah script ini

# --- KONFIGURASI ---
# Ganti dengan path model JUARA ANDA (Spectrogram Fold 3)
MODEL_PATH = "models_saved/spectrogram_fold2_best.pth" 
TEST_FILE = "audio_test/siren.wav"  # Ganti dengan nama file suara yang mau dites
MODEL_TYPE = "spectrogram"          # Wajib spectrogram

# Label Kelas (Urutan harus sama dengan saat training!)
LABELS = ['Car Horn', 'Dog Bark', 'Gun Shot', 'Siren']

def preprocess_audio(file_path):
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=22050)
    
    # 2. Potong/Padding jadi 4 detik (sama seperti saat training)
    max_len = 22050 * 4
    if len(y) > max_len:
        y = y[:max_len]
    else:
        padding = max_len - len(y)
        y = np.pad(y, (0, padding))
        
    # 3. Ubah ke Spectrogram (Sama persis logic training)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    
    # Normalisasi ke 0-255
    spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min()) * 255
    spec_img = spec_norm.astype(np.uint8)
    
    # Ubah jadi 3 Channel (RGB) agar bisa masuk ke ResNet
    spec_img = cv2.cvtColor(spec_img, cv2.COLOR_GRAY2RGB)
    
    # Ubah ke Tensor (C, H, W)
    input_tensor = torch.tensor(spec_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return input_tensor.unsqueeze(0) # Tambah batch dimension

def predict():
    print(f"🔍 Memuat model dari {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Arsitektur
    model = AudioClassifier(model_type=MODEL_TYPE, num_classes=4).to(device)
    
    # Load Otak (Weights)
    # map_location='cpu' agar aman dijalankan di laptop tanpa GPU
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"🎧 Sedang mendengarkan file: {TEST_FILE}")
    try:
        input_tensor = preprocess_audio(TEST_FILE).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        idx = predicted_class.item()
        score = confidence.item() * 100
        
        print("\n" + "="*30)
        print(f"🗣️  HASIL PREDIKSI: {LABELS[idx].upper()}")
        print(f"📊  Yakin: {score:.2f}%")
        print("="*30)
        
        # Tampilkan detail persen semua kelas
        print("\nDetail:")
        for i, label in enumerate(LABELS):
            probs = probabilities[0][i].item() * 100
            print(f"- {label}: {probs:.2f}%")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Pastikan file audio ada dan formatnya .wav")

if __name__ == "__main__":
    predict()