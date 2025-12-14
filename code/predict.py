import torch
import librosa
import numpy as np
from model import AudioClassifier 

# --- KONFIGURASI ---
# Pastikan path model benar
MODEL_PATH = "models_saved/spectrogram_fold2_best.pth" 
TEST_FILE = "audio_test/siren.wav"  # File audio yang mau dites
MODEL_TYPE = "spectrogram"          # Tetap "spectrogram" (karena arsitekturnya ResNet38)

# Label Kelas
LABELS = ['Car Horn', 'Dog Bark', 'Gun Shot', 'Siren']

def preprocess_audio(file_path):
    # 1. Load Audio
    # PANNs dilatih pada sample rate 32000 Hz, jadi kita ikut standar itu
    y, sr = librosa.load(file_path, sr=32000)
    
    # 2. Potong/Padding jadi 4 detik (Standar PANNs UrbanSound)
    # 32000 Hz x 4 detik = 128000 samples
    max_len = 32000 * 4
    
    if len(y) > max_len:
        # Kalau kepanjangan, potong
        y = y[:max_len]
    else:
        # Kalau kependekan, tambah nol (padding)
        padding = max_len - len(y)
        y = np.pad(y, (0, padding))
        
    # 3. Ubah ke Tensor
    # PANNs menerima input Raw Audio (Batch, Time)
    # Kita ubah jadi (1, Time) -> Batch size 1
    input_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    
    return input_tensor

def predict():
    print(f"🔍 Memuat model dari {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load Arsitektur
        model = AudioClassifier(model_type=MODEL_TYPE, num_classes=4).to(device)
        
        # Load Otak (Weights)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"🎧 Sedang mendengarkan file: {TEST_FILE}")
        
        # Preprocess (Sekarang jauh lebih simpel, cuma potong durasi)
        input_tensor = preprocess_audio(TEST_FILE).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        idx = predicted_class.item()
        score = confidence.item() * 100
        
        print("\n" + "="*40)
        print(f"🗣️  HASIL PREDIKSI: {LABELS[idx].upper()}")
        print(f"📊  Tingkat Keyakinan: {score:.2f}%")
        print("="*40)
        
        # Tampilkan detail persen semua kelas
        print("\nDetail Probabilitas:")
        for i, label in enumerate(LABELS):
            probs = probabilities[0][i].item() * 100
            print(f"- {label:<10}: {probs:.2f}%")
            
    except FileNotFoundError:
        print(f"❌ Error: File audio '{TEST_FILE}' tidak ditemukan.")
    except Exception as e:
        print(f"❌ Error Sistem: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict()