import os

# --- AUDIO SETTINGS ---
SAMPLE_RATE = 32000    # Standar PANNs (32kHz)
DURATION = 1           # Detik (PANNs biasanya 10s, tapi US8K rata-rata <4s. Kita set 5s biar aman)
NUM_SAMPLES = SAMPLE_RATE * DURATION # 160.000 samples

# --- PATHS ---
ROOT_DIR = ".."
DATA_DIR = os.path.join(ROOT_DIR, "data")  # Folder hasil organize kamu
JSON_PATH = os.path.join(ROOT_DIR, "configs", "split_data.json") # File peta json
PRETRAINED_PATH = os.path.join(ROOT_DIR, "pretrained")
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints")
REPORT_PATH = os.path.join(ROOT_DIR, "reports")

# --- LABELS ---
# Urutan ini PENTING jangan sampai tertukar!
# 0: siren, 1: car_horn, dst.
TARGET_CLASSES = ['siren', 'car_horn', 'gun_shot', 'dog_bark', 'normal']
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(TARGET_CLASSES)}
CLASS_TO_IDX   = {cls_name: i for i, cls_name in enumerate(TARGET_CLASSES)}

# --- TRAINING SETTINGS ---
BATCH_SIZE = 32
NUM_WORKERS = 4        # Tergantung CPU laptop kamu
RANDOM_SEED = 42