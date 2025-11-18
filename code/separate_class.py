import os
import shutil
import pandas as pd

# Configuration - adjust to your paths
DATASET_ROOT = "E:/Ramon/KULIAH/TA/Code/UrbanSound8K"  # change to your dataset root if different
METADATA_PATH = os.path.join(DATASET_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_SUBDIR = "audio"  # relative to DATASET_ROOT
OUTPUT_DIR = os.path.join(DATASET_ROOT, "selected_classes")  # where class folders will be created
COPY_INSTEAD_OF_MOVE = True  # True to copy, False to move

# target classes
TARGET_CLASSES = ['car_horn', 'dog_bark', 'gun_shot', 'siren']

def collect_classes():
    if not os.path.exists(METADATA_PATH):
        print("ERROR: metadata CSV not found:", METADATA_PATH)
        return

    df = pd.read_csv(METADATA_PATH)
    df = df[df['class'].isin(TARGET_CLASSES)].reset_index(drop=True)
    if df.empty:
        print("No rows for target classes found in metadata.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    counts = {c: 0 for c in TARGET_CLASSES}
    missing_files = 0

    for _, row in df.iterrows():
        class_name = row['class']
        fold = row['fold']
        fname = row['slice_file_name']

        src = os.path.join(DATASET_ROOT, AUDIO_SUBDIR, f"fold{fold}", fname)
        if not os.path.exists(src):
            # Try alternative path (some setups may use different separators)
            missing_files += 1
            print("MISSING:", src)
            continue

        dest_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        # To avoid accidental name collisions, prefix with fold (optional)
        dest_fname = f"fold{fold}_{fname}"
        dest = os.path.join(dest_dir, dest_fname)

        try:
            if COPY_INSTEAD_OF_MOVE:
                shutil.copy2(src, dest)
            else:
                shutil.move(src, dest)
            counts[class_name] += 1
        except Exception as e:
            print("Failed to copy/move", src, "->", dest, ":", e)

    print("\nFinished.")
    print("Copied/Moved counts per class:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if missing_files:
        print(f"Missing files: {missing_files}")

if __name__ == "__main__":
    collect_classes()
