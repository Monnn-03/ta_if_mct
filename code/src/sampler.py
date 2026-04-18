import random
from collections import defaultdict
import config

def print_distribution(class_files, label=""):
    """Helper print distribusi per kelas."""
    print(f"\n   📊 Distribusi {label}:")
    total = sum(len(v) for v in class_files.values())
    for label_id, files in sorted(class_files.items()):
        class_name = config.IDX_TO_CLASS[label_id]
        count = len(files)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"      {class_name:<12} : {count:>5} file ({pct:5.1f}%)  {bar}")
    print(f"      {'TOTAL':<12} : {total:>5} file")

def undersample_normal(file_paths, labels, normal_idx, random_state=42, verbose=True):
    random.seed(random_state)

    class_files = defaultdict(list)
    for f, l in zip(file_paths, labels):
        class_files[l].append(f)

    if verbose:
        print_distribution(class_files, label="sebelum undersampling")

    # Hitung jumlah terbanyak di antara kelas BAHAYA saja (bukan normal)
    danger_counts = [
        len(v) for k, v in class_files.items() if k != normal_idx
    ]

    if not danger_counts:
        return file_paths, labels

    max_danger = max(danger_counts) + 400  # ← pakai MAX, bukan min

    original_count = len(class_files[normal_idx])
    if original_count > max_danger:
        class_files[normal_idx] = random.sample(class_files[normal_idx], max_danger)
        print(f"\n   ✂️  Undersample 'normal': {original_count} → {max_danger} file "
              f"(mengikuti kelas bahaya terbanyak)")
    else:
        print(f"\n   ✅ Kelas 'normal' tidak perlu undersample ({original_count} file).")

    if verbose:
        print_distribution(class_files, label="setelah undersampling")

    final_files, final_labels = [], []
    for label_id, file_list in class_files.items():
        final_files.extend(file_list)
        final_labels.extend([label_id] * len(file_list))

    combined = list(zip(final_files, final_labels))
    random.shuffle(combined)
    final_files, final_labels = zip(*combined)

    return list(final_files), list(final_labels)