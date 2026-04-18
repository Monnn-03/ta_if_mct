import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import config

def count_distribution(data_dir):
    distribution = defaultdict(lambda: defaultdict(int))
    class_total = defaultdict(int)

    if not os.path.exists(data_dir):
        print(f"❌ Folder tidak ditemukan: {data_dir}")
        return None, None

    fold_dirs = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    for fold_name in fold_dirs:
        fold_path = os.path.join(data_dir, fold_name)
        class_dirs = sorted([c for c in os.listdir(fold_path) if os.path.isdir(os.path.join(fold_path, c))])

        for class_name in class_dirs:
            class_path = os.path.join(fold_path, class_name)
            count = len([
                f for f in os.listdir(class_path)
                if os.path.splitext(f)[1] in {'.wav', '.ogg', '.flac', '.mp3'}
            ])
            distribution[fold_name][class_name] = count
            class_total[class_name] += count

    return distribution, class_total


def plot_distribution(distribution, class_total, save_dir):
    all_classes = sorted(class_total.keys())
    all_folds   = sorted(distribution.keys())
    n_classes   = len(all_classes)
    n_folds     = len(all_folds)

    # Warna per fold
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_folds))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Distribusi Data UrbanSound8K (Filtered)", fontsize=14, fontweight='bold')

    # ── PLOT 1: Per fold, grouped by class ──────────────────────────────────
    ax1 = axes[0]
    x = np.arange(n_classes)
    bar_width = 0.8 / n_folds  # total lebar 0.8 dibagi rata antar fold

    for i, fold in enumerate(all_folds):
        counts = [distribution[fold].get(cls, 0) for cls in all_classes]
        offset = (i - n_folds / 2 + 0.5) * bar_width
        bars = ax1.bar(x + offset, counts, width=bar_width, color=colors[i], label=fold)

        # Label angka di atas bar
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1, str(int(h)),
                    ha='center', va='bottom', fontsize=7
                )

    ax1.set_title("Distribusi per Fold & Kelas")
    ax1.set_xlabel("Kelas")
    ax1.set_ylabel("Jumlah File")
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_classes, rotation=15, ha='right')
    ax1.legend(title="Fold", fontsize=8)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # ── PLOT 2: Total per kelas ──────────────────────────────────────────────
    ax2 = axes[1]
    total_counts = [class_total[cls] for cls in all_classes]
    total_all    = sum(total_counts)

    # Warna berbeda untuk kelas "normal" supaya mudah dibedakan
    bar_colors = ['#e05c5c' if cls == 'normal' else '#5c8ae0' for cls in all_classes]
    bars2 = ax2.bar(all_classes, total_counts, color=bar_colors, edgecolor='white', linewidth=0.8)

    for bar, count in zip(bars2, total_counts):
        pct = count / total_all * 100
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{count}\n({pct:.1f}%)",
            ha='center', va='bottom', fontsize=9
        )

    # Garis rata-rata
    avg = total_all / n_classes
    ax2.axhline(avg, color='gray', linestyle='--', linewidth=1.2, label=f'Rata-rata ({avg:.0f})')

    legend_patches = [
        mpatches.Patch(color='#5c8ae0', label='Kelas Bahaya'),
        mpatches.Patch(color='#e05c5c', label='Kelas Normal'),
    ]
    ax2.legend(handles=legend_patches + [plt.Line2D([0], [0], color='gray', linestyle='--', label=f'Rata-rata ({avg:.0f})')], fontsize=8)

    ax2.set_title("Total Keseluruhan per Kelas")
    ax2.set_xlabel("Kelas")
    ax2.set_ylabel("Jumlah File")
    ax2.set_xticklabels(all_classes, rotation=15, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Simpan
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "distribusi_data.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot disimpan ke: {out_path}")
    plt.show()


if __name__ == "__main__":
    data_dir = config.DATA_DIR  # biarkan string biasa
    save_dir = os.path.join(config.ROOT_DIR, "outputs", "eda")

    print(f"📂 Membaca data dari: {data_dir}")
    distribution, class_total = count_distribution(data_dir)

    if distribution:
        plot_distribution(distribution, class_total, save_dir)