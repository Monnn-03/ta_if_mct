import os
import shutil
import pandas as pd

SOURCE_METADATA_FILE = 'UrbanSound8K/metadata/UrbanSound8K.csv'
SOURCE_AUDIO_DIR = 'UrbanSound8K/audio'
DEST_DIR = 'data'

TARGET_CLASSES = ['siren', 'car_horn', 'gun_shot', 'dog_bark']

def main():
	if not os.path.exists(SOURCE_METADATA_FILE):
		print(f"FILE METADATA TIDAK DITEMUKAN DI : {SOURCE_METADATA_FILE}")
		return

	print("Mulai Penyusunan Ulang Dataset...")

	# 1. Membaca CSV metadata
	df = pd.read_csv(SOURCE_METADATA_FILE)

	# 2. Memfilter kelas target
	df_filtered = df[df['class'].isin(TARGET_CLASSES)]

	print(f"Ditemukan {len(df_filtered)} yang cocok kelasnya.")

	count_success = 0

	# 3. Pindahkan satu per satu file audio
	for index, row in df_filtered.iterrows():
		class_name = row['class']
		fold_asli = row['fold']
		file_name = row['slice_file_name']

		# LOGIKA PENGGABUNGAN
		# Fold 1 & 2 -> Fold 1
		# Fold 3 & 4 -> Fold 2
		# DST...

		if fold_asli in [1, 2]: fold_baru = 1
		elif fold_asli in [3, 4]: fold_baru = 2
		elif fold_asli in [5, 6]: fold_baru = 3
		elif fold_asli in [7, 8]: fold_baru = 4
		elif fold_asli in [9,10]: fold_baru = 5
		else: continue

		# Menentukan direktori tujuan
		folder_tujuan = os.path.join(DEST_DIR, f"Fold{fold_baru}", class_name)

		# Buat direktori jika belum ada
		os.makedirs(folder_tujuan, exist_ok=True)

		# --- MEMINDAHKAN FILE ---
		src_path = os.path.join(SOURCE_AUDIO_DIR, f"fold{fold_asli}", file_name)
		dest_path = os.path.join(folder_tujuan, file_name)

		try:
			# Pakai copy
			shutil.copy2(src_path, dest_path)
			count_success += 1

			if count_success % 100 == 0:
				print(f"{count_success} file berhasil dipindahkan...")

		except FileNotFoundError:
			print(f"File tidak ditemukan: {src_path}")

	print(f"Selesai! Total {count_success} file berhasil dipindahkan ke '{DEST_DIR}'.")

if __name__ == "__main__":
	main()
