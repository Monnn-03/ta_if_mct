import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import json
from create_5fold_split import make_5fold_split

class AudioDataset(Dataset):
	def __init__(self, root_dir, fold=0, split_json='split.json', split_type='train', segment_length=41):
		"""
		Init adalah method untuk inisialisasi Dataset.
		Dalam kata lain, ini adalah konstruktor (mempersiapkan data) dari kelas AudioDataset.
		"""
		self.labels_map = {
			'car_horn': 0,
    	'dog_bark': 1,
    	'gun_shot': 2,
    	'siren': 3
		}

		self.segment_length = segment_length  # panjang segmen dalam frame

		### --- SPLIT HANDLING ---
		split_path = os.path.join(os.getcwd(), split_json)

		# Jika file split_json ada
		if os.path.exists(split_path):
			with open(split_path, 'r') as f:
				self.splits = json.load(f)

		# Jika file split_json tidak ada
		else :
			all_samples = []
			for label in self.labels_map:
				label_dir = os.path.join(root_dir, label)
				for fname in os.listdir(label_dir):
					fpath = os.path.join(label_dir, fname)
					if fname.endswith('.wav') and os.path.isfile(fpath):
						all_samples.append((fpath, label))

			# Panggil fungsi untuk membuat split 5 fold
			folds = make_5fold_split(all_samples, n_folds=5)
			with open(split_path, 'w') as f:
				json.dump(folds, f, indent=2)
			self.splits = folds
		### --- END SPLIT HANDLING ---

		self.samples = [(item["file_path"], item["label"]) for item in self.splits[fold][split_type]]

	def __len__(self):
		"""
		Method ini mengembalikan panjang dataset, jumlah total sampel yang ada.
		"""
		return len(self.samples)
	
	def __getitem__(self, idx):
		"""
		Method ini digunakan untuk mengambil item dari dataset berdasarkan indeks tertentu.
		"""
		return self.samples[idx]
	
if __name__ == "__main__":
	data_dir = os.path.join(os.getcwd(), 'data')
	dataset = AudioDataset(root_dir=data_dir, fold=0, split_json='split.json', split_type='train')
	print(f"Jumlah sampel dalam dataset: {len(dataset)}")