import json
import torch
from torch.utils.data import Dataset

class LyricDataset(Dataset):
    def __init__(self, json_path: str, start_idx: int = 0, end_idx: int = None):
        with open(json_path, "r") as f:
            all_metadata = json.load(f)

        if end_idx is None:
            end_idx = len(all_metadata)

        # Data Slicing part
        self.metadata = all_metadata[start_idx:end_idx]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        asr_corr = float(item.get("asr_correctness", 0.0))
        inv_bpm = float(item.get("inv_bpm", 1.0))
        bpm = float(item.get("bpm", 120.0))
        stoi_score = float(item.get("stoi", 0.0))

        features = torch.tensor(
            [asr_corr, inv_bpm, bpm, stoi_score],
            dtype=torch.float32
        )
        label = torch.tensor(float(item.get("label", 0.0)), dtype=torch.float32)
        return features, label
