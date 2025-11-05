import json
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor, WhisperTokenizer

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")

class AudioWithMetadataDataset(Dataset):
    def __init__(self, root_dir, split="train", target_sr=16000):
        self.root_dir = Path(root_dir)
        self.split = split
        self.target_sr = target_sr

        # Load metadata
        metadata_file = self.root_dir / "metadata" / f"{split}_metadata.json"
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        # Audio folder
        self.audio_dir = self.root_dir / split / "signals"
        torchaudio.set_audio_backend("soundfile")

        # Cache resamplers for speed
        self._resamplers = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        signal_id = entry["signal"]
        audio_path = self.audio_dir / f"{signal_id}.flac"

        # Load & preprocess audio
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

        # Resample → 16kHz
        if sr != self.target_sr:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = self._resamplers[sr](waveform)

        # Convert to Whisper log-mel features
        input_features = feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=self.target_sr,
        ).input_features[0]

        # Tokenize text
        prompt_text = entry.get("prompt", "")
        labels = tokenizer(prompt_text).input_ids

        return {
            "input_features": torch.tensor(input_features, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long),
            "metadata": entry,
        }

def collate_fn(batch):
    # Separate items
    features = [b["input_features"] for b in batch]
    labels = [b["labels"] for b in batch]
    metadata = [b["metadata"] for b in batch]

    # Pad features (time dimension)
    max_len_feat = max(f.shape[-1] for f in features)
    padded_features = [
        torch.nn.functional.pad(f, (0, max_len_feat - f.shape[-1]))
        for f in features
    ]
    padded_features = torch.stack(padded_features)

    # Pad labels
    max_len_lbl = max(len(l) for l in labels)
    padded_labels = [
        torch.nn.functional.pad(l, (0, max_len_lbl - len(l)), value=tokenizer.pad_token_id)
        for l in labels
    ]
    padded_labels = torch.stack(padded_labels)

    return {
        "input_features": padded_features,  # [B, 80, T]
        "labels": padded_labels,            # [B, L]
        "metadata": metadata,
    }


def get_data_loader(split, args):
    dataset = AudioWithMetadataDataset(root_dir=args["data_root"], split=split)
    return DataLoader(
        dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
