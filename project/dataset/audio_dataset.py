import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from tokenizers import Tokenizer, models, pre_tokenizers, decoders

def move_data_to_device(data, device):
    ret = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value.to(device)
    return ret

def get_data_loader(split, args):
    dataset = AudioWithMetadataDataset(
        split=split,
        root_dir=args['data_root']
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader

def collate_fn(batch):
    """
    Pads audio waveforms in the batch to the max length and groups metadata/sample rates.
    Returns a dict with:
        - waveform: [batch, channels, max_samples]
        - sample_rate: list of sample rates
        - metadata: list of dicts
    """
    # print(f'first data in batch: {batch[0]}')

    waveforms = [item["waveform"] for item in batch]
    sample_rates = [item["sample_rate"] for item in batch]
    metadata = [item["metadata"] for item in batch]

    # for i, w in enumerate(waveforms):
    #     print(f'w-{i}: {w.shape}')
    #     print(f'w-{i}: {w.shape[1]}')

    # Find max length of waveform
    max_len_audio = max(w.shape[1] for w in waveforms)

    # Find max length of input_ids
    tokenized_prompts = [m["tokenized_prompt"] for m in metadata]
    tokenized_responses = [m["tokenized_response"] for m in metadata]

    # Find the max sequence length across both
    max_len_prompt = max(len(p) for p in tokenized_prompts)
    max_len_response = max(len(r) for r in tokenized_responses)
    max_len_text = max(max_len_prompt, max_len_response)

    # Pad waveforms to max_len_audio
    padded_waveforms = []
    for w in waveforms:
        pad_len = max_len_audio - w.shape[1]
        if pad_len > 0:
            w = torch.nn.functional.pad(w, (0, pad_len))
        padded_waveforms.append(w)

    # Pad prompt and response to max_len_text
    padded_prompt = []
    padded_response = []
    for m in metadata:
        # Prompt
        prompt = m["tokenized_prompt"]
        prompt_pad = max_len_text - len(prompt)
        prompt_padded = prompt + [0] * prompt_pad
        padded_prompt.append(torch.tensor(prompt_padded, dtype=torch.long))

        # Response
        response = m["tokenized_response"]
        response_pad = max_len_text - len(response)
        response_padded = response + [0] * response_pad
        padded_response.append(torch.tensor(response_padded, dtype=torch.long))

    # Stack into tensor
    waveforms = torch.stack(padded_waveforms)  # [batch, channels, max_samples]
    prompt = torch.stack(padded_prompt)
    response = torch.stack(padded_response)

    return {
        "waveform": waveforms,
        "sample_rate": sample_rates,
        "metadata": metadata,
        "padded_prompt": prompt,
        "padded_response": response
    }

def get_tokenizer(save_path="tokenizer.json"):
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Add special token as the blank token, can also use <pad> or <mask>.
    tokenizer.add_special_tokens(["□"])

    # Add each letters, space, and aphostrophe as a token
    tokenizer.add_tokens(list("abcdefghijklmnopqrstuvwxyz '"))

    # Define how to split the input text, in this case we use the ByteLevel PreTokenizer, splitting text into individual bytes insted of unicode characters
    # This ensures that even unseen characters (like emoji or accents) can sill be reperesented as bytes (0-255 range).
    # It also normalizes spaces and handles punctuation safely
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # Define how to Decode Tokens back
    tokenizer.decoder = decoders.ByteLevel()

    # Use the special token we made as the blank token
    tokenizer.blank_token = tokenizer.token_to_id("□")

    # Save the tokenizer setting as a json for reusability
    tokenizer.save(save_path)
    
    return tokenizer


class AudioWithMetadataDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, num_hearing_loss=False, tokenizer=None):
        """
        Args:
            root_dir (str or Path): Path to dataset root (containing train/test folders and metadata)
            split (str): "train" or "test"
            transform (callable, optional): Optional transform for audio tensor
        """
        self.root_dir = Path(root_dir)
        self.split = split

        self.transform = transform
        self.num_hearing_loss = num_hearing_loss

        if tokenizer is None:
            tokenizer_path = Path(__file__).parent / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"Tokenizer file not found at: {tokenizer_path}\n"
                    "Make sure tokenizer.json exists in the dataset folder."
                )
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            self.tokenizer = tokenizer

        # Load metadata JSON
        metadata_file = self.root_dir / "metadata" / f"{split}_metadata.json"
        print(f'WOYYY CARI DI: {metadata_file}')
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        # Path to audio files
        self.audio_dir = self.root_dir / split / "signals"

        torchaudio.set_audio_backend("soundfile")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        signal_id = entry["signal"]

        # Load audio
        audio_path = self.audio_dir / f"{signal_id}.flac"
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono audio
        if waveform.shape[0] == 2:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply optional transform
        if self.transform:
            waveform = self.transform(waveform)

        # Tokenize prompt and response
        if "prompt" in entry:
            entry["tokenized_prompt"] = self.tokenizer.encode(entry["prompt"]).ids
        if "response" in entry:
            entry["tokenized_response"] = self.tokenizer.encode(entry["response"]).ids

        # Converting the hearing_loss level in the metadata to numerical
        # My plan on this is to have a numerical representation of the losses that can be feed as an input
        # My idea: Use the ground truth, and calculate the mean score of the correctness for each class, then we use that number to represent the noise level coefficient
        # Another way is to use STOI metric, but this requires the original audio and the processed audio, which might cause data leakage.
        if self.num_hearing_loss:
            hearing_loss_map = {'No Loss':0, 'Mild':1, 'Moderate':2}
            entry['num_hearing_loss'] = hearing_loss_map[entry['hearing_loss']]


        # Return waveform, sample_rate, and metadata
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "metadata": entry
        }
