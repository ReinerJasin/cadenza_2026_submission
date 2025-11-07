"""
hdemucs.py
--------------------------------
Utility for vocal separation using any Demucs family model
(hdemucs, hdemucs_mmi, htdemucs, htdemucs_ft).
"""

import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model


def separate_vocals(audio_path: str, model_name: str = "htdemucs"):
    """
    Separate vocals from an audio file using the specified Demucs model.

    Args:
        audio_path (str): Path to input audio file (.wav or .flac)
        model_name (str): One of ["hdemucs", "hdemucs_mmi", "htdemucs", "htdemucs_ft"]

    Returns:
        vocals (torch.Tensor): [channels, samples] tensor containing separated vocals
        sr (int): Sampling rate
    """
    print(f"🎧 Using model: {model_name}")
    print(f"Audio path: {audio_path}\n")

    # 1. Load pretrained model bag, then pick the first model
    bag = get_model(model_name)
    model = bag.models[0]
    model.eval()

    # 2. Load audio
    wav, sr = torchaudio.load(audio_path)

    # Demucs expects 2 channels. If your audio is mono, make it stereo.
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # [1, T] -> [2, T]

    # Normalize to avoid clipping
    wav = wav / wav.abs().max()

    # 3. Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    wav = wav.to(device)

    # 4. Run separation
    # apply_model returns: [batch, sources, channels, time]
    with torch.no_grad():
        estimates = apply_model(model, wav[None], device=device)[0]

    # 5. Get index of "vocals" stem
    sources = bag.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
    vocals_idx = sources.index("vocals")
    vocals = estimates[vocals_idx]  # shape: [channels, time]

    return vocals.cpu(), sr


if __name__ == "__main__":
    AUDIO_PATH = "/Users/reiner/Documents/GitHub/cadenza_2026_submission/project/dataset/cadenza_data/train/unprocessed/a2bf283251ea0a8fffd405f3_unproc.flac"
    vocals, sr = separate_vocals(AUDIO_PATH, model_name="htdemucs")
    print(f"✅ Separated vocals shape: {vocals.shape}, sample rate: {sr}")
    torchaudio.save("vocals_output.wav", vocals, sr)  # vocals already [C, T]
    print("💾 Saved to vocals_output.wav")
