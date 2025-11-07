import numpy as np
import torch
from pystoi import stoi as stoi_fn

def compute_stoi(ref_audio, est_audio, sr):
    """
    Compute the Short-Time Objective Intelligibility (STOI) score
    between reference and estimated audio.

    Both signals are converted to mono and trimmed to the same length.
    """
    # Convert to numpy
    if isinstance(ref_audio, torch.Tensor):
        ref_audio = ref_audio.cpu().numpy()
    if isinstance(est_audio, torch.Tensor):
        est_audio = est_audio.cpu().numpy()

    # Stereo → mono
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=0)
    if est_audio.ndim > 1:
        est_audio = np.mean(est_audio, axis=0)

    # Match length
    min_len = min(len(ref_audio), len(est_audio))
    ref_audio = ref_audio[:min_len]
    est_audio = est_audio[:min_len]

    # Compute STOI
    score = stoi_fn(ref_audio, est_audio, sr, extended=False)
    return float(score)
