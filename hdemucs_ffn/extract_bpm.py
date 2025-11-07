import librosa
import numpy as np
import torch

def compute_bpm(audio, sr):
    """
    Compute BPM (beats per minute) of the given audio.
    Converts stereo to mono before processing.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()

    # Convert audio from stereo to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    return float(tempo)
