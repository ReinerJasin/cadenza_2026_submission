"""
reconstruct_per_freq.py
-----------------------
Reconstruct noisy audio using per-frequency complex gain estimated from clean & noisy pair.

Usage:
    python reconstruct_per_freq.py clean.wav noisy.wav out_reconstructed.wav
"""

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, get_window, correlate


def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


def align_by_xcorr(ref, sig, max_lag=None):
    n = min(len(ref), len(sig))
    ref, sig = ref[:n], sig[:n]
    xcorr = correlate(sig, ref, mode="full")
    lags = np.arange(-n + 1, n)
    lag = lags[np.argmax(xcorr)]
    if lag > 0:
        sig = np.pad(sig, (lag, 0))[:n]
    elif lag < 0:
        sig = np.pad(sig[-lag:], (0, -lag))[:n]
    return sig, lag


def reconstruct_from_gain(clean_path, noisy_path, out_path):
    x, sr_c = sf.read(clean_path)
    y, sr_n = sf.read(noisy_path)
    x, y = to_mono(x.astype(np.float32)), to_mono(y.astype(np.float32))
    if sr_c != sr_n:
        raise ValueError("Sample rates must match for Option 1.")
    y, _ = align_by_xcorr(x, y, max_lag=sr_c)

    # --- STFT ---
    n_fft, hop = 4096, 1024
    win = get_window("hann", n_fft, fftbins=True)
    f, t, X = stft(x, fs=sr_c, window=win, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Y = stft(y, fs=sr_c, window=win, nperseg=n_fft, noverlap=n_fft - hop)

    # --- Compute per-frequency complex gain ---
    eps = 1e-12
    num = np.sum(Y * np.conj(X), axis=1)
    den = np.sum(np.abs(X) ** 2, axis=1) + eps
    G = num / den

    # --- Reconstruct ---
    Y_rec = G[:, None] * X
    _, y_rec = istft(Y_rec, fs=sr_c, nperseg=n_fft, noverlap=n_fft - hop)

    # --- Normalize & save ---
    y_rec /= np.max(np.abs(y_rec) + 1e-12)
    sf.write(out_path, y_rec, sr_c)
    print(f"Reconstructed audio saved to {out_path}")


if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 4:
        print("Usage: python reconstruct_per_freq.py clean.wav noisy.wav out.wav")
        sys.exit(1)
    clean, noisy, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if not os.path.exists(clean) or not os.path.exists(noisy):
        print("Error: input file(s) not found.")
        sys.exit(1)
    reconstruct_from_gain(clean, noisy, out)
