"""
reconstruct_eq_filter.py
------------------------
Apply a 31-band EQ curve (in dB) to a clean audio file.

Usage:
    python reconstruct_eq_filter.py clean.wav band_noise.json output.wav
"""

import numpy as np
import json, sys, os
import soundfile as sf
from scipy.signal import firwin2, sosfiltfilt


def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


def apply_eq_from_json(audio_path, band_json, out_path):
    # --- load audio ---
    x, sr = sf.read(audio_path)
    x = to_mono(x.astype(np.float32))
    x /= np.max(np.abs(x) + 1e-12)

    # --- load EQ bands ---
    with open(band_json, "r") as f:
        data = json.load(f)
    bands = np.array([b["band_hz"] for b in data["bands"]], dtype=float)
    gains_db = np.array([b["noise_level_db"] for b in data["bands"]], dtype=float)

    # convert dB → linear amplitude gain
    gains_lin = 10 ** (gains_db / 20)

    # normalize frequencies to [0, 1]
    f_norm = bands / (sr / 2)
    f_norm = np.clip(f_norm, 0, 1)

    # --- design smooth FIR filter ---
    taps = 4096  # higher = smoother response
    eq = firwin2(taps, f_norm, gains_lin)
    # apply linear-phase filter
    from scipy.signal import lfilter
    y = lfilter(eq, [1.0], x)

    # --- normalize & save ---
    y /= np.max(np.abs(y) + 1e-12)
    sf.write(out_path, y, sr)
    print(f"EQ-applied audio saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python reconstruct_eq_filter.py clean.wav band_noise.json output.wav")
        sys.exit(1)

    clean, json_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if not os.path.exists(clean) or not os.path.exists(json_path):
        print("Error: input file(s) not found.")
        sys.exit(1)

    apply_eq_from_json(clean, json_path, out)
