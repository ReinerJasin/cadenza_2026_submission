# """
# noise_analysis.py
# -----------------
# Estimate per-frequency noise levels between a clean and noisy audio file
# and aggregate them into 10 fixed synthesizer bands.

# Dependencies:
#     pip install numpy scipy soundfile

# Usage:
#     python noise_analysis.py clean.wav noisy.wav
# """

# import numpy as np
# # from scipy.io import wavfile
# import soundfile as sf
# from scipy.signal import stft, get_window, correlate, resample_poly
# import json
# import sys
# import os


# # ---------- Utility functions ----------

# def to_mono(x):
#     """Convert stereo -> mono if needed."""
#     return x.mean(axis=1) if x.ndim == 2 else x


# def align_by_xcorr(ref, sig, max_lag=None):
#     """
#     Align sig to ref by maximizing cross-correlation.
#     Returns (aligned_signal, lag_samples)
#     """
#     n = min(len(ref), len(sig))
#     ref, sig = ref[:n], sig[:n]
#     xcorr = correlate(sig, ref, mode="full")
#     lags = np.arange(-n + 1, n)

#     if max_lag is not None:
#         mask = (lags >= -max_lag) & (lags <= max_lag)
#         xcorr, lags = xcorr[mask], lags[mask]

#     lag = lags[np.argmax(xcorr)]

#     if lag > 0:  # shift right
#         sig = np.pad(sig, (lag, 0))[:n]
#     elif lag < 0:  # shift left
#         sig = np.pad(sig[-lag:], (0, -lag))[:n]

#     return sig, lag


# def per_freq_gain_and_noise(x, y, sr, n_fft=4096, hop=1024, window="hann", use_median=True):
#     """
#     Estimate per-frequency gain (complex) and noise power.
#     """
#     win = get_window(window, n_fft, fftbins=True)
#     f, t, X = stft(x, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
#     _, _, Y = stft(y, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)

#     eps = 1e-12
#     num = np.sum(Y * np.conj(X), axis=1)             # numerator of LS fit
#     den = np.sum(np.abs(X) ** 2, axis=1) + eps
#     G = num / den                                    # complex gain
#     R = Y - (G[:, None] * X)                         # residual

#     agg = np.median if use_median else np.mean
#     P_noise = agg(np.abs(R) ** 2, axis=1)
#     P_sig = agg(np.abs(G[:, None] * X) ** 2, axis=1)
#     SNR_dB = 10 * np.log10((P_sig + eps) / (P_noise + eps))

#     return {
#         "freq_hz": f,
#         "G_complex": G,
#         "G_mag": np.abs(G),
#         "noise_power": P_noise,
#         "signal_power": P_sig,
#         "snr_db": SNR_dB,
#     }


# def band_average(f, values, band_centers):
#     """
#     Average per-frequency values into bands defined by their center frequencies.
#     Returns average per band.
#     """
#     band_centers = np.asarray(band_centers)
#     # boundaries halfway (geometric mean)
#     band_limits = np.sqrt(band_centers[:-1] * band_centers[1:])
#     low_edges = np.concatenate(([0], band_limits))
#     high_edges = np.concatenate((band_limits, [f[-1]]))

#     band_vals = []
#     for lo, hi in zip(low_edges, high_edges):
#         mask = (f >= lo) & (f < hi)
#         if np.any(mask):
#             band_vals.append(np.sqrt(values[mask].mean()))
#         else:
#             band_vals.append(0.0)
#     return np.array(band_vals)


# # ---------- Main processing ----------

# def main(clean_path, noisy_path, output_json="band_noise.json"):
#     # --- load files ---
#     # sr_c, x = wavfile.read(clean_path)
#     # sr_n, y = wavfile.read(noisy_path)
#     x, sr_c = sf.read(clean_path)
#     y, sr_n = sf.read(noisy_path)

#     x, y = to_mono(x.astype(np.float32)), to_mono(y.astype(np.float32))

#     # --- resample if needed ---
#     target_sr = sr_c
#     if sr_c != sr_n:
#         y = resample_poly(y, sr_c, sr_n)

#     # --- align signals ---
#     n = min(len(x), len(y))
#     x, y = x[:n], y[:n]
#     y_aligned, lag = align_by_xcorr(x, y, max_lag=target_sr)

#     # --- estimate per-frequency stats ---
#     res = per_freq_gain_and_noise(x, y_aligned, sr=target_sr)
#     f = res["freq_hz"]
#     noise_power = res["noise_power"]

#     # --- aggregate into 10 synth bands ---
#     synth_bands = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
#     band_noise = band_average(f, noise_power, synth_bands)

#     # convert to dB relative to max
#     # band_noise_db = 20 * np.log10(band_noise / (band_noise.max() + 1e-12))
#     band_noise_db = 20 * np.log10(np.maximum(band_noise, 1e-12) / (band_noise.max() + 1e-12))

#     band_table = [
#         {"band_hz": int(fc), "noise_level_db": float(db)}
#         for fc, db in zip(synth_bands, band_noise_db)
#     ]

#     # --- print results ---
#     print("\nNoise level per synthesizer band:")
#     print("┌────────────┬──────────────────┐")
#     print("│  Band (Hz) │ Noise Level (dB) │")
#     print("├────────────┼──────────────────┤")
#     for row in band_table:
#         print(f"│ {row['band_hz']:>10} │ {row['noise_level_db']:>16.2f} │")
#     print("└────────────┴──────────────────┘")
#     print(f"\nLag alignment: {lag} samples")

#     # --- save to JSON ---
#     with open(output_json, "w") as f_out:
#         json.dump({"bands": band_table}, f_out, indent=2)
#     print(f"\nSaved results to {output_json}")


# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python noise_analysis.py clean.wav noisy.wav")
#         sys.exit(1)

#     clean_wav = sys.argv[1]
#     noisy_wav = sys.argv[2]

#     if not os.path.exists(clean_wav) or not os.path.exists(noisy_wav):
#         print("Error: one or both WAV files not found.")
#         sys.exit(1)

#     main(clean_wav, noisy_wav)

"""
noise_analysis.py
-----------------
Estimate per-frequency noise levels between a clean and noisy audio file,
then aggregate them into 31 ISO standard (1/3-octave) bands.

Dependencies:
    pip install numpy scipy soundfile

Usage:
    python noise_analysis.py clean.wav noisy.wav
"""

import numpy as np
import soundfile as sf
from scipy.signal import stft, get_window, correlate, resample_poly
import json
import sys
import os


# ---------- Utility functions ----------

def to_mono(x):
    """Convert stereo -> mono if needed."""
    return x.mean(axis=1) if x.ndim == 2 else x


def align_by_xcorr(ref, sig, max_lag=None):
    """
    Align 'sig' to 'ref' by maximizing cross-correlation.
    Returns (aligned_signal, lag_samples).
    """
    n = min(len(ref), len(sig))
    ref, sig = ref[:n], sig[:n]
    xcorr = correlate(sig, ref, mode="full")
    lags = np.arange(-n + 1, n)

    if max_lag is not None:
        mask = (lags >= -max_lag) & (lags <= max_lag)
        xcorr, lags = xcorr[mask], lags[mask]

    lag = lags[np.argmax(xcorr)]

    if lag > 0:  # shift right
        sig = np.pad(sig, (lag, 0))[:n]
    elif lag < 0:  # shift left
        sig = np.pad(sig[-lag:], (0, -lag))[:n]

    return sig, lag


def per_freq_gain_and_noise(x, y, sr, n_fft=4096, hop=1024, window="hann", use_median=True):
    """
    Estimate per-frequency gain (complex) and noise power between two signals.
    """
    win = get_window(window, n_fft, fftbins=True)
    f, t, X = stft(x, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    _, _, Y = stft(y, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)

    eps = 1e-12
    num = np.sum(Y * np.conj(X), axis=1)
    den = np.sum(np.abs(X) ** 2, axis=1) + eps
    G = num / den                                   # complex gain
    R = Y - (G[:, None] * X)                        # residual (difference)

    agg = np.median if use_median else np.mean
    P_noise = agg(np.abs(R) ** 2, axis=1)
    P_sig = agg(np.abs(G[:, None] * X) ** 2, axis=1)
    SNR_dB = 10 * np.log10((P_sig + eps) / (P_noise + eps))

    return {
        "freq_hz": f,
        "G_complex": G,
        "G_mag": np.abs(G),
        "noise_power": P_noise,
        "signal_power": P_sig,
        "snr_db": SNR_dB,
    }


def band_average(f, values, band_centers):
    """
    Average per-frequency values into 1/3-octave bands defined by center frequencies.
    Returns array of averaged values per band.
    """
    band_centers = np.asarray(band_centers)
    # Define boundaries halfway between bands (geometric mean)
    band_limits = np.sqrt(band_centers[:-1] * band_centers[1:])
    low_edges = np.concatenate(([0], band_limits))
    high_edges = np.concatenate((band_limits, [f[-1]]))

    band_vals = []
    for lo, hi in zip(low_edges, high_edges):
        mask = (f >= lo) & (f < hi)
        if np.any(mask):
            band_vals.append(np.sqrt(np.mean(values[mask])))
        else:
            band_vals.append(0.0)
    return np.array(band_vals)


# ---------- Main processing ----------

def main(clean_path, noisy_path, output_json="band_noise.json"):
    # --- load files ---
    x, sr_c = sf.read(clean_path)
    y, sr_n = sf.read(noisy_path)

    # --- convert to mono ---
    x, y = to_mono(x.astype(np.float32)), to_mono(y.astype(np.float32))

    # --- resample if needed ---
    target_sr = sr_c
    if sr_c != sr_n:
        y = resample_poly(y, sr_c, sr_n)

    # --- align signals ---
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    y_aligned, lag = align_by_xcorr(x, y, max_lag=target_sr)

    # --- trim & normalize ---
    min_len = min(len(x), len(y_aligned))
    x, y_aligned = x[:min_len], y_aligned[:min_len]
    x /= np.max(np.abs(x) + 1e-12)
    y_aligned /= np.max(np.abs(y_aligned) + 1e-12)

    # --- estimate per-frequency stats ---
    res = per_freq_gain_and_noise(x, y_aligned, sr=target_sr)
    f = res["freq_hz"]
    noise_power = res["noise_power"]

    # --- ISO standard 31-band 1/3-octave EQ centers ---
    iso_31_bands = [
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000,
        6300, 8000, 10000, 12500, 16000, 20000
    ]

    # --- aggregate noise into 31 bands ---
    band_noise = band_average(f, noise_power, iso_31_bands)

    # --- convert to relative dB ---
    band_noise_db = 20 * np.log10(np.maximum(band_noise, 1e-12) / (band_noise.max() + 1e-12))

    # --- prepare output table ---
    band_table = [
        {"band_hz": float(fc), "noise_level_db": float(db)}
        for fc, db in zip(iso_31_bands, band_noise_db)
    ]

    # --- print results ---
    print("\nNoise level per ISO 1/3-octave band:")
    print("┌────────────┬──────────────────┐")
    print("│  Band (Hz) │ Noise Level (dB) │")
    print("├────────────┼──────────────────┤")
    for row in band_table:
        print(f"│ {row['band_hz']:>10.1f} │ {row['noise_level_db']:>16.2f} │")
    print("└────────────┴──────────────────┘")
    print(f"\nLag alignment: {lag} samples")

    # --- save to JSON ---
    with open(output_json, "w") as f_out:
        json.dump({"bands": band_table}, f_out, indent=2)
    print(f"\nSaved results to {output_json}")


# ---------- Entry point ----------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python noise_analysis.py clean.wav noisy.wav")
        sys.exit(1)

    clean_wav = sys.argv[1]
    noisy_wav = sys.argv[2]

    if not os.path.exists(clean_wav) or not os.path.exists(noisy_wav):
        print("Error: one or both audio files not found.")
        sys.exit(1)

    main(clean_wav, noisy_wav)
