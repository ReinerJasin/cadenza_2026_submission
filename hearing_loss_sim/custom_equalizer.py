import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

class CustomEqualizer:
    def __init__(self, sr, bands, n_fft=4096, hop=1024):
        """
        Args:
            sr: sample rate
            bands: list of (center_freq, gain_db, bandwidth_hz)
            n_fft, hop: STFT parameters
        """
        self.sr = sr
        self.bands = bands
        self.n_fft = n_fft
        self.hop = hop

    def apply(self, audio):
        """
        Apply the equalizer to a mono signal.
        """
        f, t, Z = stft(audio, fs=self.sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop)
        gain_curve = np.ones_like(f)

        # Apply each band as a Gaussian gain curve
        for center, gain_db, bw in self.bands:
            gain = 10 ** (gain_db / 20.0)
            # Gaussian weight around the center
            weight = np.exp(-0.5 * ((f - center) / (bw / 2)) ** 2)
            gain_curve *= 1 + (gain - 1) * weight  # smooth blend

        # Apply gain to magnitude
        Z_eq = Z * gain_curve[:, None]

        _, y = istft(Z_eq, fs=self.sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop)
        return y

    def update_band(self, idx, gain_db=None, center=None, bandwidth=None):
        """Update a band’s parameters easily."""
        center0, gain0, bw0 = self.bands[idx]
        self.bands[idx] = (
            center if center is not None else center0,
            gain_db if gain_db is not None else gain0,
            bandwidth if bandwidth is not None else bw0
        )

    def print_bands(self):
        print("Current Equalizer Bands:")
        for i, (c, g, bw) in enumerate(self.bands):
            print(f"[{i}] Center={c:>6.0f} Hz | Gain={g:>6.1f} dB | BW={bw:>5.0f} Hz")

# Load your audio
x, sr = sf.read("/Users/reiner/Documents/GitHub/cadenza_2026_submission/project/dataset/cadenza_data/train/unprocessed/6cc97125bb708b1fa5a55e7e_unproc.flac")
if x.ndim == 2:
    x = x.mean(axis=1)  # convert to mono

# Define bands manually
bands = [
    (125,  0.0, 150),
    (250,  0.0, 200),
    (500,  0.0, 300),
    (1000, 0.0, 400),
    (1500, 0.0, 400),
    (2000, 0.0, 500),
    (3000, 0.0, 600),
    (4000, 0.0, 800),
    (6000, 0.0, 1000),
    (8000, 0.0, 1200),
]

eq = CustomEqualizer(sr, bands)

# Example: bring down 1k–2k drastically (simulate hearing loss)
eq.update_band(3, gain_db=-80)   # 1000 Hz band → -80 dB
eq.update_band(5, gain_db=-50)   # 2000 Hz band → -50 dB

eq.print_bands()

y = eq.apply(x)
sf.write("output_equalized.flac", y, sr)
