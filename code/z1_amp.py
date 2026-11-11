import numpy as np
from scipy import signal
from scipy.fftpack import fft
from scipy.ndimage import gaussian_filter1d

class AmplitudeSignalProcessor:

    def __init__(self, fs=6000.0):
        self.fs = fs

    def get_amplitude_channel(self, x, alpha=0.95):
        N = len(x)
        x_fft = np.fft.fft(x)
        freqs = np.fft.fftfreq(N, d=1.0 / self.fs)
        f_low = 5.0
        f_high = 95.0
        mask = np.zeros(N)
        band_pos = (freqs >= f_low) & (freqs <= f_high)
        band_neg = (freqs <= -f_low) & (freqs >= -f_high)
        mask[band_pos | band_neg] = 1.0
        tapered_mask = gaussian_filter1d(mask, sigma=3)
        X_masked = x_fft * tapered_mask
        x_fundamental = np.real(np.fft.ifft(X_masked))
        X_fund_fft = np.fft.fft(x_fundamental)
        sgn = np.zeros(N)
        sgn[freqs > 0] = 1.0
        sgn[freqs < 0] = -1.0
        operator = np.exp(-1j * alpha * (np.pi / 2.0) * sgn)
        x_frht = np.real(np.fft.ifft(X_fund_fft * operator))
        envelope = np.abs(x_fundamental + 1j * x_frht)
        return np.stack([x_fundamental, envelope], axis=0)
