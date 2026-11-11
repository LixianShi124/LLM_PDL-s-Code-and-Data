import numpy as np
from scipy import signal
from scipy.fftpack import fft
from scipy.ndimage import gaussian_filter1d

class HarmonicSignalProcessor:

    def __init__(self, fs=6000.0):
        self.fs = fs
        self.f_grid = 50

    def get_harmonic_channel(self, x):
        N = len(x)
        window = np.hanning(N)
        xf = fft(x * window)
        spectrum = np.abs(xf)[:N // 2] * (2.0 / N)
        freq_res = self.fs / N
        idx_fund = int(self.f_grid / freq_res)
        v_fund = np.max(spectrum[max(0, idx_fund - 1):idx_fund + 2]) + 1e-09

        def get_harmonic_amp(order):
            center = order * idx_fund
            if center < len(spectrum):
                return np.max(spectrum[max(0, center - 1):min(len(spectrum), center + 2)])
            return 0.0
        v_h3 = get_harmonic_amp(3)
        v_h5 = get_harmonic_amp(5)
        v_h7 = get_harmonic_amp(7)
        hr3 = v_h3 / v_fund
        hr5 = v_h5 / v_fund
        hr7 = v_h7 / v_fund
        temp_spec = spectrum.copy()
        temp_spec[max(0, idx_fund - 2):idx_fund + 3] = 0
        temp_spec[0] = 0
        limit_idx = int(2000 / freq_res)
        distortion_energy = np.sum(temp_spec[:limit_idx] ** 2)
        thd = np.sqrt(distortion_energy) / v_fund
        phys_feats = np.array([thd, hr3, hr5, hr7], dtype=np.float32)
        return np.concatenate([spectrum, phys_feats])
