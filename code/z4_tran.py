import numpy as np
from scipy import signal
from scipy.fftpack import fft
from scipy.ndimage import gaussian_filter1d

class TranSignalProcessor:

    def __init__(self, fs=6000.0, f0=50.0):
        self.fs = fs
        self.points_per_cycle = int(fs / f0)
        self.n_cycles = 10
        self.target_len = self.points_per_cycle * self.n_cycles

    def estimate_fundamental(self, x):
        from scipy.ndimage import gaussian_filter1d
        spectrum = np.fft.fft(x)
        frequencies = np.fft.fftfreq(len(x), d=1.0 / self.fs)
        mask = np.zeros(len(x))
        positive = (frequencies >= 5.0) & (frequencies <= 95.0)
        negative = (frequencies <= -5.0) & (frequencies >= -95.0)
        mask[positive | negative] = 1.0
        tapered_mask = gaussian_filter1d(mask, sigma=3)
        return np.real(np.fft.ifft(spectrum * tapered_mask)).astype(np.float32)

    def get_transient_channel(self, x, fundamental=None):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1 or not len(x) or (not np.isfinite(x).all()):
            raise ValueError('电压必须是一维、非空、有限数值序列')
        if len(x) < self.target_len:
            x = np.pad(x, (0, self.target_len - len(x)))
        else:
            x = x[:self.target_len]
        if fundamental is None:
            fundamental = self.estimate_fundamental(x)
        else:
            fundamental = np.asarray(fundamental, dtype=np.float32)
            if fundamental.shape != x.shape or not np.isfinite(fundamental).all():
                raise ValueError('共享基波估计必须与规整后的电压等长且数值有限')
        z = x - fundamental
        cycles = z.reshape(self.n_cycles, self.points_per_cycle)
        median_cycle = np.median(cycles, axis=0)
        residual = (cycles - median_cycle[None, :]).reshape(self.target_len)
        return np.stack([x, residual], axis=0).astype(np.float32)
