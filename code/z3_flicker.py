import numpy as np

class FlickerSignalProcessor:

    def __init__(self, fs=6000.0):
        self.fs = fs

    def get_flicker_channel(self, envelope):
        return np.asarray(envelope) - 1.0
