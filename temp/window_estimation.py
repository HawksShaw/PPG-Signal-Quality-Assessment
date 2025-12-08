from prepare_wildppg import fs
import matplotlib.pyplot as plt
import numpy as np

def signal_cutoff(signal, window_start, window_length):

    window_samples = fs*window_length
    window_signal = signal[:window_samples]
    window_time = np.linspace(window_start, window_start + window_length, window_samples)

    return window_signal, window_time

