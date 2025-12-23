import numpy as np

def signal_window(signal, window_start, window_length, fs):

    window_samples = fs*window_length
    window_signal = signal[:window_samples]
    window_time = np.linspace(window_start, window_start + window_length, window_samples)

    return window_signal, window_time

