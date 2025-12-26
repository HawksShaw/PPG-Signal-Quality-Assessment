import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def hr_quality(ppg_peaks, ecg_signal, fs, tolerance=8):
    
    ecg_peaks, _ = find_peaks(ecg_signal, distance=fs/2.5, height=0.5)

    ppg_intervals = np.diff(ppg_peaks)
    ecg_intervals = np.diff(ecg_peaks)

    ppg_hr = 60 / np.mean(ppg_intervals)
    ecg_hr = 60 / np.mean(ecg_intervals)

    if len(ecg_peaks) < 2:
        print("Uncertain comparison - bad ECG signal")
        return None

    error = abs(ecg_hr - ppg_hr)

    if error >= tolerance:
        return "Signal Rejected", error
    else:
        return "Signal Accepted", error

