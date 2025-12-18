from scipy.signal import find_peaks
import numpy as np

def diastolic_peaks(signal, systolic_index, fs, prominence=None):
    diastoles = []

    for i in range(len(systolic_index) - 1):

        search_start = systolic_index[i]
        search_end = int((search_start + systolic_index[i+1])/2)
        search_segment = signal[search_start : search_end]

        if(len(search_segment)) < 0.1*fs:
            continue

        peaks, props = find_peaks(search_segment, prominence=prominence)

        if len(peaks) == 0:
            continue

        found_peak = peaks[np.argmax(search_segment[peaks])]
        diastoles.append(search_start+found_peak)

    return diastoles