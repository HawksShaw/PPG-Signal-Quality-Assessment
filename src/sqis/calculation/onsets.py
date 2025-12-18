import numpy as np
from scipy.signal import find_peaks

def pulse_onsets(signal, fs, systolic_index, pulse_fraction_percentage=0.3):
    onsets = []

    for onset in range(1, len(systolic_index)):
        previous_systole = systolic_index[onset-1]
        current_systole = systolic_index[onset]

        search_start = previous_systole + int(pulse_fraction_percentage*(current_systole-previous_systole))
        search_end = current_systole

        if search_end <= search_start:
            continue

        search_segment = signal[search_start : search_end]

        minima, _ = find_peaks(-search_segment)

        if len(minima) == 0:
            continue

        found_minima = minima[np.argmin(search_segment[minima])]
        onsets.append(found_minima + search_start)

    return np.array(onsets, dtype=int)