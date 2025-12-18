import numpy as np
from scipy.signal import find_peaks

def dicrotic_notches(signal, systolic_index, diastolic_index):
    notches_index = []

    n_beats = min(len(systolic_index), len(diastolic_index))

    for i in range(n_beats):
        search_start = systolic_index[i]
        search_end = diastolic_index[i]

        if search_end <= search_start:
            continue

        search_segment = signal[search_start:search_end]

        if len(search_segment) < 3:
            continue

        minima, props = find_peaks(-search_segment)

        if len(minima) == 0:
            continue

        main_min = minima[np.argmin(search_segment[minima])]
        notches_index.append(search_start + main_min)

    return np.array(notches_index, dtype=int)
