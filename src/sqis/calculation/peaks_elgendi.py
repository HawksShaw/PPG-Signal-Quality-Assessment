import numpy as np
import pandas as pd

def peaks_elgendi(signal, fs):
    """
    Ref: Elgendi M. et al., 'Systolic Peak Detection in Acceleration Photoplethysmograms...'
    """

    #Square the signal
    signal_sq = np.power(signal, 2)

    #Define moving average window sizes
    fast_window = int(0.11 * fs) | 1 #fast window for expected ~0.11s-wide systolic peak
    slow_window = int(0.60 * fs) | 1 #slow window for expected ~0.60s-long heartbeat

    ma_peak = pd.Series(signal_sq).rolling(window=fast_window, center=True).mean().fillna(0).values
    ma_beat = pd.Series(signal_sq).rolling(window=slow_window, center=True).mean().fillna(0).values

    beta = 0.02 * np.mean(signal_sq)
    roi  = ma_peak > (ma_beat + beta)

    peaks = []
    peak_search = True

    start_region = 0
    in_region = False

    for i, is_active in enumerate(roi):
        if is_active and not in_region:
            in_region = True
            start_region = i
        elif not is_active and in_region:
            in_region = False
            end_region = i
            if end_region - start_region > 0:
                local_max_idx = np.argmax(signal[start_region:end_region]) + start_region
                peaks.append(local_max_idx)

    return np.array(peaks)