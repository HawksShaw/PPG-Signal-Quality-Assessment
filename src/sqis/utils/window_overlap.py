import numpy as np

def window_overlap(ppg_signal, fs, window_overlap=0.7, window_length=8):

    # --- Added an overlap to windows for better artifact detection ---
    window_samples = int(fs*window_length)
    stride_samples = int(window_samples*(1-window_overlap))

    if stride_samples <= 0:
        raise ValueError("Overlap too high - step size is 0.")
    if len(signal) < window_samples:
        print(f"Warning - signal length of {len(signal)} is shorter than the window length of {len(ppg_window)}")
        return []

    windows = []

    for start in range(0, len(ppg_signal)-window_samples+1, stride_samples):
        end = start + window_samples
        segment = ppg_signal[start:end]

        windows.append({
            "data" : ppg_signal,
            "start_idx" : start,
            "end_idx" : end,
            "timestamp" : start/fs
        })

    return windows
