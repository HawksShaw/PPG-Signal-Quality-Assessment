from scipy.signal import find_peaks
from signal_filtering import *
from butterworth import bandpass

def beat_detection(filtered_signal, fs, min_distance = 0.33, min_prominence = None):
    distance = int(min_distance*fs)
    if min_prominence is None:
        min_prominence = 0.1 * filtered_signal.std()
    peaks, props = find_peaks(filtered_signal, distance = distance, prominence = min_prominence)
    return peaks, props

def pulse_detection_rate(signal, fs, lowcut = 0.5, highcut = 3.7, order = 3, min_distance_s = 0.33, min_prominence=None, expected_hr_range=(40, 200)):
    # --- Bandpass filter ---
    signal_filtered = bandpass(signal, fs, lowcut, highcut, order)
    # --- Peak detection ---
    peaks, props = beat_detection(signal_filtered, fs, min_distance_s, min_prominence)

    n_beats = len(peaks)
    duration_s = len(signal) / fs

    # --- Define physiological range ---
    min_beats = expected_hr_range[0] * duration_s / 60.0
    max_beats = expected_hr_range[1] * duration_s / 60.0

    # --- Define range ---
    if n_beats < min_beats or n_beats > max_beats:
        sqi = 0.0
    else:
        sqi = 1.0

    return sqi, n_beats, peaks, signal_filtered

