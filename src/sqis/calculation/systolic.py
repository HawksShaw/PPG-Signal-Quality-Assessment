from scipy.signal import find_peaks

def systolic_peaks(signal, fs, prominence=None, height=None):

    peaks, props = find_peaks(
        signal,
        distance=fs*0.34,
        prominence=prominence,
        height=height
    )

    return peaks, props