from scipy.signal import butter, filtfilt

def bandpass(signal, fs, lowcut=0.7, highcut=3.75, order=3):
    nyquist = 0.5*fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

    