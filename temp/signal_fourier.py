from patient_1load import *
from scipy.fft import fft, ifft, fftfreq
import numpy as np

# Forward FFT
signal_fourier = np.fft.fft(filtered_signal)
signal_magnitude = np.abs(signal_fourier) / windowed_samples

# Frequency domain for plotting (only positive frequencies)
# No modification of the signal_fourier needed for iFFT!
frequency_domain = np.fft.fftfreq(windowed_samples, d=1/fs)
positive_freqs = frequency_domain[:windowed_samples//2]
positive_mag = 2 * signal_magnitude[:windowed_samples//2]
positive_mag[0] = positive_mag[0]/2

# ...plotting can use positive_freqs/positive_mag...

# Inverse FFT (full spectrum)
inverted_fourier = np.fft.ifft(signal_fourier)
# This will be the same length, and for a real input signal, the result is (very close to) real again.

# If you want the real part only:
inverted_fourier = inverted_fourier.real

print(len(positive_mag))
print("Inverse FFT shape:", inverted_fourier.shape)