from prepare_wildppg import *
import matplotlib.pyplot as plt
import numpy as np

# --- define sampling rate, time domain and number of samples ---
# the signal was flattened to access all channels in a single-dimension axis

fs = 128
t = 1/fs
signal = ppg_head.flatten()
N = len(signal)
time = np.linspace(0, (N-1)*t, N)

plt.figure()
plt.plot(time, signal)
plt.show()