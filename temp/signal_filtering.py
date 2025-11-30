from wildppg_load import *
from scipy import signal

window_start = 5
time = 5
samples = fs*time 
cutoff_time = np.linspace(0, (N-1)*t, samples)
window = ppg_signal_1[:samples]
time_window = np.linspace(window_start, window_start + time, samples)

low_cutoff = 10
high_cutoff = 0.5
filter_order = 3

b, a = signal.butter(filter_order, [high_cutoff, low_cutoff], fs = fs, btype='band')
window_filtered = signal.filtfilt(b, a, window)

# plt.subplot(2,1,1)
# plt.plot(time_window, window)
# plt.title("No filter")

# plt.subplot(2,1,2)
# plt.plot(time_window, window_filtered)
# plt.title("Butterworth 0.5-3.7Hz")

# plt.show()

