from example_load import *
from find_peaks import *
import matplotlib.pyplot as plt

sqi, n_beats, peaks, filtered_signal = pulse_detection_rate(ppg_signal_1, fs=128)

print(f"Beat detectability SQI: {sqi}")
print(f"Number of detected beats: {n_beats}")

# --- Plot visuals ---
t = np.arange(len(filtered_signal)) / fs
plt.plot(t, filtered_signal)

plt.scatter(t[peaks], filtered_signal[peaks], color = 'r', label = 'Detected Peaks')
plt.xlabel('Time (s)')
plt.title(f"Bandpass PPG, detected beats\nSQI: {sqi}, Beats: {n_beats}")
plt.legend()
plt.show()

