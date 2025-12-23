from prepare_wildppg import *
from find_peaks import *
import matplotlib.pyplot as plt
from window_estimation import signal_cutoff


window_signal, window_time = signal_cutoff(signal, 0, 10)
sqi, n_beats, peaks, filtered_signal = pulse_detection_rate(window_signal, fs=128)

print(f"Beat detectability SQI: {sqi}")
print(f"Number of detected beats: {n_beats}")
print(f"Estimated BPM: {n_beats*6}")

estimated_bpm = n_beats*6
sum_bpm = 0
for i in range(1280):
    sum_bpm += signal_bpm[i]

sum_bpm = sum_bpm/len(window_signal)

print(f"real bpm: {sum_bpm}")

# --- Plot visuals ---

t = np.arange(len(filtered_signal)) / fs
plt.plot(t, filtered_signal)

plt.scatter(t[peaks], filtered_signal[peaks], color = 'r', label = 'Detected Peaks')
plt.xlabel('Time (s)')
plt.title(f"Bandpass PPG, detected beats\nSQI: {sqi}, Beats: {n_beats}")
plt.legend()
plt.show()

