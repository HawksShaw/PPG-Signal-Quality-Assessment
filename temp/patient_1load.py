from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from window import signal_cutoff
from butterworth import bandpass

file = loadmat('data\\WildPPG_Part_an0.mat', struct_as_record = False, squeeze_me = True)

ppg_wrist_g = file['wrist'].ppg_g
ecg_bpm = file['sternum'].ecg

fs = ppg_wrist_g.fs
descr = ppg_wrist_g.descr
signal = ppg_wrist_g.v

bpm_descr = ecg_bpm.descr
signal_bpm = ecg_bpm.v

print(f"Information for green PPG signal:\nSampling rate: {fs},\nDescription: {descr},\nSignal shape: {signal.shape}")
print(f"Information for ECG signal:\nSampling rate: {fs},\nDescription: {bpm_descr},\nSignal shape: {signal_bpm.shape}")

t = 1/fs
n = len(signal)
time = np.linspace(0, (n-1)*t, n)
# print(f"recording time: {t*n/3600} hours")

windowed_signal, windowed_time = signal_cutoff(signal, 0, 20, fs)
filtered_signal = bandpass(windowed_signal, fs, lowcut=0.5, highcut=3.75, order=3)

windowed_samples = len(filtered_signal)

# for i in range(0,len(filtered_signal)):
#     if filtered_signal[i] < 0:
#         filtered_signal[i] = 0

# plt.figure()
# plt.plot(time, signal)
# plt.xlabel('time [s]')
# plt.ylabel('PPG [au]')
# plt.show()
