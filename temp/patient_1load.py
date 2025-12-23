from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from window import signal_cutoff
from butterworth import bandpass

file = loadmat('data\\WildPPG_Part_an0.mat', struct_as_record = False, squeeze_me = True)

ppg_wrist_g = file['wrist'].ppg_g
ppg_wrist_r = file['wrist'].ppg_r
ppg_wrist_ir = file['wrist'].ppg_ir
ecg_bpm = file['sternum'].ecg

fs = ppg_wrist_g.fs
descr_wrist_g = ppg_wrist_g.descr
descr_wrist_r = ppg_wrist_r.descr
descr_wrist_ir = ppg_wrist_ir.descr

signal_wrist_g = ppg_wrist_g.v
signal_wrist_r = ppg_wrist_r.v
signal_wrist_ir = ppg_wrist_ir.v

signal_wrist_total = signal_wrist_g + signal_wrist_r + signal_wrist_ir

bpm_descr = ecg_bpm.descr
signal_bpm = ecg_bpm.v

print(f"Information for green PPG signal:\nSampling rate: {fs},\nDescription: {descr_wrist_g},\nSignal shape: {signal_wrist_g.shape}")
print(f"Information for infrared PPG signal:\nSampling rate: {fs},\nDescription: {descr_wrist_ir},\nSignal shape: {signal_wrist_ir.shape}")
print(f"Information for red PPG signal:\nSampling rate: {fs},\nDescription: {descr_wrist_r},\nSignal shape: {signal_wrist_r.shape}")
print(f"Information for ECG signal:\nSampling rate: {fs},\nDescription: {bpm_descr},\nSignal shape: {signal_bpm.shape}")

t = 1/fs
n = len(signal_wrist_g)
time = np.linspace(0, (n-1)*t, n)
# print(f"recording time: {t*n/3600} hours")

start_time = 40
length = 30

window_wrist_g, window_time = signal_cutoff(signal_wrist_g, start_time, length, fs)
window_wrist_ir, window_time = signal_cutoff(signal_wrist_ir, start_time, length, fs)
window_wrist_r, window_time = signal_cutoff(signal_wrist_r, start_time, length, fs)
window_wrist_total, widnow_time = signal_cutoff(signal_wrist_total, start_time, length, fs)
window_bpm, window_time = signal_cutoff(signal_bpm, start_time, length, fs)

filter_wrist_g = bandpass(window_wrist_g, fs, lowcut=0.5, highcut=8.0, order=3)
filter_wrist_ir = bandpass(window_wrist_ir, fs, lowcut=0.5, highcut=8.0, order=3)
filter_wrist_r = bandpass(window_wrist_r, fs, lowcut=0.5, highcut=8.0, order=3)
filter_wrist_total = bandpass(window_wrist_total, fs, lowcut=0.5, highcut=8.0, order=3)
total_after_filter = filter_wrist_g + filter_wrist_ir + filter_wrist_r

print(total_after_filter.shape)

window_samples = len(filter_wrist_g)

plt.figure()
plt.subplot(2,2,1)
plt.plot(window_time, filter_wrist_g)
plt.xlabel('time [s]')
plt.ylabel('PPG [au]')
plt.title('Green PPG for wrist')

plt.subplot(2,2,2)
plt.plot(window_time, filter_wrist_ir)
plt.xlabel('time [s]')
plt.ylabel('PPG [au]')
plt.title('Infrared PPG for wrist')

plt.subplot(2,2,3)
plt.plot(window_time, filter_wrist_r)
plt.xlabel('time [s]')
plt.ylabel('PPG [au]')
plt.title('Red PPG for wrist')

plt.subplot(2,2,4)
plt.plot(window_time, total_after_filter)
plt.xlabel('time [s]')
plt.ylabel('ECG val [V]')
plt.title('Sum PPG for wrist')

plt.show()
