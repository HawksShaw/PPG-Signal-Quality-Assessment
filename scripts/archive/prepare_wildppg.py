from load_wildppg import *
import numpy as np

# --- Each variable below is a list of length 16 - change index from 0 to 15 to check each subject ---

# Subject no. 1

subject_number = 0

data_ppg_head   = data['data_ppg_head'][subject_number]        # PPG signals (head-worn sensor)
data_ppg_chest  = data['data_ppg_chest'][subject_number]       # PPG signals (wrist-worn sensor)
data_ppg_wrist  = data['data_ppg_wrist'][subject_number]       # PPG signals (chest-worn sensor)
data_ppg_ankle  = data['data_ppg_ankle'][subject_number]       # PPG signals (ankle-worn sensor)

data_bpm_values = data['data_bpm_values'][subject_number]      # BPM ground-truth
data_altitude   = data['data_altitude_values'][subject_number] # Altitude readings

data_temp_head  = data['data_temp_head'][subject_number]       # Head temperature readings
data_temp_chest = data['data_temp_chest'][subject_number]      # Chest temperature readings
data_temp_wrist = data['data_temp_wrist'][subject_number]      # Wrist temperature readings
data_temp_ankle = data['data_temp_ankle'][subject_number]      # Ankle temperature readings

data_imu_head  = data['data_imu_head'][subject_number]         # IMU head sensor
data_imu_chest = data['data_imu_chest'][subject_number]        # IMU chest sensor
data_imu_wrist = data['data_imu_wrist'][subject_number]        # IMU wrist sensor
data_imu_ankle = data['data_imu_ankle'][subject_number]        # IMU ankle sensor


# --- Accessing a subject's signal --- 

bpm        = data_bpm_values[0]
ppg_head   = data_ppg_head[0]
ppg_wrist  = data_ppg_wrist[0]
ppg_chest  = data_ppg_chest[0]
ppg_ankle  = data_ppg_ankle[0]
bpm        = data_bpm_values[0]
altitude   = data_altitude[0]
temp       = data_temp_wrist[0]
imu_chest  = data_imu_chest[0]

# ---- Show summary ----

print(f'Loaded WildPPG.mat: {len(data_ppg_ankle)} subjects')
print('Variables loaded:')
for key in data:
    if not key.startswith('__'):
        print(f' - {key}: shape {data[key].shape}')

# --- define sampling rate, time domain and number of samples ---
# the signal was flattened to access all channels in a single-dimension axis

fs = 128
t = 1/fs
signal_bpm = bpm
signal = ppg_head.flatten()
n_samples_bpm = len(bpm)
n_samples = len(signal)
print(f"time of the signal: {n_samples*t/60} minutes")
time = np.linspace(0, (n_samples-1)*t, n_samples)
time_bpm = np.linspace(0, (n_samples_bpm-1)*t, n_samples_bpm)