import os
import urllib.request
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# ---- Set data file name and Hugging Face URL ----

datafile = 'WildPPG.mat'
hf_url = 'https://huggingface.co/datasets/eth-siplab/WildPPG/resolve/main/WildPPG.mat'

# ---- Check if file exists; if not, download ----

if not os.path.isfile(datafile):
    print(f'Data file {datafile} not found.')
    print('Downloading from Hugging Face...')
    urllib.request.urlretrieve(hf_url, datafile)
    print('Download complete!')
else:
    print(f'Found data file: {datafile}')

# ---- Load the data ----

data = scipy.io.loadmat(datafile)

# Each variable below is a list of length 16 (one per subject)

data_ppg_head   = data['data_ppg_head'][0]        # PPG signals (head-worn sensor)
data_ppg_wrist  = data['data_ppg_wrist'][0]       # PPG signals (wrist-worn sensor)
data_ppg_chest  = data['data_ppg_chest'][0]       # PPG signals (chest-worn sensor)
data_ppg_ankle  = data['data_ppg_ankle'][0]       # PPG signals (ankle-worn sensor)
data_bpm_values = data['data_bpm_values'][0]      # Cleaned heart rate values
data_altitude   = data['data_altitude_values'][0] # Altitude readings
data_temp_wrist = data['data_temp_wrist'][0]      # Wrist temperature readings
data_imu_chest  = data['data_imu_chest'][0]       # IMU (chest sensor)

# ---- Example: Accessing a Subject ----

# print(len(data_ppg_head))
subject_idx = 0 
ppg_head_1   = data_ppg_head[subject_idx]
ppg_wrist_1  = data_ppg_wrist[subject_idx]
ppg_chest_1  = data_ppg_chest[subject_idx]
ppg_ankle_1  = data_ppg_ankle[subject_idx]
bpm_1        = data_bpm_values[subject_idx]
altitude_1   = data_altitude[subject_idx]
temp_1       = data_temp_wrist[subject_idx]
imu_chest_1  = data_imu_chest[subject_idx]

# ---- Show summary ----
print(f'Loaded WildPPG.mat: {len(data_ppg_ankle)} subjects')
print('Variables loaded:')
for key in data:
    if not key.startswith('__'):
        print(f' - {key}: shape {data[key].shape}')


# ---- Plot raw signal ----
fs = 128
t = 1/fs
ppg_signal_1 = []
for channel in (0, 199):
    ppg_signal_1.append(ppg_head_1[:, channel])

N = len(ppg_signal_1)
time_domain_1 = np.linspace(0, (N-1)*t, N)

plt.figure()
plt.plot(time_domain_1, ppg_signal_1)
plt.xlabel("time [s]")
plt.ylabel("Signal amplitude [V]")
plt.title("1st patient, 1st channel")

plt.show()