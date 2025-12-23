import os
import urllib.request
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# ---- Set data file name and Hugging Face URL ----
datafile = 'assets/datasets/WildPPG.mat'
hf_url = 'https://huggingface.co/datasets/eth-siplab/WildPPG/resolve/main/WildPPG.mat'

# ---- Check if file exists; if not, download ----
if not os.path.isfile(datafile):
    print(f'Data file {datafile} not found.')
    #print('Downloading from Hugging Face...')
    #urllib.request.urlretrieve(hf_url, datafile)
    #print('Download complete!')
else:
    print(f'Found data file: {datafile}')

# ---- Load the data ----
data = scipy.io.loadmat(datafile)

# Each variable below is a list of length 16 (one per subject)
data_ppg_head   = data['data_ppg_head']       # PPG signals (head-worn sensor)
data_ppg_wrist  = data['data_ppg_wrist']       # PPG signals (wrist-worn sensor)
data_bpm_values = data['data_bpm_values']      # Cleaned heart rate values
data_altitude   = data['data_altitude_values'] # Altitude readings
data_temp_wrist = data['data_temp_wrist']      # Wrist temperature readings
data_imu_chest  = data['data_imu_chest']       # IMU (chest sensor)

# ---- Example: Accessing Subject #1 ----
subject_idx = 0  # Python is 0-based!
ppg_head_1   = data_ppg_head[subject_idx, 0]
ppg_wrist_1  = data_ppg_wrist[subject_idx, 0]
bpm_1        = data_bpm_values[subject_idx, 0]
altitude_1   = data_altitude[subject_idx, 0]
temp_1       = data_temp_wrist[subject_idx, 0]
imu_chest_1  = data_imu_chest[subject_idx, 0]

# ---- Show summary ----
print(f'Loaded WildPPG.mat: {len(data_ppg_head)} subjects')
print('Variables loaded:')
for key in data:
    if not key.startswith('__'):
        print(f' - {key}: shape {data[key].shape}')

channel_idx = 0
ppg = ppg_head_1[:, channel_idx]
fs = 128
t = 1/fs
N = len(ppg)
time_domain = np.linspace(0, (N-1)*t, N)

plt.figure()
plt.plot(time_domain, ppg)
plt.show()