import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

def bandpass_filter(signal, fs, lowcut=0.5, highcut=3.7, order=3):
    nyquist_freq = 0.5*fs
    lowpass = lowcut/nyquist_freq
    highpass = highcut/nyquist_freq
    sos = butter(order, [lowpass, highpass], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def wildppg_stream(data_dir, sensors=['wrist'], window_seconds=8, overlap=0.6, preprocess=True):
    files_dir = os.path.join(data_dir, 'WildPPG_Part_*.mat')
    files = sorted(glob.glob(files_dir))

    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}")

    print(f"Found {len(files)} files. Starting data stream.")

    for file_path in files:
        subject_id = os.path.basename(file_path).split('.')[0]
        try:
            mat_file = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        except Exception as ex:
            print(f"Skipping subject {subject_id}: {ex}")
            continue

        for sensor_location in sensors:
            if not hasattr(mat_file, sensor_location):
                print(f"No attribute {sensor_location} for subject {subject_id}")
                continue
            sensor_data = getattr(mat_file, sensor_location)
            ppg_fs = sensor_data.ppg_g.fs
            ecg_fs = sensor_data.ecg.fs
            acc_fs = sensor_data.acc_x.fs

            full_signals = {
                "ppg_ir" : sensor_data.ppg_ir.v,
                "ppg_r"  : sensor_data.ppg_r.v,
                "ppg_g"  : sensor_data.ppg_g.v,
                "acc_x"  : sensor_data.acc_x.v,
                "acc_y"  : sensor_data.acc_y.v,
                "acc_z"  : sensor_data.acc_z.v,
                "ecg"    : None
            }

            if hasattr(mat.ecg, 'v'):
                full_signals['ecg'] = sensor_data.ecg.v

            if preprocess:
                for char in ['ppg_ir', 'ppg_r', 'ppg_g']:
                    full_signals[char] = bandpass_filter(full_signals[char], fs)
                    
            num_samples = len(full_signals['ppg_ir'])

            
            


        