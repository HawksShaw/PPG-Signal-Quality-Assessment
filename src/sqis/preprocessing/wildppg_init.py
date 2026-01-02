import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

# --- THIS ONLY WORKS FOR THE WILDPPG DATASET ---

def z_score_normalize(signal):
    if np.std(signal) <= 1e-6:
        return signal
    else:
        return (signal-np.mean(signal))/np.std(signal)

def bandpass_filter(signal, fs, lowcut=0.5, highcut=3.7, order=3):
    nyquist_freq = 0.5*fs
    lowpass = lowcut/nyquist_freq
    highpass = highcut/nyquist_freq
    sos = butter(order, [lowpass, highpass], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def wildppg_stream(data_dir, sensors=['wrist'], window_seconds=8, overlap=0.6, preprocess=True):
    #Feed the entire directory and sort for reproducible executions
    files_dir = os.path.join(data_dir, 'WildPPG_Part_*.mat')
    files = sorted(glob.glob(files_dir))

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"No such directory as {data_dir}")
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}")

    print(f"Found {len(files)} files. Starting data stream.")

    for file_path in files:
        subject_id = os.path.basename(file_path).split('.')[0] #Skip the .mat part for the subject's ID
        try:
            mat_file = loadmat(file_path, squeeze_me=True, struct_as_record=False) #Load individual files
        except Exception as ex:
            print(f"Skipping subject {subject_id}: {ex}")
            continue

        for sensor_location in sensors:
            if sensor_location not in mat_file:
                print(f"No attribute '{sensor_location}' for subject {subject_id}")
                continue

            sensor_data = mat_file[sensor_location]
            ppg_fs = sensor_data.ppg_g.fs

            full_signals = {
                "ppg_ir" : sensor_data.ppg_ir.v,
                "ppg_r"  : sensor_data.ppg_r.v,
                "ppg_g"  : sensor_data.ppg_g.v,
                "acc_x"  : sensor_data.acc_x.v,
                "acc_y"  : sensor_data.acc_y.v,
                "acc_z"  : sensor_data.acc_z.v,
                "ecg"    : None
            }

            if 'ecg' in mat_file and hasattr(mat_file["ecg"].v):
                full_signals['ecg'] = mat_file['ecg'].v

            if preprocess:
                for char in ['ppg_ir', 'ppg_r', 'ppg_g']:
                    full_signals[char] = bandpass_filter(full_signals[char], ppg_fs)
                    
            num_samples = len(full_signals['ppg_ir'])
            window_samples = int(window_seconds*ppg_fs)
            step_size = int(window_samples*(1-overlap))

            for start in range(0, num_samples-window_samples+1, step_size):
                end = start+window_samples
                
                norm_ir = z_score_normalize(full_signals["ppg_ir"][start:end])
                norm_r  = z_score_normalize(full_signals["ppg_r"][start:end])
                norm_g  = z_score_normalize(full_signals["ppg_g"][start:end])


                #Create a dictionary for easier metadata access
                window_signal = {
                    "metadata" : {
                        "subject_id"    : subject_id,
                        "sensor"        : sensor_location,
                        "sampling_rate" : ppg_fs,
                        "timestamp"     : start/ppg_fs
                    },
                    "ppg_signal" : {
                        "ir" : norm_ir,
                        "r"  : norm_r,
                        "g"  : norm_g
                    },
                    "accel" : {
                        "x" : full_signals['acc_x'][start:end],
                        "y" : full_signals['acc_y'][start:end],
                        "z" : full_signals['acc_z'][start:end]
                    }
                }

                if full_signals["ecg"] is not None:
                    window_signal["ecg"] = full_signals["ecg"][start:end]

                yield window_signal


        