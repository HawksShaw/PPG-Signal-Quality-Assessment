import os
import glob
import time
import requests
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

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"No such directory as {data_dir}")
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
            if sensor_location not in mat_file:
                continue

            sensor_data = mat_file[sensor_location]
            ppg_fs = float(sensor_data.ppg_g.fs)

            full_signals = {
                "ppg_ir" : sensor_data.ppg_ir.v,
                "ppg_r"  : sensor_data.ppg_r.v,
                "ppg_g"  : sensor_data.ppg_g.v,
                # "ecg"    : sensor_data.ecg.v,
                "acc_x"  : sensor_data.acc_x.v,
                "acc_y"  : sensor_data.acc_y.v,
                "acc_z"  : sensor_data.acc_z.v,
            }

            if preprocess:
                for char in ['ppg_ir', 'ppg_r', 'ppg_g']:
                    full_signals[char] = bandpass_filter(full_signals[char], ppg_fs)
                    
            num_samples = len(full_signals['ppg_ir'])
            window_samples = int(window_seconds*ppg_fs)
            step_size = int(window_samples*(1-overlap))

            for start in range(0, num_samples-window_samples+1, step_size):
                end = start+window_samples
                
                yield {
                    "metadata" : {
                        "subject_id"    : subject_id,
                        "sensor"        : sensor_location,
                        "sampling_rate" : ppg_fs,
                        "timestamp"     : start/ppg_fs
                    },
                    "ppg_signal" : {
                        "ir" : full_signals["ppg_ir"][start:end],
                        "r"  : full_signals["ppg_r"][start:end],
                        "g"  : full_signals["ppg_g"][start:end],
                        # "ecg_gt" : full_signals["ecg"][start:end]
                    },
                    "accel" : {
                        "x" : full_signals['acc_x'][start:end],
                        "y" : full_signals['acc_y'][start:end],
                        "z" : full_signals['acc_z'][start:end]
                    }
                }

DATA_DIR = "./data/raw/" 
API_URL = "http://127.0.0.1:8000/assess"

def run_feeder():
    print(f"Initializing stream from: {DATA_DIR}")
    
    stream = wildppg_stream(DATA_DIR)
    
    count = 0
    for window in stream:
        time_start = time.time()
        count += 1
        chosen_window = 3
        plot_window = True
        
        payload = {
            "subject_id": window['metadata']['subject_id'],
            "sampling_rate": float(window['metadata']['sampling_rate']),
            "ppg_ir": window['ppg_signal']['ir'].tolist(),
            "acc_x": window['accel']['x'].tolist(),
            "acc_y": window['accel']['y'].tolist(),
            "acc_z": window['accel']['z'].tolist()
        }

        # --- PRINT FOR POSTMAN VISUALIZATION ---
        # if count == chosen_window: 
        #     print(json.dumps(payload))
        #     plt.plot

        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"[Window {count}] Status: {result['status']} | Conf: {result['confidence']:.2f}, | Reason: {result['reasons']}")
            else:
                print(f"[Window {count}] Failed: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            print("ERROR: Could not connect to API. Is 'uvicorn' running?")
            break
        except Exception as e:
            print(f"Error processing window: {e}")
            break
        
        time_end = time.time()
        print(f"Elapsed stream time: {time_end-time_start:.4f} seconds")

        if count > 500:
            break
           
    print(response)
if __name__ == "__main__":
    run_feeder()