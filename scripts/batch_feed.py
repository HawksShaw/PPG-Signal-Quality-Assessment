import os
import glob
import time
import requests
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

def bandpass_filter(signal, fs, lowcut=0.5, highcut=3.7, order=3):
    nyquist_freq = 0.5*fs
    lowpass = lowcut/nyquist_freq
    highpass = highcut/nyquist_freq
    sos = butter(order, [lowpass, highpass], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def wildppg_stream(data_dir, sensors=['sternum'], window_seconds=8, overlap=0.6, preprocess=True):
    files_dir = os.path.join(data_dir, 'WildPPG_Part_*.mat')
    files = sorted(glob.glob(files_dir))
    if sensors in files:
        print(sensors)
    
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}.")
        return

    for file_path in files:
        subject_id = os.path.basename(file_path).split('.')[0]
        try:
            mat_file = loadmat(file_path, squeeze_me=True, struct_as_record=False)
            raw_ecg = mat_file['sternum'].ecg.v
        except Exception as ex:
            print(f"Skipping subject {subject_id}: {ex}")
            continue

        for sensor_location in sensors:
            if sensor_location not in mat_file: 
                continue

            sensor_data = mat_file[sensor_location]
            ppg_fs = float(sensor_data.ppg_g.fs)

            full_signals = {
                "ppg_ir": sensor_data.ppg_ir.v,
                "ppg_r": sensor_data.ppg_r.v,
                "ppg_g": sensor_data.ppg_g.v,
                "ecg"  : raw_ecg,
                "acc_x": sensor_data.acc_x.v,
                "acc_y": sensor_data.acc_y.v,
                "acc_z": sensor_data.acc_z.v,
            }

            if preprocess:
                for char in ['ppg_ir', 'ppg_r', 'ppg_g']:
                    full_signals[char] = bandpass_filter(full_signals[char], ppg_fs)
            
            num_samples = len(full_signals['ppg_ir'])
            window_samples = int(window_seconds*ppg_fs)
            step_size = int(window_samples*(1-overlap))

            for start in range(0, num_samples-window_samples+1, step_size):
                end = start+window_samples

                ecg_slice = []
                if len(full_signals["ecg"]) > end:
                    ecg_slice = full_signals["ecg"][start:end]

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
                        "ecg_gt" : ecg_slice
                    },
                    "accel" : {
                        "x" : full_signals['acc_x'][start:end],
                        "y" : full_signals['acc_y'][start:end],
                        "z" : full_signals['acc_z'][start:end]
                    }
                }

DATA_DIR = "./data/raw/" 
API_URL = "http://127.0.0.1:8000/assess/batch"
BATCH_SIZE = 5
OUTPUT_FILENAME = "heartpy_results.csv"

def run_batch_feeder():
    print(f"Initializing BATCH stream from: {DATA_DIR}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    stream = wildppg_stream(DATA_DIR)
    
    buffer = []
    batch_count = 0
    all_results = []

    for window in stream:
        # time_start = time.time()
        batch_count += 1
        flattenned_window = {
            "subject_id": window['metadata']['subject_id'],
            "sampling_rate": float(window['metadata']['sampling_rate']),
            "ppg_ir": window['ppg_signal']['ir'].tolist(),
            "ecg_gt": window['ppg_signal']['ecg_gt'].tolist(),
            "acc_x": window['accel']['x'].tolist(),
            "acc_y": window['accel']['y'].tolist(),
            "acc_z": window['accel']['z'].tolist()
        }
        buffer.append(flattenned_window)
        
        if len(buffer) >= BATCH_SIZE:
            batch_results = send_batch(buffer)
            if batch_results:
                all_results.extend(batch_results)
            buffer = [] 
            # Optional sleep
            #time.sleep(0.5)
        # time_end = time.time()
        # print(f"Elapsed batch computation time: {time_end-time_start:.4f} seconds")
        # if batch_count >= 500:
        #     break

    if buffer:
        batch_results = send_batch(buffer)
        if batch_results:
            all_results.extend(batch_results)

    if all_results:
        flat_data = []
        for res in all_results:
            row = {
                "subject": res['metadata'].get('subject_id', 'unknown'),
                "status": res['status'],
                "confidence": res['confidence'],
                "hr_error": res['metadata'].get('hr_error', 'Unknown'),
                "gt_label": res['metadata'].get('gt_label', 'Unknown'),
                "skewness": res['metrics'].get('skewness'),
                "snr": res['metrics'].get('spectral_snr'),
                "motion_detected": res['metadata'].get('motion_detected')
            }
            flat_data.append(row)

        df = pd.DataFrame(flat_data)
        df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"Saved results to: {OUTPUT_FILENAME}")
    else:
        print("No available results to save")

def send_batch(window_list):
    count = 0
    try:
        response = requests.post(API_URL, json=window_list)
        
        if response.status_code == 200:
            data = response.json()
            count = count + data.get("processed_count", 0)
            print(f"Batch Success! Processed {count} windows.")
            return data.get("results", [])
        else:
            print(f"Batch Failed: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    run_batch_feeder()