import os
import numpy as np
import requests
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

FILE_PATH = "data/raw/Sample_PPG_MAT_125Hz.mat"
API_URL = "http://127.0.0.1:8000/assess"
WINDOW_SEC = 8

def bandpass_filter(signal, fs, lowcut=0.5, highcut=3.7, order=3):
    nyquist_freq = 0.5 * fs
    lowpass = lowcut / nyquist_freq
    highpass = highcut / nyquist_freq
    sos = butter(order, [lowpass, highpass], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def load_reference_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file at: {path}")

    mat_data = loadmat(path, struct_as_record=False, squeeze_me=True)
    fs = float(mat_data['Fs'])
    signal = mat_data['Data']
    
    if signal.ndim > 1:
        print(f"Warning: Signal has shape {signal.shape}. Taking first column/row.")
        signal = signal.flatten()
        
    return signal, fs

def run_test():
    print(f"--- Loading Reference Data: {FILE_PATH} ---")
    
    try:
        raw_ppg, fs = load_reference_data(FILE_PATH)
        print(f"Loaded. Fs: {fs} Hz | Length: {len(raw_ppg)} samples")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    ppg_filtered = bandpass_filter(raw_ppg, fs)

    dummy_acc = np.zeros(len(ppg_filtered)).tolist()
    window_samples = int(WINDOW_SEC * fs)
    total_len = len(ppg_filtered)
    
    print(f"--- Starting Stream ({total_len // window_samples} windows) ---")

    for start in range(0, total_len - window_samples, window_samples):
        end = start + window_samples
        chunk_ppg = ppg_filtered[start:end]
        chunk_acc = dummy_acc[start:end]
        
        payload = {
            "subject_id": "REFERENCE_GOLD_STD",
            "sampling_rate": fs,
            "ppg_ir": chunk_ppg.tolist(),
            "acc_x": chunk_acc,
            "acc_y": chunk_acc,
            "acc_z": chunk_acc
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()
            status_icon = "✅" if result['status'] in ['GOOD', 'ACCEPTABLE'] else "❌"
            print(f"Window {start//window_samples}: {status_icon} {result['status']} | Conf: {result['confidence']:.2f}")
            
            if result['status'] == 'BAD':
                print(f"   Reason: {result['reasons']}")

        except Exception as e:
            print(f"Connection Error: {e}")
            break

if __name__ == "__main__":
    run_test()