import os
import numpy as np
import requests
from scipy.io import loadmat


FILE_PATH = "data/raw/Sample_PPG_MAT_125Hz.mat"
API_URL = "http://127.0.0.1:8000/assess"
WINDOW_SEC = 8

def load_reference_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file at: {path}")

    # Load exactly as you described
    mat_data = loadmat(path, struct_as_record=False, squeeze_me=True)
    
    # Extract Fs and Signal
    fs = float(mat_data['Fs'])
    signal = mat_data['Data']
    
    # Safety Check: Ensure signal is 1D array
    if signal.ndim > 1:
        print(f"⚠️ Warning: Signal has shape {signal.shape}. Taking first column/row.")
        signal = signal.flatten()
        
    return signal, fs

def z_score_normalize(signal):
    """
    Scales signal so Mean=0 and Std=1. 
    Crucial for making the Peak Detector work across different datasets.
    """
    if np.std(signal) == 0:
        return signal # Avoid division by zero
    return (signal - np.mean(signal)) / np.std(signal)

def run_test():
    print(f"--- Loading Reference Data: {FILE_PATH} ---")
    
    try:
        raw_ppg, fs = load_reference_data(FILE_PATH)
        print(f"✅ Loaded. Fs: {fs} Hz | Length: {len(raw_ppg)} samples")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # 1. Normalize the Reference Signal
    # This solves the "Peak Threshold" issue if the signal is too small/large
    ppg_norm = z_score_normalize(raw_ppg)

    # 2. Create Dummy Accelerometer (Perfect Stillness)
    dummy_acc = np.zeros(len(ppg_norm)).tolist()

    # 3. Process Window by Window
    window_samples = int(WINDOW_SEC * fs)
    total_len = len(ppg_norm)
    
    print(f"--- Starting Stream ({total_len // window_samples} windows) ---")

    for start in range(0, total_len - window_samples, window_samples):
        end = start + window_samples
        
        # Slice data
        chunk_ppg = ppg_norm[start:end]
        chunk_acc = dummy_acc[start:end]
        
        payload = {
            "subject_id": "REFERENCE_GOLD_STD",
            "sampling_rate": fs,
            "ppg_ir": chunk_ppg.tolist(),
            # Since we don't have real acc, we send zeros (Subject is perfectly still)
            "acc_x": chunk_acc,
            "acc_y": chunk_acc,
            "acc_z": chunk_acc
        }
        
        try:
            # Send to Docker API
            response = requests.post(API_URL, json=payload)
            result = response.json()
            
            # --- VALIDATION REPORT ---
            print(f"Window {start//window_samples}: {result['status'].upper()} | Conf: {result['confidence']:.2f}")
            
            if result['status'] != 'accept':
                print(f"   Reason: {result['reasons']}")
                if 'metrics' in result:
                    print(f"   Metrics: {result['metrics']}")

        except Exception as e:
            print(f"Connection Error: {e}")
            break

if __name__ == "__main__":
    run_test()