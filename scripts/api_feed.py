import pandas as pd
import requests
import time

# 1. Load your actual data
CSV_PATH = "your_data_file.csv"  # Update this to your filename
df = pd.read_csv(CSV_PATH)

# 2. Define the API settings
API_URL = "http://127.0.0.1:8000/assess"
FS = 25  # Your sampling rate
WINDOW_SIZE = int(FS * 8)  # 8-second windows (200 samples)

def feed_data():
    # Iterate through the CSV in 8-second chunks
    for start in range(0, len(df), WINDOW_SIZE):
        end = start + WINDOW_SIZE
        chunk = df.iloc[start:end]

        # Ensure we have a full window to avoid the "Window too short" error
        if len(chunk) < WINDOW_SIZE:
            break

        # 3. Format the data into the JSON structure your API expects
        payload = {
            "subject_id": "subject_01",
            "sampling_rate": float(FS),
            "ppg_ir": chunk['ppg'].tolist(),  # Replace 'ppg' with your actual column name
            "acc_x": chunk['acc_x'].tolist(),
            "acc_y": chunk['acc_y'].tolist(),
            "acc_z": chunk['acc_z'].tolist()
        }

        # 4. POST the data to the API
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"Window {start//WINDOW_SIZE}: Status: {result['status']}, Confidence: {result['confidence']:.2f}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Connection failed: {e}")
            break

        # Optional: slow down the feeder to simulate real-time processing
        time.sleep(1)

if __name__ == "__main__":
    feed_data()