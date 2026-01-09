import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_baseline = "scipy_results.csv"
file_heartpy  = "heartpy_results.csv"
file_elgendi  = "elgendi_results.csv"

def analyze_performance(file_path, name):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise FileNotFoundError(f"File not found: {e}")
        return None

    df['hr_error'] = pd.to_numeric(df['hr_error'], errors='coerce')

    total_windows = len(df)
    valid_df = df[
        (df['status'].isin(['GOOD', 'ACCEPTABLE'])) &
        (df['hr_error'].notna())
    ].copy()

    retention_rate = (len(valid_df)/total_windows)*100 if total_windows > 0 else 0
    if len(valid_df) == 0:
        print("No valid comparisons found (check if ECG data was processed)")
        return None

    mae = valid_df['hr_error'].abs().mean()
    rmse = np.sqrt((valid_df['hr_error']**2).mean())

    print(f"Processed windows: {total_windows}")
    print(f"Retention rate: {retention_rate:.1f}%")
    print(f"Mean Absolute Error: {mae:.2f} BPM")
    print(f"RMSE: {rmse:.2f} BPM")
    print("-"*30)

    return {
        "name" : name,
        "df"   : valid_df,
        "mae"  : mae,
        "retention" : retention_rate
    }

def plot_comparison(plot_a, plot_b, plot_c):
    if not plot_a or not plot_b or not plot_c: return

    plt.figure(figsize=(10,6))
    data_a = plot_a['df']['hr_error'].abs()
    data_b = plot_b['df']['hr_error'].abs()
    data_c = plot_c['df']['hr_error'].abs()

    plt.boxplot([data_a, data_b, data_c], labels=[plot_c['name'], plot_b['name'], plot_a['name']], showfliers=False)
    plt.title("Accuracy comparison for: Elgendi method, heartPy toolbox, SciPy's find_peaks detection")
    plt.ylabel("Absolute Error (BPM)")
    plt.grid(True)

    plt.text(1, data_a.median(), f"MAE: {plot_a['mae']:.2f}", ha='center', va='bottom', fontweight='bold')
    plt.text(2, data_b.median(), f"MAE: {plot_b['mae']:.2f}", ha='center', va='bottom', fontweight='bold')
    plt.text(3, data_c.median(), f"MAE: {plot_c['mae']:.2f}", ha='center', va='bottom', fontweight='bold')

    plt.show()

if __name__ == '__main__':
    plot_baseline = analyze_performance(file_baseline, "Scipy's find_peaks")
    plot_heartpy = analyze_performance(file_heartpy, "HeartPy")
    plot_elgendi = analyze_performance(file_elgendi, "Elgendi")

    plot_comparison(plot_baseline, plot_heartpy, plot_elgendi)
