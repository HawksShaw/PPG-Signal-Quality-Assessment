import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION: Map Labels to Your Filenames ---
# Edit the paths below to match your actual files
FILES = {
    "Ideal (Lab)": "ideal.csv", 
    "Sternum":     "heartpy_results_sternum.csv",
    "Forehead":    "heartpy_results_head.csv",
    "Wrist":       "heartpy_results_wrist.csv",
    "Ankle":       "heartpy_results_ankle.csv"
}

# Define colors for consistent visualization
COLORS = {
    "Ideal (Lab)": "black",     # Baseline
    "Sternum":     "#2ca02c",   # Green (Good)
    "Forehead":    "#1f77b4",   # Blue
    "Wrist":       "#ff7f0e",   # Orange
    "Ankle":       "#d62728"    # Red (Worst)
}

OUTPUT_FILENAME = "Figure_Sensor_Density_Comparison.png"

def plot_combined_density():
    plt.figure(figsize=(12, 7))
    
    data_found = False

    # Loop through the dictionary and plot each file
    for label, filepath in FILES.items():
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: File not found: {filepath} (Skipping {label})")
            continue
            
        try:
            df = pd.read_csv(filepath)
            
            # Check if 'confidence' column exists
            if 'confidence' not in df.columns:
                print(f"⚠️ Warning: Column 'confidence' missing in {filepath} (Skipping)")
                continue

            # Special styling for the Ideal/Reference signal
            if label == "Ideal (Lab)":
                linestyle = "--"
                linewidth = 2.5
                alpha = 0.1 # Very transparent fill for ideal
            else:
                linestyle = "-"
                linewidth = 2
                alpha = 0.2 # Standard transparency
            
            # Plot the Density (KDE)
            sns.kdeplot(
                data=df, 
                x="confidence", 
                fill=True, 
                color=COLORS.get(label, "gray"), 
                alpha=alpha, 
                linewidth=linewidth, 
                linestyle=linestyle,
                label=label, 
                clip=(-0.2, 1.0) # Force graph to stay within valid probability range
            )
            
            print(f"✅ Plotted: {label} ({len(df)} windows)")
            data_found = True
            
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")

    if not data_found:
        print("No valid data files found. Check your paths in the FILES dictionary.")
        return

    # --- Formatting the Graph ---
    plt.title("Impact of Sensor Location on Signal Confidence", fontsize=14, fontweight='bold')
    plt.xlabel("Confidence Score (Quality Probability)", fontsize=12)
    plt.ylabel("Density of Windows", fontsize=12)
    plt.xlim(-0.2, 1.15)
    
    # Add a vertical line for the Decision Threshold
    plt.axvline(0.5, color='gray', linestyle=':', label="Decision Threshold (0.5)")

    plt.legend(title="Sensor Location", loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"\n🎉 Graph saved to {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    plot_combined_density()