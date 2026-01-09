import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
WILD_DATA_PATH = "scipy_results.csv"
IDEAL_DATA_PATH = "ideal.csv"
OUTPUT_FILENAME = "Figure_4_1_Environmental_Sensitivity.png"

def load_data():

    df_wild = pd.DataFrame()
    df_ideal = pd.DataFrame()
    
    if os.path.exists(WILD_DATA_PATH):
        df_wild = pd.read_csv(WILD_DATA_PATH)
    else:
        print(f"Warning: {WILD_DATA_PATH} not found.")

    if os.path.exists(IDEAL_DATA_PATH):
        df_ideal = pd.read_csv(IDEAL_DATA_PATH)
    else:
        print(f"Warning: {IDEAL_DATA_PATH} not found.")
        
    return df_wild, df_ideal

def generate_combined_figure(df_wild, df_ideal):
    if df_wild.empty and df_ideal.empty:
        print("No data available to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    ax0 = axes[0]
    
    if not df_ideal.empty:
        sns.kdeplot(
            data=df_ideal, x="confidence", fill=True, 
            color="#1f77b4", alpha=0.6, linewidth=2,
            label="Ideal Lab Reference", clip=(0, 1.01), ax=ax0
        )
        # Add mean line for ideal
        ax0.axvline(df_ideal['confidence'].mean(), color='#1f77b4', linestyle=':', linewidth=2)

    if not df_wild.empty:
        sns.kdeplot(
            data=df_wild, x="confidence", fill=True, 
            color="#ff7f0e", alpha=0.4, linewidth=2,
            label="Ambulatory (Wild Aggregate)", clip=(0, 1.01), ax=ax0
        )

    ax0.set_title("A. Macro-level Environment Dependency\n(Ideal vs. Real-World)", fontsize=12, fontweight='bold')
    ax0.set_ylabel("Probability Density", fontsize=11)
    ax0.set_xlabel("Confidence Score", fontsize=11)
    ax0.legend(loc='upper left')
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim(0, 1.05)

    # =============================================
    # PANEL B (Right): Internal Ambulatory Sensitivity (Motion Split)
    # =============================================
    ax1 = axes[1]

    if not df_wild.empty and 'motion_detected' in df_wild.columns:
        # Plot Stationary (Clean-ish Wild data)
        sns.kdeplot(
            data=df_wild[df_wild['motion_detected'] == False], 
            x="confidence", fill=True, 
            color="#2ca02c", alpha=0.5, linewidth=2,
            label="Ambulatory - Stationary Periods", clip=(0, 1.01), ax=ax1
        )

        # Plot Moving (Noisy Wild data)
        sns.kdeplot(
            data=df_wild[df_wild['motion_detected'] == True], 
            x="confidence", fill=True, 
            color="#d62728", alpha=0.5, linewidth=2,
            label="Ambulatory - High Motion Periods", clip=(0, 1.01), ax=ax1
        )
        
        # Add decision threshold line
        ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label="Decision Threshold (0.5)")

        ax1.set_title("B. Micro-level System Sensitivity\n(Impact of Motion Artifacts)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("") # ShareY handles this label
        ax1.set_xlabel("Confidence Score", fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1.05)
    else:
        ax1.text(0.5, 0.5, "Wild data missing 'motion_detected' column", 
                 ha='center', va='center', transform=ax1.transAxes)

    plt.suptitle("Figure 4.1: Quantitative Analysis of Environmental Noise Sensitivity", fontsize=14, y=0.98)
    plt.tight_layout()
    
    plt.subplots_adjust(top=0.90)

    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    df_wild, df_ideal = load_data()
    generate_combined_figure(df_wild, df_ideal)