import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
DB_PATH = "quality_assessment.db"

def load_data():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT subject_id, status, metrics FROM assessments"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Extract metrics from JSON
    metrics_list = []
    for _, row in df.iterrows():
        try:
            m = json.loads(row['metrics'])
            m['subject_id'] = row['subject_id']
            m['status'] = row['status']
            metrics_list.append(m)
        except:
            continue
    return pd.DataFrame(metrics_list)

def plot_subject_robustness(df):
    """
    Requirement: Subject-level splits.
    Shows the Acceptance Rate for each subject.
    """
    # Calculate Pass Rate per Subject
    subject_stats = df.groupby('subject_id')['status'].value_counts(normalize=True).unstack().fillna(0)
    
    # Handle case where 'accept' or 'GOOD' might be missing columns
    if 'accept' not in subject_stats.columns: subject_stats['accept'] = 0
    if 'GOOD' in subject_stats.columns: subject_stats['accept'] += subject_stats['GOOD']
    
    # Sort by Pass Rate
    subject_stats = subject_stats.sort_values(by='accept', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=subject_stats.index, y=subject_stats['accept'], palette="viridis")
    
    plt.title("Subject-Level Robustness: Acceptance Rate per Subject")
    plt.ylabel("Acceptance Rate (0.0 - 1.0)")
    plt.xlabel("Subject ID")
    plt.xticks(rotation=45, ha='right')
    plt.axhline(subject_stats['accept'].mean(), color='red', linestyle='--', label='Average Pass Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_threshold_guidelines(df, metric_name='motion_energy'):
    """
    Requirement: Practical integration guidelines (thresholds).
    Finds the intersection point to recommend a cutoff.
    """
    if metric_name not in df.columns:
        print(f"Skipping threshold plot: {metric_name} not found.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot overlapping distributions
    sns.histplot(data=df, x=metric_name, hue='status', kde=True, bins=30, 
                 palette={'GOOD': 'green', 'BAD': 'red', 'ACCEPTABLE' : 'orange'},
                 alpha=0.5)

    # Calculate suggested threshold (simple mean of means heuristic)
    mean_accept = df[df['status'].isin(['GOOD', 'ACCEPTABLE'])][metric_name].mean()
    mean_reject = df[df['status'].isin(['BAD'])][metric_name].mean()
    
    threshold_est = (mean_accept + mean_reject) / 2
    
    plt.axvline(threshold_est, color='black', linestyle='--', linewidth=2)
    plt.text(threshold_est, plt.ylim()[1]*0.9, f' Suggested Cutoff\n ~{threshold_est:.2f}', 
             ha='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f"Integration Guideline: Defining the '{metric_name}' Threshold")
    plt.xlabel(f"{metric_name} (Calculated)")
    plt.xlim(0, df[metric_name].quantile(0.95)) # Zoom in on the relevant area (ignore outliers)
    plt.show()

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        # 1. Subject Robustness
        plot_subject_robustness(df)
        
        # 2. Threshold Definition (Using your main metric)
        plot_threshold_guidelines(df, metric_name='motion_energy')
    else:
        print("No data found.")