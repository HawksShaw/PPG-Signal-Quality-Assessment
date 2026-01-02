import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sqlite3
import json

db_path = "quality_assessment.db"

def load_data():
    try:
        connect = sqlite3.connect(db_path)
        query = "SELECT status, confidence, metrics FROM assessments"
        df = pd.read_sql_query(query, connect)
        connect.close()

        if df.empty:
            print("Database is empty.")
            return pd.DataFrame()

        metrics_data = []
        for idx, row in df.iterrows():
            try:
                mtr = json.loads(row['metrics'])
                mtr['status'] = row['status']
                mtr['confidence'] = row['confidence']

                metrics_data.append(mtr)
            except Exception as e:
                print(f"Error found: {e}")
                continue

        clean_df = pd.DataFrame(metrics_data)
        print(f"Successfully loaded {len(clean_df)} records.")
        return clean_df

    except Exception as e:
        print(f"Database error: {e}")
        print(f"Make sure 'quality_assessment.db' exists and is not opened in another running process.")
        return pd.DataFrame()

def run_validation():
    df = load_data()

    if df.empty:
        return

    all_features = ['spectral_snr', 'skewness', 'kurtosis', 'relative_power', 'zero_crossing_rate', 'motion_energy', 'average_jerk', 'max_magnitude']
    features = [col for col in all_features if col in df.columns]
    print(f"Extracted features: {features}")

    if len(features) < 2:
        print("Not enough metric features for PCA validation.")
        return

    df_clean = df.dropna(subset=features)
    df_clean = df_clean[~df_clean[features].isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    if df_clean.empty:
        print("All data contains NaN or Inf values.")
        return

    x = df_clean[features].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(x)

    df_clean['PCA1'] = pc[:, 0]
    df_clean['PCA2'] = pc[:, 1]

    variance_ratio = pca.explained_variance_ratio_
    total_variance = sum(variance_ratio)*100
    print(f"PCA Variance: {total_variance:.2f}% (PC1: {variance_ratio[0]:.2f}, PC2: {variance_ratio[1]:.2f})")

    plt.figure()

    sns.scatterplot(
        x="PCA1", y="PCA2",
        hue='status',
        palette={
            'GOOD'  : '#FFB8D9',
            'ACCEPTABLE' : '#A3004C',
            'BAD'   : '#470021'
        },
        data=df_clean,
        s=100,
        alpha=0.7
    )

    plt.title(f"Unsupervised validation for signal quality clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_validation()
    