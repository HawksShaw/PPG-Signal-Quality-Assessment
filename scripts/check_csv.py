import pandas as pd

df = pd.read_csv("benchmark_results.csv", nrows=2)

print("Column names:")
print(list(df.columns))