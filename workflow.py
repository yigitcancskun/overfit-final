import os
import pandas as pd
import matplotlib.pyplot as plt

files = [
    "./data/fusion_author_scores.parquet",
    "./data/fusion_scored_messages.parquet",
    "./data/datathonFINAL.parquet",
    "./data/datathonFINAL_cleaned_fast.parquet"
]

target_col = 'author_score'
found_file = None

for f in files:
    if not os.path.exists(f):
        continue
    try:
        # Check schema
        df_sample = pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f, nrows=1)
        if target_col in df_sample.columns:
            found_file = f
            break
    except Exception as e:
        print(f"Error reading {f}: {e}")

if found_file:
    print(f"Found column '{target_col}' in: {found_file}")
    df = pd.read_parquet(found_file, columns=[target_col])
    col = df[target_col].dropna()
    
    stats = {
        "min": col.min(),
        "max": col.max(),
        "mean": col.mean(),
        "median": col.median()
    }
    print(f"Stats: {stats}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(col, bins=50, edgecolor='black')
    plt.title(f'Histogram of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.savefig('author_score_histogram.png')
    print("Histogram saved as author_score_histogram.png")
else:
    print(f"Column '{target_col}' not found in listed files.")
