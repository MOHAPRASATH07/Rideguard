import pandas as pd
import numpy as np
import os
import glob

# ─────────────────────────────
# LOAD DATASET 2 — KAGGLE
# ─────────────────────────────
def load_kaggle_dataset(folder):
    all_files = glob.glob(os.path.join(folder, '*.csv'))
    dfs = []

    for f in all_files:
        try:
            df = pd.read_csv(f)
            df = df.rename(columns={
                'acc_x': 'accel_x',
                'acc_y': 'accel_y',
                'acc_z': 'accel_z',
                'gyro_x': 'gyro_x',
                'gyro_y': 'gyro_y',
                'gyro_z': 'gyro_z'
            })
            df['source'] = 'kaggle'
            dfs.append(df)
        except Exception as e:
            print(f"Error: {f} — {e}")
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Kaggle dataset loaded: {combined.shape}")
        if 'event_class' in combined.columns:
            print(f"Event classes: {combined['event_class'].value_counts()}")
    else:
        combined = pd.DataFrame()
        print(f"Kaggle dataset: no CSV files found or all reads failed in {folder}")

    return combined

# ─────────────────────────────
# LOAD DATASET 1 — MOTORCYCLE
# ─────────────────────────────
def load_motorcycle_dataset(folder, label):
    all_files = glob.glob(os.path.join(folder, '*.csv'))
    dfs = []

    for f in all_files:
        try:
            df = pd.read_csv(f, sep='\t')
            df.columns = df.columns.str.strip()
            df = df.rename(columns={
                'time(s)': 'timestamp',
                'Ax(m/s²)': 'accel_x',
                'Ay(m/s²)': 'accel_y',
                'Az(m/s²)': 'accel_z',
                'Rx(°/s)': 'gyro_x',
                'Ry.(°/s)': 'gyro_y',
                'Rz(°/s)': 'gyro_z'
            })
            df['event_class'] = label
            df['source'] = 'motorcycle'
            dfs.append(df)
        except Exception as e:
            print(f"Error: {f} — {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Motorcycle dataset loaded: {combined.shape}")
    else:
        combined = pd.DataFrame()
        print(f"Motorcycle dataset: no CSV files found or all reads failed in {folder}")

    return combined

# ─────────────────────────────
# COMBINE BOTH
# ─────────────────────────────
def load_all():
    # Load kaggle dataset
    kaggle_df = load_kaggle_dataset('../data/kaggle/')

    # Load motorcycle falls — HIGH RISK
    fall_df = load_motorcycle_dataset(
        '../data/motorcycle/falls/', 
        label='fall'
    )

    # Load motorcycle manoeuvres — MEDIUM RISK
    manoeuvre_df = load_motorcycle_dataset(
        '../data/motorcycle/manoeuvres/', 
        label='extreme_manoeuvre'
    )

    # Combine all
    keep_cols = [
        'timestamp', 'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z', 'event_class', 'source'
    ]

    kaggle_clean = kaggle_df[[
        c for c in keep_cols if c in kaggle_df.columns
    ]]
    fall_clean = fall_df[[
        c for c in keep_cols if c in fall_df.columns
    ]]
    manoeuvre_clean = manoeuvre_df[[
        c for c in keep_cols if c in manoeuvre_df.columns
    ]]

    final = pd.concat(
        [kaggle_clean, fall_clean, manoeuvre_clean],
        ignore_index=True
    )

    if final.empty:
        print("Final combined dataset is empty — no data to map or save.")
        return final

    # Map to risk labels
    risk_map = {
        'safe':                0,
        'normal':              0,
        'sudden_acc':          1,
        'sudden_brake':        1,
        'sudden_turn':         1,
        'extreme_manoeuvre':   1,
        'harsh_brake':         2,
        'harsh_acc':           2,
        'fall':                2
    }

    final['risk_level'] = final['event_class'].map(
        lambda x: risk_map.get(str(x).lower().strip(), 1)
    )

    print(f"\nFinal combined dataset: {final.shape}")
    print(f"\nRisk distribution:")
    print(final['risk_level'].value_counts())
    print(f"\nEvent classes found:")
    print(final['event_class'].value_counts())

    final.to_csv('../data/combined_dataset.csv', index=False)
    print("\n Saved to data/combined_dataset.csv")

    return final

if __name__ == '__main__':
    df = load_all()