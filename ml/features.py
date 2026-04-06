import pandas as pd
import numpy as np
from pathlib import Path

def engineer_features(df):
    # Ensure numeric sensor columns are floats (coerce mixed/string values)
    numeric_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Replace NaNs in sensor columns with 0 for safe math operations
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # ─────────────────────────
    # MAGNITUDE FEATURES
    # ─────────────────────────
    df['accel_mag'] = np.sqrt(
        df['accel_x']**2 +
        df['accel_y']**2 +
        df['accel_z']**2
    )

    df['gyro_mag'] = np.sqrt(
        df['gyro_x']**2 +
        df['gyro_y']**2 +
        df['gyro_z']**2
    )

    # ─────────────────────────
    # JERK — rate of change
    # ─────────────────────────
    df['jerk'] = df['accel_mag'].diff().abs().fillna(0)

    # ─────────────────────────
    # BRAKING INTENSITY
    # ─────────────────────────
    df['braking_intensity'] = df['accel_y'].clip(upper=0).abs()

    # ─────────────────────────
    # TURN SHARPNESS
    # ─────────────────────────
    df['turn_sharpness'] = df['gyro_z'].abs()

    # ─────────────────────────
    # HARSH EVENT FLAGS
    # ─────────────────────────
    df['harsh_accel_flag'] = (df['accel_mag'] > 15).astype(int)
    df['harsh_brake_flag'] = (df['accel_y'] < -8).astype(int)
    df['harsh_turn_flag']  = (df['gyro_mag'] > 2).astype(int)

    # ─────────────────────────
    # TILT ANGLE
    # ─────────────────────────
    df['tilt_angle'] = np.arctan2(
        df['accel_x'],
        np.sqrt(df['accel_y']**2 + df['accel_z']**2)
    ) * 180 / np.pi

    # ─────────────────────────
    # IMPACT SCORE
    # ─────────────────────────
    df['impact_score'] = (
        df['accel_mag'] * 0.4 +
        df['gyro_mag'] * 0.3 +
        df['jerk'] * 0.3
    )

    return df

def get_feature_columns():
    return [
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'accel_mag', 'gyro_mag',
        'jerk', 'braking_intensity',
        'turn_sharpness', 'tilt_angle',
        'impact_score',
        'harsh_accel_flag',
        'harsh_brake_flag',
        'harsh_turn_flag'
    ]

if __name__ == '__main__':
    # Resolve path relative to this file to be robust in different cwd
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'combined_dataset.csv'
    print(f"Loading combined dataset from {data_path}...")
    if not data_path.exists():
        print(f"File not found: {data_path}")
        print("Generate it by running: python ml/load_data.py from project root")
        raise SystemExit(1)

    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded: {df.shape}")

    print("Engineering features...")
    df = engineer_features(df)

    feature_cols = get_feature_columns()
    print(f"\nFeatures created: {len(feature_cols)}")
    print(feature_cols)

    out_path = data_path.parent / 'featured_dataset.csv'
    try:
        df.to_csv(out_path, index=False)
        print(f"\n Saved to {out_path}")
    except PermissionError:
        print(f"Permission denied when writing {out_path}.")
        print("Check file permissions or close any program using the file and try again.")
        raise SystemExit(1)
    except Exception as e:
        print(f"Failed to save featured dataset: {e}")
        raise