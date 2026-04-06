"""
train_mileage.py — Complete ML pipeline for mileage prediction

ALL preprocessing steps included:
1. Missing value handling  — SimpleImputer (median strategy)
2. Feature engineering     — 10 sensor-derived features
3. Feature selection       — Correlation threshold + SelectKBest
4. Feature scaling         — StandardScaler in Pipeline
5. Hyperparameter tuning   — GridSearchCV for GB, CV for KNN
6. Feature modeling        — KMeans + KNN + GradientBoosting
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, '../data')
MODELS = os.path.join(BASE, '../models')

print("=" * 60)
print("RIDEGUARD — MILEAGE MODEL TRAINER")
print("All preprocessing steps included")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────
print("\nSTEP 1 — Loading mileage dataset...")

path = os.path.join(DATA, 'mileage_dataset.csv')
if not os.path.exists(path):
    print("ERROR: Run build_mileage_dataset.py first")
    exit()

df = pd.read_csv(path)
print(f"  Shape : {df.shape}")
print(f"  Target: actual_kmpl")
print(f"  Range : {df['actual_kmpl'].min():.1f} - {df['actual_kmpl'].max():.1f} kmpl")

# ─────────────────────────────────────────────
# STEP 2 — MISSING VALUE HANDLING
# Strategy: median imputation
# Why median: sensor data has outliers from crashes
#             median is robust to outliers
#             mean would be pulled by extreme values
# ─────────────────────────────────────────────
print("\nSTEP 2 — Missing value handling...")

ALL_FEATURES = [
    'accel_mag', 'gyro_mag', 'jerk',
    'braking_intensity', 'turn_sharpness',
    'impact_score',
    'harsh_accel_flag', 'harsh_brake_flag', 'harsh_turn_flag',
    'speed_kmh'
]

print(f"  Missing before:")
missing = df[ALL_FEATURES].isnull().sum()
print(f"  {missing[missing > 0].to_dict() or 'None'}")

# Apply median imputation
imputer = SimpleImputer(strategy='median')
df[ALL_FEATURES] = imputer.fit_transform(df[ALL_FEATURES])

print(f"  Missing after : {df[ALL_FEATURES].isnull().sum().sum()}")
print(f"  Strategy      : median (robust to sensor outliers)")
joblib.dump(imputer, os.path.join(MODELS, 'mileage_imputer.pkl'))
print(f"  Imputer saved")

# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# Already done in build_mileage_dataset.py
# Documenting what was created and why
# ─────────────────────────────────────────────
print("\nSTEP 3 — Feature engineering summary...")

feature_descriptions = {
    'accel_mag':          'Resultant acceleration magnitude — detects overall force',
    'gyro_mag':           'Resultant gyroscope magnitude — detects rotation intensity',
    'jerk':               'Rate of change of acceleration — detects sudden moves',
    'braking_intensity':  'Negative Y acceleration magnitude — detects hard braking',
    'turn_sharpness':     'Absolute Z gyroscope — detects sharp turns',
    'impact_score':       'Weighted combination: 0.4×accel + 0.3×gyro + 0.3×jerk',
    'harsh_accel_flag':   'Binary: accel_mag > 15 m/s² — harsh acceleration event',
    'harsh_brake_flag':   'Binary: accel_y < -8 m/s² — harsh braking event',
    'harsh_turn_flag':    'Binary: gyro_mag > 2 rad/s — harsh turn event',
    'speed_kmh':          'Speed in kmph — major mileage factor',
}

for feat, desc in feature_descriptions.items():
    print(f"  {feat:<22} : {desc}")

# ─────────────────────────────────────────────
# STEP 4 — FEATURE SELECTION
# Method 1: Correlation with target
# Method 2: SelectKBest with f_regression
# Why: Remove weakly correlated features
#      Reduces model complexity
#      Prevents overfitting
# ─────────────────────────────────────────────
print("\nSTEP 4 — Feature selection...")

X_all = df[ALL_FEATURES]
y     = df['actual_kmpl']

# Correlation analysis
corr = X_all.corrwith(y).abs().sort_values(ascending=False)
print(f"\n  Correlation with actual_kmpl:")
for feat, val in corr.items():
    status = "KEEP" if val >= 0.05 else "DROP"
    print(f"  {feat:<22} : {val:.4f}  [{status}]")

# Keep features with correlation >= 0.05
selected_by_corr = corr[corr >= 0.05].index.tolist()
print(f"\n  Selected by correlation : {len(selected_by_corr)} features")
print(f"  {selected_by_corr}")

# SelectKBest for additional validation
selector = SelectKBest(f_regression, k=min(8, len(selected_by_corr)))
selector.fit(X_all[selected_by_corr], y)
kbest_mask    = selector.get_support()
selected_final = [f for f, m in zip(selected_by_corr, kbest_mask) if m]

# Always keep speed_kmh — most important feature
if 'speed_kmh' not in selected_final:
    selected_final.append('speed_kmh')

print(f"\n  Final selected features ({len(selected_final)}):")
for f in selected_final:
    print(f"    {f}")

joblib.dump(selected_final, os.path.join(MODELS, 'mileage_features.pkl'))
print(f"  Feature list saved")

X = df[selected_final]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Train: {X_train.shape}  Test: {X_test.shape}")

# ─────────────────────────────────────────────
# STEP 5 — MODEL 1: K-MEANS CLUSTERING
# Unsupervised rider style grouping
# No hyperparameter tuning needed (silhouette)
# ─────────────────────────────────────────────
print("\nSTEP 5 — K-Means Clustering...")

CLUSTER_FEATURES = [f for f in selected_final if f != 'speed_kmh']
# Ensure we have sensor features for clustering (rider style detection)
if not CLUSTER_FEATURES:
    CLUSTER_FEATURES = ['accel_mag', 'gyro_mag', 'impact_score']
    print(f"  Warning: No sensor features passed selection. Using fallback for clustering: {CLUSTER_FEATURES}")

scaler_km = StandardScaler()
X_cluster = scaler_km.fit_transform(df[CLUSTER_FEATURES])

best_k, best_sil = 4, -1
for k in range(3, 7):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_cluster)
    sil = silhouette_score(X_cluster[:5000], lbl[:5000])
    print(f"  k={k}  silhouette={sil:.4f}")
    if sil > best_sil:
        best_sil, best_k = sil, k

best_k = max(best_k, 4)
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster)

impact_mean = (
    pd.Series(df[CLUSTER_FEATURES].mean(axis=1).values)
    .groupby(cluster_labels).mean().sort_values()
)

style_names = ['Eco Rider', 'Normal Rider', 'Aggressive Rider', 'Dangerous Rider']
cluster_to_style = {int(cid): style_names[min(i, 3)]
                    for i, (cid, _) in enumerate(impact_mean.items())}

cluster_kmpl = pd.Series(y.values).groupby(cluster_labels).mean()
cluster_info = {
    cid: {
        'style':    cluster_to_style[cid],
        'avg_kmpl': round(float(cluster_kmpl[cid]), 1),
        'count':    int((cluster_labels == cid).sum())
    }
    for cid in cluster_to_style
}

print(f"\n  Clusters (k={best_k}):")
for cid, info in cluster_info.items():
    print(f"  {info['style']:20s}  avg={info['avg_kmpl']} kmpl  n={info['count']}")

joblib.dump(kmeans,           os.path.join(MODELS, 'kmeans_mileage.pkl'))
joblib.dump(scaler_km,        os.path.join(MODELS, 'kmeans_scaler.pkl'))
joblib.dump(cluster_info,     os.path.join(MODELS, 'cluster_info.pkl'))
joblib.dump(cluster_to_style, os.path.join(MODELS, 'cluster_styles.pkl'))
print("  K-Means saved")

# ─────────────────────────────────────────────
# STEP 6 — MODEL 2: KNN REGRESSION
# Hyperparameter tuning: cross validation
# to find best k
# Scaling: StandardScaler in Pipeline
# Missing: SimpleImputer in Pipeline
# ─────────────────────────────────────────────
print("\nSTEP 6 — KNN Regression with hyperparameter tuning...")

best_knn_k, best_knn_mae = 5, float('inf')

for k in [3, 5, 7, 10, 15, 20]:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('knn',     KNeighborsRegressor(n_neighbors=k, n_jobs=-1))
    ])
    scores = cross_val_score(
        pipe, X_train, y_train,
        cv=5, scoring='neg_mean_absolute_error'
    )
    mae = -scores.mean()
    print(f"  k={k:2d}  MAE={mae:.2f} kmpl  std={scores.std():.3f}")
    if mae < best_knn_mae:
        best_knn_mae, best_knn_k = mae, k

print(f"\n  Best k = {best_knn_k}")

knn_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('knn',     KNeighborsRegressor(
        n_neighbors=best_knn_k,
        metric='euclidean',
        weights='distance',
        n_jobs=-1
    ))
])
knn_pipe.fit(X_train, y_train)
y_pred_knn = knn_pipe.predict(X_test)
knn_mae = mean_absolute_error(y_test, y_pred_knn)
knn_r2  = r2_score(y_test, y_pred_knn)

print(f"\n  KNN Results:")
print(f"  MAE : {knn_mae:.2f} kmpl")
print(f"  R2  : {knn_r2:.4f}")

joblib.dump(knn_pipe, os.path.join(MODELS, 'knn_mileage.pkl'))
print("  KNN saved")

# ─────────────────────────────────────────────
# STEP 7 — MODEL 3: GRADIENT BOOSTING
# Hyperparameter tuning: GridSearchCV
# Scaling: StandardScaler in Pipeline
# Missing: SimpleImputer in Pipeline
# ─────────────────────────────────────────────
print("\nSTEP 7 — Gradient Boosting with GridSearchCV...")

gb_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   GradientBoostingRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators':  [100, 200],
    'model__max_depth':     [3, 5, 7],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample':     [0.8, 1.0]
}

print(f"  Running GridSearchCV (this takes 2-3 minutes)...")
grid_search = GridSearchCV(
    gb_pipe,
    param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

print(f"\n  Best params : {grid_search.best_params_}")
print(f"  Best CV MAE : {-grid_search.best_score_:.2f} kmpl")

y_pred_gb = grid_search.best_estimator_.predict(X_test)
gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_r2  = r2_score(y_test, y_pred_gb)

print(f"\n  Gradient Boosting Results:")
print(f"  MAE : {gb_mae:.2f} kmpl")
print(f"  R2  : {gb_r2:.4f}")

joblib.dump(grid_search.best_estimator_, os.path.join(MODELS, 'gb_mileage.pkl'))
print("  Gradient Boosting saved")

# ─────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREPROCESSING STEPS SUMMARY")
print("=" * 60)
print(f"""
Step                  Method                      Status
─────────────────────────────────────────────────────────
Missing values        SimpleImputer (median)      Done
Feature engineering   10 sensor features          Done
Feature selection     Correlation + SelectKBest   Done
Feature scaling       StandardScaler (Pipeline)   Done
Hyperparameter KNN    Cross-validation k          Done
Hyperparameter GB     GridSearchCV                Done
""")

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<25} {'MAE (kmpl)':<15} {'R2':<10} {'Type'}")
print("-" * 65)
print(f"{'K-Means':<25} {'N/A':<15} {'N/A':<10} Unsupervised clustering")
print(f"{'KNN (k={best_knn_k})':<25} {knn_mae:<15.2f} {knn_r2:<10.4f} Instance-based regression")
print(f"{'Gradient Boosting':<25} {gb_mae:<15.2f} {gb_r2:<10.4f} Ensemble regression")
print(f"\nBest model: {'KNN' if knn_mae < gb_mae else 'Gradient Boosting'}")

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1 — K-Means clusters
counts = pd.Series(cluster_labels).value_counts().sort_index()
styles = [cluster_to_style.get(i, f'C{i}') for i in counts.index]
colors = ['#00e5a0', '#4d9fff', '#ffb830', '#ff3d5a'][:len(styles)]
axes[0,0].bar(styles, counts.values, color=colors)
axes[0,0].set_title('K-Means: Rider Style Distribution', fontweight='bold')
axes[0,0].set_ylabel('Readings')
axes[0,0].tick_params(axis='x', rotation=12)

# Plot 2 — Feature importance (correlation)
corr_selected = corr[selected_final].sort_values(ascending=True)
axes[0,1].barh(corr_selected.index, corr_selected.values, color='#4d9fff')
axes[0,1].set_title('Feature Selection: Correlation with Mileage', fontweight='bold')
axes[0,1].set_xlabel('Absolute correlation')

# Plot 3 — KNN actual vs predicted
idx = np.random.choice(len(y_test), min(300, len(y_test)), replace=False)
axes[1,0].scatter(y_test.iloc[idx], y_pred_knn[idx], alpha=0.4, color='#4d9fff', s=15)
axes[1,0].plot([10,80],[10,80],'--r',lw=1)
axes[1,0].set_title(f'KNN: Actual vs Predicted\nMAE={knn_mae:.2f} kmpl', fontweight='bold')
axes[1,0].set_xlabel('Actual kmpl'); axes[1,0].set_ylabel('Predicted kmpl')

# Plot 4 — GB actual vs predicted
axes[1,1].scatter(y_test.iloc[idx], y_pred_gb[idx], alpha=0.4, color='#00e5a0', s=15)
axes[1,1].plot([10,80],[10,80],'--r',lw=1)
axes[1,1].set_title(f'Gradient Boosting: Actual vs Predicted\nMAE={gb_mae:.2f} kmpl', fontweight='bold')
axes[1,1].set_xlabel('Actual kmpl'); axes[1,1].set_ylabel('Predicted kmpl')

plt.suptitle(
    'RideGuard Mileage Models — All Preprocessing Steps Applied\n'
    'Ground truth: Real Nairobi motorcycle fuel data',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(MODELS, 'mileage_comparison.png'), dpi=150, bbox_inches='tight')
print("\nPlot saved: models/mileage_comparison.png")

print("\n" + "=" * 60)
print("ALL MODELS SAVED")
print("=" * 60)
print("  mileage_imputer.pkl   — SimpleImputer")
print("  mileage_features.pkl  — selected features")
print("  kmeans_mileage.pkl    — K-Means")
print("  knn_mileage.pkl       — KNN Pipeline")
print("  gb_mileage.pkl        — Gradient Boosting Pipeline")
print("  mileage_comparison.png")
print("\nNo synthetic data. Real Nairobi motorcycle fuel data used.")