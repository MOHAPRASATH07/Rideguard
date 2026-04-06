"""
ml/train_model.py — RideGuard Risk Model Training

Fixed:
1. Removed synthetic / duplicated resampling
2. Feature selection before scaling
3. Isolation Forest on scaled selected features
4. Scaler saved after feature selection
5. Fixed XGBoost overfitting with validation split + early stopping
6. Removed test-set leakage from Optuna and final XGBoost training
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import shap
import joblib
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, '../data')
MODELS = os.path.join(BASE, '../models')
os.makedirs(MODELS, exist_ok=True)

print("=" * 55)
print("STEP 1 — LOADING DATA")
print("=" * 55)

df = pd.read_csv(os.path.join(DATA, 'featured_dataset.csv'))
print(f"Loaded: {df.shape}")
print(df['risk_level'].value_counts())

print("\n" + "=" * 55)
print("STEP 2 — MISSING VALUE HANDLING")
print("=" * 55)

missing = df.isnull().sum()
missing_cols = missing[missing > 0]
print(f"Missing: {len(missing_cols)} columns" if len(missing_cols) > 0 else "No missing values")

df = df.ffill().fillna(0)
print(f"Missing after: {df.isnull().sum().sum()}")

print("\n" + "=" * 55)
print("STEP 3 — USING ORIGINAL DATA (NO SYNTHETIC VALUES)")
print("=" * 55)

df_balanced = df.copy()
print(f"Dataset: {df_balanced.shape}")
print(df_balanced['risk_level'].value_counts())

FEATURE_COLS = [c for c in [
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x',  'gyro_y',  'gyro_z',
    'accel_mag', 'gyro_mag',
    'jerk', 'braking_intensity',
    'turn_sharpness', 'tilt_angle',
    'impact_score',
    'harsh_accel_flag', 'harsh_brake_flag', 'harsh_turn_flag'
] if c in df_balanced.columns]

X = df_balanced[FEATURE_COLS].fillna(0)
y = df_balanced['risk_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

print("\n" + "=" * 55)
print("STEP 5 — FEATURE SELECTION (before scaling)")
print("=" * 55)

selector_rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
selector_rf.fit(X_train, y_train)

importances = pd.Series(
    selector_rf.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)

print("Feature importances:")
print(importances.round(4))

threshold = importances.median()
selected_features = importances[importances >= threshold].index.tolist()
print(f"\nSelected {len(selected_features)} features: {selected_features}")

X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]

joblib.dump(selected_features, os.path.join(MODELS, 'selected_features.pkl'))
print("selected_features.pkl saved")

print("\n" + "=" * 55)
print("STEP 6 — FEATURE SCALING (after selection)")
print("=" * 55)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled  = scaler.transform(X_test_sel)

joblib.dump(scaler, os.path.join(MODELS, 'scaler.pkl'))
print(f"scaler.pkl saved — fitted on {len(selected_features)} selected features")

plt.figure(figsize=(10, 8))
sns.heatmap(X_train_sel.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(MODELS, 'correlation_matrix.png'), dpi=100)
plt.close()
print("Correlation matrix saved")

print("\n" + "=" * 55)
print("STEP 7 — ISOLATION FOREST")
print("=" * 55)

iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
iso.fit(X_train_scaled)

anomaly_preds = iso.predict(X_test_scaled)
anomaly_count = (anomaly_preds == -1).sum()
print(f"Anomalies in test: {anomaly_count} / {len(X_test_scaled)} ({anomaly_count/len(X_test_scaled)*100:.1f}%)")

joblib.dump(iso, os.path.join(MODELS, 'iso_forest.pkl'))
print("iso_forest.pkl saved")
print("IMPORTANT: At runtime always scale before iso_forest.predict()")

print("\n" + "=" * 55)
print("STEP 8 — VALIDATION SPLIT FOR XGBOOST")
print("=" * 55)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_sel, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train split for tuning: {X_tr.shape}")
print(f"Validation split: {X_val.shape}")

print("\n" + "=" * 55)
print("STEP 9 — HYPERPARAMETER TUNING (Optuna 20 trials)")
print("=" * 55)

def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 400),
        'max_depth':        trial.suggest_int('max_depth', 3, 6),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.05),
        'subsample':        trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),

        'reg_alpha':        trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),

        'random_state': 42,
        'n_jobs':        -1,
        'verbosity':     0
    }

    m = XGBClassifier(**params, early_stopping_rounds=20)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return m.score(X_val, y_val)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=False)

print(f"Best params  : {study.best_params}")
print(f"Best accuracy: {study.best_value:.4f}")

print("\n" + "=" * 55)
print("STEP 10 — RANDOM FOREST")
print("=" * 55)

labels_all = [0, 1, 2]
target_names_all = ['Low', 'Moderate', 'High']

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sel, y_train)
y_pred_rf = rf.predict(X_test_sel)
rf_accuracy = rf.score(X_test_sel, y_test)

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(classification_report(
    y_test, y_pred_rf,
    labels=labels_all,
    target_names=target_names_all
))

plt.figure(figsize=(7, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred_rf, labels=labels_all),
    annot=True, fmt='d',
    xticklabels=target_names_all,
    yticklabels=target_names_all,
    cmap='Blues'
)
plt.title(f'Random Forest — {rf_accuracy:.1%}')
plt.tight_layout()
plt.savefig(os.path.join(MODELS, 'rf_confusion.png'), dpi=100)
plt.close()

joblib.dump(rf, os.path.join(MODELS, 'rf_model.pkl'))
print("rf_model.pkl saved")

print("\n" + "=" * 55)
print("STEP 11 — XGBOOST (best params)")
print("=" * 55)

best_params = dict(study.best_params)
best_params.update({
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
})

xgb = XGBClassifier(**best_params, early_stopping_rounds=20)
xgb.fit(
    X_train_sel, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred_xgb = xgb.predict(X_test_sel)
xgb_accuracy = xgb.score(X_test_sel, y_test)

print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print(classification_report(
    y_test, y_pred_xgb,
    labels=labels_all,
    target_names=target_names_all
))

plt.figure(figsize=(7, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred_xgb, labels=labels_all),
    annot=True, fmt='d',
    xticklabels=target_names_all,
    yticklabels=target_names_all,
    cmap='Greens'
)
plt.title(f'XGBoost — {xgb_accuracy:.1%}')
plt.tight_layout()
plt.savefig(os.path.join(MODELS, 'xgb_confusion.png'), dpi=100)
plt.close()

joblib.dump(xgb, os.path.join(MODELS, 'xgb_model.pkl'))
print("xgb_model.pkl saved")

print("\n" + "=" * 55)
print("STEP 12 — SCIKIT-LEARN PIPELINES")
print("=" * 55)

rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])
rf_pipe.fit(X_train_sel, y_train)
rf_pipe_acc = rf_pipe.score(X_test_sel, y_test)
print(f"Pipeline RF  : {rf_pipe_acc:.4f}")

xgb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  XGBClassifier(**best_params))
])
xgb_pipe.fit(X_train_sel, y_train)
xgb_pipe_acc = xgb_pipe.score(X_test_sel, y_test)
print(f"Pipeline XGB : {xgb_pipe_acc:.4f}")

joblib.dump(rf_pipe, os.path.join(MODELS, 'rf_pipeline.pkl'))
joblib.dump(xgb_pipe, os.path.join(MODELS, 'xgb_pipeline.pkl'))
print("Pipelines saved")

print("\n" + "=" * 55)
print("STEP 13 — SHAP EXPLAINABILITY")
print("=" * 55)

explainer = shap.TreeExplainer(xgb)
n = min(500, len(X_test_sel))
shap_values = explainer.shap_values(X_test_sel.iloc[:n])

# Handle multiclass output safely
if isinstance(shap_values, list):
    shap_values = shap_values[0]

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test_sel.iloc[:n],
    plot_type="bar",
    show=False
)
plt.title("What causes riding risk? — SHAP Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(MODELS, 'shap_summary.png'), dpi=100)
plt.close()
print("shap_summary.png saved")

print("\n" + "=" * 55)
print("COMPLETE — ALL MODELS SAVED")
print("=" * 55)
print(f"\nResults:")
print(f"  Random Forest  : {rf_accuracy:.1%}")
print(f"  XGBoost        : {xgb_accuracy:.1%}  -> production model")
print(f"  Pipeline RF    : {rf_pipe_acc:.1%}")
print(f"  Pipeline XGB   : {xgb_pipe_acc:.1%}")
print(f"\nSelected features ({len(selected_features)}): {selected_features}")
print(f"\nSaved in models/:")
for f in [
    'scaler.pkl', 'selected_features.pkl', 'iso_forest.pkl',
    'rf_model.pkl', 'xgb_model.pkl', 'rf_pipeline.pkl', 'xgb_pipeline.pkl',
    'correlation_matrix.png', 'rf_confusion.png', 'xgb_confusion.png', 'shap_summary.png'
]:
    print(f"  {f}")
