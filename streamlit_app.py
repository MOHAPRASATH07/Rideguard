"""
streamlit_app.py — RideGuard Hugging Face Deployment
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import os
import matplotlib.pyplot as plt

# ─────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────
st.set_page_config(
    page_title="RideGuard — Riding Risk & Mileage",
    page_icon="🛵",
    layout="wide"
)

# ─────────────────────────────────
# STYLING
# ─────────────────────────────────
st.markdown("""
<style>
.main-title{font-size:2rem;font-weight:700;color:#1a1917;margin-bottom:4px;}
.sub-title{color:#6b6860;font-size:1rem;margin-bottom:2rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────
# LOAD MODELS (SAFE)
# ─────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_models():
    try:
        models = {
            'xgb': joblib.load(os.path.join(MODELS, 'xgb_model.pkl')),
            'iso': joblib.load(os.path.join(MODELS, 'iso_forest.pkl')),
            'scaler': joblib.load(os.path.join(MODELS, 'scaler.pkl')),
            'features': joblib.load(os.path.join(MODELS, 'selected_features.pkl')),
            'kmeans': joblib.load(os.path.join(MODELS, 'kmeans_mileage.pkl')),
            'km_scaler': joblib.load(os.path.join(MODELS, 'kmeans_scaler.pkl')),
            'cluster': joblib.load(os.path.join(MODELS, 'cluster_info.pkl')),
            'm_feats': joblib.load(os.path.join(MODELS, 'mileage_features.pkl')),
        }

        # Safe SHAP init
        try:
            models['explainer'] = shap.TreeExplainer(models['xgb'])
        except:
            models['explainer'] = None

        return models, None

    except Exception as e:
        return None, str(e)

models, err = load_models()

# ─────────────────────────────────
# HEADER
# ─────────────────────────────────
st.markdown('<div class="main-title">🛵 RideGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time Riding Risk & Mileage Efficiency</div>', unsafe_allow_html=True)

if err:
    st.error(f"Model loading error: {err}")
    st.stop()

# ─────────────────────────────────
# INPUT
# ─────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    accel_x = st.slider("Accel X", -20.0, 20.0, 0.0)
    accel_y = st.slider("Accel Y", -20.0, 20.0, 0.0)
    accel_z = st.slider("Accel Z", -20.0, 20.0, 0.0)

with col2:
    gyro_x = st.slider("Gyro X", -5.0, 5.0, 0.0)
    gyro_y = st.slider("Gyro Y", -5.0, 5.0, 0.0)
    gyro_z = st.slider("Gyro Z", -5.0, 5.0, 0.0)

speed = st.slider("Speed", 0, 120, 30)
base_kmpl = st.slider("Base Mileage", 10, 100, 60)

# ─────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────
accel_mag = float(np.sqrt(accel_x**2 + accel_y**2 + accel_z**2))
gyro_mag  = float(np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2))
jerk      = float(abs(accel_mag - 9.8))
braking   = float(abs(min(accel_y, 0)))
turn      = float(abs(gyro_z))
tilt      = float(np.arctan2(accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 180 / np.pi)
impact    = float(accel_mag*0.4 + gyro_mag*0.3 + jerk*0.3)

features = {
    'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
    'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
    'accel_mag': accel_mag, 'gyro_mag': gyro_mag,
    'jerk': jerk,
    'braking_intensity': braking,
    'turn_sharpness': turn,
    'tilt_angle': tilt,
    'impact_score': impact,
    'harsh_accel_flag': int(accel_mag > 12),
    'harsh_brake_flag': int(accel_y < -5),
    'harsh_turn_flag': int(gyro_mag > 1.5)
}

# ─────────────────────────────────
# PREDICT
# ─────────────────────────────────
if st.button("Predict"):

    selected = models['features']
    df = pd.DataFrame([features])

    # ✅ FIX 1: Ensure feature alignment
    for col in selected:
        if col not in df:
            df[col] = 0.0

    df = df[selected]

    # Prediction
    probs = models['xgb'].predict_proba(df)[0]
    probs = np.nan_to_num(probs).flatten()   # ✅ FIX 2

    risk_idx = int(np.argmax(probs))
    risk_lbl = ['LOW', 'MODERATE', 'HIGH'][risk_idx]

    # Isolation Forest
    try:
        scaled = models['scaler'].transform(df)
        is_anomaly = models['iso'].predict(scaled)[0] == -1
    except:
        is_anomaly = False

    # ✅ FIX 3: Robust SHAP
    explanation = ['accel_mag', 'gyro_mag', 'jerk']
    try:
        if models['explainer'] is not None:
            sv = models['explainer'](df)

            if hasattr(sv, "values"):
                arr = sv.values[0]
            elif isinstance(sv, list):
                arr = sv[risk_idx][0]
            else:
                arr = sv[0]

            imp = dict(zip(selected, abs(arr)))
            top3 = sorted(imp.items(), key=lambda x: float(x[1]), reverse=True)[:3]
            explanation = [f[0] for f, _ in top3]
    except:
        pass

    # Mileage
    efficiency = 1.0
    if speed > 60: efficiency -= 0.15
    if accel_mag > 10: efficiency -= 0.2
    efficiency = max(0.4, efficiency)
    predicted_kmpl = round(base_kmpl * efficiency, 1)

    # ─────────────────────────────
    # OUTPUT
    # ─────────────────────────────
    st.subheader("Results")
    st.write("Risk:", risk_lbl)
    st.write("Anomaly:", is_anomaly)
    st.write("Top factors:", explanation)
    st.write("Mileage:", predicted_kmpl, "kmpl")

    # Simple chart
    fig, ax = plt.subplots()
    ax.bar(['Low','Moderate','High'], probs*100)
    st.pyplot(fig)
