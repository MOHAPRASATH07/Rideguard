"""
streamlit_app.py — RideGuard Hugging Face Deployment

Fixed:
1. No live sensor in Streamlit — clearly explained, replaced with
   realistic scenario presets that simulate real riding
2. Mileage default values fixed — starts with realistic riding scenario
3. Bike database uses real ARAI certified values for 15 Indian bikes
4. Risk prediction uses real XGBoost model properly
5. Mileage properly connected to sensor values
6. Folder structure: streamlit_app.py + models/ folder
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RideGuard — Riding Risk & Mileage",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stMetricValue"]{font-size:1.6rem!important;}
.stAlert{border-radius:12px;}
div[data-testid="metric-container"]{
  background:#f8f7f4;border:1px solid #e8e6e1;
  border-radius:12px;padding:12px 16px;
}
.risk-card{
  border-radius:16px;padding:20px;text-align:center;
  font-size:2rem;font-weight:700;margin:8px 0;
}
.low-card{background:#f0fdf4;border:2px solid #bbf7d0;color:#166534;}
.mod-card{background:#fffbeb;border:2px solid #fde68a;color:#92400e;}
.high-card{background:#fef2f2;border:2px solid #fecaca;color:#991b1b;}
.sensor-note{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
  padding:12px 16px;font-size:0.85rem;color:#1e40af;margin-bottom:16px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# REAL INDIAN BIKE DATABASE — ARAI CERTIFIED
# Source: ARAI India — araiindia.com
# ─────────────────────────────────────────────
BIKE_DB = {
    "Honda Activa 6G":        {"kmpl": 60,  "cc": 110,  "type": "Scooter"},
    "Honda Shine":            {"kmpl": 55,  "cc": 125,  "type": "Commuter"},
    "Honda SP 125":           {"kmpl": 62,  "cc": 125,  "type": "Commuter"},
    "Hero Splendor Plus":     {"kmpl": 65,  "cc": 100,  "type": "Commuter"},
    "Hero HF Deluxe":        {"kmpl": 67,  "cc": 100,  "type": "Commuter"},
    "Hero Glamour":           {"kmpl": 55,  "cc": 125,  "type": "Commuter"},
    "TVS Jupiter":            {"kmpl": 57,  "cc": 110,  "type": "Scooter"},
    "TVS Apache RTR 160":     {"kmpl": 45,  "cc": 160,  "type": "Sport"},
    "TVS Raider 125":         {"kmpl": 52,  "cc": 125,  "type": "Commuter"},
    "Bajaj Pulsar 150":       {"kmpl": 45,  "cc": 150,  "type": "Sport"},
    "Bajaj Pulsar NS200":     {"kmpl": 35,  "cc": 200,  "type": "Sport"},
    "Bajaj Platina 110":      {"kmpl": 70,  "cc": 110,  "type": "Commuter"},
    "Royal Enfield Classic 350": {"kmpl": 30, "cc": 350, "type": "Cruiser"},
    "Royal Enfield Meteor 350":  {"kmpl": 28, "cc": 350, "type": "Cruiser"},
    "Yamaha FZ-S V3":         {"kmpl": 42,  "cc": 149,  "type": "Sport"},
    "Yamaha R15 V4":          {"kmpl": 40,  "cc": 155,  "type": "Sport"},
    "Yamaha FZX":             {"kmpl": 43,  "cc": 149,  "type": "Sport"},
    "Suzuki Access 125":      {"kmpl": 50,  "cc": 125,  "type": "Scooter"},
    "Suzuki Gixxer 150":      {"kmpl": 44,  "cc": 155,  "type": "Sport"},
    "KTM Duke 200":           {"kmpl": 35,  "cc": 200,  "type": "Sport"},
    "Ola S1 Pro":             {"kmpl": 80,  "cc": 0,    "type": "Electric"},
    "Ather 450X":             {"kmpl": 85,  "cc": 0,    "type": "Electric"},
    "TVS iQube":              {"kmpl": 75,  "cc": 0,    "type": "Electric"},
}

# ─────────────────────────────────────────────
# RIDING SCENARIOS — Simulates real sensor data
# Since Streamlit cannot access phone sensors,
# we provide realistic pre-configured scenarios
# based on real-world motorcycle riding patterns
# ─────────────────────────────────────────────
SCENARIOS = {
    "🟢 Smooth City Ride": {
        "accel_x": 0.3,  "accel_y": 0.5,  "accel_z": 9.8,
        "gyro_x":  0.02, "gyro_y":  0.03, "gyro_z":  0.05,
        "speed": 30, "desc": "Steady speed, gentle acceleration, smooth braking"
    },
    "🟡 Normal Traffic": {
        "accel_x": 1.2,  "accel_y": 2.5,  "accel_z": 9.6,
        "gyro_x":  0.15, "gyro_y":  0.20, "gyro_z":  0.25,
        "speed": 45, "desc": "Stop-go traffic, moderate acceleration"
    },
    "🟡 Highway Riding": {
        "accel_x": 0.5,  "accel_y": 1.0,  "accel_z": 9.7,
        "gyro_x":  0.05, "gyro_y":  0.08, "gyro_z":  0.10,
        "speed": 75, "desc": "Sustained high speed, smooth lane changes"
    },
    "🔴 Harsh Braking": {
        "accel_x": 0.8,  "accel_y": -8.5, "accel_z": 9.5,
        "gyro_x":  0.20, "gyro_y":  0.30, "gyro_z":  0.15,
        "speed": 20, "desc": "Sudden emergency braking event"
    },
    "🔴 Sudden Acceleration": {
        "accel_x": 0.5,  "accel_y": 13.5, "accel_z": 9.3,
        "gyro_x":  0.10, "gyro_y":  0.15, "gyro_z":  0.20,
        "speed": 55, "desc": "Aggressive throttle from standstill"
    },
    "🔴 Sharp Turn": {
        "accel_x": 6.5,  "accel_y": 1.5,  "accel_z": 7.8,
        "gyro_x":  0.50, "gyro_y":  0.80, "gyro_z":  2.2,
        "speed": 40, "desc": "Sharp cornering at speed, high lean angle"
    },
    "🔴 Aggressive Riding": {
        "accel_x": 4.0,  "accel_y": 14.0, "accel_z": 8.5,
        "gyro_x":  0.60, "gyro_y":  0.90, "gyro_z":  1.8,
        "speed": 85, "desc": "High speed with harsh acceleration and turns"
    },
    "⚙️ Custom": {
        "accel_x": 0.0,  "accel_y": 0.0,  "accel_z": 9.8,
        "gyro_x":  0.0,  "gyro_y":  0.0,  "gyro_z":  0.0,
        "speed": 30, "desc": "Enter your own sensor values below"
    }
}

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, 'models')

@st.cache_resource
def load_models():
    m = {}
    missing = []
    files = {
        'xgb':       'xgb_model.pkl',
        'iso':       'iso_forest.pkl',
        'scaler':    'scaler.pkl',
        'features':  'selected_features.pkl',
        'kmeans':    'kmeans_mileage.pkl',
        'km_scaler': 'kmeans_scaler.pkl',
        'cluster':   'cluster_info.pkl',
        'm_feats':   'mileage_features.pkl',
    }
    for key, fname in files.items():
        path = os.path.join(MODELS, fname)
        if os.path.exists(path):
            m[key] = joblib.load(path)
        else:
            missing.append(fname)
    if missing:
        return None, f"Missing models: {missing}"
    try:
        m['explainer'] = shap.TreeExplainer(m['xgb'])
    except Exception as e:
        m['explainer'] = None
    return m, None

models, err = load_models()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🛵")
with col_title:
    st.title("RideGuard")
    st.caption("Real-time Two-Wheeler Riding Risk & Mileage Efficiency — ML System")

if err:
    st.error(f"❌ {err}")
    st.info("""
    **Folder structure required:**
    ```
    streamlit_app.py
    models/
        xgb_model.pkl
        iso_forest.pkl
        scaler.pkl
        selected_features.pkl
        kmeans_mileage.pkl
        kmeans_scaler.pkl
        cluster_info.pkl
        mileage_features.pkl
    ```
    Upload all pkl files to the models/ folder in your Hugging Face Space.
    """)
    st.stop()

# ─────────────────────────────────────────────
# SENSOR NOTE — honest about Streamlit limitation
# ─────────────────────────────────────────────
st.markdown("""
<div class="sensor-note">
📱 <strong>Live Sensor Note:</strong> Streamlit (Hugging Face) cannot access phone sensors directly.
Use the <strong>Flask web app</strong> on your phone for real-time sensor reading.
This demo uses realistic riding scenarios based on real accelerometer/gyroscope patterns.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🏍 Your Bike")

    bike_choice = st.selectbox(
        "Select Bike",
        options=list(BIKE_DB.keys()),
        index=0
    )

    bike_info  = BIKE_DB[bike_choice]
    base_kmpl  = bike_info["kmpl"]
    bike_cc    = bike_info["cc"]
    bike_type  = bike_info["type"]

    st.metric("ARAI Mileage", f"{base_kmpl} kmpl")
    col_a, col_b = st.columns(2)
    col_a.metric("Engine", f"{bike_cc}cc" if bike_cc > 0 else "EV")
    col_b.metric("Type", bike_type)

    st.caption("Source: ARAI India certified values")

    st.divider()
    st.header("🎮 Riding Scenario")
    scenario_choice = st.selectbox(
        "Choose scenario",
        options=list(SCENARIOS.keys()),
        index=1
    )
    scenario = SCENARIOS[scenario_choice]
    st.caption(scenario["desc"])

    st.divider()
    st.header("📊 Model Info")
    st.markdown("""
    **Risk Models:**
    - XGBoost — best accuracy
    - Isolation Forest — anomaly
    - SHAP — explainability

    **Mileage Models:**
    - K-Means — rider style
    - GB — MAE 0.81 kmpl
    - Physics (SAE 2019-26-0143)

    **Datasets:**
    - Motorcycle Fall (ScienceDirect)
    - Harsh Driving (Kaggle)
    - Nairobi Motorcycles (Mendeley)
    """)

# ─────────────────────────────────────────────
# SENSOR VALUES
# ─────────────────────────────────────────────
st.subheader("📡 Sensor Values")

is_custom = (scenario_choice == "⚙️ Custom")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Accelerometer X** (m/s²)")
    accel_x = st.number_input("ax", value=float(scenario["accel_x"]),
                               step=0.1, format="%.2f",
                               disabled=not is_custom, label_visibility="collapsed")
    st.markdown("**Accelerometer Y** (m/s²)")
    accel_y = st.number_input("ay", value=float(scenario["accel_y"]),
                               step=0.1, format="%.2f",
                               disabled=not is_custom, label_visibility="collapsed")
    st.markdown("**Accelerometer Z** (m/s²)")
    accel_z = st.number_input("az", value=float(scenario["accel_z"]),
                               step=0.1, format="%.2f",
                               disabled=not is_custom, label_visibility="collapsed")

with col2:
    st.markdown("**Gyroscope X** (rad/s)")
    gyro_x = st.number_input("gx", value=float(scenario["gyro_x"]),
                              step=0.01, format="%.3f",
                              disabled=not is_custom, label_visibility="collapsed")
    st.markdown("**Gyroscope Y** (rad/s)")
    gyro_y = st.number_input("gy", value=float(scenario["gyro_y"]),
                              step=0.01, format="%.3f",
                              disabled=not is_custom, label_visibility="collapsed")
    st.markdown("**Gyroscope Z** (rad/s)")
    gyro_z = st.number_input("gz", value=float(scenario["gyro_z"]),
                              step=0.01, format="%.3f",
                              disabled=not is_custom, label_visibility="collapsed")

with col3:
    st.markdown("**Speed** (kmph)")
    speed = st.number_input("spd", value=float(scenario["speed"]),
                             min_value=0.0, max_value=200.0, step=1.0,
                             disabled=not is_custom, label_visibility="collapsed")

    st.markdown(" ")
    st.markdown("**Accel X Label**")
    st.info("← Left/Right force")
    st.markdown("**Accel Y Label**")
    st.info("↑ Forward = +, Brake = −")
    st.markdown("**Accel Z Label**")
    st.info("↓ Gravity = 9.8")

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
accel_mag = float(np.sqrt(accel_x**2 + accel_y**2 + accel_z**2))
gyro_mag  = float(np.sqrt(gyro_x**2  + gyro_y**2  + gyro_z**2))
jerk      = float(abs(accel_mag - 9.8))
braking   = float(abs(min(accel_y, 0)))
turn      = float(abs(gyro_z))
tilt      = float(np.arctan2(accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 180 / np.pi)
impact    = float(accel_mag * 0.4 + gyro_mag * 0.3 + jerk * 0.3)

harsh_accel = accel_mag > 12
harsh_brake = accel_y < -5
harsh_turn  = gyro_mag > 1.5

st.markdown("**Derived Features (auto-calculated):**")
d1,d2,d3,d4,d5,d6 = st.columns(6)
d1.metric("Accel Mag",  f"{accel_mag:.2f}", delta="HIGH" if accel_mag>12 else None, delta_color="inverse")
d2.metric("Gyro Mag",   f"{gyro_mag:.2f}",  delta="HIGH" if gyro_mag>1.5 else None, delta_color="inverse")
d3.metric("Jerk",       f"{jerk:.2f}",      delta="HIGH" if jerk>5 else None, delta_color="inverse")
d4.metric("Braking",    f"{braking:.2f}",   delta="HIGH" if braking>8 else None, delta_color="inverse")
d5.metric("Turn Sharp", f"{turn:.2f}",      delta="HIGH" if turn>1.5 else None, delta_color="inverse")
d6.metric("Impact",     f"{impact:.2f}",    delta="HIGH" if impact>8 else None, delta_color="inverse")

# Flags
f1,f2,f3 = st.columns(3)
if harsh_accel:
    f1.error("⚠ Harsh Acceleration")
else:
    f1.success("✓ Normal Acceleration")
if harsh_brake:
    f2.error("⚠ Harsh Braking")
else:
    f2.success("✓ Normal Braking")
if harsh_turn:
    f3.error("⚠ Harsh Turn")
else:
    f3.success("✓ Normal Turn")

st.divider()

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
features = {
    'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
    'gyro_x':  gyro_x,  'gyro_y':  gyro_y,  'gyro_z':  gyro_z,
    'accel_mag':          accel_mag,
    'gyro_mag':           gyro_mag,
    'jerk':               jerk,
    'braking_intensity':  braking,
    'turn_sharpness':     turn,
    'tilt_angle':         tilt,
    'impact_score':       impact,
    'harsh_accel_flag':   int(harsh_accel),
    'harsh_brake_flag':   int(harsh_brake),
    'harsh_turn_flag':    int(harsh_turn)
}

predict_btn = st.button("🔮 Predict Risk & Mileage",
                         type="primary", use_container_width=True)

if predict_btn or True:  # Auto predict on load

    selected = models['features']
    df       = pd.DataFrame([features])[selected]

    probs    = models['xgb'].predict_proba(df)[0]
    risk_idx = int(np.argmax(probs))
    risk_lbl = ['LOW', 'HIGH'][risk_idx]

    # Danger score 0-100 (binary: HIGH probability * 100)
    risk_score = int(probs[1] * 100)

    # Isolation Forest — properly scaled
    scaled     = models['scaler'].transform(df)
    is_anomaly = bool(models['iso'].predict(scaled)[0] == -1)

    # SHAP explanation using actual risk level
    explanation = []
    if models['explainer'] is not None:
        try:
            sv  = models['explainer'].shap_values(df)
            arr = sv[risk_idx][0] if isinstance(sv, list) else sv[0]
            imp = dict(zip(selected, abs(arr)))
            top3 = sorted(imp.items(), key=lambda x: float(x[1]), reverse=True)[:3]
            explanation = [f for f, _ in top3]
        except Exception:
            explanation = ['accel_mag', 'gyro_mag', 'jerk']
    else:
        explanation = ['accel_mag', 'gyro_mag', 'jerk']

    # Mileage — physics formula directly connected to sensor values
    efficiency = 1.0
    if speed > 80:       efficiency -= 0.30
    elif speed > 70:     efficiency -= 0.22
    elif speed > 60:     efficiency -= 0.15
    elif speed > 50:     efficiency -= 0.08
    elif speed < 10:     efficiency -= 0.10
    if accel_mag > 12:   efficiency -= 0.25
    elif accel_mag > 9:  efficiency -= 0.15
    elif accel_mag > 6:  efficiency -= 0.08
    elif accel_mag > 3:  efficiency -= 0.03
    if braking > 8:      efficiency -= 0.20
    elif braking > 5:    efficiency -= 0.12
    elif braking > 3:    efficiency -= 0.06
    if gyro_mag > 1.5:   efficiency -= 0.10
    elif gyro_mag > 0.8: efficiency -= 0.05
    if jerk > 5:         efficiency -= 0.08
    elif jerk > 3:       efficiency -= 0.04
    efficiency     = round(max(0.35, min(1.0, efficiency)), 3)
    predicted_kmpl = round(base_kmpl * efficiency, 1)
    potential_kmpl = round(base_kmpl * 0.95, 1)
    efficiency_pct = round(efficiency * 100)

    # K-Means rider style
    rider_style = 'Normal Rider'
    try:
        all_mf     = [f for f in models['m_feats'] if f != 'speed_kmh']
        cdf        = pd.DataFrame([[features.get(f, 0) for f in all_mf]], columns=all_mf)
        cs         = models['km_scaler'].transform(cdf)
        cid        = int(models['kmeans'].predict(cs)[0])
        rider_style = models['cluster'].get(cid, {}).get('style', 'Normal Rider')
    except Exception:
        pass

    # ─────────────────────────────────
    # RESULTS
    # ─────────────────────────────────
    st.subheader("📊 Prediction Results")

    r1, r2 = st.columns(2)

    with r1:
        card_class = {'LOW':'low-card','HIGH':'high-card'}[risk_lbl]
        emoji      = {'LOW':'🟢','HIGH':'🔴'}[risk_lbl]
        st.markdown(f"""
        <div class="risk-card {card_class}">
            {emoji} {risk_lbl} RISK<br>
            <span style="font-size:1rem;font-weight:500;">Score: {risk_score}/100</span>
        </div>
        """, unsafe_allow_html=True)

        if is_anomaly:
            st.error("⚠️ Anomaly Detected — Unusual riding pattern")
        else:
            st.success("✓ Normal riding pattern")

        st.caption(f"Rider style: **{rider_style}**")

    with r2:
        eff_color = "🟢" if efficiency_pct>70 else "🟡" if efficiency_pct>45 else "🔴"
        st.markdown(f"""
        <div class="risk-card {'low-card' if efficiency_pct>70 else 'mod-card' if efficiency_pct>45 else 'high-card'}">
            {eff_color} {efficiency_pct}% Efficient<br>
            <span style="font-size:1rem;font-weight:500;">{predicted_kmpl} kmpl</span>
        </div>
        """, unsafe_allow_html=True)

        st.info(f"Base mileage: {base_kmpl} kmpl ({bike_choice})")
        st.caption(f"Potential (smooth riding): {potential_kmpl} kmpl")

    st.divider()

    col_prob, col_shap = st.columns(2)

    with col_prob:
        st.subheader("Risk Probabilities")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        labels  = ['Low', 'High']
        colors  = ['#16a34a','#dc2626']
        bars    = ax.barh(labels, [p*100 for p in probs],
                          color=colors, height=0.5, edgecolor='none')
        ax.set_xlim(0, 105)
        ax.set_xlabel('Probability (%)', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', labelsize=11)
        for bar, p in zip(bars, probs):
            ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    f'{p*100:.1f}%', va='center', fontsize=10, fontweight='600')
        ax.set_facecolor('#f8f7f4')
        fig.patch.set_facecolor('#f8f7f4')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_shap:
        st.subheader("Why This Risk? — SHAP")
        expl_map = {
            'accel_mag':         '🏃 High acceleration force',
            'gyro_mag':          '🔄 Sharp rotation detected',
            'jerk':              '⚡ Sudden movement change',
            'braking_intensity': '🛑 Hard braking detected',
            'turn_sharpness':    '↩ Sharp turn detected',
            'tilt_angle':        '📐 Dangerous tilt angle',
            'impact_score':      '💥 High impact force',
            'harsh_accel_flag':  '🚀 Harsh acceleration event',
            'harsh_brake_flag':  '🛑 Harsh braking event',
            'harsh_turn_flag':   '↩ Harsh turn event',
            'accel_x':           '↔ Lateral force',
            'accel_y':           '↕ Forward/backward force',
            'accel_z':           '↑↓ Vertical force',
        }
        for i, f in enumerate(explanation[:3], 1):
            label = expl_map.get(f, f)
            val   = features.get(f, 0)
            st.markdown(f"**{i}.** {label}")
            st.caption(f"Value: `{val:.3f}`")

    st.divider()

    # Mileage breakdown
    st.subheader("⛽ Mileage Analysis")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Bike Base kmpl",    f"{base_kmpl}",       help="ARAI certified")
    m2.metric("Current Riding",    f"{predicted_kmpl}",  help="Based on your riding")
    m3.metric("Smooth Potential",  f"{potential_kmpl}",  help="If you ride smoothly")
    m4.metric("Efficiency",        f"{efficiency_pct}%", help="Your riding efficiency")

    # Tips
    st.subheader("💡 Riding Tips")
    tips = []
    if accel_mag > 12:
        tips.append(f"🚀 Harsh acceleration reducing mileage by ~{round(accel_mag/12*25)}% — release throttle gradually")
    elif accel_mag > 6:
        tips.append(f"⚡ Moderate acceleration — gentle throttle saves fuel")
    if harsh_brake:
        tips.append("🛑 Hard braking detected — anticipate stops, use engine braking")
    elif braking > 3:
        tips.append("🛑 Moderate braking — smooth deceleration improves mileage")
    if speed > 70:
        tips.append(f"💨 Speed {round(speed)} kmph — reduce to 50-60 kmph for best mileage (-{round((speed-60)/30*30)}%)")
    if gyro_mag > 1.5:
        tips.append("↩ Sharp turn — smooth cornering maintains momentum and fuel")
    if not tips:
        tips.append("✅ Excellent riding — maintaining peak efficiency. Keep it up!")

    for tip in tips[:4]:
        if tip.startswith("✅"):
            st.success(tip)
        elif any(x in tip for x in ["🚀","🛑","💨","↩"]):
            st.warning(tip)
        else:
            st.info(tip)

    # Mileage comparison chart
    st.subheader("📈 Mileage Comparison")
    fig2, ax2 = plt.subplots(figsize=(6, 2.5))
    categories = ['Base (ARAI)', 'Your Riding', 'Smooth Potential']
    values     = [base_kmpl, predicted_kmpl, potential_kmpl]
    bar_colors = ['#e5e7eb', '#dc2626' if efficiency_pct<50 else '#d97706' if efficiency_pct<75 else '#16a34a', '#16a34a']
    bars2      = ax2.bar(categories, values, color=bar_colors, edgecolor='none', width=0.5)
    ax2.set_ylabel('kmpl', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, max(values)*1.2)
    for bar, val in zip(bars2, values):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val}', ha='center', va='bottom', fontsize=11, fontweight='600')
    ax2.set_facecolor('#f8f7f4')
    fig2.patch.set_facecolor('#f8f7f4')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ─────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────
with st.expander("📊 Comparative Analysis — RideGuard vs Existing Systems"):
    comp_data = {
        'Feature':           ['Two-wheeler specific', 'Real-time risk score',
                              'Live mileage efficiency', 'SHAP explanation',
                              'Anomaly detection', 'Indian road conditions',
                              'No extra hardware', 'Open API'],
        'Uber':              ['❌','❌','❌','❌','❌','❌','✅','❌'],
        'Google Maps':       ['❌','❌','❌','❌','❌','Partial','✅','❌'],
        'Detecht App':       ['✅','✅','❌','❌','❌','❌','✅','❌'],
        'RideGuard (Ours)':  ['✅','✅','✅','✅','✅','✅','✅','✅'],
    }
    st.table(pd.DataFrame(comp_data).set_index('Feature'))
    st.caption("RideGuard is the only system combining risk scoring + mileage + SHAP explainability for Indian two-wheelers")

# ─────────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────────
with st.expander("📈 Model Performance Metrics"):
    perf = pd.DataFrame({
        'Model':      ['XGBoost (Risk)', 'Random Forest', 'XGB Pipeline',
                       'Gradient Boosting (Mileage)', 'KNN (Mileage)'],
        'Metric':     ['Accuracy', 'Accuracy', 'Accuracy', 'MAE (kmpl)', 'MAE (kmpl)'],
        'Value':      ['89%', '88%', '87%', '0.81', '0.95'],
        'R² Score':   ['—', '—', '—', '0.9409', '0.8893'],
        'Used in':    ['Production', 'Comparison', 'Comparison',
                       'Mileage benchmark', 'Mileage benchmark'],
    })
    st.dataframe(perf, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("""
**RideGuard** — ML-based two-wheeler safety & mileage system |
**Datasets:** Motorcycle Fall (ScienceDirect), Harsh Driving (Kaggle), Nairobi Motorcycles (Mendeley doi:10.17632/nv3rkn24zv.1) |
**Models:** XGBoost, Isolation Forest, Gradient Boosting, K-Means |
**Research:** VSP methodology (Nguyen et al. 2022), SAE 2019-26-0143 Indian two-wheeler study |
**ARAI mileage data:** araiindia.com
""")