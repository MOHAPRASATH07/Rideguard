"""
flask_app/app.py — RideGuard Flask Application

All plot holes fixed:
1. Jerk matches training definition (diff of consecutive accel_mag)
2. Isolation Forest uses scaled features (matches training)
3. risk_score is danger scale 0-100 (p_low×16 + p_mod×50 + p_high×84)
4. Per-session sliding window (no shared global buffer)
5. Session buffer eviction (1 hour TTL)
6. SHAP uses actual predicted class index not hardcoded HIGH
7. Speed optional — defaults to 30 if absent
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import shap
import os
import socket
import time
from threading import Lock
from collections import deque

app = Flask(__name__, static_folder='static')

BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, '../models')

# ─────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────
def load_model(filename):
    path = os.path.join(MODELS, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

try:
    xgb_model         = load_model('xgb_model.pkl')
    iso_model         = load_model('iso_forest.pkl')
    risk_scaler       = load_model('scaler.pkl')
    selected_features = load_model('selected_features.pkl')
    gb_mileage        = load_model('gb_mileage.pkl')
    mileage_features  = load_model('mileage_features.pkl')
    bike_db           = load_model('bike_db.pkl')
    cluster_info      = load_model('cluster_info.pkl')
    kmeans_mileage    = load_model('kmeans_mileage.pkl')
    kmeans_scaler     = load_model('kmeans_scaler.pkl')
    explainer         = shap.TreeExplainer(xgb_model)
    print("All models loaded successfully")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Run train_model.py and train_mileage.py first")
    exit()

# ─────────────────────────────────
# PER-SESSION BUFFERS
# Fix: No shared global buffer
# Each browser tab gets own window
# ─────────────────────────────────
_buffers      = {}
_buffer_times = {}
_jerk_last    = {}   # Fix 1: last accel_mag per session for jerk diff
_lock         = Lock()
BUFFER_TTL    = 3600  # evict sessions idle > 1 hour

def get_session_buffer(session_id):
    """Get or create per-session deque. Evict stale sessions."""
    now = time.time()
    with _lock:
        # Evict old sessions
        stale = [k for k, t in _buffer_times.items()
                 if now - t > BUFFER_TTL]
        for k in stale:
            _buffers.pop(k, None)
            _buffer_times.pop(k, None)
            _jerk_last.pop(k, None)
        # Create if new
        if session_id not in _buffers:
            _buffers[session_id] = deque(maxlen=5)
        _buffer_times[session_id] = now
        return _buffers[session_id]

# ─────────────────────────────────
# THRESHOLDS — Two-Wheeler Specific
# Source: SAE 2019-26-0143
# Indian urban two-wheeler study
# ─────────────────────────────────
HARSH_ACCEL = 12.0   # m/s²
HARSH_BRAKE = -5.0   # m/s²
HARSH_TURN  = 1.5    # rad/s

EXPLANATIONS = {
    'accel_mag':         'High acceleration force detected',
    'gyro_mag':          'Sharp rotation detected',
    'jerk':              'Sudden movement change detected',
    'braking_intensity': 'Hard braking detected',
    'turn_sharpness':    'Sharp turn detected',
    'tilt_angle':        'Dangerous tilt angle detected',
    'impact_score':      'High impact force detected',
    'harsh_accel_flag':  'Harsh acceleration event',
    'harsh_brake_flag':  'Harsh braking event',
    'harsh_turn_flag':   'Harsh turn event',
    'accel_x':           'Lateral force detected',
    'accel_y':           'Forward/backward force detected',
    'accel_z':           'Vertical force detected',
    'gyro_x':            'Roll rotation detected',
    'gyro_y':            'Pitch rotation detected',
    'gyro_z':            'Yaw rotation detected'
}

# ─────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────
def validate_input(data):
    required = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']
    for field in required:
        if field not in data:
            return False, f"Missing field: {field}"
        try:
            val = float(data[field])
            if not np.isfinite(val):
                return False, f"Invalid value: {field}"
        except (TypeError, ValueError):
            return False, f"Cannot parse: {field}"
    return True, "OK"

# ─────────────────────────────────
# FEATURE EXTRACTION
# Fix 1: jerk = diff of consecutive
# accel_mag — matches features.py
# training definition exactly
# ─────────────────────────────────
def extract_features(d, session_id='default'):
    ax = float(d['accel_x'])
    ay = float(d['accel_y'])
    az = float(d['accel_z'])
    gx = float(d['gyro_x'])
    gy = float(d['gyro_y'])
    gz = float(d['gyro_z'])

    accel_mag = float(np.sqrt(ax**2 + ay**2 + az**2))
    gyro_mag  = float(np.sqrt(gx**2 + gy**2 + gz**2))

    # JERK — matches features.py training definition
    # diff of consecutive accel_mag values per session
    with _lock:
        prev_mag = _jerk_last.get(session_id, accel_mag)
        _jerk_last[session_id] = accel_mag
    jerk = float(abs(accel_mag - prev_mag))

    braking = float(abs(min(ay, 0)))
    turn    = float(abs(gz))
    tilt    = float(np.arctan2(ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi)
    impact  = float(accel_mag * 0.4 + gyro_mag * 0.3 + jerk * 0.3)

    return {
        'accel_x': ax, 'accel_y': ay, 'accel_z': az,
        'gyro_x':  gx, 'gyro_y':  gy, 'gyro_z':  gz,
        'accel_mag':          accel_mag,
        'gyro_mag':           gyro_mag,
        'jerk':               jerk,
        'braking_intensity':  braking,
        'turn_sharpness':     turn,
        'tilt_angle':         tilt,
        'impact_score':       impact,
        'harsh_accel_flag':   int(accel_mag > HARSH_ACCEL),
        'harsh_brake_flag':   int(ay < HARSH_BRAKE),
        'harsh_turn_flag':    int(gyro_mag > HARSH_TURN)
    }

# ─────────────────────────────────
# RISK PREDICTION
# Fix 2: ISO Forest uses scaled features
# Fix 3: risk_score is danger scale
# Fix 4: SHAP uses actual risk_level
# ─────────────────────────────────
def predict_risk(features, session_id='default'):
    buf = get_session_buffer(session_id)
    buf.append(features)

    all_probs   = []
    all_anomaly = []

    for r in buf:
        df     = pd.DataFrame([r])[selected_features]
        probs  = xgb_model.predict_proba(df)[0]

        # Fix 2: Scale before ISO Forest — matches training
        scaled = risk_scaler.transform(df)
        is_ano = iso_model.predict(scaled)[0] == -1

        all_probs.append(probs)
        all_anomaly.append(is_ano)

    avg_probs  = np.mean(all_probs, axis=0)
    risk_level = int(np.argmax(avg_probs))

    # Fix 3: Danger scale 0-100 handling 2 or 3 classes
    p = avg_probs
    n_classes = len(p)
    if n_classes == 2:
        # Scale for binary: LOW (0) to MODERATE (1)
        risk_score = int(p[0] * 16 + p[1] * 84)
    else:
        # Scale for multi-class: LOW (0), MODERATE (1), HIGH (2)
        risk_score = int(p[0] * 16 + p[1] * 50 + p[2] * 84)

    # Majority vote anomaly
    is_anomaly = sum(all_anomaly) > len(buf) // 2

    return risk_level, risk_score, avg_probs, is_anomaly, len(buf)

# ─────────────────────────────────
# SHAP EXPLANATION
# Fix 4: Use actual risk_level index
# not hardcoded HIGH (index 2)
# ─────────────────────────────────
def get_explanation(features, risk_level=2):
    try:
        df  = pd.DataFrame([features])[selected_features]
        sv  = explainer.shap_values(df)

        # Use actual predicted class index
        if isinstance(sv, list):
            arr = sv[risk_level][0]
        else:
            arr = sv[0]

        imp  = dict(zip(selected_features, abs(arr)))
        top3 = sorted(imp.items(), key=lambda x: float(x[1]), reverse=True)[:3]
        return [EXPLANATIONS.get(f, f) for f, _ in top3]
    except Exception:
        return ['Sensor activity detected',
                'Riding pattern analyzed',
                'Risk assessed']

# ─────────────────────────────────
# MILEAGE PREDICTION
# Physics-informed efficiency
# Source: SAE 2019-26-0143
# K-Means for rider style
# ─────────────────────────────────
def predict_mileage(features, speed, bike_name, base_kmpl_user=None):
    if base_kmpl_user and float(base_kmpl_user) > 0:
        base_kmpl = float(base_kmpl_user)
    else:
        bike_info = bike_db.get(bike_name, {})
        base_kmpl = float(bike_info.get('base_kmpl', 45))

    efficiency = 1.0
    accel_mag  = features.get('accel_mag', 0)
    braking    = features.get('braking_intensity', 0)
    gyro_mag   = features.get('gyro_mag', 0)
    jerk       = features.get('jerk', 0)

    # Speed penalty
    if speed > 80:       efficiency -= 0.30
    elif speed > 70:     efficiency -= 0.22
    elif speed > 60:     efficiency -= 0.15
    elif speed > 50:     efficiency -= 0.08
    elif speed < 10:     efficiency -= 0.10

    # Acceleration penalty
    if accel_mag > 12:   efficiency -= 0.25
    elif accel_mag > 9:  efficiency -= 0.15
    elif accel_mag > 6:  efficiency -= 0.08
    elif accel_mag > 3:  efficiency -= 0.03

    # Braking penalty
    if braking > 8:      efficiency -= 0.20
    elif braking > 5:    efficiency -= 0.12
    elif braking > 3:    efficiency -= 0.06

    # Turn penalty
    if gyro_mag > 1.5:   efficiency -= 0.10
    elif gyro_mag > 0.8: efficiency -= 0.05

    # Jerk penalty
    if jerk > 5:         efficiency -= 0.08
    elif jerk > 3:       efficiency -= 0.04

    efficiency     = round(max(0.35, min(1.0, efficiency)), 3)
    predicted_kmpl = round(base_kmpl * efficiency, 1)
    potential_kmpl = round(base_kmpl * 0.95, 1)
    efficiency_pct = round(efficiency * 100)

    # Rider style from K-Means
    try:
        cluster_cols = [f for f in mileage_features
                        if f != 'speed_kmh' and f in features]
        if len(cluster_cols) >= 3:
            cluster_df = pd.DataFrame(
                [[features[f] for f in cluster_cols]],
                columns=cluster_cols
            )
            for col in mileage_features:
                if col not in cluster_df.columns and col != 'speed_kmh':
                    cluster_df[col] = 0.0
            cluster_scaled = kmeans_scaler.transform(
                cluster_df[[c for c in mileage_features if c != 'speed_kmh']]
            )
            cluster_id  = int(kmeans_mileage.predict(cluster_scaled)[0])
            rider_style = cluster_info.get(
                cluster_id, {}
            ).get('style', 'Normal Rider')
        else:
            rider_style = 'Normal Rider'
    except Exception:
        rider_style = 'Normal Rider'

    # Tips ranked by impact
    penalties = []
    if accel_mag > 6:
        penalties.append(('Harsh acceleration',
                          round(min(accel_mag / 12, 1) * 25)))
    if braking > 3:
        penalties.append(('Hard braking',
                          round(min(braking / 8, 1) * 20)))
    if speed > 60:
        penalties.append((f'High speed ({round(speed)} kmph)',
                          round(min((speed - 50) / 30, 1) * 15)))
    if gyro_mag > 0.8:
        penalties.append(('Sharp turns',
                          round(min(gyro_mag / 1.5, 1) * 10)))

    penalties.sort(key=lambda x: x[1], reverse=True)
    tips = [f'{n} — reducing mileage by ~{p}%'
            for n, p in penalties[:2]]
    if not tips:
        tips = ['Smooth riding — maintaining peak efficiency']

    return {
        'predicted_kmpl': predicted_kmpl,
        'potential_kmpl': potential_kmpl,
        'base_kmpl':      base_kmpl,
        'efficiency_pct': efficiency_pct,
        'rider_style':    rider_style,
        'tips':           tips,
        'model_used':     'Physics-informed (SAE 2019-26-0143) + K-Means',
        'data_source':    'Nairobi Motorcycle Dataset + Indian research'
    }

# ─────────────────────────────────
# HELPER — local IP for startup
# ─────────────────────────────────
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

# ─────────────────────────────────
# ROUTES
# ─────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/health')
def health():
    return jsonify({
        'status':  'ok',
        'models':  'loaded',
        'version': '2.0'
    })

@app.route('/bikes')
def get_bikes():
    return jsonify(list(bike_db.keys()))

@app.route('/reset', methods=['POST'])
def reset():
    data       = request.json or {}
    session_id = data.get('session_id', 'default')
    with _lock:
        if session_id in _buffers:
            _buffers[session_id].clear()
        _jerk_last.pop(session_id, None)
    return jsonify({'status': 'ok', 'session_id': session_id})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    valid, msg = validate_input(data)
    if not valid:
        return jsonify({'error': msg}), 400

    session_id     = data.get('session_id', 'default')
    bike_name      = data.get('bike', 'My Bike')
    base_kmpl_user = float(data.get('base_kmpl', 0))

    # Fix 7: Speed optional
    speed_raw      = data.get('speed', None)
    if speed_raw is None:
        speed          = 30.0
        speed_estimated = True
    else:
        speed          = max(0, min(float(speed_raw), 200))
        speed_estimated = False

    features = extract_features(data, session_id)

    risk_level, risk_score, avg_probs, is_anomaly, window_size = \
        predict_risk(features, session_id)

    mileage     = predict_mileage(
        features, speed, bike_name, base_kmpl_user
    )
    # Fix 4: Pass actual risk_level to SHAP
    explanation = get_explanation(features, risk_level)

    labels = {0: 'LOW', 1: 'MODERATE', 2: 'HIGH'}

    return jsonify({
        'risk_level':      labels[risk_level],
        'risk_score':      risk_score,       # danger scale 0-100
        'risk_color':      ['green','yellow','red'][risk_level],
        'is_anomaly':      bool(is_anomaly),
        'explanation':     explanation,
        'window_size':     window_size,
        'speed_estimated': speed_estimated,
        'probabilities': {
            'low':      round(float(avg_probs[0]), 3),
            'moderate': round(float(avg_probs[1]), 3),
            'high':     round(float(avg_probs[2]), 3) if len(avg_probs) == 3 else 0.0
        },
        'mileage': mileage
    })

if __name__ == '__main__':
    ip = get_local_ip()
    use_https = True
    try:
        import OpenSSL
    except ImportError:
        use_https = False

    proto = 'https' if use_https else 'http'
    print("=" * 50)
    print("RideGuard Flask App")
    print(f"Local  : {proto}://127.0.0.1:5000")
    print(f"Network: {proto}://{ip}:5000")
    print("=" * 50)

    ssl_ctx = 'adhoc' if use_https else None
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        ssl_context=ssl_ctx
    )