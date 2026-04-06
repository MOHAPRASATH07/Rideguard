"""
ml/api.py — RideGuard FastAPI Service
Run: uvicorn api:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs

All bugs fixed:
1. Jerk = diff of consecutive accel_mag per session
2. Isolation Forest uses scaler.transform() before predict
3. Batch max_risk uses numeric severity not string comparison
4. risk_score = danger scale 0-100 same as Flask
5. rf_model removed — was loaded but never used
6. Docs URL is port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import shap
import os

app = FastAPI(
    title="RideGuard API",
    description="Two-wheeler riding risk and mileage prediction",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, '../models')

def load_model(f):
    p = os.path.join(MODELS, f)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Not found: {p}")
    return joblib.load(p)

try:
    xgb_model         = load_model('xgb_model.pkl')
    iso_model         = load_model('iso_forest.pkl')
    risk_scaler       = load_model('scaler.pkl')
    selected_features = load_model('selected_features.pkl')
    explainer         = shap.TreeExplainer(xgb_model)
    print("API models loaded")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    raise

# Fix 5: rf_model NOT loaded — never used

HARSH_ACCEL = 12.0
HARSH_BRAKE = -5.0
HARSH_TURN  = 1.5

# Fix 3: Numeric severity map
SEVERITY = {'LOW': 0, 'MODERATE': 1, 'HIGH': 2}

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

# Fix 1: Per-session jerk buffer
_prev_accel_mag = {}


class SensorReading(BaseModel):
    accel_x:    float
    accel_y:    float
    accel_z:    float
    gyro_x:     float
    gyro_y:     float
    gyro_z:     float
    speed:      Optional[float] = None
    session_id: Optional[str]   = 'default'


class BatchRequest(BaseModel):
    readings: List[SensorReading]


def extract_features(d: SensorReading) -> dict:
    ax = d.accel_x; ay = d.accel_y; az = d.accel_z
    gx = d.gyro_x;  gy = d.gyro_y;  gz = d.gyro_z

    accel_mag = float(np.sqrt(ax**2 + ay**2 + az**2))
    gyro_mag  = float(np.sqrt(gx**2 + gy**2 + gz**2))

    # Fix 1: Jerk = diff of consecutive — matches features.py training
    sid      = d.session_id or 'default'
    prev_mag = _prev_accel_mag.get(sid, accel_mag)
    _prev_accel_mag[sid] = accel_mag
    jerk     = float(abs(accel_mag - prev_mag))

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


def run_prediction(features: dict) -> dict:
    df    = pd.DataFrame([features])[selected_features]
    probs = xgb_model.predict_proba(df)[0]
    n_classes = len(probs)

    risk_idx   = int(np.argmax(probs))
    labels     = ['LOW', 'MODERATE', 'HIGH'][:n_classes]
    risk_label = labels[risk_idx]

    # Danger scale 0-100 — handle 2 or 3 class models
    if n_classes == 2:
        risk_score = int(probs[0] * 16 + probs[1] * 84)
    else:
        risk_score = int(probs[0] * 16 + probs[1] * 50 + probs[2] * 84)  # n_classes == 3

    # Scale before Isolation Forest
    scaled     = risk_scaler.transform(df)
    is_anomaly = bool(iso_model.predict(scaled)[0] == -1)

    try:
        sv = explainer.shap_values(df)
        # sv shape: (1, n_features) for single sample — always use sv[0]
        arr = sv[0]
        imp = dict(zip(selected_features, abs(arr)))
        top3 = sorted(imp.items(), key=lambda x: float(x[1]), reverse=True)[:3]
        explanation = [EXPLANATIONS.get(f, f) for f, _ in top3]
    except Exception:
        explanation = ['Sensor activity detected', 'Pattern analyzed', 'Risk assessed']

    return {
        'risk_level':    risk_label,
        'risk_score':    risk_score,
        'is_anomaly':    is_anomaly,
        'explanation':   explanation,
        'probabilities': {
            'low':      round(float(probs[0]), 3),
            'moderate': round(float(probs[1]), 3),
            'high':     round(float(probs[n_classes - 1]), 3) if n_classes == 3 else 0.0
        }
    }


@app.get("/")
def root():
    return {"service": "RideGuard API", "version": "2.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "models": "loaded"}


@app.post("/predict")
def predict(reading: SensorReading):
    try:
        features = extract_features(reading)
        result   = run_prediction(features)
        speed    = reading.speed if reading.speed is not None else 30.0
        result['speed_used']      = speed
        result['speed_estimated'] = reading.speed is None
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    if not batch.readings:
        raise HTTPException(status_code=400, detail="No readings")

    results = []
    for r in batch.readings:
        results.append(run_prediction(extract_features(r)))

    # Fix 3: Numeric severity — not string comparison
    max_risk  = max(results, key=lambda x: SEVERITY.get(x['risk_level'], 0))['risk_level']
    avg_score = round(sum(r['risk_score'] for r in results) / len(results))

    return {
        'count':          len(results),
        'max_risk':       max_risk,
        'avg_risk_score': avg_score,
        'anomaly_count':  sum(1 for r in results if r['is_anomaly']),
        'results':        results
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)