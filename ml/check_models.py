import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))
selected_features = joblib.load(os.path.join(BASE, '../models/selected_features.pkl'))
scaler = joblib.load(os.path.join(BASE, '../models/scaler.pkl'))

print("Selected features:", selected_features)
print("Total:", len(selected_features))
print("Scaler n_features:", scaler.n_features_in_)
