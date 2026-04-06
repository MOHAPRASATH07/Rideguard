"""
Build bike_db.pkl — database of Indian two-wheelers with base ARAI mileage.
Run from project root: python scripts/create_bike_db.py
"""

import os
import joblib

BIKE_DB = {
    "Honda Activa":    {"base_kmpl": 45, "model_year": 2018},
    "Bajaj Pulsar":    {"base_kmpl": 38, "model_year": 2017},
    "TVS Apache":      {"base_kmpl": 40, "model_year": 2019},
    "Royal Enfield Classic": {"base_kmpl": 30, "model_year": 2018},
    "Honda CB Shine":  {"base_kmpl": 50, "model_year": 2017},
    "Bajaj Platina":   {"base_kmpl": 55, "model_year": 2016},
    "TVS Jupiter":     {"base_kmpl": 45, "model_year": 2018},
    "Hero Splendor":   {"base_kmpl": 50, "model_year": 2017},
    "Hero HF Deluxe":  {"base_kmpl": 50, "model_year": 2016},
    "Bajaj CT 100":    {"base_kmpl": 55, "model_year": 2015},
    "Yamaha FZ":       {"base_kmpl": 40, "model_year": 2019},
    "Suzuki Gixxer":   {"base_kmpl": 40, "model_year": 2019},
    "KTM Duke 200":   {"base_kmpl": 28, "model_year": 2020},
    "Jawa 42":         {"base_kmpl": 35, "model_year": 2019},
    "Honda Dio":       {"base_kmpl": 40, "model_year": 2018},
    "TVS NTORQ":       {"base_kmpl": 38, "model_year": 2019},
    "Yamaha Fascino":  {"base_kmpl": 45, "model_year": 2018},
    "Aprilia SR 150":  {"base_kmpl": 35, "model_year": 2018},
    "TVS XL 100":      {"base_kmpl": 50, "model_year": 2017},
    "Bajaj Avenger":   {"base_kmpl": 35, "model_year": 2017},
    "Honda CBR":       {"base_kmpl": 28, "model_year": 2020},
    "KTM RC 200":     {"base_kmpl": 26, "model_year": 2020},
    "Bajaj Dominar":   {"base_kmpl": 30, "model_year": 2019},
    "Royal Enfield Thunderbird": {"base_kmpl": 30, "model_year": 2018},
    "Yamaha R15":      {"base_kmpl": 28, "model_year": 2020},
}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, "models", "bike_db.pkl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(BIKE_DB, output_path)
    print(f"Saved {len(BIKE_DB)} bikes to {output_path}")
    print("\nBikes in database:")
    for name, info in sorted(BIKE_DB.items()):
        print(f"  {name}: {info['base_kmpl']} kmpl ({info['model_year']})")

if __name__ == "__main__":
    main()
