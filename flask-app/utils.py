import numpy as np
from joblib import load
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = load(BASE_DIR / "models" / "gaming_vs_academic" / "svm_gaming_academic_linear_42.joblib")
scaler = load(BASE_DIR / "models" / "gaming_vs_academic" / "scaler_gaming_academic.joblib")

def preprocess_input(data):
    data["age"] = min(data["age"], 24)
    num_features = np.array([[
        data["age"],
        data["gaming_hours"],
        data["study_hours"],
        data["sleep_hours"],
        data["attendance"],
        data["social_activity"],
        data["device_usage"],
        data["reaction_time_ms"],
        data["addiction_score"]
    ]])
    
    num_scaled = scaler.transform(num_features)[0]
    
    # One-Hot para gender
    gender_Female = 1 if data["gender"] == "Female" else 0
    gender_Male   = 1 if data["gender"] == "Male" else 0
    gender_Other  = 1 if data["gender"] == "Other" else 0

    # One-Hot para gaming_genre
    genre_Casual = 1 if data["gaming_genre"] == "Casual" else 0
    genre_FPS    = 1 if data["gaming_genre"] == "FPS" else 0
    genre_RPG    = 1 if data["gaming_genre"] == "RPG" else 0

    # One-Hot para stress_level
    stress_High   = 1 if data["stress_level"] == "High" else 0
    stress_Low    = 1 if data["stress_level"] == "Low" else 0
    stress_Medium = 1 if data["stress_level"] == "Medium" else 0

    features = np.concatenate([
        num_scaled,
        [gender_Female, gender_Male, gender_Other,
         genre_Casual, genre_FPS, genre_RPG,
         stress_High, stress_Low, stress_Medium]
    ])
    
    return features.reshape(1, -1)

def predict(data):
    features = preprocess_input(data)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]
    percentage = round(probability * 100, 1)
    label = "Aprobado ✅" if prediction == 1 else "Suspendido ❌"
    return label, percentage