import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "energy_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "energy_scaler.pkl"

# Load trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# Ask user for information
print("=== Energy Score Predictor ===\n")

hours = float(input("Hours slept: "))
quality = float(input("Sleep quality (1-10): "))
exercise = float(input("Exercise hours: "))
caffeine = float(input("Caffeine intake (cups): "))
stress = float(input("Stress level (1-10): "))
screen = float(input("Screen time (hours): "))


# Create input data
user_data = pd.DataFrame([[
    hours,
    quality,
    exercise,
    caffeine,
    stress,
    screen
]], columns=[
    "hours_slept",
    "sleep_quality",
    "exercise_hours",
    "caffeine",
    "stress_level",
    "screen_time"
])


# Scale input
user_data_scaled = scaler.transform(user_data)


# Make prediction
prediction = model.predict(user_data_scaled)


print(f"\nPredicted Energy Score: {prediction[0]:.2f}")