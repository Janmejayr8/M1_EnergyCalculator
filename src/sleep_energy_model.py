import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# File paths
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "sleep_energy_data.csv"
MODEL_PATH = BASE_DIR / "models" / "energy_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "energy_scaler.pkl"


# =========================
# Load dataset
# =========================

df = pd.read_csv(DATA_PATH)


# =========================
# Features and target
# =========================

X = df[
    [
        "hours_slept",
        "sleep_quality",
        "exercise_hours",
        "caffeine",
        "stress_level",
        "screen_time"
    ]
]

y = df["energy_score"]


# =========================
# Split dataset
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# Scale features
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# Create and train model
# =========================

model = LinearRegression()

model.fit(X_train_scaled, y_train)


# =========================
# Make predictions
# =========================

predictions = model.predict(X_test_scaled)


# =========================
# Evaluate model
# =========================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)


# =========================
# Feature coefficients
# =========================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance:")
print(coefficients)


# =========================
# Evaluation results
# =========================

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# =========================
# Save model and scaler
# =========================

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
