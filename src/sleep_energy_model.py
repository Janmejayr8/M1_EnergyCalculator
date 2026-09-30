import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


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
# This is the model currently used by the prediction app.
final_model = model
model.fit(X_train_scaled, y_train)

# Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Baseline model for comparison
baseline_model = DummyRegressor(strategy="mean")
baseline_model.fit(X_train, y_train)


# =========================
# Make predictions
# =========================

predictions = model.predict(X_test_scaled)

rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)
baseline_predictions = baseline_model.predict(X_test)

# =========================
# Evaluate model
# =========================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = mean_squared_error(y_test, rf_predictions) ** 0.5
rf_r2 = r2_score(y_test, rf_predictions)

# Evaluate Gradient Boosting
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_rmse = mean_squared_error(y_test, gb_predictions) ** 0.5
gb_r2 = r2_score(y_test, gb_predictions)

# Evaluate Baseline
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
baseline_rmse = mean_squared_error(y_test, baseline_predictions) ** 0.5
baseline_r2 = r2_score(y_test, baseline_predictions)

comparison = pd.DataFrame({
    "Model": ["Baseline", "Linear Regression", "Random Forest", "Gradient Boosting"],
    "MAE": [baseline_mae, mae, rf_mae, gb_mae],
    "MSE": [baseline_mse, mse, rf_mse, gb_mse],
    "RMSE": [baseline_rmse, rmse, rf_rmse, gb_rmse],
    "R²": [baseline_r2, r2, rf_r2, gb_r2]
})

print("\nModel Comparison:")
print(comparison.round(2))


plt.figure(figsize=(8, 5))

plt.scatter(y_test, predictions)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Energy Score")
plt.ylabel("Predicted Energy Score")
plt.title("Actual vs Predicted Energy Scores")

plt.savefig(BASE_DIR / "visual" / "actual_vs_predicted.png")

plt.show()

# =========================
# Feature coefficients
# =========================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nLinear Regression Coefficients:")
print(coefficients)


# =========================
# Save model and scaler
# =========================

joblib.dump(final_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
