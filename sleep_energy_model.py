import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("sleep_energy_data.csv")


# Features (X)
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


# Target (y)
y = df["energy_score"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

# See what the model learned
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance:")
print(coefficients)

# Make predictions
predictions = model.predict(X_test_scaled)


# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("Predicted:", predictions[:10].round(2))
print("Actual:   ", y_test.values[:10])

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")