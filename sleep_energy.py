import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Dataset
hours_slept = [4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9]
sleep_quality = [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
exercise_hours = [0, 0.2, 0.3, 0.5, 0.5, 0.8, 1, 1, 1.2, 1.5]
caffeine = [4, 4, 3, 3, 3, 2, 2, 1, 1, 1]
stress_level = [9, 8, 8, 7, 7, 6, 5, 4, 3, 2]
screen_time = [8, 7.5, 7, 6.5, 6, 5, 4.5, 4, 3.5, 3]

energy_scores = [35, 42, 48, 52, 58, 65, 70, 78, 84, 90]

# Create X
x = [
    [hours_slept[i],
     sleep_quality[i],
     exercise_hours[i],
     caffeine[i],
     stress_level[i],
     screen_time[i]]
    for i in range(len(hours_slept))
]

y = energy_scores

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Predicted energy scores:", predictions.round(2))
print("Actual energy scores:", y_test)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot
plt.scatter(y_test, predictions)

plt.xlabel("Actual Energy Score")
plt.ylabel("Predicted Energy Score")
plt.title("Actual vs Predicted Energy Score")

plt.show()

# Get user input
hours = float(input("Hours slept: "))
quality = float(input("Sleep quality (1-10): "))
exercise = float(input("Exercise hours: "))
caffeine = float(input("Caffeine intake (cups): "))
stress = float(input("Stress level (1-10): "))
screen = float(input("Screen time (hours): "))

# Create input for the model
user_data = [[
    hours,
    quality,
    exercise,
    caffeine,
    stress,
    screen
]]

# Predict energy score
user_prediction = model.predict(user_data)

print(f"\nPredicted Energy Score: {user_prediction[0]:.2f}")