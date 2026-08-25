import tkinter as tk
import pandas as pd
import joblib


# Load model and scaler
model = joblib.load("energy_model.pkl")
scaler = joblib.load("energy_scaler.pkl")


# -----------------------------
# Prediction function
# -----------------------------

def predict_energy():

    try:
        hours = float(hours_entry.get())
        quality = float(quality_entry.get())
        exercise = float(exercise_entry.get())
        caffeine = float(caffeine_entry.get())
        stress = float(stress_entry.get())
        screen = float(screen_entry.get())

        # Validate inputs
        if not 0 <= hours <= 24:
            raise ValueError("Hours slept must be between 0 and 24.")

        if not 1 <= quality <= 10:
            raise ValueError("Sleep quality must be between 1 and 10.")

        if not 0 <= exercise <= 24:
            raise ValueError("Exercise hours must be between 0 and 24.")

        if not 0 <= caffeine <= 20:
            raise ValueError("Caffeine must be between 0 and 20 cups.")

        if not 1 <= stress <= 10:
            raise ValueError("Stress level must be between 1 and 10.")

        if not 0 <= screen <= 24:
            raise ValueError("Screen time must be between 0 and 24 hours.")

        # Create DataFrame
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

        # Predict
        prediction = model.predict(user_data_scaled)[0]

        # Keep score between 0 and 100
        prediction = max(0, min(100, prediction))

        # Determine energy level
        if prediction >= 70:
            level = "High Energy ⚡"
        elif prediction >= 40:
            level = "Moderate Energy 🙂"
        else:
            level = "Low Energy 😴"

        # Display result
        result_label.config(
            text=f"Energy Score: {prediction:.2f}"
        )

        level_label.config(
            text=level
        )

    except ValueError as error:

        result_label.config(
            text=f"Error: {error}"
        )

        level_label.config(text="")


# -----------------------------
# Clear function
# -----------------------------

def clear_fields():

    hours_entry.delete(0, tk.END)
    quality_entry.delete(0, tk.END)
    exercise_entry.delete(0, tk.END)
    caffeine_entry.delete(0, tk.END)
    stress_entry.delete(0, tk.END)
    screen_entry.delete(0, tk.END)

    result_label.config(
        text="Energy Score: --"
    )

    level_label.config(text="")


# -----------------------------
# GUI
# -----------------------------

window = tk.Tk()

window.title("Energy Score Predictor")
window.geometry("450x650")


# Title
title = tk.Label(
    window,
    text="Energy Score Predictor",
    font=("Arial", 22, "bold")
)

title.pack(pady=20)


subtitle = tk.Label(
    window,
    text="Enter your daily lifestyle information",
    font=("Arial", 11)
)

subtitle.pack(pady=5)


# -----------------------------
# Input fields
# -----------------------------

def create_input(label_text):

    tk.Label(
        window,
        text=label_text,
        font=("Arial", 11)
    ).pack(pady=(10, 2))

    entry = tk.Entry(
        window,
        width=25,
        font=("Arial", 11)
    )

    entry.pack()

    return entry


hours_entry = create_input("Hours slept")

quality_entry = create_input("Sleep quality (1-10)")

exercise_entry = create_input("Exercise hours")

caffeine_entry = create_input("Caffeine intake (cups)")

stress_entry = create_input("Stress level (1-10)")

screen_entry = create_input("Screen time (hours)")


# -----------------------------
# Buttons
# -----------------------------

button_frame = tk.Frame(window)

button_frame.pack(pady=25)


predict_button = tk.Button(
    button_frame,
    text="Predict Energy",
    command=predict_energy,
    width=15
)

predict_button.grid(row=0, column=0, padx=5)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_fields,
    width=10
)

clear_button.grid(row=0, column=1, padx=5)


# -----------------------------
# Result
# -----------------------------

result_label = tk.Label(
    window,
    text="Energy Score: --",
    font=("Arial", 18, "bold")
)

result_label.pack(pady=10)


level_label = tk.Label(
    window,
    text="",
    font=("Arial", 15)
)

level_label.pack(pady=5)


window.mainloop()