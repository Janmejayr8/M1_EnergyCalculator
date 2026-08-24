**Energy Calculator:**
A machine learning model that predicts a person's daily energy score based on lifestyle habits — sleep, exercise, caffeine intake, stress, and screen time.

**Overview:**
This project uses a linear regression model (scikit-learn) trained on lifestyle data to predict an energy_score output. The goal was to explore how everyday habits statistically relate to perceived energy levels, and to practice the full ML workflow: data generation, preprocessing, training, and evaluation.

**Features Used:**
hours_slept
sleep_quality
exercise_hours
caffeine
stress_level
screen_time
Tech Stack
Python
pandas
scikit-learn

**Results**
R² Score: 0.95
Evaluated using an 80/20 train-test split with Mean Squared Error and R² as metrics

**Project Structure:**
├── generate_data.py         # Generates the synthetic dataset
├── sleep_energy_data.csv    # Dataset used for training/testing
├── sleep_energy_model.py    # Trains and evaluates the regression model
└── sleep_energy.py          # (add a short note on what this script does)

**How to Run:**
pip install pandas scikit-learn
python generate_data.py       # generates the dataset (if not already present)
python sleep_energy_model.py  # trains the model and prints evaluation results

**What I Learned:**
Building an end-to-end ML pipeline: data prep → training → evaluation
Interpreting model coefficients to understand feature importance
Evaluating regression models using R² and MSE
Future Improvements
Replace synthetic data with real, self-tracked data
Try additional models (Random Forest, Gradient Boosting) for comparison
Build a simple interface (CLI or web app) to input daily habits and get a live prediction
